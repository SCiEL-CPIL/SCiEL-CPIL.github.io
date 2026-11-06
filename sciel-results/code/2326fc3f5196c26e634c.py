import time
import random
import copy

def _arr_to_mask(arr):
    """Convert a list of 1..512 values to a Python int bitmask (bit i = value i+1)."""
    mask = 0
    for x in arr:
        mask |= (1 << x)
    return mask

def _mask_to_arr(mask):
    """Convert a Python int bitmask back to sorted list of values."""
    arr = []
    x = 1
    while x <= 512:
        if mask & (1 << x):
            arr.append(x)
        x += 1
    return arr

def _score_bitset(mask):
    """Count distinct d>0 such that exists a with a, a+d, a+2d all in set."""
    count = 0
    for d in range(1, 257):
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            count += 1
    return count

def _local_search(arr, deadline, rng):
    """
    Perform local search to improve the set.
    Uses iterative single-swap hill climbing with random restarts if stuck.
    """
    cur_arr = list(arr)
    cur_mask = _arr_to_mask(cur_arr)
    cur_score = _score_bitset(cur_mask)
    best_arr = cur_arr[:]
    best_score = cur_score
    
    n = 32
    # We will do full passes of single-swap improvements
    # A full pass is 32 * 512 checks.
    
    while time.time() < deadline:
        improved = False
        # Shuffle order to break symmetry and avoid local minima traps in ordering
        indices = list(range(n))
        rng.shuffle(indices)
        
        for idx in indices:
            old = cur_arr[idx]
            # Try to find a better replacement
            # Randomize the order of new values to explore different directions
            new_values = list(range(1, 513))
            rng.shuffle(new_values)
            
            for new in new_values:
                if new == old:
                    continue
                if (cur_mask >> new) & 1:
                    continue
                
                new_mask = cur_mask & ~(1 << old) | (1 << new)
                trial_score = _score_bitset(new_mask)
                
                if trial_score > cur_score:
                    cur_mask = new_mask
                    cur_arr[idx] = new
                    cur_arr.sort()
                    cur_score = trial_score
                    improved = True
                    if cur_score > best_score:
                        best_score = cur_score
                        best_arr = cur_arr[:]
                    break # Restart the index loop or just move to next index? 
                    # Usually, restarting the whole pass or continuing is fine.
                    # Continuing to next index is faster.
            
            if time.time() > deadline:
                break
        
        if not improved:
            # Stuck in local optimum. 
            # Option 1: Stop.
            # Option 2: Random restart (perturb and try again).
            # Given the ruggedness, a small random perturbation might help escape.
            if rng.random() < 0.1: # 10% chance to perturb
                # Pick 2-3 random elements and change them
                for _ in range(3):
                    idx = rng.randrange(n)
                    old = cur_arr[idx]
                    new = rng.randint(1, 512)
                    if new == old or (cur_mask >> new) & 1:
                        continue
                    cur_mask = cur_mask & ~(1 << old) | (1 << new)
                    cur_arr[idx] = new
                    cur_arr.sort()
                    cur_score = _score_bitset(cur_mask)
            else:
                break # No improvement and no perturbation chosen, so stop
            
    return best_arr, best_score

def _crossover(parent1, parent2, rng):
    """
    Uniform crossover: pick each element from parent1 or parent2.
    Ensure result is a valid set of 32 distinct elements.
    """
    child_set = set()
    for i in range(32):
        if rng.random() < 0.5:
            child_set.add(parent1[i])
        else:
            child_set.add(parent2[i])
    
    # If we have duplicates or less than 32, fill randomly
    if len(child_set) < 32:
        # Fill with random elements
        while len(child_set) < 32:
            child_set.add(rng.randint(1, 512))
    
    # If we have more than 32 (impossible with set of 32 picks, but just in case logic errors)
    # Actually, set of 32 picks from two sets of 32 can result in fewer than 32 unique elements.
    # It cannot result in more than 32 unique elements because we only pick 32 times.
    # Wait, if parent1 and parent2 are different, the union can be up to 64.
    # But we are picking 32 times. So the set size is at most 32.
    # If size < 32, we need to add more.
    
    child_list = sorted(list(child_set))
    return child_list

def _mutate(arr, rng, num_mutations=2):
    """
    Randomly replace 'num_mutations' elements with new random values.
    """
    arr = list(arr)
    for _ in range(num_mutations):
        idx = rng.randrange(32)
        old = arr[idx]
        new = rng.randint(1, 512)
        if new == old:
            continue
        # Ensure new is not already in the set (except at idx)
        if new in arr and arr.index(new) != idx:
            continue
        arr[idx] = new
        arr.sort()
    return arr

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    rng = random.Random(42)
    
    # Initial population
    pop_size = 30
    population = []
    scores = []
    
    # Generate diverse initial individuals
    for i in range(pop_size):
        if i < 5:
            # Dense intervals
            start = rng.randint(1, 481)
            ind = list(range(start, start + 32))
        elif i < 10:
            # Structured (arithmetic progression based)
            d = rng.randint(1, 15)
            max_start = 512 - 31 * d
            if max_start < 1:
                d = 1
                max_start = 481
            start = rng.randint(1, max_start)
            ind = list(range(start, start + 32 * d, d))[:32]
            # If length < 32, fill
            if len(ind) < 32:
                ind_set = set(ind)
                while len(ind_set) < 32:
                    ind_set.add(rng.randint(1, 512))
                ind = sorted(list(ind_set))
        else:
            # Random
            ind = sorted(rng.sample(range(1, 513), 32))
        
        population.append(ind)
        mask = _arr_to_mask(ind)
        scores.append(_score_bitset(mask))
    
    best_overall = population[0]
    best_overall_score = scores[0]
    for i in range(1, pop_size):
        if scores[i] > best_overall_score:
            best_overall_score = scores[i]
            best_overall = population[i]
            
    # Main GA loop
    generation = 0
    while time.time() < deadline - 10.0: # Leave 10s for final polish
        generation += 1
        new_population = []
        new_scores = []
        
        # Elitism: keep top 5
        sorted_indices = sorted(range(pop_size), key=lambda i: scores[i], reverse=True)
        for i in range(5):
            idx = sorted_indices[i]
            new_population.append(population[idx])
            new_scores.append(scores[idx])
            
        # Generate offspring
        while len(new_population) < pop_size:
            # Select parents (tournament selection)
            p1_idx = rng.randrange(pop_size)
            p2_idx = rng.randrange(pop_size)
            # Simple tournament of 2
            if scores[p1_idx] < scores[p2_idx]:
                p1_idx, p2_idx = p2_idx, p1_idx
            
            # Crossover
            child = _crossover(population[p1_idx], population[p2_idx], rng)
            
            # Mutation
            if rng.random() < 0.5:
                child = _mutate(child, rng, num_mutations=2)
            
            # Local Search (brief)
            # We don't want to spend too much time here per individual.
            # Let's just do a few quick improvements or leave it for the next round's evaluation?
            # Better: Evaluate, then if it's good, local search it.
            
            mask = _arr_to_mask(child)
            score = _score_bitset(mask)
            
            new_population.append(child)
            new_scores.append(score)
            
            if score > best_overall_score:
                best_overall_score = score
                best_overall = child
        
        # Apply local search to the new population members that are promising
        # To save time, only local search the top 10 of the new population
        new_sorted_indices = sorted(range(len(new_population)), key=lambda i: new_scores[i], reverse=True)
        
        for i in range(min(10, len(new_population))):
            idx = new_sorted_indices[i]
            if time.time() > deadline - 10.0:
                break
            # Local search with a small time budget per individual
            # Or just a fixed number of passes?
            # Let's use a fixed number of passes for consistency
            arr, score = _local_search(new_population[idx], deadline - 10.0, rng)
            new_population[idx] = arr
            new_scores[idx] = score
            
            if score > best_overall_score:
                best_overall_score = score
                best_overall = arr
        
        population = new_population
        scores = new_scores
        
    # Final Polish
    if time.time() < deadline:
        arr, score = _local_search(best_overall, deadline, rng)
        if score > best_overall_score:
            best_overall = arr
            best_overall_score = score
            
    # Ensure validity
    result = sorted(set(best_overall))
    if len(result) != 32:
        # Repair
        result_set = set(result)
        while len(result_set) < 32:
            result_set.add(rng.randint(1, 512))
        result = sorted(list(result_set))
        
    result = [x for x in result if 1 <= x <= 512]
    if len(result) < 32:
        extra = [x for x in range(1, 513) if x not in result]
        result.extend(extra[:32 - len(result)])
        result = sorted(result[:32])
        
    return result

import time
import random
import math

N = 256
TIME_LIMIT = 88.0  # Safety margin below 90s

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    # Precompute 3-APs and 4-APs as bitmasks
    # Bit i corresponds to element i (0-indexed), so element i+1 in 1-indexed
    # Mask for an AP (a, b, c) is (1<<a) | (1<<b) | (1<<c)
    
    three_ap_masks = []
    four_ap_masks = []
    
    # 3-APs: a, a+d, a+2d
    for d in range(1, (N - 2) // 2 + 1):
        for a in range(0, N - 2 * d):
            mask = (1 << a) | (1 << (a + d)) | (1 << (a + 2 * d))
            three_ap_masks.append(mask)
            
    # 4-APs: a, a+d, a+2d, a+3d
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(0, N - 3 * d):
            mask = (1 << a) | (1 << (a + d)) | (1 << (a + 2 * d)) | (1 << (a + 3 * d))
            four_ap_masks.append(mask)

    # Precompute element weights (number of 3-APs they are part of) for greedy initialization
    elem_weights = [0] * N
    for mask in three_ap_masks:
        # Count bits set in mask? No, we need to know which elements are in which 3-APs.
        # This loop is slow if done naively. Let's just use index based weights or simple count.
        # Actually, for greedy, we can just use the count of 3-APs containing each element.
        pass # We'll calculate this efficiently or use a simpler heuristic.

    # Efficient weight calculation:
    # For each element x, how many 3-APs contain it?
    # It's the number of pairs (a,d) such that x is a, a+d, or a+2d.
    # This is roughly O(N).
    elem_weights = [0] * N
    for d in range(1, (N - 2) // 2 + 1):
        for a in range(0, N - 2 * d):
            elem_weights[a] += 1
            elem_weights[a + d] += 1
            elem_weights[a + 2 * d] += 1

    def count_3aps(mask):
        cnt = 0
        for m in three_ap_masks:
            if (mask & m) == m:
                cnt += 1
        return cnt

    def has_4ap(mask):
        for m in four_ap_masks:
            if (mask & m) == m:
                return True
        return False

    def get_4ap_violations(mask):
        """Return list of indices of 4-APs present in mask."""
        violations = []
        for i, m in enumerate(four_ap_masks):
            if (mask & m) == m:
                violations.append(i)
        return violations

    def repair(mask):
        """Remove elements to eliminate 4-APs. Returns new mask."""
        changed = True
        while changed:
            changed = False
            violations = get_4ap_violations(mask)
            if not violations:
                break
            # Pick the first violation
            ap_idx = violations[0]
            ap_mask = four_ap_masks[ap_idx]
            # Find elements in this AP that are in mask
            # We need to decide which one to remove.
            # Heuristic: Remove the one with the lowest weight (fewest 3-APs)
            # To do this, we need to extract the elements.
            # Since N=256, we can iterate through bits in ap_mask.
            candidates = []
            temp = ap_mask
            while temp:
                lsb = temp & -temp
                idx = lsb.bit_length() - 1
                candidates.append(idx)
                temp -= lsb
            
            # Remove the candidate with minimum weight
            worst_idx = min(candidates, key=lambda x: elem_weights[x])
            mask &= ~(1 << worst_idx)
            changed = True
        return mask

    def greedy_construct():
        """Greedy construction based on weights."""
        selected = 0
        # Sort indices by weight descending
        order = sorted(range(N), key=lambda x: (-elem_weights[x], x))
        for x in order:
            # Check if adding x creates a 4-AP
            # We can check if (selected | (1<<x)) has a 4-AP involving x?
            # Or just check the whole thing. It's fast enough.
            new_mask = selected | (1 << x)
            if not has_4ap(new_mask):
                selected = new_mask
        return selected

    def random_construct():
        """Random subset, then repair."""
        mask = 0
        for i in range(N):
            if rng.random() < 0.5:
                mask |= (1 << i)
        return repair(mask)

    def evaluate(mask):
        if has_4ap(mask):
            return -10000
        return count_3aps(mask)

    # Initialize population
    pop_size = 50
    population = []
    
    # Add some greedy solutions
    population.append(greedy_construct())
    population.append(random_construct())
    
    # Add diverse randoms
    for _ in range(pop_size - 2):
        population.append(random_construct())
        
    # Evaluate initial population
    best_mask = max(population, key=evaluate)
    best_score = evaluate(best_mask)
    
    # Genetic Algorithm
    max_generations = 2000
    mutation_rate = 0.1
    crossover_rate = 0.8
    
    for gen in range(max_generations):
        if time.time() - start_time > TIME_LIMIT - 2:
            break
            
        # Selection and Crossover
        new_population = []
        
        # Elitism: keep best few
        sorted_pop = sorted(population, key=evaluate, reverse=True)
        elite_count = 5
        for i in range(elite_count):
            new_population.append(sorted_pop[i])
            
        while len(new_population) < pop_size:
            # Tournament selection
            p1 = max(rng.sample(population, 3), key=evaluate)
            p2 = max(rng.sample(population, 3), key=evaluate)
            
            if rng.random() < crossover_rate:
                # Uniform crossover
                child = 0
                for i in range(N):
                    bit = (1 << i)
                    if p1 & bit:
                        if rng.random() < 0.5:
                            child |= bit
                    else:
                        if p2 & bit:
                            if rng.random() < 0.5:
                                child |= bit
            else:
                child = p1
                
            # Mutation
            if rng.random() < mutation_rate:
                # Random bit flip
                idx = rng.randint(0, N - 1)
                child ^= (1 << idx)
            elif rng.random() < mutation_rate * 0.5:
                # Swap mutation: remove one, add one
                # Find a set bit
                if child:
                    temp = child
                    lsb = temp & -temp
                    idx_remove = lsb.bit_length() - 1
                    child &= ~lsb
                    # Find an unset bit
                    unset = ~child & ((1 << N) - 1)
                    if unset:
                        # Pick a random unset bit
                        # This is tricky with large integers, but N=256 is small.
                        # We can just pick a random index and check if it's unset.
                        for _ in range(10): # Try a few times
                            idx_add = rng.randint(0, N - 1)
                            if not (child & (1 << idx_add)):
                                child |= (1 << idx_add)
                                break
            
            # Repair if necessary
            if has_4ap(child):
                child = repair(child)
                
            new_population.append(child)
            
        population = new_population
        
        # Update best
        current_best = max(population, key=evaluate)
        current_score = evaluate(current_best)
        if current_score > best_score:
            best_score = current_score
            best_mask = current_mask = current_best

    # Final polish: Local search on best_mask
    # Try single bit flips to improve
    improved = True
    while improved and (time.time() - start_time < TIME_LIMIT - 1):
        improved = False
        # Try removing bits
        for i in range(N):
            if best_mask & (1 << i):
                new_mask = best_mask & ~(1 << i)
                if not has_4ap(new_mask):
                    new_score = count_3aps(new_mask)
                    if new_score > best_score:
                        best_mask = new_mask
                        best_score = new_score
                        improved = True
                        break # Restart loop with new best
        if improved:
            continue
            
        # Try adding bits
        for i in range(N):
            if not (best_mask & (1 << i)):
                new_mask = best_mask | (1 << i)
                if not has_4ap(new_mask):
                    new_score = count_3aps(new_mask)
                    if new_score > best_score:
                        best_mask = new_mask
                        best_score = new_score
                        improved = True
                        break # Restart loop with new best
        if improved:
            continue

    # Extract result
    result = []
    for i in range(N):
        if best_mask & (1 << i):
            result.append(i + 1)
            
    return sorted(result)

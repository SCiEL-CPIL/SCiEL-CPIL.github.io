import time
import random
import sys

N = 256

def _precompute():
    """Precompute 4-AP constraints and 3-AP structure."""
    # All 4-APs: (a, a+d, a+2d, a+3d)
    all_4aps = []
    for a in range(1, N + 1):
        for d in range(1, (N - a + 1) // 3 + 1):
            b = a + d
            c = b + d
            e = c + d
            if e <= N:
                all_4aps.append((a, b, c, e))
    
    # For each element x, list of 4-APs containing x (as index into all_4aps)
    # and for each such 4-AP, the other 3 elements
    all_4aps_by_element = [[] for _ in range(N + 1)]
    for idx, (a, b, c, e) in enumerate(all_4aps):
        all_4aps_by_element[a].append((b, c, e))
        all_4aps_by_element[b].append((a, c, e))
        all_4aps_by_element[c].append((a, b, e))
        all_4aps_by_element[e].append((a, b, c))
    
    # Precompute for 3-AP counting: for each middle b, list of valid (a, c) pairs
    three_ap_pairs = [[] for _ in range(N + 1)]
    for b in range(1, N + 1):
        for a in range(1, b):
            c = 2 * b - a
            if c > b and c <= N:
                three_ap_pairs[b].append((a, c))
    
    # For fast delta computation
    x_as_first = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        for b in range(x + 1, N + 1):
            c = 2 * b - x
            if c > b and c <= N:
                x_as_first[x].append((b, c))
    
    x_as_last = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        for a in range(1, x):
            s = a + x
            if s % 2 == 0:
                b = s // 2
                if a < b < x:
                    x_as_last[x].append((a, b))
    
    return all_4aps_by_element, three_ap_pairs, x_as_first, x_as_last

def _creates_4ap(mask, x, all_4aps_by_element):
    """Check if adding x to mask creates a 4-AP."""
    for triple in all_4aps_by_element[x]:
        if (mask & (1 << triple[0])) and (mask & (1 << triple[1])) and (mask & (1 << triple[2])):
            return True
    return False

def _delta_add(mask, x, three_ap_pairs, x_as_first, x_as_last):
    """Count new 3-APs created by adding x to the set represented by mask."""
    count = 0
    for a, c in three_ap_pairs[x]:
        if (mask & (1 << a)) and (mask & (1 << c)):
            count += 1
    for b, c in x_as_first[x]:
        if (mask & (1 << b)) and (mask & (1 << c)):
            count += 1
    for a, b in x_as_last[x]:
        if (mask & (1 << a)) and (mask & (1 << b)):
            count += 1
    return count

def _count_3aps(mask, three_ap_pairs):
    """Count total 3-APs in set represented by mask."""
    count = 0
    for b in range(1, N + 1):
        if not (mask & (1 << b)):
            continue
        for a, c in three_ap_pairs[b]:
            if (mask & (1 << a)) and (mask & (1 << c)):
                count += 1
    return count

def _greedy_construct(all_4aps_by_element, order=None):
    """Greedy construction: add elements in given order if no 4-AP created."""
    if order is None:
        order = list(range(1, N + 1))
    mask = 0
    for x in order:
        if not _creates_4ap(mask, x, all_4aps_by_element):
            mask |= (1 << x)
    return mask

def _local_search(mask, all_4aps_by_element, three_ap_pairs, x_as_first, x_as_last, deadline):
    """Hill climbing local search."""
    current_mask = mask
    current_score = _count_3aps(current_mask, three_ap_pairs)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        set_elements = [i for i in range(1, N + 1) if current_mask & (1 << i)]
        unset_elements = [i for i in range(1, N + 1) if not (current_mask & (1 << i))]
        
        # Try all additions
        for y in unset_elements[:]:
            if _creates_4ap(current_mask, y, all_4aps_by_element):
                continue
            delta = _delta_add(current_mask, y, three_ap_pairs, x_as_first, x_as_last)
            if delta > 0:
                current_mask |= (1 << y)
                current_score += delta
                improved = True
                unset_elements.remove(y)
                set_elements.append(y)
        
        # Try all swaps
        if time.time() < deadline:
            for x in set_elements[:]:
                if not improved and time.time() > deadline:
                    break
                temp_mask = current_mask & ~(1 << x)
                for y in unset_elements:
                    if _creates_4ap(temp_mask, y, all_4aps_by_element):
                        continue
                    delta = _delta_add(temp_mask, y, three_ap_pairs, x_as_first, x_as_last) - _delta_add(current_mask, x, three_ap_pairs, x_as_first, x_as_last)
                    if delta > 0:
                        current_mask = temp_mask | (1 << y)
                        current_score += delta
                        improved = True
                        set_elements.remove(x)
                        set_elements.append(y)
                        unset_elements.remove(y)
                        unset_elements.append(x)
                        break
    
    return current_mask, current_score

def _genetic_algorithm(all_4aps_by_element, three_ap_pairs, x_as_first, x_as_last, deadline, seed=42):
    """Genetic algorithm with bitmask-based fitness evaluation."""
    random.seed(seed)
    
    # Initialize population
    pop_size = 30
    population = []
    
    # Start with greedy solutions
    for i in range(5):
        order = list(range(1, N + 1))
        random.shuffle(order)
        mask = _greedy_construct(all_4aps_by_element, order)
        population.append(mask)
    
    # Add some random valid sets
    for i in range(pop_size - 5):
        mask = 0
        elements = list(range(1, N + 1))
        random.shuffle(elements)
        for x in elements:
            if not _creates_4ap(mask, x, all_4aps_by_element):
                if random.random() < 0.6:  # Probability of adding
                    mask |= (1 << x)
        population.append(mask)
    
    # Evaluate initial population
    scores = [_count_3aps(m, three_ap_pairs) for m in population]
    best_idx = max(range(pop_size), key=lambda i: scores[i])
    best_mask = population[best_idx]
    best_score = scores[best_idx]
    
    # Genetic algorithm loop
    mutation_rate = 0.3
    crossover_rate = 0.7
    elite_count = 3
    
    while time.time() < deadline:
        # Sort by fitness
        indices = sorted(range(pop_size), key=lambda i: scores[i], reverse=True)
        population = [population[i] for i in indices]
        scores = [scores[i] for i in indices]
        
        # Update best
        if scores[0] > best_score:
            best_score = scores[0]
            best_mask = population[0]
        
        # Create next generation
        new_population = []
        
        # Elitism
        for i in range(elite_count):
            new_population.append(population[i])
        
        # Fill rest with crossover and mutation
        while len(new_population) < pop_size and time.time() < deadline:
            # Select parents
            p1_idx = random.randint(0, pop_size - 1)
            p2_idx = random.randint(0, pop_size - 1)
            p1 = population[p1_idx]
            p2 = population[p2_idx]
            
            if random.random() < crossover_rate:
                # Uniform crossover
                child = 0
                for x in range(1, N + 1):
                    bit = 1 << x
                    if random.random() < 0.5:
                        child |= (p1 & bit)
                    else:
                        child |= (p2 & bit)
            else:
                child = p1
            
            # Mutation
            if random.random() < mutation_rate:
                # Try to add or remove random elements
                for _ in range(random.randint(1, 3)):
                    x = random.randint(1, N)
                    bit = 1 << x
                    if child & bit:
                        # Remove
                        new_child = child & ~bit
                        if _count_3aps(new_child, three_ap_pairs) >= _count_3aps(child, three_ap_pairs) - 2:
                            child = new_child
                    else:
                        # Add if valid
                        if not _creates_4ap(child, x, all_4aps_by_element):
                            child |= bit
            
            # Ensure validity: remove any elements that create 4-APs
            # Actually, the mutation above should maintain validity, but let's be safe
            # Re-check and fix if needed
            set_elements = [i for i in range(1, N + 1) if child & (1 << i)]
            valid_mask = 0
            for x in set_elements:
                if not _creates_4ap(valid_mask, x, all_4aps_by_element):
                    valid_mask |= (1 << x)
            child = valid_mask
            
            new_population.append(child)
        
        population = new_population
        scores = [_count_3aps(m, three_ap_pairs) for m in population]
    
    return best_mask, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Precompute
    all_4aps_by_element, three_ap_pairs, x_as_first, x_as_last = _precompute()
    
    # Phase 1: Multiple greedy constructions with different orders
    best_mask = 0
    best_score = 0
    
    # Standard greedy
    mask1 = _greedy_construct(all_4aps_by_element, list(range(1, N + 1)))
    score1 = _count_3aps(mask1, three_ap_pairs)
    if score1 > best_score:
        best_mask = mask1
        best_score = score1
    
    # Reverse greedy
    mask2 = _greedy_construct(all_4aps_by_element, list(range(N, 0, -1)))
    score2 = _count_3aps(mask2, three_ap_pairs)
    if score2 > best_score:
        best_mask = mask2
        best_score = score2
    
    # Randomized greedy (multiple runs)
    random.seed(123)
    for trial in range(15):
        if time.time() > deadline - 40:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        mask_t = _greedy_construct(all_4aps_by_element, order)
        score_t = _count_3aps(mask_t, three_ap_pairs)
        if score_t > best_score:
            best_mask = mask_t
            best_score = score_t
    
    # Phase 2: Local search on best
    if time.time() < deadline - 30:
        best_mask, best_score = _local_search(best_mask, all_4aps_by_element, three_ap_pairs, x_as_first, x_as_last, deadline - 30)
    
    # Phase 3: Genetic algorithm
    if time.time() < deadline - 10:
        ga_mask, ga_score = _genetic_algorithm(all_4aps_by_element, three_ap_pairs, x_as_first, x_as_last, deadline - 10, seed=42)
        if ga_score > best_score:
            best_mask = ga_mask
            best_score = ga_score
    
    # Phase 4: Final local search polish
    if time.time() < deadline - 2:
        best_mask, best_score = _local_search(best_mask, all_4aps_by_element, three_ap_pairs, x_as_first, x_as_last, deadline - 2)
    
    # Convert mask to sorted list
    result = [i for i in range(1, N + 1) if best_mask & (1 << i)]
    
    return result

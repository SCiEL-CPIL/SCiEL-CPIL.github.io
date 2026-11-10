"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin from 90s limit
    n = 512
    
    # Precompute all 4-APs for fast counting
    # A 4-AP is (a, a+d, a+2d, a+3d)
    # We store them as tuples of indices
    all_4aps = []
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            all_4aps.append((a, b, c, e))
            
    num_aps = len(all_4aps)
    
    # Helper to count 4-APs in a bitset (represented as list of bools or int mask)
    # Using a list of bools for simplicity, but counting is O(num_aps)
    # To speed up fitness evaluation, we can precompute for each 4-AP the indices.
    # We will represent the genome as a list of 0/1.
    
    def count_aps(genome):
        count = 0
        for ap in all_4aps:
            if genome[ap[0]] and genome[ap[1]] and genome[ap[2]] and genome[ap[3]]:
                count += 1
        return count
        
    def fitness(genome):
        size = sum(genome)
        ap_count = count_aps(genome)
        # We want to maximize size, penalize APs heavily
        # If ap_count > 0, it's invalid. We want to drive ap_count to 0.
        # Penalty should be large enough that removing 1 element from an AP is always beneficial
        # if it reduces AP count, even if it reduces size.
        # Let's use a dynamic penalty or a very high static one.
        # Max possible APs for a set is small.
        # If we have 1 AP, penalty should exceed gain of 1 element.
        # Let's try penalty = 1000.
        return size - 1000 * ap_count

    # Initialize population
    pop_size = 100
    population = []
    
    # Seed with some greedy solutions to give it a head start
    # Greedy 1: Base-3 digit sum order
    def base3_sum(x):
        s = 0
        while x > 0:
            s += x % 3
            x //= 3
        return s

    order = sorted(range(1, n + 1), key=lambda x: (base3_sum(x), x))
    greedy_genome = [0] * (n + 1)
    for x in order:
        # Check if adding x creates AP
        valid = True
        # This is slow, so we just add it and let GA fix it? 
        # No, better to start with a valid set.
        # Let's just use random initialization for diversity, and one greedy.
        pass
    
    # Actually, let's generate the greedy set properly first to seed one individual.
    greedy_genome = [0] * (n + 1)
    for x in order:
        # Fast check: is there an AP containing x and 3 other elements in greedy_genome?
        # We can just add it and rely on GA to remove conflicts if penalty is high.
        # But starting valid is better.
        # Let's use the can_add logic from before but simplified for seeding.
        # For speed, let's just random init most, and seed 1 greedy.
        pass

    # Random Initialization
    for i in range(pop_size):
        if i == 0:
            # Seed with a known good heuristic: numbers with few 3s in base 3?
            # Or just random
            genome = [0] * (n + 1)
            for x in range(1, n + 1):
                if random.random() < 0.4:
                    genome[x] = 1
            population.append(genome)
        else:
            genome = [0] * (n + 1)
            for x in range(1, n + 1):
                if random.random() < 0.4:
                    genome[x] = 1
            population.append(genome)
            
    best_genome = [0] * (n + 1)
    best_fitness = -1
    best_valid_size = 0
    
    # Evaluate initial population
    for genome in population:
        f = fitness(genome)
        if f > best_fitness:
            best_fitness = f
            best_genome = list(genome)
            if count_aps(genome) == 0:
                best_valid_size = sum(genome)

    generation = 0
    while time.time() < deadline - 5:
        # Selection: Tournament
        next_population = []
        
        # Keep the best individual (Elitism)
        next_population.append(list(best_genome))
        
        # Generate rest
        while len(next_population) < pop_size:
            # Pick 2 parents from current population
            p1 = random.choice(population)
            p2 = random.choice(population)
            
            # Crossover: Uniform
            child = [0] * (n + 1)
            for i in range(1, n + 1):
                if random.random() < 0.5:
                    child[i] = p1[i]
                else:
                    child[i] = p2[i]
            
            # Mutation: Bit flip
            # Higher mutation rate for early generations, lower later?
            # Or just fixed.
            for i in range(1, n + 1):
                if random.random() < 0.05:
                    child[i] = 1 - child[i]
            
            next_population.append(child)
            
        # Evaluate new population
        population = next_population
        for genome in population:
            f = fitness(genome)
            if f > best_fitness:
                best_fitness = f
                best_genome = list(genome)
                if count_aps(genome) == 0:
                    best_valid_size = sum(genome)
        
        generation += 1
        
    # Final cleanup: Ensure best_genome is valid. 
    # If it's not valid, we need to remove elements to break APs.
    # Greedy removal: Find an AP, remove a random element from it, repeat.
    current = list(best_genome)
    current_ap_count = count_aps(current)
    while current_ap_count > 0:
        # Find a conflicting AP
        found = False
        for ap in all_4aps:
            if current[ap[0]] and current[ap[1]] and current[ap[2]] and current[ap[3]]:
                # Remove one element randomly from this AP
                idx_to_remove = random.choice(ap)
                current[idx_to_remove] = 0
                found = True
                break
        if not found:
            break
        current_ap_count = count_aps(current)
        
    # Convert to list of ints
    result_set = [i for i in range(1, n + 1) if current[i]]
    
    # Verify validity one last time
    final_check_in_set = [False] * (n + 1)
    for x in result_set:
        final_check_in_set[x] = True
        
    valid = True
    for a in range(1, n + 1):
        if not final_check_in_set[a]:
            continue
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            if final_check_in_set[a+d] and final_check_in_set[a+2*d] and final_check_in_set[a+3*d]:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback: Return a simple valid set (e.g., all numbers with only 0s and 1s in base 3? No, that's 3-AP free. 4-AP free is easier.)
        # Just return the greedy base-3 set which is likely valid or close.
        # Actually, if GA failed to produce a valid set, it's a problem.
        # Let's try a simple greedy from scratch as fallback.
        fb_in_set = [False] * (n + 1)
        fb_set = []
        for x in order:
             # Simple check
             ok = True
             # Check if x completes an AP with 3 existing
             # This is O(n^2) per x, too slow for 512 if done naively in Python?
             # 512 * 512 * 4 checks ~ 1M ops. Fine.
             for d in range(1, (x // 3) + 1):
                 if fb_in_set[x-d] and fb_in_set[x-2*d] and fb_in_set[x-3*d]: ok = False; break
             if ok:
                 for d in range(1, (n - x) // 3 + 1):
                     if fb_in_set[x+d] and fb_in_set[x+2*d] and fb_in_set[x+3*d]: ok = False; break
             if ok:
                 max_d = min(x, (n - x) // 2)
                 for d in range(1, max_d + 1):
                     if fb_in_set[x-d] and fb_in_set[x+d] and fb_in_set[x+2*d]: ok = False; break
             if ok:
                 max_d = min(x // 2, n - x)
                 for d in range(1, max_d + 1):
                     if fb_in_set[x-2*d] and fb_in_set[x-d] and fb_in_set[x+d]: ok = False; break
             if ok:
                 fb_in_set[x] = True
                 fb_set.append(x)
        return sorted(fb_set)
        
    return sorted(result_set)
# EVOLVE-BLOCK-END

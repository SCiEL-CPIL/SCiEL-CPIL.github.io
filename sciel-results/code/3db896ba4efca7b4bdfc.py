import time
import random
import numpy as np

# Precomputation
N = 256
# Generate all 3-APs and 4-APs as bitmasks
# A 3-AP is (a, a+d, a+2d). Mask has bits at a, a+d, a+2d.
# Indices are 1-based in problem, 0-based in bitmask.
# Bit i corresponds to number i+1.

def get_ap_masks(term_count):
    masks = []
    # Max difference d such that a + (term_count-1)d <= N
    # For 3-AP: a + 2d <= 256 => d <= (256-1)/2 = 127.5 -> 127
    # For 4-AP: a + 3d <= 256 => d <= (256-1)/3 = 85.3 -> 85
    max_d = (N - 1) // (term_count - 1)
    for d in range(1, max_d + 1):
        # a can be 1 to N - (term_count-1)d
        max_a = N - (term_count - 1) * d
        for a in range(1, max_a + 1):
            mask = 0
            for j in range(term_count):
                val = a + j * d
                # val is 1-based, bit index is val-1
                mask |= (1 << (val - 1))
            masks.append(mask)
    return masks

M3 = get_ap_masks(3)
M4 = get_ap_masks(4)

# Convert to numpy arrays for potential vectorization if needed, 
# but standard list iteration is often faster for small N in pure Python 
# if we don't have to allocate arrays. 
# However, 16000 items is small. Let's stick to lists for simplicity and low overhead.

def evaluate(mask):
    """
    Returns the number of 3-APs if mask is 4-AP-free, else -1.
    """
    # Check 4-AP constraint
    # If any 4-AP is fully present, it's invalid
    for m in M4:
        if (mask & m) == m:
            return -1
    
    # Count 3-APs
    count = 0
    for m in M3:
        if (mask & m) == m:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = 50.0 # 10s safety margin
    
    # Initial Population
    pop_size = 100
    population = []
    
    # 1. Greedy seed (same as original)
    greedy_mask = 0
    for x in range(1, N + 1):
        bit = 1 << (x - 1)
        # Check if adding x creates a 4-AP
        valid = True
        # We need to check if x is the 4th term of a 4-AP (x, x-d, x-2d, x-3d)
        # or 3rd term (x, x-d, x-2d, ... wait, 4-AP is a, a+d, a+2d, a+3d)
        # If we add x, it could be:
        # 1. a+3d = x => a, a+d, a+2d must be in set. x = a+3d.
        # 2. a+2d = x => a, a+d, a+3d must be in set? No, if a+2d is in set, and a, a+d, a+3d are in set, 
        #    then a, a+d, a+2d, a+3d is a 4-AP.
        # So we must check all 4-APs containing x.
        
        # Simpler: just check all 4-APs in M4 that contain bit x.
        # But M4 is precomputed. We can just check if (greedy_mask | bit) is valid.
        # This is O(|M4|) per step, which is 256 * 10000 = 2.5M ops. Fast enough.
        new_mask = greedy_mask | bit
        valid = True
        for m in M4:
            if (new_mask & m) == m:
                valid = False
                break
        if valid:
            greedy_mask = new_mask
    
    population.append(greedy_mask)
    
    # 2. Random valid sets
    # Generate random masks and filter, or generate and repair.
    # Repair is hard. Let's just generate random masks and keep valid ones.
    # To ensure we get some, we can try random subsets of the greedy set, 
    # or random small sets.
    
    for _ in range(pop_size - 1):
        # Random mask with density similar to greedy
        # Greedy size is roughly N / log(N) ? 
        # Let's just pick random bits with p=0.1
        mask = 0
        for i in range(N):
            if random.random() < 0.15:
                mask |= (1 << i)
        
        # If invalid, try to fix by removing elements involved in 4-APs
        # Or just discard and retry a few times
        attempts = 0
        while evaluate(mask) == -1 and attempts < 5:
            # Find a 4-AP and remove a random element from it
            # This is expensive. Just regenerate.
            mask = 0
            for i in range(N):
                if random.random() < 0.15:
                    mask |= (1 << i)
            attempts += 1
        
        if evaluate(mask) != -1:
            population.append(mask)
        else:
            # Fallback: add a subset of greedy
            subset_mask = 0
            for i in range(N):
                if greedy_mask & (1 << i) and random.random() < 0.5:
                    subset_mask |= (1 << i)
            population.append(subset_mask)

    best_mask = 0
    best_fitness = -1
    
    # Evaluate initial population
    for m in population:
        f = evaluate(m)
        if f > best_fitness:
            best_fitness = f
            best_mask = m
    
    generations = 0
    while time.time() - start_time < deadline:
        generations += 1
        next_pop = []
        
        # Elitism: keep best few
        scored_pop = [(evaluate(m), m) for m in population]
        scored_pop.sort(key=lambda x: x[0], reverse=True)
        
        for i in range(5): # Keep top 5
            if scored_pop[i][1] not in next_pop:
                next_pop.append(scored_pop[i][1])
        
        # Generate offspring
        while len(next_pop) < pop_size:
            # Tournament selection
            parent1_idx = random.randint(0, len(scored_pop) - 1)
            parent2_idx = random.randint(0, len(scored_pop) - 1)
            # Simple: pick random from top 20%
            top_k = max(1, len(scored_pop) // 5)
            p1 = scored_pop[random.randint(0, top_k - 1)][1]
            p2 = scored_pop[random.randint(0, top_k - 1)][1]
            
            # Crossover: Uniform
            child = 0
            for i in range(N):
                if random.random() < 0.5:
                    child |= (p1 >> i) & 1
                else:
                    child |= (p2 >> i) & 1
                # Shift for next bit? No, just construct mask.
                # This loop is slow. Let's use bitwise ops.
            
            # Faster crossover:
            # Random mask for crossover
            cross_mask = 0
            for i in range(N):
                if random.random() < 0.5:
                    cross_mask |= (1 << i)
            child = (p1 & cross_mask) | (p2 & ~cross_mask)
            
            # Mutation
            # Flip 1-3 bits
            num_mut = random.randint(1, 3)
            for _ in range(num_mut):
                bit_idx = random.randint(0, N - 1)
                child ^= (1 << bit_idx)
            
            next_pop.append(child)
        
        population = next_pop
        
        # Update best
        for m in population:
            f = evaluate(m)
            if f > best_fitness:
                best_fitness = f
                best_mask = m
        
        # Early stop if we hit a theoretical upper bound? 
        # Max 3-APs in a 4-AP-free set is not known exactly, but likely < N^2/4.
        # If best_fitness is very high, we might be close.
        
    # Convert best_mask to sorted list
    result = []
    for i in range(N):
        if best_mask & (1 << i):
            result.append(i + 1)
    
    return sorted(result)

# Helper to run construct if called directly for testing
if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    # Verify
    mask = 0
    for x in res:
        mask |= (1 << (x-1))
    print(f"Fitness: {evaluate(mask)}")

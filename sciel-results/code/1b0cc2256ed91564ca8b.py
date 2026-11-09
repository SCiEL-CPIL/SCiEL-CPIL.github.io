import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    N = 32
    MAX_VAL = 512

    def score_mask(mask):
        count = 0
        m = mask
        for d in range(1, 128):
            if (m & (m >> d) & (m >> (2 * d))) != 0:
                count += 1
        for d in range(128, 256):
            if (m & (m >> d) & (m >> (2 * d))) != 0:
                count += 1
        return count

    def mask_to_list(mask):
        res = []
        m = mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            res.append(idx)
            m ^= lsb
        return res

    def list_to_mask(lst):
        m = 0
        for x in lst:
            m |= (1 << x)
        return m

    # Initialize population
    pop_size = 20
    population = []
    
    # Seed with diverse structures
    structures = [
        list(range(1, 1 + 31 * 16, 16)),
        list(range(1, 33)),
        list(range(481, 513)),
        list(range(1, 1 + 31 * 8, 8)),
        list(range(1, 1 + 31 * 4, 4)),
        list(range(1, 1 + 16 * 32, 32)) + list(range(2, 2 + 16 * 32, 32)),
        list(range(1, 17)) + list(range(257, 273)),
        list(range(1, 1 + 31 * 12, 12)),
        list(range(1, 1 + 31 * 6, 6)),
        list(range(1, 1 + 31 * 2, 2)),
    ]
    
    for i in range(pop_size):
        if i < len(structures):
            lst = structures[i][:]
            # Add some noise to seed structures
            if random.random() < 0.5:
                for _ in range(random.randint(0, 5)):
                    idx = random.randint(0, N-1)
                    new_val = random.randint(1, MAX_VAL)
                    while new_val in lst:
                        new_val = random.randint(1, MAX_VAL)
                    lst[idx] = new_val
        else:
            lst = random.sample(range(1, MAX_VAL + 1), N)
        
        mask = list_to_mask(lst)
        score = score_mask(mask)
        population.append((mask, score))
    
    # Sort by score
    population.sort(key=lambda x: x[1], reverse=True)
    
    best_mask, best_score = population[0]
    
    # GA Parameters
    gen_size = 10
    mutation_rate = 0.3
    elite_size = 2
    
    while time.time() < deadline:
        new_pop = []
        
        # Elitism
        for i in range(elite_size):
            new_pop.append(population[i])
        
        # Fill rest with offspring
        while len(new_pop) < pop_size:
            # Selection: Tournament selection
            def select():
                idx1 = random.randint(0, pop_size-1)
                idx2 = random.randint(0, pop_size-1)
                if population[idx1][1] > population[idx2][1]:
                    return population[idx1]
                else:
                    return population[idx2]
            
            p1 = select()
            p2 = select()
            
            # Crossover: Uniform crossover on bitmask
            # Each bit has 50% chance of coming from p1 or p2
            # But we need to ensure exactly 32 bits are set.
            # Simple uniform crossover on bits might result in different pop count.
            # So we do:
            # 1. Create a mask of positions where we take from p1
            # 2. Count bits taken from p1. If not N, adjust by swapping.
            
            # Alternative: List-based crossover then convert to mask
            # Convert to sorted lists
            l1 = mask_to_list(p1[0])
            l2 = mask_to_list(p2[0])
            l1.sort()
            l2.sort()
            
            child_list = []
            for i in range(N):
                if random.random() < 0.5:
                    child_list.append(l1[i])
                else:
                    child_list.append(l2[i])
            
            # Handle duplicates
            if len(set(child_list)) < N:
                # Resolve conflicts
                used = set(child_list)
                for i in range(N):
                    if child_list.count(child_list[i]) > 1:
                        # Replace duplicate with a random new value
                        new_val = random.randint(1, MAX_VAL)
                        while new_val in used:
                            new_val = random.randint(1, MAX_VAL)
                        child_list[i] = new_val
                        used.add(new_val)
                        # Remove other occurrences
                        for j in range(i+1, N):
                            if child_list[j] == new_val:
                                child_list[j] = -1 # Mark for fix
                        break # This simple fix might not work perfectly, let's use a more robust one below
            
            # Robust duplicate fix:
            child_set = set(child_list)
            if len(child_set) < N:
                # Replace duplicates with random unused values
                current_set = set(child_list)
                for i in range(N):
                    if child_list[i] in current_set and child_list.count(child_list[i]) > 1:
                        new_val = random.randint(1, MAX_VAL)
                        while new_val in current_set:
                            new_val = random.randint(1, MAX_VAL)
                        child_list[i] = new_val
                        current_set.add(new_val)
            
            # Mutation
            if random.random() < mutation_rate:
                num_mut = random.randint(1, 3)
                for _ in range(num_mut):
                    idx = random.randint(0, N-1)
                    old_val = child_list[idx]
                    new_val = random.randint(1, MAX_VAL)
                    while new_val in child_list and new_val != old_val:
                        new_val = random.randint(1, MAX_VAL)
                    child_list[idx] = new_val
            
            # Final validation: ensure 32 distinct integers in 1..512
            if len(set(child_list)) != N:
                # Fallback to random
                child_list = random.sample(range(1, MAX_VAL + 1), N)
            
            child_mask = list_to_mask(child_list)
            child_score = score_mask(child_mask)
            new_pop.append((child_mask, child_score))
            
            if child_score > best_score:
                best_score = child_score
                best_mask = child_mask
        
        # Update population
        population = new_pop
        population.sort(key=lambda x: x[1], reverse=True)
        
        # If best score hasn't improved in a while, inject random individuals
        if population[0][1] == best_score and random.random() < 0.1:
            for i in range(pop_size // 4):
                lst = random.sample(range(1, MAX_VAL + 1), N)
                mask = list_to_mask(lst)
                score = score_mask(mask)
                population[i] = (mask, score)
            population.sort(key=lambda x: x[1], reverse=True)

    return mask_to_list(best_mask)

# EVOLVE-BLOCK-END

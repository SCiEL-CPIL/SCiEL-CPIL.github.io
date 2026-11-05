import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    N = 256
    elements = range(1, N + 1)
    
    # Precompute 3-APs and 4-APs as bitmasks
    # Bit i represents element i+1 (index 0 is element 1)
    # Element x is bit (x-1)
    
    four_masks = []
    three_masks = []
    
    # Generate all 4-APs: a, a+d, a+2d, a+3d <= N
    for d in range(1, (N // 3) + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            mask = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1)) | (1 << (e - 1))
            four_masks.append(mask)
            
    # Generate all 3-APs: a, a+d, a+2d <= N
    for d in range(1, (N // 2) + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            mask = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            three_masks.append(mask)
    
    def is_valid(mask):
        """Check if mask contains no 4-AP."""
        for fm in four_masks:
            if (mask & fm) == fm:
                return False
        return True

    def count_threes(mask):
        """Count number of 3-APs in mask."""
        count = 0
        for tm in three_masks:
            if (mask & tm) == tm:
                count += 1
        return count

    def fitness(mask):
        """Return number of 3-APs if valid, else -1."""
        if not is_valid(mask):
            return -1
        return count_threes(mask)

    def repair(mask):
        """Remove elements to break 4-APs until valid."""
        while not is_valid(mask):
            # Find a 4-AP
            found = False
            for fm in four_masks:
                if (mask & fm) == fm:
                    # Pick a random bit from this 4-AP to remove
                    # Bits in fm are the positions of the 4 elements
                    bits = []
                    temp = fm
                    while temp:
                        # Get lowest set bit
                        lsb = temp & -temp
                        idx = lsb.bit_length() - 1
                        bits.append(idx)
                        temp ^= lsb
                    
                    # Remove one random element from the 4-AP
                    remove_idx = random.choice(bits)
                    mask &= ~(1 << remove_idx)
                    found = True
                    break
            if not found:
                break # Should not happen if is_valid returned False
        return mask

    # Initialize Population
    pop_size = 60
    population = []
    
    # 1. Add the greedy seed solution
    seed_set = set()
    for x in range(1, N + 1):
        if all(not all(x - j * d in seed_set for j in (1, 2, 3)) for d in range(1, (x - 1) // 3 + 1)):
            seed_set.add(x)
    
    seed_mask = 0
    for x in seed_set:
        seed_mask |= (1 << (x - 1))
    population.append(seed_mask)
    
    # 2. Add some heuristic initializations
    # Multiples of 3
    m3_mask = 0
    for x in range(3, N + 1, 3):
        m3_mask |= (1 << (x - 1))
    population.append(repair(m3_mask))
    
    # Multiples of 4
    m4_mask = 0
    for x in range(4, N + 1, 4):
        m4_mask |= (1 << (x - 1))
    population.append(repair(m4_mask))
    
    # Random valid sets
    while len(population) < pop_size:
        # Start with random subset
        mask = 0
        for x in range(1, N + 1):
            if random.random() < 0.3:
                mask |= (1 << (x - 1))
        mask = repair(mask)
        population.append(mask)
    
    # Evaluate initial fitness
    pop_fitness = [fitness(m) for m in population]
    
    best_mask = max(population, key=lambda m: pop_fitness[population.index(m)])
    best_fit = max(pop_fitness)
    
    # Genetic Algorithm Loop
    generation = 0
    while time.time() < deadline:
        generation += 1
        
        # Selection: Tournament
        def select():
            idx1 = random.randint(0, len(population) - 1)
            idx2 = random.randint(0, len(population) - 1)
            if pop_fitness[idx1] > pop_fitness[idx2]:
                return population[idx1]
            return population[idx2]
        
        new_population = []
        
        # Elitism: Keep top 2
        sorted_idx = sorted(range(len(population)), key=lambda i: pop_fitness[i], reverse=True)
        new_population.append(population[sorted_idx[0]])
        new_population.append(population[sorted_idx[1]])
        
        # Generate offspring
        while len(new_population) < pop_size:
            p1 = select()
            p2 = select()
            
            # Crossover: Uniform
            child = 0
            for i in range(N):
                if random.random() < 0.5:
                    child |= (p1 & (1 << i))
                else:
                    child |= (p2 & (1 << i))
            
            # Mutation: Flip bits
            # Higher mutation rate for exploration
            for i in range(N):
                if random.random() < 0.05:
                    child ^= (1 << i)
            
            # Repair
            child = repair(child)
            
            new_population.append(child)
        
        # Evaluate new population
        population = new_population
        pop_fitness = [fitness(m) for m in population]
        
        current_best_fit = max(pop_fitness)
        if current_best_fit > best_fit:
            best_fit = current_best_fit
            best_mask = population[pop_fitness.index(current_best_fit)]
            
        # Check deadline
        if time.time() > deadline - 5:
            break

    # Convert best_mask to sorted list
    result_set = []
    for i in range(N):
        if best_mask & (1 << i):
            result_set.append(i + 1)
    
    return sorted(result_set)

# EVOLVE-BLOCK-END

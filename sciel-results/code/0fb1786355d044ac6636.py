def construct():
    import random
    import math
    import time

    n = 128
    elements = list(range(1, n + 1))
    
    # Precompute LCMs
    def get_lcm(a, b):
        return a * b // math.gcd(a, b)

    # bad_masks[a][b] is a bitmask of c such that (a,b,c) is a bad triple
    bad_masks = [[0] * (n + 1) for _ in range(n + 1)]
    
    # Precomputation
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            L = get_lcm(a, b)
            mask = 0
            for c in range(1, n + 1):
                if c == a or c == b:
                    continue
                if get_lcm(a, c) == L and get_lcm(b, c) == L:
                    mask |= (1 << (c - 1))
            bad_masks[a][b] = mask
            bad_masks[b][a] = mask

    def popcount(x):
        return bin(x).count('1')

    def count_violations(S_mask):
        """Count the number of bad triples in S_mask."""
        count = 0
        for a in range(1, n + 1):
            if not (S_mask & (1 << (a - 1))):
                continue
            for b in range(a + 1, n + 1):
                if not (S_mask & (1 << (b - 1))):
                    continue
                count += popcount(S_mask & bad_masks[a][b])
        return count // 3

    def is_valid(S_mask):
        return count_violations(S_mask) == 0

    def repair(S_mask):
        """Remove elements to make the set valid."""
        # Limit iterations to prevent infinite loop or excessive time
        for _ in range(200):
            if count_violations(S_mask) == 0:
                break
            elems_in_set = [x for x in range(1, n + 1) if S_mask & (1 << (x - 1))]
            if not elems_in_set:
                break
            # Remove a random element
            x = random.choice(elems_in_set)
            S_mask &= ~(1 << (x - 1))
        return S_mask

    def improve(S_mask):
        """Try to add elements to the set while maintaining validity."""
        # Limit passes to prevent excessive time
        for _ in range(5):
            changed = False
            candidates = [x for x in range(1, n + 1) if not (S_mask & (1 << (x - 1)))]
            random.shuffle(candidates)
            for x in candidates:
                new_mask = S_mask | (1 << (x - 1))
                if is_valid(new_mask):
                    S_mask = new_mask
                    changed = True
            if not changed:
                break
        return S_mask

    def greedy_init():
        S_mask = 0
        elems = elements[:]
        random.shuffle(elems)
        for x in elems:
            new_mask = S_mask | (1 << (x - 1))
            if is_valid(new_mask):
                S_mask = new_mask
        return S_mask

    start_time = time.time()
    max_time = 55.0
    
    rng = random.Random(42)
    
    # Initialize population
    pop_size = 30
    population = []
    for _ in range(pop_size):
        s = greedy_init()
        s = improve(s)
        population.append(s)
    
    best_mask = max(population, key=lambda x: popcount(x))
    best_size = popcount(best_mask)
    
    while time.time() - start_time < max_time:
        # Evaluate fitness
        population.sort(key=lambda x: popcount(x), reverse=True)
        
        # Selection and Crossover
        new_population = []
        
        # Elitism: keep top 10%
        elite_count = max(1, pop_size // 10)
        new_population.extend(population[:elite_count])
        
        while len(new_population) < pop_size:
            # Tournament Selection
            p1 = population[rng.randint(0, min(10, pop_size-1))]
            p2 = population[rng.randint(0, min(10, pop_size-1))]
            
            # Crossover: Union
            child = p1 | p2
            
            # Repair
            child = repair(child)
            
            # Improve
            child = improve(child)
            
            # Mutation
            if rng.random() < 0.5:
                # Random Swap
                in_elems = [x for x in range(1, n + 1) if child & (1 << (x - 1))]
                out_elems = [x for x in range(1, n + 1) if not (child & (1 << (x - 1)))]
                if in_elems and out_elems:
                    x_in = rng.choice(in_elems)
                    x_out = rng.choice(out_elems)
                    new_child = (child & ~(1 << (x_in - 1))) | (1 << (x_out - 1))
                    if is_valid(new_child):
                        child = new_child
            
            new_population.append(child)
            
        population = new_population
        
        # Update best
        curr_best = max(population, key=lambda x: popcount(x))
        if popcount(curr_best) > best_size:
            best_size = popcount(curr_best)
            best_mask = curr_best

    result = []
    for i in range(1, n + 1):
        if best_mask & (1 << (i - 1)):
            result.append(i)
            
    if not result:
        result = [1]
        
    return result

def construct():
    import math
    import random
    import time

    def score_of(U):
        U = sorted(set(U))
        if not U or 0 not in U:
            return 0
        max_u = max(U)
        if max_u == 0:
            return 0
        diff_set = set()
        sum_set = set()
        for a in U:
            for b in U:
                diff_set.add(a - b)
                sum_set.add(a + b)
        n_diff = len(diff_set)
        n_sum = len(sum_set)
        if n_sum == 0 or n_diff == 0:
            return 0
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1
        denom = math.log(2 * max_u + 1)
        if denom == 0:
            return 1
        return 1 + math.log(ratio) / denom

    random.seed(42)
    start_time = time.time()
    time_limit = 115  # Seconds
    
    # Initial diverse pool with structured patterns
    initial_sets = [
        [0, 1, 3, 7, 15],
        [0, 1, 2, 4, 8],
        [0, 1, 3, 9, 27],
        [0, 1, 4, 13, 40],
        [0, 2, 5, 12, 29],
        [0, 1, 3, 10, 31],
        [0, 1, 2, 5, 11],
        [0, 1, 3, 6, 12],
        [0, 1, 4, 10, 25],
        [0, 2, 6, 18],
        [0, 1, 3, 9],
        [0, 1, 4, 13],
        [0, 1, 3, 7, 13],
        [0, 1, 3, 7, 11],
        [0, 3, 8, 18, 39],
        [0, 1, 4, 16, 65],
        [0, 2, 7, 22, 67],
        [0, 1, 5, 21, 81], # Geometric 5
        [0, 1, 4, 16, 64], # Geometric 4
        [0, 1, 3, 9, 27], # Geometric 3
    ]
    
    population = []
    for s in initial_sets:
        sc = score_of(s)
        population.append((sc, s))
    
    # Add some random individuals to population
    for _ in range(10):
        n = random.randint(4, 8)
        max_val = random.randint(20, 300)
        elems = random.sample(range(1, max_val+1), n-1)
        current = sorted([0] + elems)
        sc = score_of(current)
        population.append((sc, current))
    
    population.sort(key=lambda x: x[0], reverse=True)
    
    best_score = population[0][0]
    best_candidate = population[0][1][:]
    
    pop_size = 20
    mutation_rate = 0.3
    crossover_rate = 0.7
    
    while time.time() - start_time < time_limit:
        # Selection: Keep top half, replace bottom half
        # Ensure population size is consistent
        if len(population) < pop_size:
            # Fill with random or mutated best
            while len(population) < pop_size:
                parent = population[random.randint(0, len(population)-1)][1]
                new_ind = parent[:]
                # Simple mutation
                if len(new_ind) > 3 and random.random() < mutation_rate:
                    idx = random.randint(1, len(new_ind)-1)
                    new_ind[idx] = random.randint(1, 300)
                    new_ind = sorted(set(new_ind))
                if len(new_ind) < 4:
                    new_ind = parent[:]
                sc = score_of(new_ind)
                population.append((sc, new_ind))
        
        population.sort(key=lambda x: x[0], reverse=True)
        
        # Update best
        if population[0][0] > best_score:
            best_score = population[0][0]
            best_candidate = population[0][1][:]
            
        # Generate new generation
        new_population = []
        # Elitism: Keep top 2
        for i in range(2):
            new_population.append(population[i])
            
        while len(new_population) < pop_size and time.time() - start_time < time_limit:
            # Tournament selection for parents
            p1 = population[random.randint(0, 4)][1]
            p2 = population[random.randint(0, 4)][1]
            
            if random.random() < crossover_rate and len(p1) == len(p2) and len(p1) > 4:
                # Uniform crossover
                child = []
                for i in range(len(p1)):
                    if random.random() < 0.5:
                        child.append(p1[i])
                    else:
                        child.append(p2[i])
                child = sorted(set(child))
            else:
                # Clone one parent
                child = p1[:] if random.random() < 0.5 else p2[:]
                
            # Mutation
            if random.random() < mutation_rate:
                # Randomly add, remove, or replace
                move = random.choice(['replace', 'add', 'remove'])
                if move == 'replace' and len(child) > 3:
                    idx = random.randint(1, len(child)-1)
                    child[idx] = random.randint(1, 300)
                    child = sorted(set(child))
                elif move == 'add' and len(child) < 9:
                    child.append(random.randint(1, 300))
                    child = sorted(set(child))
                elif move == 'remove' and len(child) > 4:
                    idx = random.randint(1, len(child)-1)
                    child.pop(idx)
                    child = sorted(set(child))
            
            # Ensure 0 is in set and set is valid
            if 0 not in child:
                child = [0] + [x for x in child if x > 0]
            child = sorted(set(child))
            if len(child) < 3:
                continue
                
            sc = score_of(child)
            new_population.append((sc, child))
            
        population = new_population
        
    return best_candidate

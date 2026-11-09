import random
import time
import numpy as np

def construct():
    n = 256
    random.seed(42)

    # Precompute 3-APs and 4-APs
    three_aps = []
    four_aps = []
    
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 2 + 1):
            b = a + d
            c = a + 2 * d
            if c <= n:
                three_aps.append((a, b, c))
                
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e <= n:
                four_aps.append((a, b, c, e))

    # Precompute neighbors for fast operations
    # three_aps_with[x] = list of (y, z) such that (x,y,z) is a 3-AP
    three_aps_with = {i: [] for i in range(1, n + 1)}
    for a, b, c in three_aps:
        three_aps_with[a].append((b, c))
        three_aps_with[b].append((a, c))
        three_aps_with[c].append((a, b))
        
    # four_aps_with[x] = list of (y, z, w) such that (x,y,z,w) is a 4-AP
    four_aps_with = {i: [] for i in range(1, n + 1)}
    for a, b, c, d in four_aps:
        four_aps_with[a].append((b, c, d))
        four_aps_with[b].append((a, c, d))
        four_aps_with[c].append((a, b, d))
        four_aps_with[d].append((a, b, c))

    def count_3_aps_fast(S):
        count = 0
        for x in S:
            for y, z in three_aps_with[x]:
                if y in S and z in S:
                    count += 1
        return count // 3

    def is_valid_addition(S, x):
        # Check if adding x to S creates a 4-AP
        for y, z, w in four_aps_with[x]:
            if y in S and z in S and w in S:
                return False
        return True

    def repair_smart(S):
        # Remove elements to break all 4-APs
        # Greedy: Find a 4-AP, remove the element with min loss in 3-APs
        changed = True
        while changed:
            changed = False
            for a, b, c, d in four_aps:
                if a in S and b in S and c in S and d in S:
                    # Found 4-AP
                    min_loss = float('inf')
                    victim = None
                    for x in [a, b, c, d]:
                        loss = 0
                        for y, z in three_aps_with[x]:
                            if y in S and z in S:
                                loss += 1
                        if loss < min_loss:
                            min_loss = loss
                            victim = x
                    S.remove(victim)
                    changed = True
                    break
        return S

    def generate_greedy_valid():
        S = set()
        elems = list(range(1, n + 1))
        random.shuffle(elems)
        for x in elems:
            if is_valid_addition(S, x):
                S.add(x)
        return S

    # GA Setup
    pop_size = 60
    population = [generate_greedy_valid() for _ in range(pop_size)]
    
    start_time = time.time()
    timeout = 55.0
    best_fitness = -1
    best_set = set()
    
    # Initial evaluation
    for S in population:
        f = count_3_aps_fast(S)
        if f > best_fitness:
            best_fitness = f
            best_set = set(S)

    while time.time() - start_time < timeout:
        # Evaluate fitness
        fitnesses = [count_3_aps_fast(S) for S in population]
        
        # Update best
        max_idx = np.argmax(fitnesses)
        if fitnesses[max_idx] > best_fitness:
            best_fitness = fitnesses[max_idx]
            best_set = set(population[max_idx])
            
        # Selection, Crossover, Mutation
        new_population = []
        
        # Elitism: Keep top 5
        sorted_indices = np.argsort(fitnesses)[::-1]
        for i in range(min(5, pop_size)):
            new_population.append(set(population[sorted_indices[i]]))
            
        # Generate rest
        while len(new_population) < pop_size:
            # Tournament Selection
            p1_idx = random.randint(0, pop_size - 1)
            p2_idx = random.randint(0, pop_size - 1)
            if fitnesses[p1_idx] < fitnesses[p2_idx]:
                p1_idx, p2_idx = p2_idx, p1_idx
            p1 = population[p1_idx]
            p2 = population[p2_idx]
            
            # Crossover: Mix
            child = set()
            for x in range(1, n + 1):
                if x in p1 or x in p2:
                    if random.random() < 0.5:
                        child.add(x)
            
            # Repair
            repair_smart(child)
            
            # Mutation
            # 1. Try to add a random valid element
            if random.random() < 0.5:
                x = random.randint(1, n)
                if x not in child and is_valid_addition(child, x):
                    child.add(x)
            
            # 2. Try to swap
            if random.random() < 0.3 and len(child) > 0:
                x = random.choice(list(child))
                child.remove(x)
                y = random.randint(1, n)
                if y not in child and is_valid_addition(child, y):
                    child.add(y)
            
            new_population.append(child)
        
        population = new_population

    return [i for i in range(1, n + 1) if i in best_set]

import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    def fitness(orientation, n):
        # orientation: list of bits
        # Build matrix A: A[i][j] = 1 if i beats j
        A = np.zeros((n, n), dtype=np.uint8)
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                bit = orientation[idx]
                if bit == 0:
                    A[i][j] = 1
                else:
                    A[j][i] = 1
                idx += 1
        
        bad_count = 0
        # For each 4-set S
        for combo in itertools.combinations(range(n), 4):
            # Check if there is a vertex v NOT in combo that beats all in combo
            found = False
            for v in range(n):
                if v in combo:
                    continue
                # Check if v beats all in combo
                ok = True
                for u in combo:
                    if A[v][u] == 0:
                        ok = False
                        break
                if ok:
                    found = True
                    break
            if not found:
                bad_count += 1
        return bad_count

    def mutate(orientation, rate=0.05):
        new_orient = orientation.copy()
        for i in range(len(new_orient)):
            if np.random.random() < rate:
                new_orient[i] = 1 - new_orient[i]
        return new_orient

    def crossover(o1, o2):
        mask = np.random.random(len(o1)) < 0.5
        return np.where(mask, o1, o2)

    best_score = float('inf')
    best_orient = None
    best_n = 31
    
    for n in range(31, 68):
        if time.time() > deadline - 5:
            break
            
        pop_size = 50
        time_per_n = 10
        
        pop = [np.random.randint(0, 2, size=n*(n-1)//2) for _ in range(pop_size)]
        
        best_in_pop = None
        best_fit = float('inf')
        
        gen_start = time.time()
        gen = 0
        while gen < 200 and (time.time() - gen_start) < time_per_n:
            fits = []
            for ind in pop:
                f = fitness(ind, n)
                fits.append(f)
                if f < best_fit:
                    best_fit = f
                    best_in_pop = ind.copy()
                    if f == 0:
                        return {'n': n, 'orientation': ind.tolist()}
            
            new_pop = []
            for _ in range(pop_size):
                candidates = np.random.randint(0, pop_size, size=5)
                best_cand = min(candidates, key=lambda i: fits[i])
                new_pop.append(pop[best_cand].copy())
            
            for i in range(0, pop_size, 2):
                if i+1 < pop_size:
                    c1 = crossover(new_pop[i], new_pop[i+1])
                    c2 = crossover(new_pop[i], new_pop[i+1])
                    new_pop[i] = mutate(c1)
                    new_pop[i+1] = mutate(c2)
            
            pop = new_pop
            gen += 1
            
        if best_in_pop is not None:
            if best_fit < best_score:
                best_score = best_fit
                best_orient = best_in_pop
                best_n = n

    if best_orient is not None:
        return {'n': best_n, 'orientation': best_orient.tolist()}
        
    # Fallback: Return a random one (likely invalid, but better than crash)
    # Or a transitive one? Transitive is invalid for this interpretation.
    return {'n': 31, 'orientation': [0]* (31*30//2)}


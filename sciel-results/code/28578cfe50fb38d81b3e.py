import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Helper to get a valid small coloring (seed logic)
    def get_seed_coloring(n=9):
        return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}

    # Try n from 34 down to 10
    for n in range(34, 9, -1):
        if time.time() > deadline - 10.0:
            break
            
        # Precompute structures for n
        triples = list(combinations(range(n), 3))
        M = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        k4_neighbors = [[] for _ in range(M)]
        k5_neighbors = [[] for _ in range(M)]
        
        for i, (a,b,c) in enumerate(triples):
            # K4s
            for x in range(n):
                if x in (a,b,c): continue
                t1 = tuple(sorted((a,b,x)))
                t2 = tuple(sorted((a,c,x)))
                t3 = tuple(sorted((b,c,x)))
                k4_neighbors[i].append([triple_to_idx[t1], triple_to_idx[t2], triple_to_idx[t3]])
            
            # K5s
            others = [v for v in range(n) if v not in (a,b,c)]
            for y, z in combinations(others, 2):
                all_triples_in_5 = list(combinations(sorted([a,b,c,y,z]), 3))
                others_in_5 = [t for t in all_triples_in_5 if t != (a,b,c)]
                indices = [triple_to_idx[t] for t in others_in_5]
                k5_neighbors[i].append(indices)

        # Initialize colors
        p_red = 0.4
        colors = np.random.choice([0, 1], size=M, p=[p_red, 1-p_red])
        
        def calc_energy_fast(colors):
            red_k4_count = 0
            for i in range(M):
                if colors[i] == 0:
                    for neigh in k4_neighbors[i]:
                        if all(colors[j] == 0 for j in neigh):
                            red_k4_count += 1
            red_k4_count //= 4
            
            blue_k5_count = 0
            for i in range(M):
                if colors[i] == 1:
                    for neigh in k5_neighbors[i]:
                        if all(colors[j] == 1 for j in neigh):
                            blue_k5_count += 1
            blue_k5_count //= 10
            
            return red_k4_count + blue_k5_count

        E = calc_energy_fast(colors)
        best_E = E
        best_colors = colors.copy()
        
        if E == 0:
            return {'n': n, 'colors': best_colors.tolist()}
            
        T = 1.0
        cooling = 0.99
        # Estimate steps based on time
        # Let's do 20000 steps per restart
        steps = 20000
        
        for step in range(steps):
            if time.time() > deadline - 5.0:
                break
                
            i = random.randint(0, M-1)
            old_val = colors[i]
            new_val = 1 - old_val
            
            c_red_k4 = 0
            for neigh in k4_neighbors[i]:
                if all(colors[j] == 0 for j in neigh):
                    c_red_k4 += 1
                    
            c_blue_k5 = 0
            for neigh in k5_neighbors[i]:
                if all(colors[j] == 1 for j in neigh):
                    c_blue_k5 += 1
            
            if old_val == 0:
                delta = -c_red_k4 + c_blue_k5
            else:
                delta = c_red_k4 - c_blue_k5
                
            if delta < 0:
                colors[i] = new_val
                E += delta
                if E < best_E:
                    best_E = E
                    best_colors = colors.copy()
                    if E == 0:
                        return {'n': n, 'colors': best_colors.tolist()}
            else:
                if random.random() < np.exp(-delta / T):
                    colors[i] = new_val
                    E += delta
                    
            T *= cooling
            
        if best_E == 0:
            return {'n': n, 'colors': best_colors.tolist()}
            
    # Fallback
    return get_seed_coloring()

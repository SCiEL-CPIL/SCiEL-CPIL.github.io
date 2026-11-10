import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    n = 34
    
    triples = list(combinations(range(n), 3))
    T = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    k4_sets = list(combinations(range(n), 4))
    K4 = len(k4_sets)
    
    k5_sets = list(combinations(range(n), 5))
    K5 = len(k5_sets)
    
    triple_to_k4s = [[] for _ in range(T)]
    triple_to_k5s = [[] for _ in range(T)]
    
    k4_triples = []
    for i, k4 in enumerate(k4_sets):
        k4_triples_idx = []
        for j in range(4):
            t = tuple(sorted([k4[k] for k in range(4) if k != j]))
            idx = triple_to_idx[t]
            k4_triples_idx.append(idx)
            triple_to_k4s[idx].append(i)
        k4_triples.append(k4_triples_idx)
        
    k5_triples = []
    for i, k5 in enumerate(k5_sets):
        k5_triples_idx = []
        for comb in combinations(range(5), 3):
            t = tuple(sorted([k5[comb[0]], k5[comb[1]], k5[comb[2]]]))
            idx = triple_to_idx[t]
            k5_triples_idx.append(idx)
            triple_to_k5s[idx].append(i)
        k5_triples.append(k5_triples_idx)

    colors = [random.randint(0, 1) for _ in range(T)]
    
    # Counts of red edges in each K4, blue edges in each K5
    red_count_K4 = [0] * K4
    blue_count_K5 = [0] * K5
    
    cost = 0
    
    # Initialize counts
    for i in range(K4):
        ts = k4_triples[i]
        cnt = 0
        for t in ts:
            if colors[t] == 0:
                cnt += 1
        red_count_K4[i] = cnt
        if cnt == 4:
            cost += 1
            
    for i in range(K5):
        ts = k5_triples[i]
        cnt = 0
        for t in ts:
            if colors[t] == 1:
                cnt += 1
        blue_count_K5[i] = cnt
        if cnt == 10:
            cost += 1
            
    best_cost = cost
    best_colors = colors[:]
    
    if cost == 0:
        return {'n': n, 'colors': colors}
        
    T_curr = 1.0
    T_min = 0.001
    cooling = 0.9995
    
    while time.time() < deadline:
        if T_curr < T_min:
            T_curr = T_min
            
        t_idx = random.randint(0, T - 1)
        old_c = colors[t_idx]
        new_c = 1 - old_c
        
        delta_cost = 0
        
        # Update K4s
        for k4_idx in triple_to_k4s[t_idx]:
            if old_c == 0:
                # Removing a red edge
                red_count_K4[k4_idx] -= 1
                if red_count_K4[k4_idx] == 3:
                     # It was 4 (cost 1), now 3 (cost 0)
                     delta_cost -= 1
            else:
                # Adding a red edge (was blue, now red)
                red_count_K4[k4_idx] += 1
                if red_count_K4[k4_idx] == 4:
                     # It was 3 (cost 0), now 4 (cost 1)
                     delta_cost += 1
                    
        # Update K5s
        for k5_idx in triple_to_k5s[t_idx]:
            if old_c == 1:
                # Removing a blue edge
                blue_count_K5[k5_idx] -= 1
                if blue_count_K5[k5_idx] == 9:
                     # It was 10 (cost 1), now 9 (cost 0)
                     delta_cost -= 1
            else:
                # Adding a blue edge (was red, now blue)
                blue_count_K5[k5_idx] += 1
                if blue_count_K5[k5_idx] == 10:
                     # It was 9 (cost 0), now 10 (cost 1)
                     delta_cost += 1
        
        # Accept/Reject
        if delta_cost < 0:
            accept = True
        else:
            if delta_cost == 0:
                accept = True
            else:
                ratio = math.exp(-delta_cost / T_curr)
                accept = random.random() < ratio
                
        if accept:
            colors[t_idx] = new_c
            cost += delta_cost
            
            if cost < best_cost:
                best_cost = cost
                best_colors = colors[:]
                if best_cost == 0:
                    return {'n': n, 'colors': best_colors}
                    
        T_curr *= cooling
        
    if best_cost != 0:
        n_seed = 9
        return {'n': n_seed, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n_seed), 3)]}
    else:
        return {'n': n, 'colors': best_colors}

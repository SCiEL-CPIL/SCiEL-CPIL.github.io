import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for safety
    
    # Try n from 34 down to 10
    for n in range(34, 9, -1):
        # Check if we have enough time left for this n
        # Precomputation takes ~1-2s for n=34
        # Search takes variable time
        if time.time() > deadline - 20:
            continue
            
        # Precompute indices
        t0 = time.time()
        triples = list(combinations(range(n), 3))
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        k4_indices = []
        for comb in combinations(range(n), 4):
            t1 = (comb[0], comb[1], comb[2])
            t2 = (comb[0], comb[1], comb[3])
            t3 = (comb[0], comb[2], comb[3])
            t4 = (comb[1], comb[2], comb[3])
            k4_indices.append([triple_to_idx[t1], triple_to_idx[t2], triple_to_idx[t3], triple_to_idx[t4]])
            
        k5_indices = []
        for comb in combinations(range(n), 5):
            idxs = []
            for i in range(5):
                for j in range(i+1, 5):
                    for k in range(j+1, 5):
                        idxs.append(triple_to_idx[(comb[i], comb[j], comb[k])])
            k5_indices.append(idxs)
            
        k4_arr = np.array(k4_indices, dtype=np.int32)
        k5_arr = np.array(k5_indices, dtype=np.int32)
        
        # Check time after precomputation
        if time.time() > deadline - 15:
            continue

        def get_cost(colors):
            red_k4 = np.sum(np.all(colors[k4_arr] == 0, axis=1))
            blue_k5 = np.sum(np.all(colors[k5_arr] == 1, axis=1))
            return red_k4 + blue_k5
        
        n_triples = len(triples)
        
        # Search parameters
        # Allocate remaining time for this n, capped at 30s
        time_budget = min(30, deadline - time.time() - 5)
        if time_budget <= 0:
            continue
            
        # We will try several restarts
        # Each restart gets a fraction of the time
        n_restarts = max(1, int(time_budget / 5)) # ~5s per restart
        best_cost = float('inf')
        best_colors = None
        
        for r in range(n_restarts):
            if time.time() > deadline - 10:
                break
                
            colors = np.random.randint(0, 2, n_triples, dtype=np.int8)
            cost = get_cost(colors)
            if cost == 0:
                return {'n': n, 'colors': colors.tolist()}
                
            best_local_cost = cost
            best_local_colors = colors.copy()
            
            # Hill Climbing with Random Restarts on Stagnation
            t_start = time.time()
            stagnation_counter = 0
            max_stagnation = 100 # Steps without improvement
            
            while time.time() - t_start < (time_budget / n_restarts):
                if time.time() > deadline - 10:
                    break
                
                # Pick a random triple to flip
                idx = random.randint(0, n_triples - 1)
                old_val = colors[idx]
                colors[idx] = 1 - old_val
                
                new_cost = get_cost(colors)
                
                if new_cost < cost:
                    cost = new_cost
                    stagnation_counter = 0
                    if new_cost == 0:
                        return {'n': n, 'colors': colors.tolist()}
                    if new_cost < best_local_cost:
                        best_local_cost = new_cost
                        best_local_colors = colors.copy()
                else:
                    # Revert
                    colors[idx] = old_val
                    stagnation_counter += 1
                    
                # If stuck, do a random kick (flip 5 random triples)
                if stagnation_counter > max_stagnation:
                    for _ in range(5):
                        k = random.randint(0, n_triples - 1)
                        colors[k] = 1 - colors[k]
                    cost = get_cost(colors)
                    stagnation_counter = 0
                    if cost < best_local_cost:
                        best_local_cost = cost
                        best_local_colors = colors.copy()
                    
            if best_local_cost < best_cost:
                best_cost = best_local_cost
                best_colors = best_local_colors
                
        if best_cost == 0 and best_colors is not None:
            return {'n': n, 'colors': best_colors.tolist()}
            
    # Fallback to seed
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}

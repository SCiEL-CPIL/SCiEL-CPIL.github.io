import numpy as np
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Try n from 34 down to 1
    for n in range(34, 0, -1):
        if time.time() > deadline - 5:
            break
            
        # Precompute structures
        triples = list(combinations(range(n), 3))
        triple_index = {t: i for i, t in enumerate(triples)}
        num_triples = len(triples)
        
        # Map triple index to list of K4 indices and K5 indices
        # This is needed for fast delta calculation
        # k4s containing triple i
        # k5s containing triple i
        
        # First, list all K4s and K5s
        k4_list = list(combinations(range(n), 4))
        k5_list = list(combinations(range(n), 5))
        num_k4 = len(k4_list)
        num_k5 = len(k5_list)
        
        # Precompute triple to k4/k5 mapping
        # triple_to_k4s[i] = list of k4 indices
        triple_to_k4s = [[] for _ in range(num_triples)]
        for k4_idx, quad in enumerate(k4_list):
            a, b, c, d = quad
            t1 = triple_index[(a, b, c)]
            t2 = triple_index[(a, b, d)]
            t3 = triple_index[(a, c, d)]
            t4 = triple_index[(b, c, d)]
            triple_to_k4s[t1].append(k4_idx)
            triple_to_k4s[t2].append(k4_idx)
            triple_to_k4s[t3].append(k4_idx)
            triple_to_k4s[t4].append(k4_idx)
            
        triple_to_k5s = [[] for _ in range(num_triples)]
        for k5_idx, quint in enumerate(k5_list):
            for t in combinations(quint, 3):
                t_idx = triple_index[t]
                triple_to_k5s[t_idx].append(k5_idx)
                
        # Initialize colors
        colors = np.random.randint(0, 2, num_triples).astype(np.int8)
        
        # Initialize counts
        # k4_red_counts[j]: number of red triples in k4_list[j]
        k4_red_counts = np.zeros(num_k4, dtype=np.int8)
        for k4_idx, quad in enumerate(k4_list):
            a, b, c, d = quad
            t1 = triple_index[(a, b, c)]
            t2 = triple_index[(a, b, d)]
            t3 = triple_index[(a, c, d)]
            t4 = triple_index[(b, c, d)]
            count = (colors[t1] == 0) + (colors[t2] == 0) + (colors[t3] == 0) + (colors[t4] == 0)
            k4_red_counts[k4_idx] = count
            
        # k5_blue_counts[j]: number of blue triples in k5_list[j]
        k5_blue_counts = np.zeros(num_k5, dtype=np.int8)
        for k5_idx, quint in enumerate(k5_list):
            count = 0
            for t in combinations(quint, 3):
                t_idx = triple_index[t]
                if colors[t_idx] == 1:
                    count += 1
            k5_blue_counts[k5_idx] = count
            
        current_red_violations = np.sum(k4_red_counts == 0)
        current_blue_violations = np.sum(k5_blue_counts == 10)
        current_cost = current_red_violations + current_blue_violations
        
        if current_cost == 0:
            return {'n': n, 'colors': colors.tolist()}
            
        # Simulated Annealing with fast delta
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        sa_deadline = time.time() + 10.0 # 10s per n
        
        while time.time() < sa_deadline:
            i = np.random.randint(num_triples)
            old_val = colors[i]
            new_val = 1 - old_val
            
            delta = 0
            
            # Update K4 violations
            if old_val == 0: # Red -> Blue
                # k4_red_counts decreases by 1
                for k4_idx in triple_to_k4s[i]:
                    old_count = k4_red_counts[k4_idx]
                    if old_count == 1:
                        # Was not violation (1 red), becomes violation (0 red)
                        delta += 1
                    # If old_count > 1, no change in violation status
                    # If old_count == 0, impossible since i was red
                    k4_red_counts[k4_idx] = old_count - 1
            else: # Blue -> Red
                # k4_red_counts increases by 1
                for k4_idx in triple_to_k4s[i]:
                    old_count = k4_red_counts[k4_idx]
                    if old_count == 0:
                        # Was violation (0 red), becomes not violation (1 red)
                        delta -= 1
                    k4_red_counts[k4_idx] = old_count + 1
                    
            # Update K5 violations
            if old_val == 0: # Red -> Blue
                # k5_blue_counts increases by 1
                for k5_idx in triple_to_k5s[i]:
                    old_count = k5_blue_counts[k5_idx]
                    if old_count == 9:
                        # Was not violation (9 blue), becomes violation (10 blue)
                        delta += 1
                    k5_blue_counts[k5_idx] = old_count + 1
            else: # Blue -> Red
                # k5_blue_counts decreases by 1
                for k5_idx in triple_to_k5s[i]:
                    old_count = k5_blue_counts[k5_idx]
                    if old_count == 10:
                        # Was violation (10 blue), becomes not violation (9 blue)
                        delta -= 1
                    k5_blue_counts[k5_idx] = old_count - 1
            
            new_cost = current_cost + delta
            
            if delta < 0 or (np.random.rand() < np.exp(-delta / T)):
                colors[i] = new_val
                current_cost = new_cost
            else:
                # Revert
                colors[i] = old_val
                # Revert counts
                if old_val == 0:
                    for k4_idx in triple_to_k4s[i]:
                        k4_red_counts[k4_idx] += 1
                    for k5_idx in triple_to_k5s[i]:
                        k5_blue_counts[k5_idx] -= 1
                else:
                    for k4_idx in triple_to_k4s[i]:
                        k4_red_counts[k4_idx] -= 1
                    for k5_idx in triple_to_k5s[i]:
                        k5_blue_counts[k5_idx] += 1
                        
            T *= cooling
            if T < T_min:
                T = T_min
                
            if current_cost == 0:
                return {'n': n, 'colors': colors.tolist()}
                
    return {'n': 1, 'colors': []}

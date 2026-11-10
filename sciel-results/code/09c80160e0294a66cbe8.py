"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit
    
    def get_precomputed_indices(n):
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Precompute K4 triple indices
        k4_triples = []
        for k4 in combinations(range(n), 4):
            ts = []
            for i in range(4):
                t = tuple(sorted(k4[:i] + k4[i+1:]))
                ts.append(triple_to_idx[t])
            k4_triples.append(ts)
        k4_arr = np.array(k4_triples, dtype=np.int32) if k4_triples else np.empty((0,4), dtype=np.int32)
        
        # Precompute K5 triple indices
        k5_triples = []
        for k5 in combinations(range(n), 5):
            ts = []
            for combo in combinations(range(5), 3):
                t = tuple(sorted([k5[i] for i in combo]))
                ts.append(triple_to_idx[t])
            k5_triples.append(ts)
        k5_arr = np.array(k5_triples, dtype=np.int32) if k5_triples else np.empty((0,10), dtype=np.int32)
        
        # Precompute incident triple indices for each vertex (for incremental updates)
        # For each vertex v, list of triple indices that include v
        vertex_incident = [[] for _ in range(n)]
        for i, (a, b, c) in enumerate(triples):
            vertex_incident[a].append(i)
            vertex_incident[b].append(i)
            vertex_incident[c].append(i)
        vertex_incident = [np.array(v, dtype=np.int32) for v in vertex_incident]
        
        return num_triples, k4_arr, k5_arr, vertex_incident

    def count_violations_vec(colors, k4_arr, k5_arr):
        if len(k4_arr) > 0:
            red_k4_count = np.sum(colors[k4_arr], axis=1)
            red_violations = np.sum(red_k4_count == 0)
        else:
            red_violations = 0
            
        if len(k5_arr) > 0:
            blue_k5_count = np.sum(colors[k5_arr], axis=1)
            blue_violations = np.sum(blue_k5_count == 10)
        else:
            blue_violations = 0
            
        return int(red_violations + blue_violations)

    best_n = 0
    best_colors = None
    
    # Try increasing n
    # Start from 10, go up to 34
    for n in range(10, 35):
        if time.time() > deadline - 5.0:
            break
            
        # Time budget per n
        # Smaller n are easier, larger n are harder.
        # Let's give a fixed 3s to each, but maybe less for very small ones?
        # 10 to 34 is 25 values. 25 * 3 = 75s.
        time_budget = 3.0
        n_start_time = time.time()
        
        num_triples, k4_arr, k5_arr, vertex_incident = get_precomputed_indices(n)
        
        # Initialize colors
        colors = np.random.randint(0, 2, size=num_triples).astype(np.int8)
        
        current_v = count_violations_vec(colors, k4_arr, k5_arr)
        best_v = current_v
        best_cols = colors.copy()
        
        # SA Parameters
        T = 10.0
        T_end = 0.1
        cooling_rate = 0.999
        
        # Incremental violation tracking
        # For each K4, track how many of its 4 triples are red (0)
        # For each K5, track how many of its 10 triples are blue (1)
        
        if len(k4_arr) > 0:
            # k4_arr shape: (num_k4, 4)
            # k4_red_counts[k4_idx] = number of red triples in that K4
            k4_red_counts = np.sum(colors[k4_arr] == 0, axis=1).astype(np.int32)
            # red_violations = sum of (k4_red_counts == 4)
            current_red_viol = np.sum(k4_red_counts == 4)
        else:
            k4_red_counts = None
            current_red_viol = 0
            
        if len(k5_arr) > 0:
            # k5_arr shape: (num_k5, 10)
            # k5_blue_counts[k5_idx] = number of blue triples in that K5
            k5_blue_counts = np.sum(colors[k5_arr] == 1, axis=1).astype(np.int32)
            current_blue_viol = np.sum(k5_blue_counts == 10)
        else:
            k5_blue_counts = None
            current_blue_viol = 0
            
        current_v = current_red_viol + current_blue_viol
        best_v = current_v
        best_cols = colors.copy()
        
        while time.time() < n_start_time + time_budget:
            idx = random.randint(0, num_triples - 1)
            old_val = colors[idx]
            new_val = 1 - old_val
            
            # Determine which K4s and K5s are affected by flipping triple idx
            # A triple (a,b,c) is part of K4s {a,b,c,d} for d != a,b,c
            # and part of K5s {a,b,c,d,e} for d,e != a,b,c
            
            # For incremental update, we need to know which K4s contain this triple
            # and which K5s contain this triple.
            # This requires precomputation: for each triple, which K4s/K5s include it.
            # Let's precompute this.
            
            # Actually, let's just do the full re-evaluation for simplicity and correctness.
            # The vectorized count is fast enough for n up to 34.
            
            colors[idx] = new_val
            new_v = count_violations_vec(colors, k4_arr, k5_arr)
            delta_v = new_v - current_v
            
            if delta_v < 0:
                current_v = new_v
                if new_v < best_v:
                    best_v = new_v
                    best_cols = colors.copy()
            else:
                if delta_v == 0:
                    current_v = new_v
                else:
                    if T > T_end:
                        prob = np.exp(-delta_v / T)
                        if random.random() < prob:
                            current_v = new_v
                        else:
                            colors[idx] = old_val
                    else:
                        colors[idx] = old_val
            
            T *= cooling_rate
        
        if best_v == 0:
            best_n = n
            best_colors = best_cols.tolist()
            
    if best_n == 0:
        # Fallback to seed
        n = 9
        return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
        
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END

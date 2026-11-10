"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    def solve_for_n(n, time_limit):
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        k4s = list(combinations(range(n), 4))
        k5s = list(combinations(range(n), 5))
        num_k4 = len(k4s)
        num_k5 = len(k5s)
        
        # Map K4/K5 to triple indices
        k4_triples = []
        for k4 in k4s:
            ts = []
            for i in range(4):
                t = tuple(sorted(k4[:i] + k4[i+1:]))
                ts.append(triple_to_idx[t])
            k4_triples.append(ts)
            
        k5_triples = []
        for k5 in k5s:
            ts = []
            for combo in combinations(range(5), 3):
                t = tuple(sorted([k5[i] for i in combo]))
                ts.append(triple_to_idx[t])
            k5_triples.append(ts)
            
        # Reverse maps: triple index -> list of k4 indices, triple index -> list of k5 indices
        triple_to_k4s = [[] for _ in range(num_triples)]
        for k_idx, ts in enumerate(k4_triples):
            for t_idx in ts:
                triple_to_k4s[t_idx].append(k_idx)
                
        triple_to_k5s = [[] for _ in range(num_triples)]
        for k_idx, ts in enumerate(k5_triples):
            for t_idx in ts:
                triple_to_k5s[t_idx].append(k_idx)

        # Random initialization
        colors = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
        
        # Counts
        red_count_k4 = np.zeros(num_k4, dtype=np.int8)
        blue_count_k5 = np.zeros(num_k5, dtype=np.int8)
        
        # Initialize counts
        for k_idx, ts in enumerate(k4_triples):
            for t_idx in ts:
                if colors[t_idx] == 0:
                    red_count_k4[k_idx] += 1
                    
        for k_idx, ts in enumerate(k5_triples):
            for t_idx in ts:
                if colors[t_idx] == 1:
                    blue_count_k5[k_idx] += 1
                    
        cur_violations = np.sum(red_count_k4 == 4) + np.sum(blue_count_k5 == 10)
        
        if cur_violations == 0:
            return n, colors, True

        # Local Search
        T = 1.0
        cooling = 0.9995
        min_T = 0.001
        steps = 0
        max_steps = 2000000 # Cap steps
        
        while time.time() < time_limit and steps < max_steps:
            t_idx = np.random.randint(0, num_triples)
            old_color = colors[t_idx]
            new_color = 1 - old_color
            
            delta_v = 0
            
            # Update Red K4 counts
            for k_idx in triple_to_k4s[t_idx]:
                if old_color == 0: # Red -> Blue
                    if red_count_k4[k_idx] == 4:
                        delta_v -= 1
                    red_count_k4[k_idx] -= 1
                else: # Blue -> Red
                    if red_count_k4[k_idx] == 3:
                        delta_v += 1
                    red_count_k4[k_idx] += 1

            # Update Blue K5 counts
            for k_idx in triple_to_k5s[t_idx]:
                if old_color == 0: # Red -> Blue
                    if blue_count_k5[k_idx] == 9:
                        delta_v += 1
                    blue_count_k5[k_idx] += 1
                else: # Blue -> Red
                    if blue_count_k5[k_idx] == 10:
                        delta_v -= 1
                    blue_count_k5[k_idx] -= 1
            
            accepted = False
            if delta_v < 0:
                accepted = True
            else:
                if T > min_T:
                    prob = math.exp(-delta_v / T)
                    if np.random.random() < prob:
                        accepted = True
            
            if accepted:
                colors[t_idx] = new_color
                cur_violations += delta_v
                if cur_violations == 0:
                    return n, colors, True
            else:
                # Revert counts
                for k_idx in triple_to_k4s[t_idx]:
                    if old_color == 0:
                        red_count_k4[k_idx] += 1
                    else:
                        red_count_k4[k_idx] -= 1
                for k_idx in triple_to_k5s[t_idx]:
                    if old_color == 0:
                        blue_count_k5[k_idx] -= 1
                    else:
                        blue_count_k5[k_idx] += 1

            T *= cooling
            steps += 1
            
        return n, colors, False

    # Try n=34
    # Allocate 100s for n=34
    n, colors, is_valid = solve_for_n(34, start_time + 100.0)
    if is_valid:
        return {'n': n, 'colors': colors.tolist()}
        
    # If 34 fails, try 33 with remaining time
    if time.time() < deadline - 5.0:
        n, colors, is_valid = solve_for_n(33, deadline - 5.0)
        if is_valid:
            return {'n': n, 'colors': colors.tolist()}
            
    # Fallback to n=9 (trivial)
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}

# EVOLVE-BLOCK-END

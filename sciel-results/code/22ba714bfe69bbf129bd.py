"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    best_n = 9
    best_colors = None
    
    # Precompute triple index mapping helper
    def make_triple_map(n):
        triples = list(combinations(range(n), 3))
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        return triples, triple_to_idx, len(triples)
    
    # Precompute K4 and K5 triple index arrays for a given n
    def precompute_k4_k5(n, triples, triple_to_idx):
        k4s = list(combinations(range(n), 4))
        k5s = list(combinations(range(n), 5))
        
        # K4: for each 4-set, get the 4 triple indices
        k4_triples = np.zeros((len(k4s), 4), dtype=np.int32)
        for i, k4 in enumerate(k4s):
            for j in range(4):
                t = tuple(sorted(k4[:j] + k4[j+1:]))
                k4_triples[i, j] = triple_to_idx[t]
        
        # K5: for each 5-set, get the 10 triple indices
        k5_triples = np.zeros((len(k5s), 10), dtype=np.int32)
        for i, k5 in enumerate(k5s):
            count = 0
            for combo in combinations(range(5), 3):
                t = tuple(sorted([k5[c] for c in combo]))
                k5_triples[i, count] = triple_to_idx[t]
                count += 1
        
        return k4_triples, k5_triples
    
    # Vectorized violation count
    def count_violations_np(colors, k4_triples, k5_triples):
        # colors is a numpy array of 0s and 1s
        # Red K4: all 4 triples are 0 (red)
        red_k4_violations = np.all(colors[k4_triples] == 0, axis=1).sum()
        # Blue K5: all 10 triples are 1 (blue)
        blue_k5_violations = np.all(colors[k5_triples] == 1, axis=1).sum()
        return int(red_k4_violations + blue_k5_violations)
    
    # Try n values from 12 up to 34
    for n in range(12, 35):
        if time.time() > deadline - 10.0:
            break
        
        n_start = time.time()
        time_budget = min(5.0, max(1.0, (deadline - 10.0 - n_start) / (35 - n)))
        
        try:
            triples, triple_to_idx, num_triples = make_triple_map(n)
            k4_triples, k5_triples = precompute_k4_k5(n, triples, triple_to_idx)
            
            # Multiple random restarts
            found_valid = False
            best_v_for_n = float('inf')
            best_cols_for_n = None
            
            num_restarts = 5
            for restart in range(num_restarts):
                if time.time() > n_start + time_budget:
                    break
                
                # Random initial coloring
                colors = np.random.randint(0, 2, size=num_triples).astype(np.int8)
                
                current_v = count_violations_np(colors, k4_triples, k5_triples)
                best_v = current_v
                best_cols = colors.copy()
                
                # Simulated annealing / hill climbing
                no_improve = 0
                max_no_improve = 200
                temp = 1.0
                
                while time.time() < n_start + time_budget and no_improve < max_no_improve:
                    # Flip a random triple
                    idx = np.random.randint(0, num_triples)
                    old_color = colors[idx]
                    colors[idx] = 1 - old_color
                    
                    new_v = count_violations_np(colors, k4_triples, k5_triples)
                    
                    if new_v < best_v:
                        best_v = new_v
                        best_cols = colors.copy()
                        no_improve = 0
                        temp *= 0.95
                    elif new_v <= current_v:
                        current_v = new_v
                        no_improve += 1
                    else:
                        # Accept with probability exp(-delta/temp)
                        delta = new_v - current_v
                        if random.random() < np.exp(-delta / max(temp, 0.1)):
                            current_v = new_v
                            no_improve += 1
                            temp *= 0.99
                        else:
                            colors[idx] = old_color
                            no_improve += 1
                            temp *= 0.99
                    
                    if best_v == 0:
                        break
                
                if best_v == 0:
                    found_valid = True
                    best_cols_for_n = best_cols
                    break
                elif best_v < best_v_for_n:
                    best_v_for_n = best_v
                    best_cols_for_n = best_cols
            
            if found_valid:
                best_n = n
                best_colors = best_cols_for_n.tolist()
                # Continue to try higher n
                
        except Exception:
            pass
    
    # Fallback: return a valid small coloring
    if best_n < 12 or best_colors is None:
        n = 9
        return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
    else:
        return {'n': best_n, 'colors': best_colors}

# EVOLVE-BLOCK-END

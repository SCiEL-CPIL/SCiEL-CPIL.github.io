import time
import random
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 145  # Leave 5s buffer for 150s limit
    
    # Precompute all 4-APs
    # a, a+d, a+2d, a+3d < n
    # d >= 1
    # a >= 0
    # a + 3d <= n - 1 => a <= n - 1 - 3d
    ap_list = []
    for d in range(1, (n - 1) // 3 + 1):
        max_a = n - 1 - 3 * d
        for a in range(max_a + 1):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Convert to numpy array for fast indexing
    # Shape: (num_aps, 4)
    ap_indices = np.array(ap_list, dtype=np.int32)
    num_aps = len(ap_indices)
    
    def thue_morse(n):
        # Returns list of 0/1
        return np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)

    def count_mono(x):
        # x is numpy array of 0/1
        cols = x[ap_indices]
        mono = (cols[:, 0] == cols[:, 1]) & (cols[:, 1] == cols[:, 2]) & (cols[:, 2] == cols[:, 3])
        return np.sum(mono)

    def compute_deltas(x):
        # x is numpy array of 0/1
        # Returns numpy array of deltas for each bit
        cols = x[ap_indices]
        c0, c1, c2, c3 = cols[:, 0], cols[:, 1], cols[:, 2], cols[:, 3]
        
        # Current mono status
        mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        M = mono.astype(np.int8)
        
        # Delta for flipping c0
        new_c0 = 1 - c0
        mono0 = (new_c0 == c1) & (c1 == c2) & (c2 == c3)
        C0 = mono0.astype(np.int8) - M
        
        # Delta for flipping c1
        new_c1 = 1 - c1
        mono1 = (c0 == new_c1) & (new_c1 == c2) & (c2 == c3)
        C1 = mono1.astype(np.int8) - M
        
        # Delta for flipping c2
        new_c2 = 1 - c2
        mono2 = (c0 == c1) & (c1 == new_c2) & (new_c2 == c3)
        C2 = mono2.astype(np.int8) - M
        
        # Delta for flipping c3
        new_c3 = 1 - c3
        mono3 = (c0 == c1) & (c1 == c2) & (c2 == new_c3)
        C3 = mono3.astype(np.int8) - M
        
        deltas = np.zeros(n, dtype=np.int32)
        np.add.at(deltas, ap_indices[:, 0], C0)
        np.add.at(deltas, ap_indices[:, 1], C1)
        np.add.at(deltas, ap_indices[:, 2], C2)
        np.add.at(deltas, ap_indices[:, 3], C3)
        
        return deltas

    def simulated_annealing(x_init, time_limit):
        x = x_init.copy()
        best_x = x.copy()
        best_score = count_mono(x)
        current_score = best_score
        
        # SA Parameters
        T = 5.0
        cooling_rate = 0.9995
        T_min = 0.01
        
        steps = 0
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            if T < T_min:
                # If T is too low, do a few hill climbing steps then break or restart
                # For simplicity, just break and let the caller restart
                break
            
            deltas = compute_deltas(x)
            
            # Pick a random bit
            k = random.randint(0, n - 1)
            delta = deltas[k]
            
            # Accept if improvement or with probability exp(-delta/T)
            if delta < 0:
                accept = True
            else:
                # delta >= 0
                # If delta is large, probability is small
                # Use log to avoid underflow?
                # exp(-delta/T)
                if delta > 50 * T: # exp(-50) is very small
                    accept = False
                else:
                    accept = random.random() < np.exp(-delta / T)
            
            if accept:
                x[k] = 1 - x[k]
                # Update score efficiently?
                # We can just recompute, or update incrementally.
                # Recomputing is O(N_aps). Incremental is O(degree(k)).
                # Since we already computed deltas (O(N_aps)), incremental is faster.
                # But we need the *new* score to check if it's better than best.
                # current_score += delta
                current_score += delta
                
                if current_score < best_score:
                    best_score = current_score
                    best_x = x.copy()
            
            T *= cooling_rate
            steps += 1
            
            # Check time periodically
            if steps % 1000 == 0 and time.time() > deadline - 2:
                break

        return best_x, best_score

    best_overall = None
    best_score_overall = float('inf')
    
    # Phase 1: Thue-Morse
    x_init = thue_morse(n)
    best_x, score = simulated_annealing(x_init, 60)
    if score < best_score_overall:
        best_score_overall = score
        best_overall = best_x
        
    # Phase 2: Random
    x_init = np.array([random.randint(0, 1) for _ in range(n)], dtype=np.int8)
    best_x, score = simulated_annealing(x_init, 40)
    if score < best_score_overall:
        best_score_overall = score
        best_overall = best_x
        
    # Phase 3: Thue-Morse with noise
    x_init = thue_morse(n).copy()
    # Flip 10% of bits
    mask = np.random.random(n) < 0.1
    x_init[mask] = 1 - x_init[mask]
    best_x, score = simulated_annealing(x_init, 40)
    if score < best_score_overall:
        best_score_overall = score
        best_overall = best_x

    if best_overall is None:
        return [0] * n
        
    return list(best_overall)


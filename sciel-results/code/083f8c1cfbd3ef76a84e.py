import numpy as np
import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Target n values to try
    target_ns = list(range(34, 19, -1))
    
    # Precomputation cache for structures
    # We will generate structures on the fly for each n
    
    def get_structures(n):
        """
        Returns:
        M4: np.array of shape (N_k4, 4)
        M5: np.array of shape (N_k5, 10)
        triple_to_k4s: list of lists, index by triple index
        triple_to_k5s: list of lists, index by triple index
        """
        triples = list(itertools.combinations(range(n), 3))
        triple_index = {t: i for i, t in enumerate(triples)}
        M = len(triples)
        
        # Generate K4s
        k4s = list(itertools.combinations(range(n), 4))
        N_k4 = len(k4s)
        M4 = np.zeros((N_k4, 4), dtype=np.int32)
        triple_to_k4s = [[] for _ in range(M)]
        
        for i, k4 in enumerate(k4s):
            # k4 is (a,b,c,d)
            # Triples: (a,b,c), (a,b,d), (a,c,d), (b,c,d)
            t1 = triple_index[(k4[0], k4[1], k4[2])]
            t2 = triple_index[(k4[0], k4[1], k4[3])]
            t3 = triple_index[(k4[0], k4[2], k4[3])]
            t4 = triple_index[(k4[1], k4[2], k4[3])]
            M4[i] = [t1, t2, t3, t4]
            triple_to_k4s[t1].append(i)
            triple_to_k4s[t2].append(i)
            triple_to_k4s[t3].append(i)
            triple_to_k4s[t4].append(i)
            
        # Generate K5s
        k5s = list(itertools.combinations(range(n), 5))
        N_k5 = len(k5s)
        M5 = np.zeros((N_k5, 10), dtype=np.int32)
        triple_to_k5s = [[] for _ in range(M)]
        
        for i, k5 in enumerate(k5s):
            # 10 triples from 5 vertices
            # itertools.combinations(k5, 3)
            for j, t in enumerate(itertools.combinations(k5, 3)):
                t_idx = triple_index[t]
                M5[i, j] = t_idx
                triple_to_k5s[t_idx].append(i)
                
        return M4, M5, triple_to_k4s, triple_to_k5s, triples

    def solve_for_n(n, time_limit):
        M4, M5, triple_to_k4s, triple_to_k5s, triples = get_structures(n)
        M = len(triples)
        
        # Initial random coloring
        colors = np.random.randint(0, 2, M, dtype=np.int8)
        
        # Initial sums
        sums4 = colors[M4].sum(axis=1)
        sums5 = colors[M5].sum(axis=1)
        
        # Initial violations
        V4 = np.sum(sums4 == 0)
        V5 = np.sum(sums5 == 10)
        V = V4 + V5
        
        if V == 0:
            return colors
        
        T = 1.0
        T_min = 0.001
        alpha = 0.999
        best_V = V
        best_colors = colors.copy()
        
        current_time = time.time()
        
        while time.time() < current_time + time_limit and T > T_min:
            # Pick random triple
            t = np.random.randint(0, M)
            c = colors[t]
            new_c = 1 - c
            
            delta_sum = 1 if c == 0 else -1
            
            delta_V = 0
            
            # Update K4s
            for k4_idx in triple_to_k4s[t]:
                old_sum = sums4[k4_idx]
                new_sum = old_sum + delta_sum
                old_v = 1 if old_sum == 0 else 0
                new_v = 1 if new_sum == 0 else 0
                delta_V += new_v - old_v
                sums4[k4_idx] = new_sum
                
            # Update K5s
            for k5_idx in triple_to_k5s[t]:
                old_sum = sums5[k5_idx]
                new_sum = old_sum + delta_sum
                old_v = 1 if old_sum == 10 else 0
                new_v = 1 if new_sum == 10 else 0
                delta_V += new_v - old_v
                sums5[k5_idx] = new_sum
                
            # Acceptance
            if delta_V < 0 or random.random() < np.exp(-delta_V / T):
                colors[t] = new_c
                V += delta_V
                if V < best_V:
                    best_V = V
                    best_colors = colors.copy()
                    if V == 0:
                        return colors
            else:
                # Rollback
                # Since we already updated sums, we need to revert
                for k4_idx in triple_to_k4s[t]:
                    sums4[k4_idx] -= delta_sum
                for k5_idx in triple_to_k5s[t]:
                    sums5[k5_idx] -= delta_sum
                
            T *= alpha
            
        return best_colors

    # Main loop
    for n in target_ns:
        if time.time() > deadline - 10:
            break
            
        # Estimate time per n
        # Smaller n is faster.
        # Let's allocate time based on n
        # n=34 is slowest.
        time_alloc = 5.0 if n > 30 else 10.0
        
        if time.time() + time_alloc > deadline:
             time_alloc = deadline - time.time() - 1.0
             
        if time_alloc <= 0:
             break
             
        sol = solve_for_n(n, time_alloc)
        
        # Check if valid
        # We need to re-verify because solve_for_n might return best_colors which might not be 0 violations
        # But if it returned early, it is 0.
        # Let's check V for the returned solution.
        # Actually, solve_for_n returns best_colors. We need to check if best_V was 0.
        # Let's modify solve_for_n to return (colors, V)
        
        # Re-implementing solve_for_n return logic for clarity in main loop
        # Let's just check the returned colors for validity using a fast check.
        
        M4, M5, triple_to_k4s, triple_to_k5s, triples = get_structures(n)
        sums4 = sol[M4].sum(axis=1)
        sums5 = sol[M5].sum(axis=1)
        V = np.sum(sums4 == 0) + np.sum(sums5 == 10)
        
        if V == 0:
            return {'n': n, 'colors': sol.tolist()}
            
    # If no solution found for large n, try to return a valid small one
    # Or just return the best found?
    # The task says "Maximize n". If we can't find valid for 34, we should try 33, etc.
    # If we fail all, we should return a valid small one to get score > 0.
    # The seed returns n=9.
    
    # Fallback: return a known valid small coloring
    # The seed logic: (a+b+c)%3==0 -> red(0), else blue(1)
    # Let's check if n=9 is valid.
    # For n=9, M=84.
    # It's likely valid or close.
    
    # Let's try to find ANY valid coloring for a small n quickly
    # e.g. n=10, 11...
    for n in range(15, 5, -1):
        if time.time() > deadline - 5:
            break
        sol = solve_for_n(n, 2.0)
        M4, M5, _, _, _ = get_structures(n)
        sums4 = sol[M4].sum(axis=1)
        sums5 = sol[M5].sum(axis=1)
        V = np.sum(sums4 == 0) + np.sum(sums5 == 10)
        if V == 0:
            return {'n': n, 'colors': sol.tolist()}
            
    # Last resort: seed solution
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(n), 3)]}

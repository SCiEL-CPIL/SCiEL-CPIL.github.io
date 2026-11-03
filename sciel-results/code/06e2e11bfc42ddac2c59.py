# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_budget = 115.0

    # Precompute properties for speed
    # a and b are assumed symmetric for QAPLIB wil100, but we handle general case in delta logic if needed.
    # For wil100, a and b are symmetric.
    
    # Helper to compute initial cost
    def compute_cost(perm):
        b_perm = b[np.ix_(perm, perm)]
        return np.sum(a * b_perm)

    # Optimized Delta Calculation for swapping facilities i and j
    # p is the permutation: facility i is at location p[i]
    def get_delta(i, j, p):
        li = p[i]
        lj = p[j]
        if li == lj:
            return 0
        
        # Vectors for a
        # R_i[k] = a[i, k], R_j[k] = a[j, k]
        # C_i[k] = a[k, i], C_j[k] = a[k, j]
        
        # Difference in a rows: a[i, :] - a[j, :]
        diff_a_row = a[i, :] - a[j, :]
        # Difference in b rows: b[lj, :] - b[li, :]
        diff_b_row = b[lj, :] - b[li, :]
        
        # Difference in a cols: a[:, i] - a[:, j]
        diff_a_col = a[:, i] - a[:, j]
        # Difference in b cols: b[:, lj] - b[:, li]
        diff_b_col = b[:, lj] - b[:, li]
        
        # Sum over all k
        s1 = np.dot(diff_a_row, diff_b_row)
        s2 = np.dot(diff_a_col, diff_b_col)
        
        # Subtract k=i and k=j terms
        # For s1:
        # k=i: (a[i,i] - a[j,i]) * (b[lj, li] - b[li, li])
        # k=j: (a[i,j] - a[j,j]) * (b[lj, lj] - b[li, lj])
        term_1_i = (a[i, i] - a[j, i]) * (b[lj, li] - b[li, li])
        term_1_j = (a[i, j] - a[j, j]) * (b[lj, lj] - b[li, lj])
        
        # For s2:
        # k=i: (a[i,i] - a[i,j]) * (b[li, lj] - b[li, li])
        # k=j: (a[j,i] - a[j,j]) * (b[lj, lj] - b[lj, li])
        term_2_i = (a[i, i] - a[i, j]) * (b[li, lj] - b[li, li])
        term_2_j = (a[j, i] - a[j, j]) * (b[lj, lj] - b[lj, li])
        
        return (s1 - term_1_i - term_1_j) + (s2 - term_2_i - term_2_j)

    # Initialization: Rank Matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    current_cost = compute_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    # Simulated Annealing Parameters
    # Start high enough to escape local minima, cool down slowly
    T_start = 1000.0
    T_end = 1.0
    # Number of steps to determine cooling rate
    # We don't know exact steps, so we use time-based cooling or fixed steps per run
    # Let's do a single long run with adaptive cooling based on time remaining.
    
    # To allow restarts, we wrap the SA loop in a function
    def run_sa(p_init, cost_init, duration):
        p = p_init.copy()
        current_cost = cost_init
        best_cost_local = cost_init
        best_p_local = p.copy()
        
        # Estimate steps possible. 
        # Each step involves 2 dot products of size n=100.
        # np.dot(100, 100) is very fast, ~1-2 microseconds?
        # Python overhead might be ~10-20 microseconds per iteration.
        # Let's aim for ~100k-200k iterations per second.
        
        # Use a fixed number of iterations per "chunk" or just loop until time.
        # Cooling schedule: Linear or Exponential.
        # Let's use a geometric cooling with a small factor, resetting T if stuck?
        # Or simpler: T = T_start * (T_end/T_start)^(t/T_total)
        
        # Since we don't know T_total in iterations, we use time.
        # Let's just decrease T by a factor every N iterations.
        
        T = T_start
        cooling_factor = 0.999
        # If we do 100k iterations, T drops by 0.999^100k ~ 0.
        # 0.999^10000 ~ 0.00004.
        
        while time.time() - start_time < duration:
            # Pick random distinct i, j
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
                
            delta = get_delta(i, j, p)
            
            if delta < 0 or (T > 0 and random.random() < np.exp(-delta / T)):
                # Accept move
                p[i], p[j] = p[j], p[i]
                current_cost += delta
                
                if current_cost < best_cost_local:
                    best_cost_local = current_cost
                    best_p_local = p.copy()
            
            # Cool down
            T *= cooling_factor
            
            # Optional: If T is very low, we might be stuck.
            # We can restart or just continue.
            
        return best_p_local, best_cost_local

    # Execute SA
    # We can try to do multiple restarts if time allows.
    # For now, one long run is a safe baseline for "new approach".
    # Let's try 2 restarts to improve robustness.
    
    # Split time: 60% first run, 35% second run, 5% buffer
    t1 = time_budget * 0.6
    t2 = time_budget * 0.35
    
    # Run 1
    p1, cost1 = run_sa(p, current_cost, t1)
    if cost1 < best_cost:
        best_cost = cost1
        best_p = p1
    
    # Run 2: Random restart
    if time.time() - start_time < time_budget - 2.0:
        # Generate a new random permutation or re-rank
        p2 = np.random.permutation(n)
        # Or use the previous best and perturb it
        # Perturbing the best is usually better
        p2 = best_p.copy()
        # Perform some random swaps to perturb
        for _ in range(10):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i != j:
                p2[i], p2[j] = p2[j], p2[i]
        
        cost2_init = compute_cost(p2)
        p2, cost2 = run_sa(p2, cost2_init, t2)
        
        if cost2 < best_cost:
            best_cost = cost2
            best_p = p2

    return best_p.tolist()

# EVOLVE-BLOCK-END

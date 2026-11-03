"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 140.0  # Leave 10s safety margin for 150s limit

    # 1. Initialization: Rank Matching
    # Heavy flows (high row sums in a) should go to central locations (low row sums in b)
    a_sums = a.sum(axis=1)
    b_sums = b.sum(axis=1)
    
    # Sort indices
    heavy_idx = np.argsort(-a_sums, kind="stable")
    central_idx = np.argsort(b_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy_idx] = central_idx
    
    # Calculate initial cost
    # Cost = sum a[i,j] * b[p[i], p[j]]
    # Efficient calculation: sum_i (a[i] @ b[p[i], :])
    # This is O(n^3) if done naively, but we can do it in O(n^2) with careful indexing?
    # Actually, sum_{i,j} a[i,j] b[p[i], p[j]] = sum_i (a[i] @ b[p[i], :])
    # b[p[i], :] is a row of b.
    # Let's just compute it. n=150, so 150 * (150*150) ops = 3.3M ops. Very fast.
    
    def calc_cost(p):
        # p is a permutation of 0..n-1
        # b_perm = b[p, :]
        # cost = sum(a[i] * b_perm[i]) for i in range(n)
        # Using einsum: sum(a * b_perm)
        # b_perm = b[p]
        return np.sum(a * b[p])

    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost

    # 2. Simulated Annealing
    # Parameters
    T_start = 1000.0
    T_end = 1.0
    # We need to determine the number of steps or run until time limit.
    # Let's run until time limit.
    
    # Precompute for speed?
    # a is (n, n), b is (n, n).
    # Move: swap i, j.
    # Delta = sum_k (a[i,k] - a[j,k]) * (b[p[j], p[k]] - b[p[i], p[k]]) + 2 * a[i,j] * b[p[i], p[j]]
    # Let u = p[i], v = p[j].
    # diff_a = a[i] - a[j]
    # diff_b = b[v, p] - b[u, p]  # Note: b[v, p[k]] - b[u, p[k]]
    # delta = dot(diff_a, diff_b) + 2 * a[i,j] * b[u,v]
    
    # We will iterate.
    # To make it faster, we can use a "first improvement" or "best improvement" strategy?
    # SA usually evaluates random moves.
    # For n=150, random pair selection is O(1) per move.
    # 150s / (time per move).
    # Time per move in Python:
    #   diff_a = a[i] - a[j] (150 ops)
    #   diff_b = b[v, p] - b[u, p] (300 ops + indexing)
    #   dot (150 ops)
    #   Total ~1000-2000 ns? Maybe 10-50 us in Python/NumPy overhead.
    #   1e6 moves / s is possible.
    #   150s -> 1.5e8 moves.
    #   This is very good.
    
    # However, Python loop overhead is significant.
    # Let's try to optimize the inner loop.
    
    # We'll use a linear cooling schedule.
    # Since we don't know the exact number of iterations, we can just decrement T by a small amount
    # or use a time-based schedule.
    # Let's use a fixed number of iterations per "temperature step" or just a continuous decay.
    
    # Let's try a simple approach:
    # While time < limit:
    #   Pick random i, j
    #   Calculate delta
    #   If delta < 0 or random() < exp(-delta/T):
    #       Swap
    #       Update cost
    #   T = T * 0.999 (or similar)
    
    # To avoid underflow/overflow in exp, clamp delta.
    
    rng = np.random.default_rng(42)
    
    T = T_start
    
    # We need to be careful with the cooling rate.
    # If we do 1e6 moves, and T decays too fast, we get stuck.
    # If T decays too slow, we don't converge.
    # Let's estimate the number of moves we can do.
    # Let's just run until time limit and use a time-based temperature update.
    # T(t) = T_start * exp(-alpha * t)
    # Or simpler: T = T_start * (T_end/T_start)^(t / total_time)
    # Since total_time is fixed (140s), we can calculate T based on elapsed time.
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
            
        # Calculate current temperature
        # Linear in time? Exponential?
        # Exponential is standard.
        # T = T_start * exp(-k * elapsed)
        # We want T ~ T_end at elapsed = time_limit.
        # T_end = T_start * exp(-k * time_limit)
        # k = ln(T_start/T_end) / time_limit
        k = np.log(T_start / T_end) / time_limit
        T = T_start * np.exp(-k * elapsed)
        
        # Generate random move
        i = rng.integers(n)
        j = rng.integers(n)
        if i == j:
            continue
            
        u = p[i]
        v = p[j]
        
        # Calculate delta
        # diff_a = a[i] - a[j]
        # diff_b = b[v, p] - b[u, p]
        # delta = dot(diff_a, diff_b) + 2 * a[i,j] * b[u,v]
        
        # Optimization: 
        # We can precompute row sums? No, delta depends on p.
        
        # Vectorized calculation for this specific move
        # a_i = a[i]
        # a_j = a[j]
        # b_u = b[u]
        # b_v = b[v]
        # diff_a = a_i - a_j
        # diff_b = b_v[p] - b_u[p]
        
        # This involves indexing b with p, which is O(n).
        
        delta = np.dot(a[i] - a[j], b[v, p] - b[u, p]) + 2.0 * a[i, j] * b[u, v]
        
        # Acceptance criterion
        if delta < 0 or rng.random() < np.exp(-delta / T):
            # Swap
            p[i], p[j] = p[j], p[i]
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                
    return best_p.tolist()

# EVOLVE-BLOCK-END

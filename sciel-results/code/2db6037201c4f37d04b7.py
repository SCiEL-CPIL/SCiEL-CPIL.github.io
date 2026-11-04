"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    
    # Helper to compute full cost
    def compute_cost(p):
        B = b[np.ix_(p, p)]
        return int(np.sum(a * B))
        
    # Helper to compute delta cost for swapping i and j
    def delta_cost(i, j, p):
        u = p[i]
        v = p[j]
        
        # Vectorized delta calculation
        # The change in cost when swapping facilities i and j (locations u and v)
        # Cost = sum a[i,k] b[p[i], p[k]]
        # Delta = a[i,j](b[u,v]-b[v,u]) + a[j,i](b[v,u]-b[u,v]) 
        #        + sum_{k != i,j} a[i,k](b[v,p[k]] - b[u,p[k]]) 
        #        + sum_{k != i,j} a[j,k](b[u,p[k]] - b[v,p[k]])
        #        + sum_{k != i,j} a[k,i](b[p[k],v] - b[p[k],u])
        #        + sum_{k != i,j} a[k,j](b[p[k],u] - b[p[k],v])
        
        # Let's compute the terms efficiently
        # b_v_p = b[v, p], b_u_p = b[u, p]
        # b_p_v = b[p, v], b_p_u = b[p, u]
        
        b_v_p = b[v, :]
        b_u_p = b[u, :]
        b_p_v = b[:, v]
        b_p_u = b[:, u]
        
        # Mask to exclude i and j
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        
        # Terms involving i and j with others
        # Sum over k != i,j of a[i,k] * (b[v,p[k]] - b[u,p[k]])
        s1 = np.dot(a[i, mask], b_v_p[mask] - b_u_p[mask])
        
        # Sum over k != i,j of a[j,k] * (b[u,p[k]] - b[v,p[k]])
        s2 = np.dot(a[j, mask], b_u_p[mask] - b_v_p[mask])
        
        # Sum over k != i,j of a[k,i] * (b[p[k],v] - b[p[k],u])
        s3 = np.dot(a[mask, i], b_p_v[mask] - b_p_u[mask])
        
        # Sum over k != i,j of a[k,j] * (b[p[k],u] - b[p[k],v])
        s4 = np.dot(a[mask, j], b_p_u[mask] - b_p_v[mask])
        
        # Cross terms for i and j themselves
        # Original: a[i,j]*b[u,v] + a[j,i]*b[v,u]
        # New: a[i,j]*b[v,u] + a[j,i]*b[u,v]
        cross_delta = a[i, j] * (b[v, u] - b[u, v]) + a[j, i] * (b[u, v] - b[v, u])
        
        return s1 + s2 + s3 + s4 + cross_delta

    # Initial Solution: Greedy + Local Search Warm-up
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    facility_order = np.argsort(-a_row_sums, kind='stable')
    location_order = np.argsort(b_row_sums, kind='stable')
    
    p = np.empty(n, dtype=np.int64)
    p[facility_order] = location_order
    
    current_cost = compute_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    # Quick local search warm-up (Steepest Descent) to get a strong start
    # This is important for Tabu Search to start from a good basin
    for _ in range(10):
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                d = delta_cost(i, j, p)
                if d < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += d
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    # Tabu Search Parameters
    start_time = time.time()
    time_limit = 115.0
    
    rng = np.random.default_rng(42)
    
    tabu_len = 30  # Length of tabu list
    tabu_list = {} # Maps (i, j) to the iteration when it becomes non-tabu
    current_iter = 0
    
    no_improve_count = 0
    max_no_improve = 5000 # Trigger shake after this many non-improving moves
    
    while time.time() - start_time < time_limit:
        # Find best non-tabu move
        best_delta = float('inf')
        best_i = -1
        best_j = -1
        
        # To make this fast, we don't check all pairs every time (O(n^2) per step is too slow for 150 in Python loop? 
        # n=150, n^2 = 22500. 22500 * 115s / (time per delta) 
        # If delta takes ~1us (vectorized), 22500 * 1e-6 = 0.02s per step. 
        # 115 / 0.02 = 5750 steps. This is feasible.
        
        # Optimization: We can sample moves or use a heuristic selection. 
        # However, for QAP, steepest descent within the allowed moves is standard.
        # Let's try full scan first. If too slow, we'll switch to random sampling.
        
        # Actually, Python loop over 22500 pairs calling a Python function `delta_cost` is SLOW.
        # We need to vectorize the move selection or use a faster heuristic.
        
        # Alternative: Randomly select a set of candidate moves and pick the best non-tabu one.
        # Or, use a simpler acceptance criterion like Randomized Hill Climbing with Tabu.
        
        # Let's use a hybrid: 
        # 1. Generate a batch of random pairs.
        # 2. Compute deltas for them.
        # 3. Pick the best non-tabu move from the batch.
        
        batch_size = 200
        candidates = []
        for _ in range(batch_size):
            i = rng.integers(n)
            j = rng.integers(n)
            while i == j:
                j = rng.integers(n)
            key = (min(i, j), max(i, j))
            if key in tabu_list and tabu_list[key] > current_iter:
                continue
            candidates.append((i, j))
            
        if not candidates:
            # All candidates tabu, clear tabu list or accept any
            tabu_list.clear()
            i = rng.integers(n)
            j = rng.integers(n)
            while i == j:
                j = rng.integers(n)
            candidates.append((i, j))

        best_delta_in_batch = float('inf')
        best_i_batch = -1
        best_j_batch = -1
        
        for i, j in candidates:
            d = delta_cost(i, j, p)
            if d < best_delta_in_batch:
                best_delta_in_batch = d
                best_i_batch = i
                best_j_batch = j
                
        if best_i_batch == -1:
            continue
            
        i = best_i_batch
        j = best_j_batch
        d = best_delta_in_batch
        
        # Accept move
        p[i], p[j] = p[j], p[i]
        current_cost += d
        
        # Update tabu list
        key = (min(i, j), max(i, j))
        tabu_list[key] = current_iter + tabu_len
        
        # Clean up tabu list to prevent memory growth (optional, dict is small)
        # for k in list(tabu_list.keys()):
        #     if tabu_list[k] <= current_iter:
        #         del tabu_list[k]
        
        current_iter += 1
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # Diversification: Shake if stuck
        if no_improve_count > max_no_improve:
            # Perform 10 random swaps
            for _ in range(10):
                i = rng.integers(n)
                j = rng.integers(n)
                while i == j:
                    j = rng.integers(n)
                p[i], p[j] = p[j], p[i]
            current_cost = compute_cost(p)
            no_improve_count = 0
            # Reset tabu list to allow exploration
            tabu_list.clear()
            current_iter = 0

    return best_p.tolist()
# EVOLVE-BLOCK-END

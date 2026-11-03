"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 145.0  # Leave 5s safety margin for 150s limit

    # 1. Initial Solution: Rank Matching
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
    # Can be computed as sum over i of a[i] dot b[p[i], :]
    # Or using matrix multiplication tricks, but direct sum is O(n^2)
    # For n=150, O(n^2) is fast.
    cost = 0
    for i in range(n):
        cost += np.dot(a[i], b[p[i], :])
        
    best_cost = cost
    best_p = p.copy()

    # 2. Local Search: 2-Opt (Pairwise Swap)
    # We iterate until no improvement is found in a full pass or time runs out.
    
    while True:
        if time.time() - start_time > time_limit:
            break
            
        improved = False
        # Scan all pairs
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n):
                u = p[i]
                v = p[j]
                
                # Calculate delta
                # delta = sum_k (a[i,k] - a[j,k]) * (b[v, p[k]] - b[u, p[k]]) + 2*a[i,j]*b[u,v]
                
                # Optimization: If a[i,j] is 0 and rows are similar, might be 0, but we compute anyway.
                
                diff_a = a[i] - a[j]
                diff_b = b[v, p] - b[u, p]
                
                # Use float to prevent overflow in dot product if ints are large
                delta = np.dot(diff_a.astype(np.float64), diff_b.astype(np.float64))
                delta += 2.0 * a[i, j] * b[u, v]
                
                if delta < 0:
                    # Perform swap
                    p[i], p[j] = p[j], p[i]
                    cost += delta
                    improved = True
                    
                    # Restart scan from beginning to ensure stability
                    # This is O(n^3) worst case but effective for QAP
                    break
            if improved:
                break
        
        if not improved:
            # Local optimum reached
            break
            
    return best_p.tolist() if cost > best_cost else p.tolist()
    
    # Actually, if we update best_p only when we find a new global best?
    # The local search is monotonically improving (delta < 0), so current p is always better than initial.
    # So we can just return p.

# EVOLVE-BLOCK-END

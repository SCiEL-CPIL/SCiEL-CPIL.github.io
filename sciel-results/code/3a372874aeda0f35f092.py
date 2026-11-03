"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def solve(a, b):
    n = len(a)
    deadline = time.time() + 145.0  # safety margin before 150s kill

    # --- Fast cost evaluation and delta computation helpers ---
    # We maintain p: facility i at location p[i].
    # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    
    # Precompute row sums of a for delta computation
    # For swapping facilities u and v (at locations lu=p[u], lv=p[v]):
    # The change in cost comes from terms involving u or v.
    
    # Let's define:
    # C(u, v) = sum_{j != u, v} a[u,j] * b[lu, p[j]] + sum_{j != u, v} a[v,j] * b[lv, p[j]] 
    #           + sum_{i != u, v} a[i,u] * b[p[i], lu] + sum_{i != u, v} a[i,v] * b[p[i], lv]
    #           + a[u,v] * b[lu, lv] + a[v,u] * b[lv, lu]
    # After swap:
    # C'(u, v) = sum_{j != u, v} a[u,j] * b[lv, p[j]] + sum_{j != u, v} a[v,j] * b[lu, p[j]] 
    #            + sum_{i != u, v} a[i,u] * b[p[i], lv] + sum_{i != u, v} a[i,v] * b[p[i], lu]
    #            + a[u,v] * b[lv, lu] + a[v,u] * b[lu, lv]
    
    # This is complex to update incrementally for arbitrary swaps without maintaining
    # auxiliary structures. Let's use a simpler 2-opt style local search with 
    # precomputed matrices for speed.
    
    # Alternative: Use a matrix-based approach for the local search step.
    # Let P be the permutation matrix (n x n) where P[i, p[i]] = 1.
    # Cost = sum_{i,j,k,l} a[i,j] * b[k,l] * P[i,k] * P[j,l]
    #      = trace(A * P * B * P^T) if we define things carefully?
    # Actually, Cost = sum_{i,j} a[i,j] b[p[i], p[j]]
    # Let's define M = B. Then cost = sum_{i,j} a[i,j] M[p[i], p[j]].
    
    # For a swap of locations x and y (i.e., swap p[u]=x and p[v]=y):
    # The delta can be computed efficiently if we maintain:
    # S1[i] = sum_j a[i,j] * b[p[i], p[j]]  (row contribution of i)
    # S2[j] = sum_i a[i,j] * b[p[i], p[j]]  (col contribution of j)
    # But swapping p[u] and p[v] affects many terms.
    
    # Let's use a direct but optimized delta calculation for swapping two facilities u and v.
    # Let lu = p[u], lv = p[v].
    # Terms involving u or v in the sum:
    # Original:
    #   sum_{j != u,v} a[u,j] b[lu, p[j]] + sum_{j != u,v} a[v,j] b[lv, p[j]]
    #   + sum_{i != u,v} a[i,u] b[p[i], lu] + sum_{i != u,v} a[i,v] b[p[i], lv]
    #   + a[u,v] b[lu, lv] + a[v,u] b[lv, lu]
    # New (after swap, u at lv, v at lu):
    #   sum_{j != u,v} a[u,j] b[lv, p[j]] + sum_{j != u,v} a[v,j] b[lu, p[j]]
    #   + sum_{i != u,v} a[i,u] b[p[i], lv] + sum_{i != u,v} a[i,v] b[p[i], lu]
    #   + a[u,v] b[lv, lu] + a[v,u] b[lu, lv]
    
    # Delta = New - Original
    # Delta = sum_{j != u,v} a[u,j] (b[lv, p[j]] - b[lu, p[j]])
    #       + sum_{j != u,v} a[v,j] (b[lu, p[j]] - b[lv, p[j]])
    #       + sum_{i != u,v} a[i,u] (b[p[i], lv] - b[p[i], lu])
    #       + sum_{i != u,v} a[i,v] (b[p[i], lu] - b[p[i], lv])
    #       + a[u,v] (b[lv, lu] - b[lu, lv]) + a[v,u] (b[lu, lv] - b[lv, lu])
    
    # Note: b is symmetric in many QAP instances, but not guaranteed. Let's assume general case.
    # If b is symmetric, b[lv, lu] = b[lu, lv], so the last two terms cancel.
    
    # We can precompute for each location k:
    # RowB[k] = b[k, :]
    # ColB[k] = b[:, k]
    
    # To speed up the sums, we can maintain:
    # For each facility i, we need sum_{j != u,v} a[i,j] b[p[i], p[j]] ... this is hard.
    
    # Let's try a different approach: 2-move local search with vectorized numpy operations.
    # For a candidate swap (u, v), we can compute the delta using vectorized ops on the 
    # relevant rows/columns.
    
    # Initial solution: Rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.astype(np.int64)
    
    # Compute initial cost
    def compute_cost(p):
        # cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
        # We can do this as:
        # Let B_p = b[np.ix_(p, p)]
        # cost = np.sum(a * B_p)
        B_p = b[np.ix_(p, p)]
        return np.sum(a * B_p)
    
    best_p = p.copy()
    best_cost = compute_cost(p)
    
    # Local search: Try swapping pairs
    # To make this fast, we'll use a random order and limit the number of swaps per iteration.
    
    # Precompute b rows and cols for fast access
    # Actually, let's just use the vectorized delta computation.
    
    def compute_swap_delta(u, v, p, a, b):
        lu = p[u]
        lv = p[v]
        if lu == lv:
            return 0
        
        # We need:
        # S1 = sum_{j != u,v} a[u,j] (b[lv, p[j]] - b[lu, p[j]])
        # S2 = sum_{j != u,v} a[v,j] (b[lu, p[j]] - b[lv, p[j]])
        # S3 = sum_{i != u,v} a[i,u] (b[p[i], lv] - b[p[i], lu])
        # S4 = sum_{i != u,v} a[i,v] (b[p[i], lu] - b[p[i], lv])
        # S5 = a[u,v] (b[lv, lu] - b[lu, lv]) + a[v,u] (b[lu, lv] - b[lv, lu])
        
        # Create masks for j != u, v and i != u, v
        # For S1: sum over j != u, v of a[u,j] * (b[lv, p[j]] - b[lu, p[j]])
        # Let diff_b = b[lv, :] - b[lu, :]  (vector of length n)
        # Then S1 = sum_{j != u,v} a[u,j] * diff_b[p[j]]
        
        diff_b = b[lv, :] - b[lu, :]
        # S1 = np.sum(a[u, :] * diff_b[p]) - a[u,u]*diff_b[lu] - a[u,v]*diff_b[lv]
        S1 = np.sum(a[u, :] * diff_b[p]) - a[u, u] * diff_b[lu] - a[u, v] * diff_b[lv]
        
        # S2 = sum_{j != u,v} a[v,j] * (b[lu, p[j]] - b[lv, p[j]])
        #    = - sum_{j != u,v} a[v,j] * (b[lv, p[j]] - b[lu, p[j]])
        #    = - (np.sum(a[v, :] * diff_b[p]) - a[v,u]*diff_b[lu] - a[v,v]*diff_b[lv])
        S2 = -(np.sum(a[v, :] * diff_b[p]) - a[v, u] * diff_b[lu] - a[v, v] * diff_b[lv])
        
        # S3 = sum_{i != u,v} a[i,u] * (b[p[i], lv] - b[p[i], lu])
        # Let diff_bT = b[:, lv] - b[:, lu]  (vector of length n)
        # Then S3 = sum_{i != u,v} a[i,u] * diff_bT[p[i]]
        diff_bT = b[:, lv] - b[:, lu]
        S3 = np.sum(a[:, u] * diff_bT[p]) - a[u, u] * diff_bT[lu] - a[v, u] * diff_bT[lv]
        
        # S4 = sum_{i != u,v} a[i,v] * (b[p[i], lu] - b[p[i], lv])
        #    = - sum_{i != u,v} a[i,v] * (b[p[i], lv] - b[p[i], lu])
        #    = - (np.sum(a[:, v] * diff_bT[p]) - a[u,v]*diff_bT[lu] - a[v,v]*diff_bT[lv])
        S4 = -(np.sum(a[:, v] * diff_bT[p]) - a[u, v] * diff_bT[lu] - a[v, v] * diff_bT[lv])
        
        # S5
        S5 = a[u, v] * (b[lv, lu] - b[lu, lv]) + a[v, u] * (b[lu, lv] - b[lv, lu])
        
        return S1 + S2 + S3 + S4 + S5
    
    # Local search loop
    n_iter = 0
    while time.time() < deadline:
        n_iter += 1
        improved = False
        # Try random swaps
        # For efficiency, we'll try a limited number of random swaps per iteration
        n_swaps = min(n * 10, 500)  # Limit swaps per iteration
        for _ in range(n_swaps):
            u = np.random.randint(n)
            v = np.random.randint(n)
            if u == v:
                continue
            
            delta = compute_swap_delta(u, v, p, a, b)
            if delta < 0:
                # Accept swap
                p[u], p[v] = p[v], p[u]
                best_cost += delta
                improved = True
                # Update best if better
                if best_cost < np.sum(a * b[np.ix_(p, p)]):  # Verify occasionally? No, trust delta.
                    pass
                # Actually, let's just update best_p and best_cost
                # But we need to be careful: delta is exact, so best_cost is correct.
                if best_cost < best_cost:  # This is always true if delta < 0
                    best_p = p.copy()
                    best_cost = best_cost  # Already updated
        
        if not improved:
            # No improvement in this iteration, try a different strategy or break
            # For now, break to avoid wasting time
            break
    
    return best_p.tolist()
# EVOLVE-BLOCK-END

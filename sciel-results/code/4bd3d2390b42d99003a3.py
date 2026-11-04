"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(p, a, b):
    """Exact QAP cost for permutation p (facility i -> location p[i])."""
    # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # We can compute this efficiently by building the permuted b matrix
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))


def _qap_cost_incremental_swap(p, a, b, i, j):
    """
    Compute the change in cost if we swap positions i and j in the permutation.
    p is a list (0-indexed): facility k is at location p[k].
    After swapping, facility i goes to location p[j] and facility j goes to location p[i].
    
    We compute delta = new_cost - old_cost.
    """
    li = p[i]
    lj = p[j]
    
    # The cost contribution involving facility i and j changes.
    # Specifically, we need to consider all pairs (i, k) and (j, k) for all k != i, j,
    # plus the pair (i, j) itself.
    
    # Let's derive the delta.
    # Old cost terms involving i or j:
    #   For each k != i, j:
    #     a[i,k]*b[li, p[k]] + a[k,i]*b[p[k], li] + a[j,k]*b[lj, p[k]] + a[k,j]*b[p[k], lj]
    #   For k = i: a[i,i]*b[li,li]
    #   For k = j: a[i,j]*b[li,lj] + a[j,i]*b[lj,li] + a[j,j]*b[lj,lj]
    
    # New cost terms (after swap, i is at lj, j is at li):
    #   For each k != i, j:
    #     a[i,k]*b[lj, p[k]] + a[k,i]*b[p[k], lj] + a[j,k]*b[li, p[k]] + a[k,j]*b[p[k], li]
    #   For k = i: a[i,i]*b[lj,lj]
    #   For k = j: a[i,j]*b[lj,li] + a[j,i]*b[li,lj] + a[j,j]*b[li,li]
    
    # Delta = sum over all changes.
    # This is expensive to compute naively for each pair. Let's precompute row sums.
    
    # Actually, let's just compute the full cost before and after if the neighborhood is small,
    # or use a smarter incremental update.
    
    # For a 2-opt swap of i and j:
    # Delta = sum_{k != i, j} [ a[i,k]*(b[lj,p[k]] - b[li,p[k]]) + a[k,i]*(b[p[k],lj] - b[p[k],li])
    #                 + a[j,k]*(b[li,p[k]] - b[lj,p[k]]) + a[k,j]*(b[p[k],li] - b[p[k],lj]) ]
    #       + a[i,i]*(b[lj,lj] - b[li,li]) + a[j,j]*(b[li,li] - b[lj,lj])
    #       + a[i,j]*(b[lj,li] - b[li,lj]) + a[j,i]*(b[li,lj] - b[lj,li])
    
    # This is O(n) per swap. For n=100, that's 100 operations per swap.
    # Number of pairs in a 2-opt neighborhood is ~n^2/2 = 5000.
    # Total per iteration: 5000 * 100 = 500,000 ops. In NumPy, this is fast.
    
    # Let's vectorize this.
    # Create arrays for the terms.
    
    # We need to be careful with the indices.
    
    # Let's use NumPy to compute the delta.
    # We'll compute the change for all k at once.
    
    # Mask for k != i, j
    mask = np.ones(n, dtype=bool)
    mask[i] = False
    mask[j] = False
    k_indices = np.where(mask)[0]
    
    # Terms for facility i:
    # a[i, k] * (b[lj, p[k]] - b[li, p[k]])
    # a[k, i] * (b[p[k], lj] - b[p[k], li])
    
    # Terms for facility j:
    # a[j, k] * (b[li, p[k]] - b[lj, p[k]])
    # a[k, j] * (b[p[k], li] - b[p[k], lj])
    
    # Combine:
    # (a[i, k] - a[j, k]) * (b[lj, p[k]] - b[li, p[k]])
    # (a[k, i] - a[k, j]) * (b[p[k], lj] - b[p[k], li])
    
    # Let's compute these vectorially.
    
    pk = p[k_indices]
    
    term1 = (a[i, k_indices] - a[j, k_indices]) * (b[lj, pk] - b[li, pk])
    term2 = (a[k_indices, i] - a[k_indices, j]) * (b[pk, lj] - b[pk, li])
    
    delta_sum = np.sum(term1) + np.sum(term2)
    
    # Diagonal terms:
    delta_diag = a[i, i] * (b[lj, lj] - b[li, li]) + a[j, j] * (b[li, li] - b[lj, lj])
    
    # Cross terms:
    delta_cross = a[i, j] * (b[lj, li] - b[li, lj]) + a[j, i] * (b[li, lj] - b[lj, li])
    
    return int(delta_sum + delta_diag + delta_cross)


def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    # 1. Initialization: Rank-matching heuristic
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.tolist()
    
    best_p = p.copy()
    best_cost = _qap_cost(best_p, a, b)
    
    # 2. Local Search: 2-opt swaps
    # We'll perform iterative improvement until no improvement is found or time is up.
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Try all pairs (i, j) with i < j
        for i in range(n):
            for j in range(i + 1, n):
                # Check time periodically
                if (time.time() - start_time) > time_limit - 5:
                    # Return best found so far
                    return best_p
                
                delta = _qap_cost_incremental_swap(p, a, b, i, j)
                if delta < 0:
                    # Perform the swap
                    p[i], p[j] = p[j], p[i]
                    best_cost += delta
                    if best_cost < best_cost:  # This is always true if delta < 0
                        pass
                    # Update best if this is the new best
                    # Since we're doing greedy descent, the current p is the best so far
                    best_p = p.copy()
                    best_cost = _qap_cost(p, a, b)  # Recompute to avoid drift
                    improved = True
                    # Restart the loop to pick up new interactions
                    # Actually, for efficiency, we can continue with the same pass,
                    # but restarting ensures we don't miss better moves.
                    # For now, let's just mark improved and continue the inner loop.
                    # A more aggressive approach is to restart the outer loop.
                    break
            if improved:
                # Restart the outer loop for a fresh pass
                # But this can be slow. Let's just continue.
                pass
    
    # 3. Final Polish: Try a few more passes if time allows
    while (time.time() - start_time) < time_limit - 10:
        any_improved = False
        for i in range(n):
            for j in range(i + 1, n):
                if (time.time() - start_time) > time_limit - 5:
                    return best_p
                delta = _qap_cost_incremental_swap(p, a, b, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    any_improved = True
                    best_p = p.copy()
                    best_cost = _qap_cost(p, a, b)
        if not any_improved:
            break
    
    return best_p
# EVOLVE-BLOCK-END

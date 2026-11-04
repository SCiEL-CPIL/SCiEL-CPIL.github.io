import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    
    # 1. Initial Solution: Rank Matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Helper to calculate cost
    def calc_cost(p):
        b_perm = b[p][:, p]
        return np.sum(a * b_perm)
        
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Precompute M matrix
    # M[i, l] = sum_k a[i, k] b[l, p[k]] + a[k, i] b[p[k], l]
    # We can compute this efficiently.
    # M[:, l] = a @ b[l, p] + a.T @ b[p, l]
    
    def compute_M(p):
        M = np.zeros((n, n), dtype=np.float64)
        for l in range(n):
            v1 = b[l, p]
            v2 = b[p, l]
            M[:, l] = a @ v1 + a.T @ v2
        return M

    M = compute_M(p)
    
    start_time = time.time()
    time_limit = 115.0
    
    # Tabu Search Parameters
    tabu_tenure = 100 # Number of iterations a move stays tabu
    tabu_list = {} # (i, j) -> iteration_expires
    iteration = 0
    
    while time.time() - start_time < time_limit:
        iteration += 1
        
        # Find best move
        best_delta = np.inf
        best_i = -1
        best_j = -1
        
        # Iterate over all pairs
        # To speed up, we can vectorize the delta calculation?
        # Delta_ij = M[i, p[j]] - M[i, p[i]] + M[j, p[i]] - M[j, p[j]] + 
        #            a[i, j] b[p[i], p[j]] + a[j, i] b[p[j], p[i]] + 
        #            a[i, j] b[p[j], p[i]] + a[j, i] b[p[i], p[j]]
            
        # We can compute this for all i, j pairs using numpy?
        # Let Pi = p (vector of locations)
        # M_i_pi = M[i, Pi[i]] -> M[np.arange(n), p]
        # M_i_pj = M[i, Pi[j]] -> This is hard to vectorize directly for all i, j without creating a large matrix.
        # Given n=100, a double loop in Python might be slow if not optimized.
        # 10,000 iterations * 12,000 loops = 120,000,000 operations.
        # Python is ~10^7 simple ops/sec. So 12 seconds for just the loop overhead?
        # We should try to vectorize or use a faster inner loop.
        
        # Let's try to vectorize the delta calculation for all i, j.
        # Delta[i, j] = M[i, p[j]] - M[i, p[i]] + M[j, p[i]] - M[j, p[j]] + ...
        
        # Term 1: M[i, p[j]]
        # This is M[:, p[j]] for a fixed j.
        # We can compute M_i_pj = M[:, p] which is just M (since p is a permutation, M[:, p] is columns permuted).
        # Wait, M[i, p[j]] is the value at row i, column p[j].
        # Let M_perm = M[:, p]. Then M_perm[i, j] = M[i, p[j]].
        M_perm = M[:, p]
        
        # Term 2: M[i, p[i]]
        M_ii = M[np.arange(n), p] # Vector of M[i, p[i]]
        
        # Term 3: M[j, p[i]]
        # This is M[j, p[i]].
        # Let M_perm_T = M[p, :] ? No.
        # M[j, p[i]] is row j, col p[i].
        # If we take M_perm = M[:, p], then M_perm[j, i] = M[j, p[i]].
        # So M[j, p[i]] is M_perm.T[i, j]?
        # M_perm[i, j] = M[i, p[j]].
        # M_perm[j, i] = M[j, p[i]].
        # So Term 3 is M_perm.T[i, j] (if we treat i, j as indices in the matrix).
        
        # Term 4: M[j, p[j]]
        # This is M_ii[j].
        # So Term 4 is M_ii[j] which is a vector indexed by j.
        # In the matrix Delta[i, j], Term 4 depends only on j.
        # So we can broadcast M_ii[None, :] to (1, n).
        
        # Interaction terms:
        # a[i, j] b[p[i], p[j]] + a[j, i] b[p[j], p[i]] + a[i, j] b[p[j], p[i]] + a[j, i] b[p[i], p[j]]
        # Let A = a.
        # Let B_perm = b[p][:, p].
        # b[p[i], p[j]] = B_perm[i, j].
        # b[p[j], p[i]] = B_perm[j, i].
        
        # Term 5: A[i, j] * B_perm[i, j]
        # Term 6: A[j, i] * B_perm[j, i] = A.T[i, j] * B_perm.T[i, j]
        # Term 7: A[i, j] * B_perm[j, i]
        # Term 8: A[j, i] * B_perm[i, j]
        
        # Let's construct the Delta matrix.
        # D[i, j] = M_perm[i, j] - M_ii[i] + M_perm[j, i] - M_ii[j] + ...
        # Note: M_perm[j, i] is the same as M_perm.T[i, j].
        
        D = M_perm - M_ii[:, None] + M_perm.T - M_ii[None, :]
        
        # Add interaction terms
        B_perm = b[p][:, p]
        D += a * B_perm
        D += a.T * B_perm.T
        D += a * B_perm.T
        D += a.T * B_perm
        
        # Exclude i == j
        np.fill_diagonal(D, np.inf)
        
        # Apply Tabu List
        # We need to mask out tabu moves.
        # Tabu list stores (i, j) pairs.
        # We can create a mask matrix.
        # For efficiency, we can just check the best candidate against tabu list,
        # or mask the D matrix.
        # Masking D (100x100) is fast.
        
        # Create a mask of False for allowed moves
        # Actually, we want to set D[i, j] to inf if (i, j) is tabu.
        # We can iterate over tabu_list items.
        # tabu_list is dict (i, j) -> expiry.
        
        # To avoid creating a new mask every time, we can just filter candidates.
        # Find indices where D < inf
        # But we need to respect tabu.
        
        # Let's get the flattened D and argsort?
        # Or just find min.
        
        # Efficient way:
        # Get all pairs
        # We can just loop over the sorted candidates?
        # n=100, 10k pairs.
        
        # Let's just find the min D value, and check if it's tabu.
        # If tabu, find next min.
        
        # Sort indices by D value
        # This might be overkill.
        
        # Simpler:
        # Flatten D
        D_flat = D.flatten()
        # Get indices
        # We need to ignore tabu moves.
        
        # Let's maintain a list of tabu moves (i, j)
        # And check them.
        
        # For now, let's just find the global min, and if it's tabu, we need the next best.
        # We can create a mask.
        mask = np.ones((n, n), dtype=bool)
        for (ti, tj), exp in tabu_list.items():
            if exp > iteration:
                mask[ti, tj] = False
                mask[tj, ti] = False # Swap is symmetric in tabu? Yes, usually.
        
        D_masked = np.where(mask, D, np.inf)
        
        # Find min
        min_idx = np.unravel_index(np.argmin(D_masked), D_masked.shape)
        best_i, best_j = min_idx
        best_delta = D_masked[best_i, best_j]
        
        if np.isinf(best_delta):
            break # No moves available (should not happen)
            
        # Aspiration Criterion: Accept if it improves global best
        if best_delta < 0 or (current_cost + best_delta < best_cost):
            # Accept
            p[best_i], p[best_j] = p[best_j], p[best_i]
            current_cost += best_delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                
            # Update M
            # Only rows best_i and best_j change?
            # No, M[i', l] changes for all i' because p[best_i] and p[best_j] changed.
            # But we can update M efficiently.
            # M[i', l] += a[i', best_i] (b[l, p[best_j]] - b[l, p[best_i]]) + ...
            
            # Let's just recompute M for all rows? O(n^2) = 10,000 ops.
            # Or update incrementally.
            # Incremental update is O(n^2) anyway if we do it for all i'.
            # Let's stick to incremental update for speed.
            
            li = p[best_i]
            lj = p[best_j]
            # Note: p has been swapped. So p[best_i] is now old lj, p[best_j] is now old li.
            # We need old values for delta calculation.
            # Let's store old values before swap.
            # We did p[best_i], p[best_j] = ...
            # So we need to revert or store.
            
            # Let's redo the update logic carefully.
            # Old p_i = p[best_i] (before swap)
            # Old p_j = p[best_j] (before swap)
            # New p_i = Old p_j
            # New p_j = Old p_i
            
            # We lost old values. Let's re-fetch from best_p? No, best_p is global best.
            # Let's store them before swap.
            
            # I will refactor the swap part.
            
        # Update Tabu List
        tabu_list[(best_i, best_j)] = iteration + tabu_tenure
        tabu_list[(best_j, best_i)] = iteration + tabu_tenure
        
        # Clean up old tabu entries
        # Can do this every 10 iterations
        if iteration % 10 == 0:
            tabu_list = {k: v for k, v in tabu_list.items() if v > iteration}
            
    return best_p.tolist()

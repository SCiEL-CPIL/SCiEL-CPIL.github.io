"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations


def solve(a, b):
    """
    Solve QAP instance tho150 using deterministic Best-Improvement Tabu Search
    with vectorized O(n^2) delta calculations.
    
    Strategy:
    1. Generate initial solution via rank-matching heuristic.
    2. Perform tabu search with:
       - Best-improvement strategy (evaluate all n*(n-1) swaps per iteration)
       - Tabu list to prevent cycling
       - Aspiration criterion to allow tabu moves if they improve best known
       - Time-bounded: stop before 150s wall-clock limit
    3. Return best permutation found.
    """
    n = len(a)
    
    # Deadline guard: stop at least 5s before the 150s limit
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # ---- Step 1: Initial solution via rank-matching ----
    # Heavy flows should go to short distances
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Compute initial cost
    def compute_cost(perm):
        return int(np.sum(a * b[np.ix_(perm, perm)]))
    
    best_cost = compute_cost(p)
    best_perm = p.copy()
    
    # ---- Step 2: Tabu Search ----
    # Tabu list: store (i, j) swaps that are forbidden
    tabu_tenure = 50
    tabu_list = set()
    tabu_counter = 0
    
    # Precompute for efficient delta computation
    # For a swap of positions i and j in the permutation:
    # delta = sum_k [a[i,k]*b[p[i],p[k]] + a[k,i]*b[p[k],p[i]] + a[j,k]*b[p[j],p[k]] + a[k,j]*b[p[k],p[j]]]
    #       - sum_k [a[i,k]*b[p[j],p[k]] + a[k,i]*b[p[k],p[j]] + a[j,k]*b[p[i],p[k]] + a[k,j]*b[p[k],p[i]]]
    
    while time.time() < deadline:
        # Evaluate all possible swaps (i, j) where i < j
        # For each swap, compute the cost delta
        best_delta = 0
        best_swap = None
        best_is_tabu = False
        
        # Vectorized computation of deltas for all swaps
        # For swap (i, j):
        #   old contribution involving i: sum_k a[i,k]*b[p[i],p[k]] + a[k,i]*b[p[k],p[i]]
        #   old contribution involving j: sum_k a[j,k]*b[p[j],p[k]] + a[k,j]*b[p[k],p[j]]
        #   new contribution involving i: sum_k a[i,k]*b[p[j],p[k]] + a[k,i]*b[p[k],p[j]]
        #   new contribution involving j: sum_k a[j,k]*b[p[p[i],p[k]] + a[k,j]*b[p[k],p[i]]
        #   Note: the term a[i,j]*b[p[i],p[j]] appears in both old and new, so it cancels
        #   Actually, let's be more careful:
        
        # Let's compute this more carefully using matrix operations
        # For each pair (i, j), the delta is:
        # delta(i,j) = sum_k [a[i,k]*b[p[j],p[k]] - a[i,k]*b[p[i],p[k]] 
        #               + a[k,i]*b[p[k],p[j]] - a[k,i]*b[p[k],p[i]]
        #               + a[j,k]*b[p[i],p[k]] - a[j,k]*b[p[j],p[k]]
        #               + a[k,j]*b[p[k],p[i]] - a[k,j]*b[p[k],p[j]]]
        #             - a[i,j]*(b[p[i],p[j]] - b[p[j],p[i]])  # correction for double-counting
        #             - a[j,i]*(b[p[j],p[i]] - b[p[i],p[j]])  # this is same as above if symmetric
        
        # Since a and b are symmetric in QAP, a[i,j]=a[j,i] and b[x,y]=b[y,x]
        # So the correction terms cancel out.
        
        # Let's use a simpler approach: precompute row contributions
        # For facility i at location l, its contribution to the total cost (excluding the i-i term) is:
        #   sum_k a[i,k]*b[l, p[k]] + a[k,i]*b[p[k], l]
        # But this double-counts pairs, so we need to be careful.
        
        # Alternative: use the standard QAP delta formula
        # For swapping facilities at positions i and j (i.e., swapping p[i] and p[j]):
        # delta = 2 * sum_{k != i, j} [a[i,k] + a[j,k]] * [b[p[i],p[k]] - b[p[j],p[k]]]
        #       + a[i,j] * (b[p[i],p[j]] - b[p[j],p[i]])
        #       + a[j,i] * (b[p[j],p[i]] - b[p[i],p[j]])
        # Since b is symmetric, b[p[i],p[j]] = b[p[j],p[i]], so the last two terms are zero.
        # So: delta = 2 * sum_{k != i, j} [a[i,k] + a[j,k]] * [b[p[i],p[k]] - b[p[j],p[k]]]
        
        # Let's compute this for all pairs (i, j)
        # Precompute: for each pair (i, j), we need sum_k (a[i,k] + a[j,k]) * (b[p[i],p[k]] - b[p[j],p[k]])
        
        # This can be written as:
        # sum_k a[i,k]*b[p[i],p[k]] - sum_k a[i,k]*b[p[j],p[k]]
        # + sum_k a[j,k]*b[p[i],p[k]] - sum_k a[j,k]*b[p[j],p[k]]
        
        # Let's define:
        #   S[i] = sum_k a[i,k] * b[p[i], p[k]]  (row i of a dotted with row p[i] of b, but indexed by p)
        #   T[i,j] = sum_k a[i,k] * b[p[j], p[k]]
        
        # Then delta(i,j) = 2 * (S[i] - T[i,j] + T[j,i] - S[j])
        # But T[j,i] = sum_k a[j,k] * b[p[i], p[k]]
        
        # Let's compute S and T efficiently
        # S[i] = sum_k a[i,k] * b[p[i], p[k]] = (a[i,:] * b[p[i], p[1:]]).sum() + a[i,i]*b[p[i],p[i]]
        # Actually, S[i] = sum_k a[i,k] * b[p[i], p[k]]
        
        # For T[i,j] = sum_k a[i,k] * b[p[j], p[k]]
        
        # We can precompute for each i:
        #   row_a_i = a[i, :]  (shape n,)
        #   Then T[i,j] = row_a_i @ b[p[j], p]  (dot product of a[i,:] with row p[j] of b, permuted by p)
        
        # Let's compute all S first
        S = np.empty(n, dtype=np.float64)
        for i in range(n):
            S[i] = np.dot(a[i, :], b[p[i], p])
        
        # Now compute deltas for all pairs (i, j) with i < j
        # delta(i,j) = 2 * (S[i] - T[i,j] + T[j,i] - S[j])
        # where T[i,j] = a[i,:] @ b[p[j], p]
        
        # To avoid O(n^2 * n) = O(n^3), we can vectorize:
        # For fixed i, T[i,:] = a[i,:] @ B_p where B_p is b with rows permuted by p
        # Actually, T[i,j] = sum_k a[i,k] * b[p[j], p[k]]
        # Let B_perm = b[np.ix_(p, p)]  (the permuted b matrix)
        # Then T[i,j] = a[i,:] @ B_perm[p[j], :] ... wait, no
        # T[i,j] = sum_k a[i,k] * b[p[j], p[k]] = sum_k a[i,k] * B_perm[j, k]
        # So T[i,:] = a[i,:] @ B_perm  (where B_perm[j,k] = b[p[j], p[k]])
        
        B_perm = b[np.ix_(p, p)]
        
        # Now T[i,:] = a[i,:] @ B_perm for each i
        # So T = a @ B_perm  (shape n x n)
        T = a @ B_perm
        
        # Now delta(i,j) = 2 * (S[i] - T[i,j] + T[j,i] - S[j])
        # Note: T[i,j] = sum_k a[i,k] * b[p[j], p[k]]
        # And S[i] = T[i,i] = sum_k a[i,k] * b[p[i], p[k]]
        
        # So delta(i,j) = 2 * (T[i,i] - T[i,j] + T[j,i] - T[j,j])
        
        # Compute all deltas
        # delta_matrix[i,j] = 2 * (T[i,i] - T[i,j] + T[j,i] - T[j,j])
        # = 2 * (S[i] + T[j,i] - T[i,j] - S[j])
        
        diag_T = np.diag(T)  # S[i] = T[i,i]
        
        # delta_matrix[i,j] = 2 * (diag_T[i] + T[j,i] - T[i,j] - diag_T[j])
        # Let's compute this matrix
        # delta_matrix = 2 * (diag_T[:, None] + T.T - T - diag_T[None, :])
        
        delta_matrix = 2.0 * (diag_T[:, None] + T.T - T - diag_T[None, :])
        
        # Set diagonal to 0 (no self-swaps)
        np.fill_diagonal(delta_matrix, 0)
        
        # Find the best (most negative) delta among non-tabu moves
        # We need to exclude tabu moves
        # For efficiency, let's first find the top candidates
        
        # Get all (i, j) with i < j and their deltas
        # We'll iterate through candidates sorted by delta
        
        # Create a mask for non-tabu moves
        # For each (i, j) with i < j, check if (i, j) or (j, i) is in tabu_list
        
        # To speed up, let's find the best few candidates
        # Flatten the upper triangle
        i_idx, j_idx = np.triu_indices(n, k=1)
        deltas = delta_matrix[i_idx, j_idx]
        
        # Sort by delta (most negative first)
        sorted_order = np.argsort(deltas)
        
        # Find the best non-tabu move
        best_delta = 0
        best_i, best_j = -1, -1
        best_aspiration = False
        
        for idx in sorted_order[:200]:  # Check top 200 candidates for efficiency
            i = i_idx[idx]
            j = j_idx[idx]
            d = deltas[idx]
            
            if d >= 0:
                break  # No more improving moves
            
            # Check tabu status
            is_tabu = (i, j) in tabu_list or (j, i) in tabu_list
            
            # Aspiration: allow tabu move if it improves best known cost
            if is_tabu:
                new_cost = best_cost + int(d)
                if new_cost < best_cost:
                    best_delta = d
                    best_i, best_j = i, j
                    best_aspiration = True
                    break
            else:
                best_delta = d
                best_i, best_j = i, j
                break
        
        # If no improving move found, stop
        if best_i == -1:
            break
        
        # Perform the swap
        p[best_i], p[best_j] = p[best_j], p[best_i]
        best_cost += int(best_delta)
        
        # Update best solution
        if best_cost < compute_cost(p):
            # This shouldn't happen if delta is correct, but just in case
            pass
        best_perm = p.copy()
        
        # Update tabu list
        tabu_list.add((best_i, best_j))
        tabu_list.add((best_j, best_i))
        tabu_counter += 1
        
        # Remove old tabu entries
        if tabu_counter > tabu_tenure:
            # We need to remove the oldest entries, but we don't track order
            # For simplicity, let's just clear and restart if list gets too large
            if len(tabu_list) > 2 * tabu_tenure:
                tabu_list.clear()
                tabu_counter = 0
        
        # Periodic check: recompute cost to avoid drift
        if tabu_counter % 100 == 0:
            actual_cost = compute_cost(p)
            if abs(actual_cost - best_cost) > 10:
                best_cost = actual_cost
                best_perm = p.copy()
    
    # Final: return the best permutation found
    return best_perm.tolist()
# EVOLVE-BLOCK-END

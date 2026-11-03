"""QAP solver for tai256c: Spectral Init + Vectorized Best-First Tabu Search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def _qap_cost(a, b, p):
    """Compute exact integer cost sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    # p is a permutation array
    # b_perm[i, j] = b[p[i], p[j]]
    b_perm = b[np.ix_(p, p)]
    return int(np.sum(a * b_perm))

def _spectral_initial(a, b):
    """
    Generate an initial solution using spectral embedding and linear assignment.
    1. Embed facilities based on flow matrix A.
    2. Embed locations based on distance matrix B (inverted to similarity).
    3. Match embeddings using Linear Sum Assignment.
    """
    n = len(a)
    
    # Center matrices
    a_centered = a - a.mean(axis=1, keepdims=True) - a.mean(axis=0, keepdims=True) + a.mean()
    # For distance, we want high similarity (low distance). 
    # Using -B as a proxy for similarity, or (max_B - B).
    # Let's use -b for spectral embedding of locations (maximize alignment with flow).
    b_centered = -b - (-b).mean(axis=1, keepdims=True) - (-b).mean(axis=0, keepdims=True) + (-b).mean()
    
    # SVD for embedding
    # We only need the top few dimensions. n=256 is small enough for full SVD but let's be safe.
    Ua, Sa, Va = np.linalg.svd(a_centered, full_matrices=False)
    Ub, Sb, Vb = np.linalg.svd(b_centered, full_matrices=False)
    
    # Embeddings
    E_a = Ua * np.sqrt(Sa)
    E_b = Ub * np.sqrt(Sb)
    
    # Cost matrix for assignment: negative dot product (since we want to maximize similarity)
    # Cost = - (E_a @ E_b.T)
    cost_assign = - (E_a @ E_b.T)
    
    # linear_sum_assignment returns (row_ind, col_ind)
    # We want facility i to go to location p[i].
    # If row_ind is facility index and col_ind is location index:
    row_ind, col_ind = linear_sum_assignment(cost_assign)
    
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    
    return p

def _vectorized_tabu_search(a, b, p, max_time=145.0):
    """
    Vectorized Best-First Tabu Search.
    Evaluates all swaps in O(n^2) vectorized operations.
    """
    n = len(a)
    start_time = time.time()
    current_p = p.copy()
    current_cost = _qap_cost(a, b, current_p)
    
    # Tabu list: store (i, j, time) or just (i, j) with a counter
    # We use a simple counter-based tabu list.
    # tabu[i, j] = iteration when it becomes untabu.
    # Since n=256, a 256x256 int array is tiny.
    tabu = np.zeros((n, n), dtype=np.int32)
    tabu_tenure = 100  # Number of iterations to keep a move tabu
    
    # Precompute A and B views
    A = a
    B = b
    
    # To compute delta for swap (i, j):
    # Delta = A[i, :] @ (B[p[j], :] - B[p[i], :]) + A[j, :] @ (B[p[i], :] - B[p[j], :])
    #       + A[:, i] @ (B[:, p[j]] - B[:, p[i]]) + A[:, j] @ (B[:, p[i]] - B[:, p[j]])
    
    # We can vectorize this over all pairs (i, j).
    # Let's create matrices for the terms.
    # For each i, we need A[i, :] dot (B[p[j], :] - B[p[i], :])
    
    # This is effectively:
    # Term1[i, j] = A[i, :] @ B[p[j], :] - A[i, :] @ B[p[i], :]
    # Let M1[i, l] = A[i, :] @ B[l, :]  (This is A @ B, but wait, B is indexed by location p)
    # Actually, let's define C[i, l] = sum_k A[i, k] * B[l, k]. This is A @ B.T?
    # No. sum_k A[i, k] * B[l, k] is (A @ B.T)[i, l].
    
    # Let's re-derive carefully.
    # Cost = sum_{i,j} A[i,j] B[p[i], p[j]]
    # Swap i and j in p. Let pi = p[i], pj = p[j].
    # Change in cost:
    # 1. Terms where i is first index: sum_k A[i,k] (B[pj, p[k]] - B[pi, p[k]])
    #    = sum_k A[i,k] B[pj, p[k]] - sum_k A[i,k] B[pi, p[k]]
    #    Let R[i, l] = sum_k A[i,k] B[l, k]. Note: B[l, k] is row l of B, col k.
    #    So R[i, l] = (A @ B.T)[i, l].
    #    Term1 = R[i, pj] - R[i, pi]
    
    # 2. Terms where j is first index: sum_k A[j,k] (B[pi, p[k]] - B[pj, p[k]])
    #    = R[j, pi] - R[j, pj]
    
    # 3. Terms where i is second index: sum_k A[k,i] (B[p[k], pj] - B[p[k], pi])
    #    Let C[l, i] = sum_k B[l, k] A[k, i]. Note: B[l, k] is row l, col k. A[k, i] is row k, col i.
    #    This is (B @ A)[l, i].
    #    Let S = B @ A. Then S[l, i] = sum_k B[l, k] A[k, i].
    #    Term3 = S[pj, i] - S[pi, i]
    
    # 4. Terms where j is second index: sum_k A[k,j] (B[p[k], pi] - B[p[k], pj])
    #    = S[pi, j] - S[pj, j]
    
    # Total Delta(i, j) = (R[i, pj] - R[i, pi]) + (R[j, pi] - R[j, pj]) + (S[pj, i] - S[pi, i]) + (S[pi, j] - S[pj, j])
    
    # Precompute R and S
    R = A @ B.T  # R[i, l]
    S = B @ A    # S[l, i]
    
    # Now, for a given permutation p, we want to find the best swap (i, j).
    # We can compute the delta for all pairs (i, j) in a vectorized manner.
    # However, R[i, pj] depends on p[j].
    # Let's create a matrix D where D[i, j] is the delta for swapping i and j.
    
    # We can compute this iteratively or in blocks.
    # Since n=256, a 256x256 matrix of deltas is small.
    # We can compute the full delta matrix in O(n^2) operations if we use advanced indexing.
    
    # Let's try to compute the full delta matrix at each step.
    # This requires O(n^2) memory and O(n^2) time per step, which is ~65k operations.
    # With Python overhead, this might be slow if done naively in a loop.
    # But we can use NumPy broadcasting.
    
    # Let's define:
    # p_arr = current_p
    # R_i_pi = R[np.arange(n), p_arr]  # Vector of R[i, p[i]] for all i
    # R_i_pj = R[np.arange(n)[:, None], p_arr[None, :]]  # Matrix R[i, p[j]]
    
    # Similarly for S.
    
    best_delta = 0
    best_i = -1
    best_j = -1
    iter_count = 0
    
    while True:
        if time.time() - start_time > max_time:
            break
        
        iter_count += 1
        
        p_arr = current_p
        idx = np.arange(n)
        
        # Compute R[i, p[j]] for all i, j
        # R is (n, n). p_arr is (n,).
        # We want a matrix M1 where M1[i, j] = R[i, p_arr[j]]
        M1 = R[:, p_arr]  # Shape (n, n). M1[i, j] = R[i, p_arr[j]]
        
        # We want Delta(i, j) = M1[i, j] - M1[i, i] + M1[j, i] - M1[j, j] + ...
        # Wait, M1[i, j] is R[i, p[j]].
        # Term1 (i row) = R[i, p[j]] - R[i, p[i]] = M1[i, j] - M1[i, i]
        # Term2 (j row) = R[j, p[i]] - R[j, p[j]] = M1[j, i] - M1[j, j]
        
        # For S terms:
        # S[l, i] where l = p[k].
        # Term3 (i col) = S[p[j], i] - S[p[i], i]
        # Let M2[l, i] = S[l, i].
        # We need S[p_arr[j], i] - S[p_arr[i], i].
        # Let M2 = S[p_arr, :]  # Shape (n, n). M2[j, i] = S[p_arr[j], i]
        M2 = S[p_arr, :]  # M2[j, i] = S[p_arr[j], i]
        
        # Term3 = M2[j, i] - M2[i, i]
        # Term4 (j col) = S[p[i], j] - S[p[j], j] = M2[i, j] - M2[j, j]
        
        # Total Delta Matrix D:
        # D[i, j] = (M1[i, j] - M1[i, i]) + (M1[j, i] - M1[j, j]) + (M2[j, i] - M2[i, i]) + (M2[i, j] - M2[j, j])
        
        # Let's compute this using broadcasting.
        # M1_diag = np.diag(M1)  # M1[i, i]
        # M2_diag = np.diag(M2)  # M2[i, i]
        
        # Term1_matrix = M1 - M1_diag[:, None]  # M1[i, j] - M1[i, i]
        # Term2_matrix = M1.T - M1_diag[None, :] # M1[j, i] - M1[j, j] ? 
        # Wait. M1.T[j, i] is M1[i, j]. No. M1.T[i, j] is M1[j, i].
        # We want Term2[i, j] = M1[j, i] - M1[j, j].
        # M1[j, i] is M1.T[i, j].
        # M1[j, j] is M1_diag[j].
        # So Term2_matrix = M1.T - M1_diag[None, :]
        
        # Term3_matrix[i, j] = M2[j, i] - M2[i, i]
        # M2[j, i] is M2.T[i, j].
        # M2[i, i] is M2_diag[i].
        # So Term3_matrix = M2.T - M2_diag[:, None]
        
        # Term4_matrix[i, j] = M2[i, j] - M2[j, j]
        # M2[i, j] is M2[i, j].
        # M2[j, j] is M2_diag[j].
        # So Term4_matrix = M2 - M2_diag[None, :]
        
        M1_diag = np.diag(M1)
        M2_diag = np.diag(M2)
        
        D = (M1 - M1_diag[:, None]) + (M1.T - M1_diag[None, :]) + \
            (M2.T - M2_diag[:, None]) + (M2 - M2_diag[None, :])
        
        # Set diagonal to 0 (no swap with self)
        np.fill_diagonal(D, 0)
        
        # Apply Tabu
        # tabu[i, j] > iter_count means it's tabu.
        # We want to ignore tabu moves unless they are the best (aspiration).
        # Simple approach: Set tabu moves to a large positive value (bad delta).
        # Or just mask them out.
        
        # Mask out i > j to avoid duplicate checks? 
        # Swapping (i, j) is same as (j, i).
        # Let's just look at the full matrix and pick the best valid one.
        
        # Create a mask for valid moves: not tabu, and i != j
        valid_mask = (tabu <= iter_count) & (np.arange(n)[:, None] != np.arange(n)[None, :])
        
        # Set invalid moves to a very large value (since we want to minimize delta, and negative is good)
        # We want the most negative delta.
        D_masked = np.where(valid_mask, D, np.inf)
        
        # Find the best (most negative) delta
        min_idx = np.unravel_index(np.argmin(D_masked), D_masked.shape)
        best_i, best_j = min_idx
        best_delta = D_masked[best_i, best_j]
        
        # If no valid move or best delta is positive (worsening), check for aspiration
        # If best_delta > 0, it means all valid moves worsen the cost.
        # In Tabu Search, we might still take a worsening move if it's the best available (to escape local min).
        # However, if we are stuck in a local minimum, we MUST take a worsening move.
        # So, if best_delta is inf (no valid moves), we should reset tabu or pick the best tabu move.
        # If best_delta > 0, we accept it if it's the best among all (including tabu? No, tabu is for preventing cycles).
        # Standard TS: Accept the best move in the neighborhood, even if it's tabu or worsening, 
        # BUT usually we restrict to non-tabu moves unless the best non-tabu move is worse than the best tabu move?
        # No, the rule is: Accept the best move among non-tabu moves. If that move worsens the objective, we still accept it (unless we have a specific stopping rule).
        # This ensures we escape local minima.
        
        if np.isinf(best_delta):
             # All moves are tabu. This is rare. Reset tabu list.
             tabu.fill(0)
             continue
             
        # Accept the move
        # Update p
        current_p[best_i], current_p[best_j] = current_p[best_j], current_p[best_i]
        current_cost += best_delta
        
        # Update tabu list
        # Mark (best_i, best_j) and (best_j, best_i) as tabu for next 'tenure' iterations
        tabu[best_i, best_j] = iter_count + tabu_tenure
        tabu[best_j, best_i] = iter_count + tabu_tenure
        
        # Optional: Aspiration criterion
        # If the move leads to a new global best, it's not tabu (but we already accepted it).
        # We don't need to untabu it immediately if we just set it.
        
    return current_p, current_cost

def solve(a, b):
    """Solve the QAP instance using spectral init + vectorized tabu search."""
    start_time = time.time()
    n = len(a)
    
    # Generate initial solution
    p = _spectral_initial(a, b)
    
    # Run local search with time budget
    # Leave 5s safety margin
    max_time = 145.0
    p, cost = _vectorized_tabu_search(a, b, p, max_time=max_time)
    
    # Final check: ensure we're within time limit
    if time.time() - start_time > max_time:
         # Should not happen, but just in case
         pass
         
    return p.tolist()
# EVOLVE-BLOCK-END

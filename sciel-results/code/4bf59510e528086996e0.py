"""
Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK.
"""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
import itertools

def _construct_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_merge_terms(U, V, W, idx1, idx2):
    """
    Attempts to replace terms idx1 and idx2 with a single new term.
    Returns (new_U_row, new_V_row, new_W_row) if successful, else None.
    This is a heuristic check.
    """
    u1 = np.array(U[idx1])
    u2 = np.array(U[idx2])
    v1 = np.array(V[idx1])
    v2 = np.array(V[idx2])
    w1 = np.array(W[idx1])
    w2 = np.array(W[idx2])
    
    # The sum of two rank-1 tensors is:
    # T_sum = u1 v1^T w1^T + u2 v2^T w2^T
    # We want to see if this can be written as u' v'^T w'^T.
    # This requires the matrix M = u1 v1^T + u2 v2^T (in a slice where w is non-zero) 
    # to be rank 1, which implies linear dependence.
    # A stronger condition: The vector space spanned by the slices must be 1D.
    
    # Let's check if w1 and w2 are linearly dependent.
    # If w1 = c * w2, then T_sum = w2^T (c * u1 v1^T + u2 v2^T).
    # Then we need (c * u1 v1^T + u2 v2^T) to be rank 1.
    
    # Case 1: w1 and w2 are proportional
    if np.allclose(w1, 0):
        w_prime = w2
        # Then T_sum = u2 v2^T w2^T, which is already rank 1 (just idx2). 
        # But we are merging two. If w1 is 0, term 1 is 0. We can just keep term 2?
        # The goal is to reduce rank. If one term is zero, we can drop it.
        # But in our schoolbook, terms are non-zero.
        pass
    
    # Check if w1 is a scalar multiple of w2
    if not np.allclose(w2, 0):
        # Compute c such that w1 ≈ c * w2
        # Use least squares or direct division if unique non-zero
        mask = w2 != 0
        if np.any(mask):
            # c = dot(w1, w2) / dot(w2, w2) if orthogonal basis, but let's just pick one element
            idx = np.where(mask)[0][0]
            c = w1[idx] / w2[idx]
            if np.allclose(w1, c * w2):
                # Check if M = c*u1*v1.T + u2*v2.T is rank 1
                M = c * np.outer(u1, v1) + np.outer(u2, v2)
                # Rank 1 check: all 2x2 minors are 0
                # For small matrices, check if M = a * b.T
                # If M is rank 1, we can decompose it.
                # Use SVD or simple check.
                if M.size > 0:
                    # Check if rows are proportional
                    if np.allclose(M, 0):
                        return None # Degenerate
                    # Find a non-zero row
                    row_idx = np.argmax(np.sum(np.abs(M), axis=1))
                    if M[row_idx, 0] != 0:
                        # Check if M[:, 0] is proportional to M[:, j] for all j
                        is_rank1 = True
                        v_prime_candidate = M[row_idx, :]
                        u_prime_candidate = M[:, row_idx] / M[row_idx, row_idx] # Wait, this is for u v^T = M
                        # Actually, if M is rank 1, M = u' v'^T.
                        # We can extract u' and v'.
                        # Let's just verify if M is rank 1.
                        # For 3x5 and 5x7, M is 15x35.
                        # Rank 1 means M[i,j] = A[i] B[j].
                        # Check if M[i,0] / M[0,0] == M[i, j] / M[0, j]
                        # Simpler: Check if M.T @ M is rank 1 (3x3 matrix)
                        MM = M.T @ M
                        # Rank 1 if all 2x2 minors of MM are 0
                        # MM is 3x3.
                        # det of submatrix should be 0.
                        minor1 = MM[0,0]*MM[1,1] - MM[0,1]*MM[1,0]
                        minor2 = MM[0,0]*MM[2,2] - MM[0,2]*MM[2,0]
                        minor3 = MM[1,0]*MM[2,1] - MM[1,1]*MM[2,0]
                        if abs(minor1) < 1e-9 and abs(minor2) < 1e-9 and abs(minor3) < 1e-9:
                             # M is rank 1.
                             # Extract u', v'.
                             # M = u' v'^T
                             # Let u' = M[:, 0] if M[0,0]!=0 else ...
                             # Let v' = M[0, :] / M[0,0]
                             if M[0,0] != 0:
                                 u_prime = M[:, 0]
                                 v_prime = M[0, :] / M[0,0]
                             else:
                                 # Find non-zero entry
                                 i, j = np.unravel_index(np.argmax(np.abs(M)), M.shape)
                                 u_prime = M[:, j]
                                 v_prime = M[i, :] / M[i, j]
                             
                             # We found u', v', and w' (which is w2).
                             # Check if this decomposition is valid for the whole tensor?
                             # The sum was u1v1^Tw1^T + u2v2^Tw2^T = u'v'^Tw'^T
                             # Since w1 = c w2, the sum is w2^T (c u1v1^T + u2v2^T) = w2^T M.
                             # And we found M = u'v'^T.
                             # So yes, it works.
                             # Return the new row.
                             # Note: U, V, W are lists of lists.
                             # We need to return the vectors to replace the rows.
                             return (u_prime.tolist(), v_prime.tolist(), w2.tolist())
    
    # Case 2: w1 and w2 are not proportional.
    # If w1 and w2 are linearly independent, the slice space is 2D.
    # The tensor sum would have border rank at least 2 in that slice?
    # Generally, if w1, w2 are indep, the sum cannot be rank 1.
    
    return None

def _greedy_reduce(U, V, W, time_limit):
    """
    Greedily try to merge pairs of terms to reduce rank.
    """
    start_time = time.time()
    current_U = [row[:] for row in U]
    current_V = [row[:] for row in V]
    current_W = [row[:] for row in W]
    rank = len(current_U[0]) if current_U else 0
    
    # We remove terms by replacing two indices with one.
    # It's easier to work with lists of vectors and remove indices.
    u_vecs = [np.array(u) for u in current_U]
    v_vecs = [np.array(v) for v in current_V]
    w_vecs = [np.array(w) for w in current_W]
    
    # We maintain active indices
    active = list(range(rank))
    
    improved = True
    while improved:
        if time.time() - start_time > time_limit:
            break
        improved = False
        # Try all pairs
        for i_idx in range(len(active)):
            if time.time() - start_time > time_limit:
                break
            idx1 = active[i_idx]
            for j_idx in range(i_idx + 1, len(active)):
                idx2 = active[j_idx]
                
                result = _try_merge_terms(current_U, current_V, current_W, idx1, idx2)
                if result is not None:
                    u_new, v_new, w_new = result
                    # Replace idx1 and idx2 with a new single term
                    # We need to update the arrays.
                    # Remove idx1 and idx2, add new one.
                    # This changes the structure.
                    # To keep it simple, we can just zero out the rows and add a new column?
                    # No, the interface requires fixed shape (n*m, rank).
                    # We must maintain the rank dimension size?
                    # "factor matrices of shapes (n*m, rank)".
                    # If we reduce rank, the shape changes.
                    # The evaluator likely accepts any rank R.
                    # So we can return a smaller matrix.
                    
                    # Let's rebuild the active sets.
                    new_u_vecs = []
                    new_v_vecs = []
                    new_w_vecs = []
                    
                    for k in range(len(active)):
                        if k == i_idx or k == j_idx:
                            continue
                        new_u_vecs.append(u_vecs[active[k]])
                        new_v_vecs.append(v_vecs[active[k]])
                        new_w_vecs.append(w_vecs[active[k]])
                    
                    new_u_vecs.append(np.array(u_new))
                    new_v_vecs.append(np.array(v_new))
                    new_w_vecs.append(np.array(w_new))
                    
                    # Convert back to lists of lists
                    current_U = [v.tolist() for v in new_u_vecs]
                    current_V = [v.tolist() for v in new_v_vecs]
                    current_W = [v.tolist() for v in new_w_vecs]
                    
                    u_vecs = new_u_vecs
                    v_vecs = new_v_vecs
                    w_vecs = new_w_vecs
                    active = list(range(len(u_vecs)))
                    
                    improved = True
                    # Restart pair search with new active set
                    break
            if improved:
                break
                
    # Final conversion
    if not current_U:
         return None
         
    final_rank = len(current_U[0])
    return {
        "U": current_U,
        "V": current_V,
        "W": current_W
    }

def construct():
    start_time = time.time()
    deadline = start_time + 110 # Safety margin
    
    n, m, p = DIMENSIONS
    
    # 1. Get Schoolbook
    result_schoolbook = _construct_schoolbook()
    
    # 2. Try to improve
    if time.time() < deadline:
        try:
            improved_result = _greedy_reduce(
                result_schoolbook["U"], 
                result_schoolbook["V"], 
                result_schoolbook["W"],
                time_limit=100
            )
            if improved_result:
                # Verify if the rank is actually lower
                if len(improved_result["U"][0]) < len(result_schoolbook["U"][0]):
                    return improved_result
        except Exception as e:
            # If error, fallback to schoolbook
            pass
            
    return result_schoolbook

# EVOLVE-BLOCK-END

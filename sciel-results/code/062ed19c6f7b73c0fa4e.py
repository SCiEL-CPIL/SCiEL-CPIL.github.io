"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _build_tensor(n, m, p):
    """Build the 6x9x6 tensor T where T[a,b,c]=1 iff indices are consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Verify that sum_r U[:,r] * V[:,r] * W[:,r] == T."""
    rank = U.shape[1]
    result = np.zeros(T.shape)
    for r in range(rank):
        result += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    return np.allclose(result, T, atol=tol)

def _try_rank16_via_linear_combination(U17, V17, W17, T, deadline):
    """
    Try to find a rank-16 decomposition by expressing one of the 17 rank-1 terms
    as a linear combination of the others, then checking if the resulting
    16-term decomposition still reproduces T.
    
    The idea: if there exists a vector (alpha_0, ..., alpha_16) not all zero such that
    sum_r alpha_r * (U_r, V_r, W_r) = 0 in the tensor space, then we can potentially
    eliminate one term.
    
    More practically: try to find coefficients c_0, ..., c_16 such that
    sum_r c_r * T_r = 0 where T_r = U_r ⊗ V_r ⊗ W_r, and then remove one term.
    """
    n_a, _ = U17.shape
    n_b, _ = V17.shape
    n_c, _ = W17.shape
    
    # Build the 17 rank-1 tensors as vectors in R^(n_a*n_b*n_c)
    n_terms = 17
    T_vecs = []
    for r in range(n_terms):
        vec = np.einsum('a,b,c->abc', U17[:,r], V17[:,r], W17[:,r]).flatten()
        T_vecs.append(vec)
    
    # Stack into a matrix of shape (n_a*n_b*n_c, n_terms)
    M = np.column_stack(T_vecs)
    
    # Find the null space of M^T (i.e., vectors c such that M @ c = 0)
    # This gives linear dependencies among the 17 tensors
    try:
        # SVD to find null space
        U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
        # Null space vectors are rows of Vt corresponding to zero singular values
        tol_svd = 1e-10
        rank_svd = np.sum(s > tol_svd)
        null_dim = n_terms - rank_svd
        
        if null_dim > 0:
            # Get the null space basis
            null_space = Vt[rank_svd:].T  # shape (n_terms, null_dim)
            
            # For each null space vector, try to eliminate one term
            for c in null_space:
                # Find the index with the largest absolute coefficient
                idx_max = np.argmax(np.abs(c))
                if abs(c[idx_max]) < 1e-10:
                    continue
                
                # Try to express term idx_max as a linear combination of the others
                # c[idx_max] * T_{idx_max} + sum_{r != idx_max} c[r] * T_r = 0
                # T_{idx_max} = -sum_{r != idx_max} (c[r] / c[idx_max]) * T_r
                
                # Build the new 16-term decomposition by replacing term idx_max
                # with the linear combination
                U_new = np.zeros((n_a, n_terms - 1))
                V_new = np.zeros((n_b, n_terms - 1))
                W_new = np.zeros((n_c, n_terms - 1))
                
                col = 0
                for r in range(n_terms):
                    if r == idx_max:
                        continue
                    U_new[:, col] = U17[:, r]
                    V_new[:, col] = V17[:, r]
                    W_new[:, col] = W17[:, r]
                    col += 1
                
                # Now we need to add the contribution from term idx_max
                # But wait, this approach doesn't directly give us a valid 16-term decomposition
                # because the linear combination involves all 17 terms.
                
                # Alternative: if there's a dependency, we can set one coefficient to 0
                # and adjust the others. But this is tricky.
                
                # Let's try a different approach: use the null space to find a 
                # 16-dimensional subspace that still spans T.
                pass
            
            # For now, just check if the rank is actually less than 17
            if rank_svd < 17:
                # The tensors are linearly dependent, so rank might be reducible
                # But we need to actually construct the lower-rank decomposition
                pass
    except Exception:
        pass
    
    return None

def _try_rank16_via_polynomial_search(U17, V17, W17, T, deadline):
    """
    Try a more direct approach: search for a 16-term decomposition by 
    optimizing over the entries.
    
    This is a non-convex problem, so we use a heuristic approach.
    """
    n_a, _ = U17.shape
    n_b, _ = V17.shape
    n_c, _ = W17.shape
    
    # Start with the 17-term decomposition and try to merge two terms
    # by finding a linear combination that simplifies the structure.
    
    # For each pair of terms (i, j), try to find coefficients alpha, beta
    # such that alpha * T_i + beta * T_j can be replaced by a single rank-1 term.
    # This is generally not possible, but we can check if the sum has rank 1.
    
    for i in range(17):
        for j in range(i+1, 17):
            # Check if T_i + T_j has rank 1
            vec = (np.einsum('a,b,c->abc', U17[:,i], V17[:,i], W17[:,i]) +
                   np.einsum('a,b,c->abc', U17[:,j], V17[:,j], W17[:,j])).flatten()
            
            # Check rank of the 6x9x6 tensor formed by vec
            # Reshape to 6x54 and check rank
            T_sum = vec.reshape(6, 54)
            s_vals = np.linalg.svd(T_sum, compute_uv=False)
            if len(s_vals) > 0 and s_vals[0] > 1e-10:
                rank_est = np.sum(s_vals > 1e-10)
                if rank_est == 1:
                    # T_i + T_j is rank 1, so we can merge them
                    # Find the rank-1 decomposition of T_sum
                    u, s, vt = np.linalg.svd(T_sum, full_matrices=False)
                    if u.shape[0] > 0 and vt.shape[1] > 0:
                        u_vec = u[:, 0] * np.sqrt(s[0])
                        v_vec = vt[0, :] * np.sqrt(s[0])
                        
                        # Reshape back
                        U_merged = u_vec.reshape(6)
                        VW_merged = v_vec.reshape(9, 6)
                        
                        # Now we need to factor VW_merged as V ⊗ W
                        # This is generally not possible, so skip
                        pass
    
    return None

def _try_rank16_via_alternative_construction(deadline):
    """
    Try an alternative construction that might achieve rank 16.
    
    Idea: Instead of using Strassen for the 2x2 block, try a different
    block decomposition or a more clever combination of terms.
    """
    n, m, p = DIMENSIONS
    n_a = n * m  # 6
    n_b = m * p  # 9
    n_c = p * n  # 6
    
    # Try to construct a rank-16 decomposition from scratch.
    # The tensor has 6*9*6 = 324 entries, with 18 non-zero entries.
    # Rank 16 would mean 16 rank-1 tensors sum to T.
    
    # One approach: use the fact that the last column of A (a2, a5) 
    # multiplies with the last row of B (b6, b7, b8) to give the last 
    # column of C (c4, c5) for row 0 and (c1, c3, c5) for row 1.
    
    # Actually, let's try a systematic search over small coefficient sets.
    # We'll try to find a 16-term decomposition where each term has 
    # small integer or half-integer coefficients.
    
    # This is a very hard problem, so we'll use a heuristic: 
    # start with the rank-17 decomposition and try to perturb it.
    
    U = np.zeros((n_a, 16))
    V = np.zeros((n_b, 16))
    W = np.zeros((n_c, 16))
    
    # Initialize with the first 16 terms of the rank-17 decomposition
    rank17_terms = [
        ([0, 4], [1, 1], [1, 1], [1, 1], [0, 3], [1, 1]),
        ([3, 4], [1, 1], [0], [1], [1, 3], [1, -1]),
        ([0], [1], [1, 4], [1, -1], [2, 3], [1, 1]),
        ([4], [1], [3, 0], [1, -1], [0, 1], [1, 1]),
        ([0, 1], [1, 1], [4], [1], [0, 2], [-1, 1]),
        ([3, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
        ([1, 4], [1, -1], [3, 4], [1, 1], [0], [1]),
        ([0], [1], [2], [1], [4], [1]),
        ([1], [1], [5], [1], [4], [1]),
        ([3], [1], [2], [1], [5], [1]),
        ([4], [1], [5], [1], [5], [1]),
        ([2], [1], [6], [1], [0], [1]),
        ([2], [1], [7], [1], [2], [1]),
        ([2], [1], [8], [1], [4], [1]),
        ([5], [1], [6], [1], [1], [1]),
        ([5], [1], [7], [1], [3], [1]),
        ([5], [1], [8], [1], [5], [1]),
    ]
    
    # Try dropping the last term and see if the remaining 16 terms work
    for r in range(16):
        u_idx, u_val, v_idx, v_val, w_idx, w_val = rank17_terms[r]
        for a, val in zip(u_idx, u_val):
            U[a, r] = val
        for b, val in zip(v_idx, v_val):
            V[b, r] = val
        for c, val in zip(w_idx, w_val):
            W[c, r] = val
    
    T = _build_tensor(n, m, p)
    if _verify_decomposition(U, V, W, T):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    n_a = n * m  # 6
    n_b = m * p  # 9
    n_c = p * n  # 6
    
    # Build the rank-17 decomposition
    rank17_terms = [
        ([0, 4], [1, 1], [1, 1], [1, 1], [0, 3], [1, 1]),
        ([3, 4], [1, 1], [0], [1], [1, 3], [1, -1]),
        ([0], [1], [1, 4], [1, -1], [2, 3], [1, 1]),
        ([4], [1], [3, 0], [1, -1], [0, 1], [1, 1]),
        ([0, 1], [1, 1], [4], [1], [0, 2], [-1, 1]),
        ([3, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
        ([1, 4], [1, -1], [3, 4], [1, 1], [0], [1]),
        ([0], [1], [2], [1], [4], [1]),
        ([1], [1], [5], [1], [4], [1]),
        ([3], [1], [2], [1], [5], [1]),
        ([4], [1], [5], [1], [5], [1]),
        ([2], [1], [6], [1], [0], [1]),
        ([2], [1], [7], [1], [2], [1]),
        ([2], [1], [8], [1], [4], [1]),
        ([5], [1], [6], [1], [1], [1]),
        ([5], [1], [7], [1], [3], [1]),
        ([5], [1], [8], [1], [5], [1]),
    ]
    
    U17 = np.zeros((n_a, 17))
    V17 = np.zeros((n_b, 17))
    W17 = np.zeros((n_c, 17))
    
    for r, (u_idx, u_val, v_idx, v_val, w_idx, w_val) in enumerate(rank17_terms):
        for a, val in zip(u_idx, u_val):
            U17[a, r] = val
        for b, val in zip(v_idx, v_val):
            V17[b, r] = val
        for c, val in zip(w_idx, w_val):
            W17[c, r] = val
    
    # Verify rank-17
    T = _build_tensor(n, m, p)
    if not _verify_decomposition(U17, V17, W17, T):
        # Fallback to schoolbook rank 18
        U_fb = np.zeros((n_a, 18))
        V_fb = np.zeros((n_b, 18))
        W_fb = np.zeros((n_c, 18))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # Try to find rank-16 via various methods
    if time.time() < deadline - 10:
        result16 = _try_rank16_via_linear_combination(U17, V17, W17, T, deadline)
        if result16 is not None:
            return result16
    
    if time.time() < deadline - 10:
        result16b = _try_rank16_via_alternative_construction(deadline)
        if result16b is not None:
            return result16b
    
    # Return the rank-17 decomposition
    return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
# EVOLVE-BLOCK-END

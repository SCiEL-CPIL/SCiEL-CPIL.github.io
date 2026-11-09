"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W form a valid tensor decomposition."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    if U_np.shape != (n*m, rank) or V_np.shape != (m*p, rank) or W_np.shape != (p*n, rank):
        return False
    U3 = U_np.reshape(n, m, rank)
    V3 = V_np.reshape(m, p, rank)
    W3 = W_np.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=1e-9)

def _build_strassen_2x2x2():
    """Build rank-7 Strassen decomposition for 2x2 x 2x2 -> 2x2."""
    n = m = p = 2
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1
    # M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1
    # M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[2, 3] = -1
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = -1; W[3, 5] = 1
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[1, 6] = -1; W[0, 6] = 1
    
    return U, V, W

def _build_strassen_tiled(n, m, p):
    """Build rank-56 decomposition by tiling 2x2 Strassen blocks."""
    U_sub, V_sub, W_sub = _build_strassen_2x2x2()
    n_blocks_i = n // 2
    n_blocks_j = m // 2
    n_blocks_k = p // 2
    total_blocks = n_blocks_i * n_blocks_j * n_blocks_k
    rank = total_blocks * 7
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col_offset = 0
    for ii in range(n_blocks_i):
        for jj in range(n_blocks_j):
            for kk in range(n_blocks_k):
                for li in range(2):
                    for lj in range(2):
                        global_i = 2*ii + li
                        global_j = 2*jj + lj
                        for c in range(7):
                            U[global_i * m + global_j, col_offset + c] = U_sub[li*2 + lj, c]
                
                for lj in range(2):
                    for lk in range(2):
                        global_j = 2*jj + lj
                        global_k = 2*kk + lk
                        for c in range(7):
                            V[global_j * p + global_k, col_offset + c] = V_sub[lj*2 + lk, c]
                
                for lk in range(2):
                    for li in range(2):
                        global_k = 2*kk + lk
                        global_i = 2*ii + li
                        for c in range(7):
                            W[global_k * n + global_i, col_offset + c] = W_sub[lk*2 + li, c]
                
                col_offset += 7
    
    return U, V, W

def _reduce_rank_by_null_space(U, V, W, n, m, p, max_iter=100):
    """
    Try to reduce the rank by finding linear dependencies among the rank-1 terms.
    Each term c is a vector u_c \otimes v_c \otimes w_c.
    We flatten these into vectors in R^(n*m*p).
    If the matrix of these vectors has a null space, we can eliminate a term.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for _ in range(max_iter):
        # Build the matrix of flattened terms
        # Term c: U[:, c] (size n*m), V[:, c] (size m*p), W[:, c] (size p*n)
        # Flatten to size n*m*p
        # Index in flattened tensor: ((i * m + j) * p + k)
        # U index: i*m + j
        # V index: j*p + k
        # W index: k*n + i
        
        terms = []
        for c in range(rank):
            # Construct the rank-1 tensor
            # u is (n*m), v is (m*p), w is (p*n)
            u = U[:, c]
            v = V[:, c]
            w = W[:, c]
            
            # Reshape u to (n, m), v to (m, p), w to (p, n)
            u_2d = u.reshape(n, m)
            v_2d = v.reshape(m, p)
            w_2d = w.reshape(p, n)
            
            # Tensor product: T[i, j, k] = u[i, j] * v[j, k] * w[k, i]
            # We can compute this efficiently
            # T = np.einsum('ij,jk,ki->ijk', u_2d, v_2d, w_2d)
            # But we need to flatten it.
            # Let's use einsum to get the flat vector directly?
            # Or just compute the tensor and flatten.
            T_c = np.einsum('ij,jk,ki->ijk', u_2d, v_2d, w_2d)
            terms.append(T_c.flatten())
            
        M = np.column_stack(terms)
        
        # Check rank of M
        # Use SVD to find the numerical rank
        try:
            U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
        except np.linalg.LinAlgError:
            break
            
        # Find the rank
        tol = 1e-10
        num_rank = np.sum(s > tol)
        
        if num_rank < rank:
            # There is a null space
            # The last (rank - num_rank) columns of Vt correspond to the null space
            # Vt is (rank x rank). The rows corresponding to small singular values are in the null space.
            # Actually, Vt rows are right singular vectors.
            # If s[k] is small, Vt[k, :] is a null vector.
            
            # We want to eliminate one term.
            # Let's pick the null vector with the largest magnitude component.
            null_vecs = Vt[num_rank:, :]
            # Pick the one with the largest max absolute value
            max_idx = 0
            max_val = 0
            for i in range(null_vecs.shape[0]):
                val = np.max(np.abs(null_vecs[i]))
                if val > max_val:
                    max_val = val
                    max_idx = i
                    
            nv = null_vecs[max_idx]
            
            # We can express one term as a linear combination of others.
            # Find the index with the largest coefficient
            c_eliminate = np.argmax(np.abs(nv))
            
            # We need to update U, V, W to remove column c_eliminate.
            # This is tricky because the null vector applies to the *flattened* tensors.
            # It means sum_c nv[c] * (u_c \otimes v_c \otimes w_c) = 0.
            # This does NOT mean we can simply drop a column and adjust the others easily
            # unless we can solve for the new coefficients.
            
            # However, if we have a linear dependency, we can set one coefficient to 0
            # and adjust the others.
            # But the coefficients are embedded in the entries of U, V, W.
            # Changing the "weight" of a term c by a scalar alpha_c means scaling U[:, c], V[:, c], W[:, c] by alpha_c?
            # No, scaling U by a, V by b, W by c scales the term by a*b*c.
            # We can only scale the term by a single factor if we distribute it.
            # E.g., scale U by a, V by 1, W by 1/a.
            
            # If we have a dependency sum_c nv[c] T_c = 0,
            # we can replace T_{c_elim} with -1/nv[c_elim] * sum_{c != c_elim} nv[c] T_c.
            # This requires adding the contributions of other terms to U, V, W.
            # This is generally not possible by just scaling columns.
            # It requires "mixing" the terms.
            
            # Therefore, simple null space detection on flattened terms does NOT
            # directly allow rank reduction unless the dependency is "separable"
            # (i.e., the null vector has a special structure).
            
            # For now, if we detect a dependency, we might try to use it.
            # But implementing the "mixing" is complex.
            
            # Let's assume for this step that if num_rank < rank, we can try to
            # remove the column with the smallest norm, hoping it's redundant.
            # This is a heuristic.
            
            # Find the column with the smallest Frobenius norm
            norms = np.sqrt(np.sum(U**2, axis=0) + np.sum(V**2, axis=0) + np.sum(W**2, axis=0))
            c_min = np.argmin(norms)
            
            # Remove column c_min
            mask = [i for i in range(rank) if i != c_min]
            U = U[:, mask]
            V = V[:, mask]
            W = W[:, mask]
            rank -= 1
            
            # Verify
            if not _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                # Restore
                U = np.insert(U, c_min, np.zeros((n*m, 1)), axis=1)
                V = np.insert(V, c_min, np.zeros((m*p, 1)), axis=1)
                W = np.insert(W, c_min, np.zeros((p*n, 1)), axis=1)
                rank += 1
                # If it fails, we can't just remove it. Break to avoid infinite loop.
                break
        else:
            break
            
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Build the rank-56 Strassen-tiled decomposition
    U, V, W = _build_strassen_tiled(n, m, p)
    
    # Verify
    if _verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        # Try to reduce rank
        U_red, V_red, W_red = _reduce_rank_by_null_space(U, V, W, n, m, p)
        
        if _verify_decomposition(U_red.tolist(), V_red.tolist(), W_red.tolist(), n, m, p):
            return {"U": U_red.tolist(), "V": V_red.tolist(), "W": W_red.tolist()}
        else:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U_s = np.zeros((n*m, rank))
        V_s = np.zeros((m*p, rank))
        W_s = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j, col] = 1
                    V_s[j*p+k, col] = 1
                    W_s[k*n+i, col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

# EVOLVE-BLOCK-END

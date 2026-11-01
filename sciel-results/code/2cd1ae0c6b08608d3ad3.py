"""Tensor decomposition for 2x4 x 4x4 matrix multiplication."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 2, 4, 4

    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i, j, k] = 1 if (i,j) -> C[k] in standard multiplication
    # A is n x m, B is m x p, C is n x p
    # C[r, s] = sum_{t=0}^{m-1} A[r, t] * B[t, s]
    # Indexing:
    #   U index: r * m + t  (row of A, column of A = row of B)
    #   V index: t * p + s  (row of B, column of B)
    #   W index: s * n + r  (official trace convention: W[k*n+i, r] where k=s, i=r)
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for r in range(n):
        for t in range(m):
            for s in range(p):
                u_idx = r * m + t
                v_idx = t * p + s
                w_idx = s * n + r
                T[u_idx, v_idx, w_idx] = 1.0

    # Try to find a rank-27 decomposition with integer/half-integer entries
    # using random initialization and iterative refinement.
    
    def try_decompose(rank, max_iters, seed=42, tol=1e-10):
        """Try to find a rank-r decomposition of T."""
        rng = np.random.RandomState(seed)
        
        # Random initialization
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
        
        # Iterative optimization: alternate fixing two factors and solving for the third
        # This is a form of tensor ring/ALS for exact decomposition
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update W: for each (u, v, w), T[u,v,w] = sum_k U[u,k]*V[v,k]*W[w,k]
            # Fix U, V; solve for W
            # This is: for each w, W[w, :] = argmin_W sum_{u,v} (T[u,v,w] - sum_k U[u,k]*V[v,k]*W[w,k])^2
            # But this is NOT linear in W when U,V are fixed... wait, it IS linear!
            # T[u,v,w] = sum_k (U[u,k]*V[v,k]) * W[w,k]
            # Let M[u,v,k] = U[u,k]*V[v,k], then T[u,v,w] = sum_k M[u,v,k]*W[w,k]
            # For each w: sum_{u,v} (T[u,v,w] - sum_k M[u,v,k]*W[w,k])^2 = 0
            # This is a system of n*m*(m*p) equations in rank unknowns... but we need EXACT.
            # Actually for exact: T = sum_k U[:,k] ⊗ V[:,k] ⊗ W[:,k]
            # Given U, V, W is determined if the system is consistent.
            
            # Better approach: use the fact that for exact rank-r decomposition:
            # T[u,v,w] = sum_{k=0}^{r-1} U[u,k] V[v,k] W[w,k]
            # 
            # Fix U and V. Then for each w, we need:
            #   T[:, :, w] = sum_k (U ⊙ V)_k * W[:, k]
            # where (U ⊙ V)_k is the outer product U[:,k] ⊗ V[:,k] flattened.
            # So T[:, :, w] (flattened to n*m*m*p) = sum_k W[w,k] * (U[:,k] ⊗ V[:,k])
            # This is a linear system in W[w, :] for each w.
            
            # Let's use a different approach: alternating least squares for the bilinear form
            # Actually, let me try a smarter approach.
            
            # Given the structure, let's try to fix U and V and check if W exists.
            # But random U, V won't work. Let me try a different strategy.
            
            pass
        
        return None
    
    # Alternative approach: use the known rank-28 Strassen decomposition and try to merge
    # two of the rank-1 terms to get rank 27.
    
    # First, build the rank-28 Strassen decomposition
    U28 = np.zeros((n * m, 28), dtype=np.float64)
    V28 = np.zeros((m * p, 28), dtype=np.float64)
    W28 = np.zeros((p * n, 28), dtype=np.float64)
    
    def add_strassen_block(offset, a_row_base, a_col_base, b_row_base, b_col_base, c_row_base, c_col_base):
        a00_u = a_row_base * m + a_col_base
        a01_u = a_row_base * m + a_col_base + 1
        a10_u = (a_row_base + 1) * m + a_col_base
        a11_u = (a_row_base + 1) * m + a_col_base + 1
        
        b00_v = b_row_base * p + b_col_base
        b01_v = b_row_base * p + b_col_base + 1
        b10_v = (b_row_base + 1) * p + b_col_base
        b11_v = (b_row_base + 1) * p + b_col_base + 1
        
        c00_w = c_col_base * n + c_row_base
        c01_w = (c_col_base + 1) * n + c_row_base
        c10_w = c_col_base * n + (c_row_base + 1)
        c11_w = (c_col_base + 1) * n + (c_row_base + 1)
        
        # M1: (A00+A11)(B00+B11) -> C00, C11
        U28[a00_u, offset + 0] += 1
        U28[a11_u, offset + 0] += 1
        V28[b00_v, offset + 0] += 1
        V28[b11_v, offset + 0] += 1
        W28[c00_w, offset + 0] += 1
        W28[c11_w, offset + 0] += 1
        
        # M2: (A10+A11)B00 -> C10, -C11
        U28[a10_u, offset + 1] += 1
        U28[a11_u, offset + 1] += 1
        V28[b00_v, offset + 1] += 1
        W28[c10_w, offset + 1] += 1
        W28[c11_w, offset + 1] -= 1
        
        # M3: A00(B01-B11) -> C01, C11
        U28[a00_u, offset + 2] += 1
        V28[b01_v, offset + 2] += 1
        V28[b11_v, offset + 2] -= 1
        W28[c01_w, offset + 2] += 1
        W28[c11_w, offset + 2] += 1
        
        # M4: A11(B10-B00) -> C00, C10
        U28[a11_u, offset + 3] += 1
        V28[b10_v, offset + 3] += 1
        V28[b00_v, offset + 3] -= 1
        W28[c00_w, offset + 3] += 1
        W28[c10_w, offset + 3] += 1
        
        # M5: (A00+A01)B11 -> -C00, C01
        U28[a00_u, offset + 4] += 1
        U28[a01_u, offset + 4] += 1
        V28[b11_v, offset + 4] += 1
        W28[c00_w, offset + 4] -= 1
        W28[c01_w, offset + 4] += 1
        
        # M6: (A10-A00)(B00+B01) -> C11
        U28[a10_u, offset + 5] += 1
        U28[a00_u, offset + 5] -= 1
        V28[b00_v, offset + 5] += 1
        V28[b01_v, offset + 5] += 1
        W28[c11_w, offset + 5] += 1
        
        # M7: (A01-A11)(B10+B11) -> C00
        U28[a01_u, offset + 6] += 1
        U28[a11_u, offset + 6] -= 1
        V28[b10_v, offset + 6] += 1
        V28[b11_v, offset + 6] += 1
        W28[c00_w, offset + 6] += 1
    
    add_strassen_block(0, 0, 0, 0, 0, 0, 0)
    add_strassen_block(7, 0, 2, 2, 0, 0, 0)
    add_strassen_block(14, 0, 0, 0, 2, 0, 2)
    add_strassen_block(21, 0, 2, 2, 2, 0, 2)
    
    # Verify the rank-28 decomposition
    T_recon = np.zeros_like(T)
    for k in range(28):
        T_recon += np.einsum('i,j->ij', U28[:, k], V28[:, k])[:, :, np.newaxis] * W28[:, k][np.newaxis, np.newaxis, :]
    # Actually need to be more careful with the einsum
    T_recon = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for k in range(28):
        outer = np.einsum('i,j->ij', U28[:, k], V28[:, k])
        T_recon += outer[:, :, np.newaxis] * W28[:, k][np.newaxis, np.newaxis, :]
    
    if not np.allclose(T, T_recon, atol=1e-12):
        # Fallback: return the Strassen rank-28
        return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
    
    # Now try to reduce to rank 27 by finding a linear dependency among the 28 terms
    # If we can find coefficients c_k such that sum_k c_k * (U_k ⊗ V_k ⊗ W_k) = 0,
    # then we can eliminate one term.
    
    # Check if the 28 rank-1 tensors are linearly independent
    # Flatten each rank-1 tensor to a vector of size n*m*m*p = 8*16*8 = 1024
    vecs = np.zeros((28, n*m*m*p))
    for k in range(28):
        outer = np.einsum('i,j->ij', U28[:, k], V28[:, k])
        vecs[k] = (outer[:, :, np.newaxis] * W28[:, k][np.newaxis, np.newaxis, :]).flatten()
    
    # Check rank of the 28x1024 matrix
    rank_mat = np.linalg.matrix_rank(vecs, tol=1e-10)
    
    if rank_mat < 28:
        # There's a linear dependency! Find null space vector
        # SVD of vecs^T (1024 x 28)
        U_svd, s_vals, Vt_svd = np.linalg.svd(vecs.T, full_matrices=False)
        # Null space vectors are rows of Vt_svd corresponding to zero singular values
        null_dim = 28 - rank_mat
        if null_dim > 0:
            # Get the null space basis
            null_vecs = Vt_svd[rank_mat:, :]  # shape (null_dim, 28)
            # Pick a null vector with largest coefficient for some index to eliminate
            for nv in null_vecs:
                # Try eliminating the term with the largest |coefficient|
                max_idx = np.argmax(np.abs(nv))
                c = nv / nv[max_idx]  # normalize so c[max_idx] = 1
                # New decomposition: for k != max_idx, U_new[:,k] = U[:,k] - c[k]*U[:,max_idx], etc.
                # Actually: sum_k c_k T_k = 0 => T_max = -sum_{k!=max} (c_k/c_max) T_k
                # So we can express T_max in terms of others, giving rank 27.
                # New factors: for k != max_idx, same; remove max_idx
                U27 = np.delete(U28, max_idx, axis=1)
                V27 = np.delete(V28, max_idx, axis=1)
                W27 = np.delete(W28, max_idx, axis=1)
                
                # Adjust the remaining factors
                for k in range(27):
                    if k < max_idx:
                        orig_k = k
                    else:
                        orig_k = k + 1
                    U27[:, k] += (-c[orig_k]) * U28[:, max_idx]
                    V27[:, k] += (-c[orig_k]) * V28[:, max_idx]
                    W27[:, k] += (-c[orig_k]) * W28[:, max_idx]
                
                # Verify
                T_recon27 = np.zeros((n*m, m*p, p*n), dtype=np.float64)
                for k in range(27):
                    outer = np.einsum('i,j->ij', U27[:, k], V27[:, k])
                    T_recon27 += outer[:, :, np.newaxis] * W27[:, k][np.newaxis, np.newaxis, :]
                
                if np.allclose(T, T_recon27, atol=1e-10):
                    # Check if entries are integer or half-integer
                    if np.allclose(U27 * 2, np.round(U27 * 2), atol=1e-8) and \
                       np.allclose(V27 * 2, np.round(V27 * 2), atol=1e-8) and \
                       np.allclose(W27 * 2, np.round(W27 * 2), atol=1e-8):
                        U27 = np.round(U27 * 2) / 2
                        V27 = np.round(V27 * 2) / 2
                        W27 = np.round(W27 * 2) / 2
                        return {"U": U27.tolist(), "V": V27.tolist(), "W": W27.tolist()}
    
    # Try random search for rank 27
    rng = np.random.RandomState(12345)
    
    def try_random_rank27(num_trials=500):
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            
            # Random integer/half-integer initialization
            scale = rng.randint(1, 4)
            U_rand = (rng.randint(0, 2, size=(n*m, 27)) - 0.5) * scale
            V_rand = (rng.randint(0, 2, size=(m*p, 27)) - 0.5) * scale
            W_rand = (rng.randint(0, 2, size=(p*n, 27)) - 0.5) * scale
            
            # Check if this decomposes T
            T_recon = np.zeros((n*m, m*p, p*n), dtype=np.float64)
            for k in range(27):
                outer = np.einsum('i,j->ij', U_rand[:, k], V_rand[:, k])
                T_recon += outer[:, :, np.newaxis] * W_rand[:, k][np.newaxis, np.newaxis, :]
            
            if np.allclose(T, T_recon, atol=1e-10):
                return {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_rand.tolist()}
        
        return None
    
    result = try_random_rank27()
    if result is not None:
        return result
    
    # Fallback to rank 28
    return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}

# EVOLVE-BLOCK-END

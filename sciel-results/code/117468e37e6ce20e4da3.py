"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses structured block decomposition combining Strassen for 2x2x2 blocks
and direct terms, targeting rank < 68. Also attempts randomized ALS
to find lower-rank exact decompositions."""

DIMENSIONS = [3, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    deadline = 140  # seconds
    
    def check_time():
        return (time.time() - start_time) < deadline
    
    def strassen_decomp(row_off_a, col_off_a, row_off_b, col_off_b, U_t, V_t, W_t):
        """Add Strassen decomposition for 2x2*2x2 block at given offsets."""
        a = (row_off_a, col_off_a)
        b = (row_off_a, col_off_a + 1)
        c = (row_off_a + 1, col_off_a)
        d = (row_off_a + 1, col_off_a + 1)
        e = (row_off_b, col_off_b)
        f = (row_off_b, col_off_b + 1)
        g = (row_off_b + 1, col_off_b)
        h = (row_off_b + 1, col_off_b + 1)
        
        C00 = (row_off_a, col_off_b)
        C01 = (row_off_a, col_off_b + 1)
        C10 = (row_off_a + 1, col_off_b)
        C11 = (row_off_a + 1, col_off_b + 1)
        
        # m1 = (a+d)(e+h) -> C00 += 1, C11 += 1
        U_t.append([(a, 1), (d, 1)])
        V_t.append([(e, 1), (h, 1)])
        W_t.append([(C00, 1), (C11, 1)])
        
        # m2 = (c+d)*e -> C10 += 1, C11 -= 1
        U_t.append([(c, 1), (d, 1)])
        V_t.append([(e, 1)])
        W_t.append([(C10, 1), (C11, -1)])
        
        # m3 = a*(f-h) -> C01 += 1, C11 += 1
        U_t.append([(a, 1)])
        V_t.append([(f, 1), (h, -1)])
        W_t.append([(C01, 1), (C11, 1)])
        
        # m4 = d*(g-e) -> C00 += 1, C10 += 1
        U_t.append([(d, 1)])
        V_t.append([(g, 1), (e, -1)])
        W_t.append([(C00, 1), (C10, 1)])
        
        # m5 = (a+b)*h -> C00 -= 1, C01 += 1
        U_t.append([(a, 1), (b, 1)])
        V_t.append([(h, 1)])
        W_t.append([(C00, -1), (C01, 1)])
        
        # m6 = (c-a)*(e+f) -> C11 += 1
        U_t.append([(c, 1), (a, -1)])
        V_t.append([(e, 1), (f, 1)])
        W_t.append([(C11, 1)])
        
        # m7 = (b-d)*(g+h) -> C00 += 1
        U_t.append([(b, 1), (d, -1)])
        V_t.append([(g, 1), (h, 1)])
        W_t.append([(C00, 1)])
    
    def direct_term(i, j, k, U_t, V_t, W_t):
        """Add a direct term A[i,j]*B[j,k] to C[i,k]."""
        U_t.append([((i, j), 1)])
        V_t.append([((j, k), 1)])
        W_t.append([((i, k), 1)])
    
    def build_matrices(U_t, V_t, W_t, rank):
        """Build dense matrices from term lists."""
        nm = n * m
        mp = m * p
        pn = p * n
        
        U_mat = np.zeros((nm, rank))
        V_mat = np.zeros((mp, rank))
        W_mat = np.zeros((pn, rank))
        
        for idx in range(rank):
            for (i, j), coeff in U_t[idx]:
                U_mat[i * m + j, idx] = coeff
            for (j, k), coeff in V_t[idx]:
                V_mat[j * p + k, idx] = coeff
            for (i, k), coeff in W_t[idx]:
                W_mat[k * n + i, idx] = coeff
        
        return U_mat, V_mat, W_mat
    
    def verify_decomp(Ua, Va, Wa):
        """Verify that the decomposition is exact using vectorized ops."""
        # For each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
        # Build the tensor product efficiently
        # U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
        # For each r, the outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r] contributes to T
        # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        
        r = Ua.shape[1]
        # Vectorized check: for each r, compute the contribution
        # T_recon = sum_r outer(U[:,r], V[:,r], W[:,r]) reshaped
        # More efficiently: for each (i,j), T[i,j,:] = sum_r U[i*m+j,r] * (V[j*p:(j+1)*p, r] * W[:, r] reshaped)
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                # U[i*m+j, :] has shape (r,)
                # V[j*p:(j+1)*p, :] has shape (p, r)
                # W[:, :] has shape (p*n, r), we need W[k*n+i, :] for k in range(p)
                # W[k*n+i, :] for k=0..p-1: indices i, n+i, 2n+i, ...
                w_idx = np.arange(p) * n + i  # shape (p,)
                W_sel = Wa[w_idx, :]  # shape (p, r)
                
                # Contribution to T[i,j,k] for each k:
                # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                # = sum_r U[i*m+j,r] * (V[j*p:(j+1)*p, r] * W_sel) [elementwise, then sum over r]
                # V_sel has shape (p, r), W_sel has shape (p, r)
                V_sel = Va[j*p:(j+1)*p, :]  # shape (p, r)
                
                # Elementwise product: (p, r) * (p, r) = (p, r)
                prod = V_sel * W_sel  # shape (p, r)
                # Multiply by U[i*m+j, r] (broadcast over p):
                contrib = prod * Ua[i*m+j, :]  # shape (p, r)
                # Sum over r:
                t_val = contrib.sum(axis=1)  # shape (p,)
                
                err = np.max(np.abs(t_val - 1.0))
                if err > max_err:
                    max_err = err
        
        return max_err < 1e-9, max_err
    
    def build_schoolbook():
        """Build the schoolbook (rank 72) decomposition as fallback."""
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col] = 1
                    V_fb[j * p + k][col] = 1
                    W_fb[k * n + i][col] = 1
                    col += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    def to_nice_list(arr):
        """Convert numpy array to list of integer/half-integer values."""
        result = []
        for row in arr:
            nice_row = []
            for val in row:
                val2 = val * 2
                if abs(val2 - round(val2)) < 1e-6:
                    nice_val = round(val2) / 2.0
                    if abs(nice_val - round(nice_val)) < 1e-9:
                        nice_val = int(round(nice_val))
                    nice_row.append(nice_val)
                else:
                    nice_row.append(val)
            result.append(nice_row)
        return result
    
    # Build the base decomposition: Strassen for 2x2 blocks + direct terms
    U_t = []
    V_t = []
    W_t = []
    
    # Use Strassen for the 2x2*2x2 blocks
    # A[0:2, 0:2] * B[0:2, k:k+2] for k in {0, 2, 4, 6}
    for k0 in [0, 2, 4, 6]:
        if not check_time():
            break
        strassen_decomp(0, 0, 0, k0, U_t, V_t, W_t)
    
    # Direct terms for A[0:2, 2] * B[2, :]
    for k0 in [0, 2, 4, 6]:
        for i in range(2):
            for k in range(k0, k0 + 2):
                if not check_time():
                    break
                direct_term(i, 2, k, U_t, V_t, W_t)
    
    # Direct terms for A[2, 0:2] * B[0:2, :]
    for k0 in [0, 2, 4, 6]:
        for j in range(2):
            for k in range(k0, k0 + 2):
                if not check_time():
                    break
                direct_term(2, j, k, U_t, V_t, W_t)
    
    # Direct terms for A[2, 2] * B[2, :]
    for k0 in [0, 2, 4, 6]:
        for k in range(k0, k0 + 2):
            if not check_time():
                break
            direct_term(2, 2, k, U_t, V_t, W_t)
    
    rank = len(W_t)
    U_mat, V_mat, W_mat = build_matrices(U_t, V_t, W_t, rank)
    
    # Verify
    valid, max_err = verify_decomp(U_mat, V_mat, W_mat)
    if not valid:
        return build_schoolbook()
    
    best_result = {"U": to_nice_list(U_mat), "V": to_nice_list(V_mat), "W": to_nice_list(W_mat)}
    best_rank = rank
    
    # Now try to reduce rank by finding linear dependencies
    # The tensor T[i,j,k] = 1 has a specific structure.
    # We can try to find if any column of U,V,W is a linear combination of others
    # such that the tensor is preserved.
    
    # Approach: Try to find a rank-r sub-decomposition by solving a linear system.
    # For a given set of r terms, we need:
    #   sum_{s=0}^{r-1} U_s[i*m+j] * V_s[j*p+k] * W_s[k*n+i] = 1  for all (i,j,k)
    
    # This is a system of 72 equations in the entries of U_s, V_s, W_s.
    # But it's nonlinear. Instead, let's try a structured approach.
    
    # Key observation: The 8 columns of B can be paired. For each pair (2k, 2k+1),
    # we use Strassen on the 2x2 A block. The direct terms handle the third row/col.
    
    # Let's try to see if we can share some structure between the 4 blocks.
    # The A matrix is the same for all 4 blocks. The B matrix differs.
    
    # Alternative: Try to find a lower-rank decomposition using the fact that
    # the tensor is "all ones" in a structured way.
    
    # Let's try a more aggressive ALS with vectorized operations:
    def als_decompose_vectorized(target, n_dim, m_dim, p_dim, rank, max_iter=200, tol=1e-10, seed=42):
        """
        Vectorized ALS for tensor decomposition.
        target: 3D array of shape (n, m, p)
        Returns (U, V, W) or None if failed.
        """
        rng = np.random.RandomState(seed)
        
        # Initialize
        U = rng.randn(n_dim * m_dim, rank) * 0.5
        V = rng.randn(m_dim * p_dim, rank) * 0.5
        W = rng.randn(p_dim * n_dim, rank) * 0.5
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iter):
            # Update U
            for r in range(rank):
                for i in range(n_dim):
                    for j in range(m_dim):
                        # num = sum_k target[i,j,k] * V[j*p+k, r] * W[k*n+i, r]
                        # den = sum_k V[j*p+k, r]^2 * W[k*n+i, r]^2
                        V_sel = V[j * p_dim:(j+1) * p_dim, r]  # shape (p,)
                        W_idx = np.arange(p_dim) * n_dim + i
                        W_sel = W[W_idx, r]  # shape (p,)
                        
                        num = np.dot(target[i, j, :], V_sel * W_sel)
                        den = np.dot(V_sel * V_sel, W_sel * W_sel)
                        if abs(den) > 1e-15:
                            U[i * m_dim + j, r] = num / den
                        else:
                            U[i * m_dim + j, r] = 0.0
            
            # Update V
            for r in range(rank):
                for j in range(m_dim):
                    for k in range(p_dim):
                        # num = sum_i target[i,j,k] * U[i*m+j, r] * W[k*n+i, r]
                        # den = sum_i U[i*m+j, r]^2 * W[k*n+i, r]^2
                        U_sel = U[np.arange(n_dim) * m_dim + j, r]  # shape (n,)
                        W_sel = W[k * n_dim + np.arange(n_dim), r]  # shape (n,)
                        
                        num = np.dot(target[:, j, k], U_sel * W_sel)
                        den = np.dot(U_sel * U_sel, W_sel * W_sel)
                        if abs(den) > 1e-15:
                            V[j * p_dim + k, r] = num / den
                        else:
                            V[j * p_dim + k, r] = 0.0
            
            # Update W
            for r in range(rank):
                for k in range(p_dim):
                    for i in range(n_dim):
                        # num = sum_j target[i,j,k] * U[i*m+j, r] * V[j*p+k, r]
                        # den = sum_j U[i*m+j, r]^2 * V[j*p+k, r]^2
                        U_sel = U[i * m_dim * n_dim + np.arange(m_dim) * 0 + np.arange(m_dim), r]  # wrong
                        # U[i*m+j, r] for j in range(m): indices i*m+0, i*m+1, ..., i*m+m-1
                        U_sel = U[i * m_dim:(i+1) * m_dim, r]  # shape (m,)
                        V_sel = V[np.arange(m_dim) * p_dim + k, r]  # shape (m,)
                        
                        num = np.dot(target[i, :, k], U_sel * V_sel)
                        den = np.dot(U_sel * U_sel, V_sel * V_sel)
                        if abs(den) > 1e-15:
                            W[k * n_dim + i, r] = num / den
                        else:
                            W[k * n_dim + i, r] = 0.0
            
            # Check convergence (every few iterations to save time)
            if it % 5 == 0 or it == max_iter - 1:
                # Vectorized error computation
                max_err = 0.0
                for i in range(n_dim):
                    for j in range(m_dim):
                        V_sel = V[j * p_dim:(j+1) * p_dim, :]  # (p, r)
                        W_idx = np.arange(p_dim) * n_dim + i
                        W_sel = W[W_idx, :]  # (p, r)
                        prod = V_sel * W_sel  # (p, r)
                        contrib = prod * U[i*m_dim+j, :]  # (p, r)
                        t_val = contrib.sum(axis=1)  # (p,)
                        err = np.max(np.abs(t_val - 1.0))
                        if err > max_err:
                            max_err = err
                
                if max_err < best_err:
                    best_err = max_err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if max_err < tol:
                    return best_U, best_V, best_W, max_err
        
        return best_U, best_V, best_W, best_err
    
    # Try ALS from various starting ranks
    if check_time():
        # Try to improve from current rank
        for r in range(best_rank - 1, max(35, best_rank - 15), -1):
            if not check_time():
                break
            
            found = False
            for restart in range(2):
                if not check_time():
                    break
                
                U_als, V_als, W_als, err = als_decompose_vectorized(
                    np.ones((n, m, p)), n, m, p, r,
                    max_iter=100, tol=1e-10,
                    seed=42 + restart * 1000
                )
                
                if U_als is not None and err < 1e-9:
                    if verify_decomp(U_als, V_als, W_als)[0]:
                        best_result = {
                            "U": to_nice_list(U_als),
                            "V": to_nice_list(V_als),
                            "W": to_nice_list(W_als)
                        }
                        best_rank = r
                        found = True
                        break
            
            if found:
                continue
            else:
                if r <= 50:
                    break
    
    return best_result
# EVOLVE-BLOCK-END

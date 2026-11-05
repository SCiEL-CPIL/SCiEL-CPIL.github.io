import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # The target tensor T has entries T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Wait, the matrix multiplication tensor M_{n,m,p} is defined such that:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor is T = sum_{i,j,k} E_{i,j} \otimes E_{j,k} \otimes E_{k,i}? 
    # No, the standard tensor for matrix multiplication M_{n,m,p} is in V^* \otimes W^* \otimes X^*
    # T = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p e_i \otimes f_j \otimes g_k
    # where e_i is the i-th row of A, f_j is the j-th column of B, g_k is the k-th column of C?
    # Let's check the interface:
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # The decomposition is: sum_r U[:,r] V[:,r] W[:,r] = T
    # The entry of T at (i*m+j, j*p+k, k*n+i) should be 1?
    # Let's verify the schoolbook:
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    # Then sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 if there is a term for (i,j,k).
    # In schoolbook, there is exactly one term for each (i,j,k).
    # So the target tensor T is 1 at indices (i*m+j, j*p+k, k*n+i) and 0 elsewhere?
    # No, the tensor is a 3-way array of size (n*m) x (m*p) x (p*n).
    # The entry T[a, b, c] is non-zero only if a, b, c are compatible indices.
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    # So T is a sparse tensor with 1s at these specific indices.

    def create_target_tensor(n, m, p):
        # We don't need to store the full dense tensor if it's large, but for 3x4x8 it's small.
        # Size: 12 x 32 x 24 = 9216 entries.
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    T[a, b, c] = 1.0
        return T

    T = create_target_tensor(n, m, p)
    
    # Strategy: Try to find a rank R decomposition.
    # Start with R = 87 (one less than the known 88).
    # If we can drive the error to 0, we have a rank 87 decomposition.
    
    def try_rank(R, max_iters=200, tol=1e-12):
        if time.time() > deadline:
            return None, np.inf
            
        # Initialize U, V, W with small random values
        # Using a small initialization to avoid saturation
        U = np.random.randn(n*m, R) * 0.1
        V = np.random.randn(m*p, R) * 0.1
        W = np.random.randn(p*n, R) * 0.1
        
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Precompute indices for the target tensor
        # We need to solve for U, V, W such that sum_r U[:,r] V[:,r] W[:,r] approx T
        # This is a non-convex problem. ALS is a standard heuristic.
        # For each r, we update U[:,r] by fixing V and W, etc.
        # However, standard ALS for CP decomposition minimizes ||T - sum U_r V_r W_r||_F.
        # Here T is sparse.
        
        # Efficient ALS update:
        # To update U[:, r], we fix V[:, r'] and W[:, r'] for all r'.
        # The term for U[:, r] in the sum is U[:, r] \otimes V[:, r] \otimes W[:, r].
        # We want to minimize || T - (U[:, r] \otimes V[:, r] \otimes W[:, r]) - Sum_{r' != r} ... ||
        # Let Res = T - Sum_{r' != r} U[:, r'] \otimes V[:, r'] \otimes W[:, r']
        # We want to find U[:, r] to minimize || Res - U[:, r] \otimes V[:, r] \otimes W[:, r] ||_F
        # This is a linear least squares problem if we flatten.
        # Let v_r = V[:, r] (size m*p), w_r = W[:, r] (size p*n)
        # Let M = v_r \otimes w_r (size (m*p)*(p*n))
        # We want to minimize || Res_flat - M \otimes u_r^T ||_F? 
        # Actually, the outer product u \otimes v \otimes w flattened is (u \otimes v) \otimes w.
        # Let's reshape Res into a matrix of size (n*m) x ((m*p)*(p*n))
        # Res_mat[i, (b, c)] = Res[i, b, c]
        # The model term is u[i] * v[b] * w[c]
        # This can be written as u[i] * (v \otimes w)[(b, c)]
        # So we want to minimize || Res_mat - u \otimes (v \otimes w) ||_F
        # This is a rank-1 approximation of Res_mat.
        # The optimal u is given by the projection of Res_mat onto the vector (v \otimes w).
        # u = (Res_mat @ (v \otimes w)) / ||v \otimes w||^2
        
        # Precompute v \otimes w for each r?
        # v is size 32, w is size 24. v \otimes w is size 768.
        # Res_mat is 12 x 768.
        
        # We need to maintain the sum of other terms efficiently.
        # Sum_other[i, b, c] = Sum_{r' != r} U[i, r'] V[b, r'] W[c, r']
        
        # Let's implement a simple ALS loop.
        
        # To speed up, we can use the fact that T is sparse.
        # But 9216 entries is small, so dense operations are fine.
        
        # Precompute v_w_r = V[:, r] \otimes W[:, r] for all r
        # This is a matrix of size R x (m*p * p*n)
        # v_w = np.einsum('ir,jr->rij', V, W).reshape(R, -1)
        
        # We will iterate over r, updating U[:, r], V[:, r], W[:, r].
        
        # To avoid recalculating the sum of all other terms from scratch,
        # we can maintain the full reconstructed tensor S = Sum U_r V_r W_r.
        # When updating U[:, r], we compute S_new = S - U[:, r] \otimes V[:, r] \otimes W[:, r]
        # Then we solve for new U[:, r] to best match T - S_new.
        # Then S = S_new + new_U[:, r] \otimes V[:, r] \otimes W[:, r].
        
        S = np.zeros_like(T)
        
        # Initialize S
        for r in range(R):
            S += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            
        err = np.linalg.norm(T - S)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            # Update U
            for r in range(R):
                # Remove current term
                S -= np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                
                # Residual
                Res = T - S
                
                # Solve for U[:, r]
                # Minimize || Res - u \otimes v \otimes w ||
                # v = V[:, r], w = W[:, r]
                # v_w = v \otimes w
                v_w = np.outer(V[:, r], W[:, r]).flatten()
                
                # Res_mat is Res reshaped to (n*m, -1)
                Res_mat = Res.reshape(n*m, -1)
                
                # u = Res_mat @ v_w / (v_w @ v_w)
                denom = np.dot(v_w, v_w)
                if denom < 1e-12:
                    continue
                u_new = Res_mat @ v_w / denom
                
                # Add new term
                U[:, r] = u_new
                S += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                
            # Update V
            for r in range(R):
                S -= np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                Res = T - S
                # Minimize || Res - u \otimes v \otimes w ||
                # u = U[:, r], w = W[:, r]
                # u_w = u \otimes w? No, we need to group indices.
                # Res[j, (i, k)] = Res[i, j, k]
                # Model: u[i] v[j] w[k]
                # For fixed u, w, we want to find v.
                # Res_mat_v[j, (i, k)] = Res[i, j, k]
                # We need to reshape Res to (m*p, n*m * p*n)
                # Indices: j is row, (i, k) is col.
                # Res_reshaped = Res.transpose(1, 0, 2).reshape(m*p, -1)
                Res_reshaped = Res.transpose(1, 0, 2).reshape(m*p, -1)
                
                # u_w_flat: u[i] * w[k]
                # We need a vector of size n*m * p*n
                # Index (i, k) corresponds to i*(p*n) + k?
                # Let's check the flattening order.
                # Res.transpose(1, 0, 2) has shape (m*p, n, p*n)
                # Reshape to (m*p, n * p*n)
                # The column index is i * (p*n) + k.
                # So the vector u_w_flat has entry at i*(p*n)+k equal to u[i]*w[k].
                u_w_flat = np.outer(U[:, r], W[:, r]).flatten()
                
                denom = np.dot(u_w_flat, u_w_flat)
                if denom < 1e-12:
                    continue
                v_new = Res_reshaped @ u_w_flat / denom
                
                V[:, r] = v_new
                S += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                
            # Update W
            for r in range(R):
                S -= np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                Res = T - S
                # Minimize || Res - u \otimes v \otimes w ||
                # u = U[:, r], v = V[:, r]
                # Res[k, (i, j)] = Res[i, j, k]
                # Res_reshaped = Res.transpose(2, 0, 1).reshape(p*n, -1)
                # Column index is i * m + j?
                # Res.transpose(2, 0, 1) has shape (p*n, n, m)
                # Reshape to (p*n, n*m)
                # Column index is i * m + j.
                Res_reshaped = Res.transpose(2, 0, 1).reshape(p*n, -1)
                
                u_v_flat = np.outer(U[:, r], V[:, r]).flatten()
                
                denom = np.dot(u_v_flat, u_v_flat)
                if denom < 1e-12:
                    continue
                w_new = Res_reshaped @ u_v_flat / denom
                
                W[:, r] = w_new
                S += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                
            err = np.linalg.norm(T - S)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if best_err < tol:
                break
                
        return best_U, best_V, best_W, best_err

    # Try rank 87
    # We can also try to verify the rank 88 solution first to be safe, but we know it works.
    # The goal is to beat 88, so we try 87.
    
    # To increase the chance of success, we can run multiple random restarts.
    # But ALS is sensitive to initialization.
    # A better initialization might be to take the rank 88 solution and remove one column,
    # then re-optimize.
    
    # Let's generate the rank 88 solution first.
    def build_strassen_88():
        rank = 88
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        strassen_terms = [
            ([1,  0,  0,  1], [1,  0,0,  1], [1,  0,0,  1]),
            ([0,  0,  1,  1], [1,  0,0,  0], [0,  0,1, -1]),
            ([1,  1,  0,  0], [0,  0,1,  1], [0,  1,0,  1]),
            ([0,  0,  0,  1], [0, -1,0,  1], [1,  0,-1,  0]),
            ([1,  0,  0,  0], [1,  0,-1,  0], [-1,1,0,  0]),
            ([-1, 0,  1,  0], [0,  1,0,  1], [0,  0,0,  1]),
            ([0,  1,  0, -1], [1,  0,1,  0], [1,  0,0,  0]),
        ]
        
        col = 0
        
        def add_strassen_block(i0, j0, k0):
            nonlocal col
            for t in range(7):
                a_coeffs = strassen_terms[t][0]
                b_coeffs = strassen_terms[t][1]
                c_coeffs = strassen_terms[t][2]
                
                for di in range(2):
                    for dj in range(2):
                        U[(i0+di)*m + (j0+dj), col] = a_coeffs[di*2+dj]
                
                for dj in range(2):
                    for dk in range(2):
                        V[(j0+dj)*p + (k0+dk), col] = b_coeffs[dj*2+dk]
                        
                for dk in range(2):
                    for di in range(2):
                        W[(k0+dk)*n + (i0+di), col] = c_coeffs[di*2+dk]
                        
                col += 1

        def add_schoolbook_block(i0, j0, k0, n_rows=1, m_cols=2, p_cols=2):
            nonlocal col
            for di in range(n_rows):
                for dj in range(m_cols):
                    for dk in range(p_cols):
                        U[(i0+di)*m + (j0+dj), col] = 1
                        V[(j0+dj)*p + (k0+dk), col] = 1
                        W[(k0+dk)*n + (i0+di), col] = 1
                        col += 1

        for j0 in [0, 2]:
            for k0 in [0, 2, 4, 6]:
                add_strassen_block(0, j0, k0)
                add_schoolbook_block(2, j0, k0, n_rows=1, m_cols=2, p_cols=2)
                
        return U, V, W

    U_88, V_88, W_88 = build_strassen_88()
    
    # Verify 88
    def verify(U, V, W):
        S = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(S, T, atol=1e-9)
        
    if not verify(U_88, V_88, W_88):
        # Fallback to schoolbook if something went wrong
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Now try to reduce to 87 using ALS, initialized from the 88 solution.
    # We take the 88 solution, drop one column (e.g., the last one), and optimize the remaining 87.
    # This is a "warm start" which is much better than random init.
    
    U_init = U_88[:, :87]
    V_init = V_88[:, :87]
    W_init = W_88[:, :87]
    
    # We can also try dropping different columns.
    # Let's try dropping the last column first.
    
    # Modify try_rank to accept initial values
    def try_rank_warm(R, U_init, V_init, W_init, max_iters=100, tol=1e-12):
        if time.time() > deadline:
            return None, np.inf
            
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        S = np.einsum('ir,jr,kr->ijk', U, V, W)
        err = np.linalg.norm(T - S)
        best_err = err
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            # Update U
            for r in range(R):
                S -= np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                Res = T - S
                v_w = np.outer(V[:, r], W[:, r]).flatten()
                Res_mat = Res.reshape(n*m, -1)
                denom = np.dot(v_w, v_w)
                if denom < 1e-12:
                    continue
                u_new = Res_mat @ v_w / denom
                U[:, r] = u_new
                S += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                
            # Update V
            for r in range(R):
                S -= np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                Res = T - S
                Res_reshaped = Res.transpose(1, 0, 2).reshape(m*p, -1)
                u_w_flat = np.outer(U[:, r], W[:, r]).flatten()
                denom = np.dot(u_w_flat, u_w_flat)
                if denom < 1e-12:
                    continue
                v_new = Res_reshaped @ u_w_flat / denom
                V[:, r] = v_new
                S += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                
            # Update W
            for r in range(R):
                S -= np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                Res = T - S
                Res_reshaped = Res.transpose(2, 0, 1).reshape(p*n, -1)
                u_v_flat = np.outer(U[:, r], V[:, r]).flatten()
                denom = np.dot(u_v_flat, u_v_flat)
                if denom < 1e-12:
                    continue
                w_new = Res_reshaped @ u_v_flat / denom
                W[:, r] = w_new
                S += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
                
            err = np.linalg.norm(T - S)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if best_err < tol:
                break
                
        return best_U, best_V, best_W, best_err

    U_87, V_87, W_87, err_87 = try_rank_warm(87, U_init, V_init, W_init)
    
    if U_87 is not None and err_87 < 1e-9:
        # We found a rank 87 decomposition!
        return {"U": U_87.tolist(), "V": V_87.tolist(), "W": W_87.tolist()}
    else:
        # If we didn't find a rank 87, return the rank 88 solution.
        return {"U": U_88.tolist(), "V": V_88.tolist(), "W": W_88.tolist()}

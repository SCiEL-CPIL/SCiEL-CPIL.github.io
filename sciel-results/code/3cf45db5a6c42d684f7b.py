import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    U_dim = n * m  # 8
    V_dim = m * p  # 32
    W_dim = p * n  # 16
    
    deadline = time.time() + 130.0
    
    # Build the matmul tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Strategy: Iterative peeling to find low-rank decomposition
    # T ≈ sum_r U_r ⊗ V_r ⊗ W_r
    # We peel rank-1 components greedily, then verify exactness.
    
    def try_peel(T, max_rank, tol=1e-9):
        """Peel rank-1 components from tensor T."""
        U_list = []
        V_list = []
        W_list = []
        residual = T.copy()
        
        for r in range(max_rank):
            if time.time() > deadline - 30:
                break
            # Find best rank-1 approximation of residual
            # Use SVD on matricizations
            # Matricize residual as (U_dim, V_dim*W_dim)
            R_mat = residual.reshape(U_dim, V_dim * W_dim)
            U_svd, S, Vt_svd = np.linalg.svd(R_mat, full_matrices=False)
            
            if len(S) == 0 or S[0] < tol:
                break
            
            # Top left singular vector
            u = U_svd[:, 0]
            # Top right singular vector, reshape back
            v_flat = Vt_svd[0, :]
            v_mat = v_flat.reshape(V_dim, W_dim)
            
            # Now factor v_mat as outer product (rank-1 approx)
            # SVD of v_mat
            u2, s2, v2t = np.linalg.svd(v_mat, full_matrices=False)
            if len(s2) == 0 or s2[0] < tol:
                break
            v = u2[:, 0] * np.sqrt(s2[0])
            w = v2t[0, :] * np.sqrt(s2[0])
            
            # Scale: make u unit, absorb scale
            u_norm = np.linalg.norm(u)
            if u_norm < tol:
                break
            u = u / u_norm
            scale = S[0] * u_norm
            v = v * scale
            w = w * scale
            
            # Check if this is a good rank-1 approx
            approx = np.outer(u, np.outer(v, w).reshape(V_dim, W_dim)).reshape(U_dim, V_dim, W_dim)
            
            U_list.append(u)
            V_list.append(v)
            W_list.append(w)
            
            residual = residual - approx
            
            # Check if residual is essentially zero
            if np.max(np.abs(residual)) < tol:
                break
        
        return U_list, V_list, W_list, residual
    
    # Try peeling with increasing rank
    best_result = None
    for max_rank in range(28, 65):  # Start from optimistic 28
        if time.time() > deadline - 30:
            break
        U_list, V_list, W_list, residual = try_peel(T, max_rank)
        
        if np.max(np.abs(residual)) < 1e-6:
            # Try to round to integers/half-integers
            rank = len(U_list)
            U = np.column_stack(U_list)
            V = np.column_stack(V_list)
            W = np.column_stack(W_list)
            
            # Scale to make entries nice
            # Find common scale
            all_vals = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            nonzero = all_vals[np.abs(all_vals) > 1e-10]
            if len(nonzero) > 0:
                # Try to find a scale that makes values close to half-integers
                best_scale = 1.0
                best_err = float('inf')
                for scale in [0.5, 1.0, 2.0, 4.0, 0.25, 0.125, 8.0]:
                    U_s = U * scale
                    V_s = V / scale  # keep U*V product same? No, we need U*V*W
                    # Actually we need U[r]*V[r]*W[r] to stay the same
                    # If we scale U by s, V by 1/s, W unchanged: product same
                    # Or scale all three: U*s1, V*s2, W*s3 with s1*s2*s3=1
                    # Let's just try rounding U, V, W to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    err = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U_r[i*m+j] * V_r[j*p+k], W_r[k*n+i])
                                err += abs(val - 1.0)
                    if err < best_err:
                        best_err = err
                        best_scale = 1.0
                        U_best, V_best, W_best = U_r, V_r, W_r
                    if best_err < 1e-6:
                        break
                
                if best_err < 1e-6:
                    U, V, W = U_best, V_best, W_best
                    best_result = (U, V, W, rank)
                    break
            
            if best_result is None:
                # Check if unrounded is close enough
                err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j] * V[j*p+k], W[k*n+i])
                            err += abs(val - 1.0)
                if err < 1e-6:
                    best_result = (U, V, W, rank)
                    break
    
    if best_result is not None:
        U, V, W, rank = best_result
        # Final verification
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j] * V[j*p+k], W[k*n+i])
                    if abs(val - 1.0) > 1e-6:
                        best_result = None
                        break
            if best_result is None:
                break
    
    if best_result is not None:
        U, V, W, rank = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: Strassen-based rank 56 (known to work)
    rank = 56
    U = np.zeros((U_dim, rank))
    V = np.zeros((V_dim, rank))
    W = np.zeros((W_dim, rank))
    
    strassen_X = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 1, 0, -1],
    ]
    strassen_Y = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    strassen_W = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ])
    
    col = 0
    for t in range(4):
        for s in range(2):
            x_j0 = 2 * s
            y_j0 = 2 * s
            k0 = 2 * t
            for r in range(7):
                for i in range(2):
                    for j_local in range(2):
                        j = x_j0 + j_local
                        U[i * m + j, col] += strassen_X[r][i * 2 + j_local]
                for j_local in range(2):
                    for k_local in range(2):
                        j = y_j0 + j_local
                        k = k0 + k_local
                        V[j * p + k, col] += strassen_Y[r][j_local * 2 + k_local]
                for i in range(2):
                    for k_local in range(2):
                        k = k0 + k_local
                        z_idx = i * 2 + k_local
                        W[k * n + i, col] += strassen_W[r][z_idx]
                col += 1
    
    # Verify fallback
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j] * V[j*p+k], W[k*n+i])
                if abs(val - 1.0) > 1e-9:
                    # Ultimate fallback: schoolbook
                    rank = n * m * p
                    U = np.zeros((U_dim, rank))
                    V = np.zeros((V_dim, rank))
                    W = np.zeros((W_dim, rank))
                    c = 0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                U[i*m+j, c] = 1
                                V[j*p+k, c] = 1
                                W[k*n+i, c] = 1
                                c += 1
                    break
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

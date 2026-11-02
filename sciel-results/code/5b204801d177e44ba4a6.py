import numpy as np
import time
import scipy.linalg as sla

def construct():
    n, m, p = 2, 4, 6
    t0 = time.time()
    deadline = t0 + 140.0
    
    # Dimensions
    dim_u = n * m  # 8
    dim_v = m * p  # 24
    dim_w = p * n  # 12
    
    # Construct the target tensor T
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) are consistent
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    T = np.zeros((dim_u, dim_v, dim_w))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    # Flattenings for rank lower bound
    # Mode-1 unfolding: (dim_u, dim_v * dim_w)
    T1 = T.reshape(dim_u, dim_v * dim_w)
    # Mode-2 unfolding: (dim_v, dim_u * dim_w)
    T2 = T.transpose(1, 0, 2).reshape(dim_v, dim_u * dim_w)
    # Mode-3 unfolding: (dim_w, dim_u * dim_v)
    T3 = T.transpose(2, 0, 1).reshape(dim_w, dim_u * dim_v)
    
    rank1 = np.linalg.matrix_rank(T1)
    rank2 = np.linalg.matrix_rank(T2)
    rank3 = np.linalg.matrix_rank(T3)
    
    R_min = max(rank1, rank2, rank3)
    
    # Upper bound is schoolbook rank
    R_max = n * m * p  # 48
    
    best_result = None
    best_rank = R_max + 1
    
    # Try ranks from R_min to R_max
    for R in range(R_min, R_max + 1):
        if time.time() > deadline - 10:
            break
            
        # Run ALS for this rank
        # Initialize factors
        np.random.seed(R) # Deterministic initialization based on rank
        U = np.random.randn(dim_u, R)
        V = np.random.randn(dim_v, R)
        W = np.random.randn(dim_w, R)
        
        # Normalize to avoid divergence
        for r in range(R):
            norm = np.linalg.norm(W[:, r])
            if norm > 1e-10:
                W[:, r] /= norm
                U[:, r] *= norm
                V[:, r] *= norm
            
        converged = False
        # Max iterations for ALS
        max_iter = 50
        tol = 1e-10
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            # Update U
            # T1 = U * (V ⊙ W)^T
            # Z_vw = V ⊙ W (Kronecker product), shape (dim_v*dim_w, R)
            Z_vw = np.kron(V, W) # (dim_v*dim_w, R)
            # U = T1 * Z_vw * (Z_vw^T Z_vw)^-1
            # Use least squares: T1 = U Z_vw^T => T1 Z_vw = U Z_vw^T Z_vw
            # U = T1 Z_vw (Z_vw^T Z_vw)^-1
            try:
                Z_vw_pinv = sla.pinv(Z_vw)
                U_new = T1 @ Z_vw @ Z_vw_pinv
                # Actually, simpler: solve U Z_vw^T = T1
                # This is equivalent to (Z_vw U^T)^T = T1^T
                # Z_vw U^T = T1^T
                # U^T = Z_vw^+ T1^T
                # U = (Z_vw^+ T1^T)^T = T1 Z_vw^+
                # Let's use lstsq for stability
                # T1 (dim_u x dim_vw) = U (dim_u x R) * Z_vw^T (R x dim_vw)
                # For each row i of T1: T1[i,:] = sum_r U[i,r] Z_vw[:,r]^T
                # This is a linear system in U[i,:].
                # [Z_vw[:,0]^T; ...; Z_vw[:,R-1]^T] is not quite right.
                # It is T1[i, :] = U[i, :] @ Z_vw^T
                # So U[i, :] is the solution to Z_vw^T X^T = T1[i, :]^T ? No.
                # X = U[i, :]. X @ Z_vw^T = T1[i, :].
                # Z_vw @ X^T = T1[i, :]^T.
                # X^T = Z_vw^+ T1[i, :]^T.
                # U[:, i] ... wait.
                # Let's use the formula: U = T1 @ Z_vw @ inv(Z_vw^T Z_vw)
                # This assumes Z_vw has full column rank.
                # We can use lstsq on the whole system.
                # Reshape T1 to (dim_u, dim_vw).
                # We want U such that U @ Z_vw.T approx T1.
                # This is equivalent to Z_vw @ U.T approx T1.T.
                U_new, _, _, _ = np.linalg.lstsq(Z_vw, T1.T, rcond=None)
                U = U_new.T
            except Exception:
                pass
                
            # Update V
            # T2 = V * (U ⊙ W)^T
            # T2 is (dim_v, dim_u * dim_w)
            Z_uw = np.kron(U, W) # (dim_u*dim_w, R)
            try:
                V_new, _, _, _ = np.linalg.lstsq(Z_uw, T2.T, rcond=None)
                V = V_new.T
            except Exception:
                pass
                
            # Update W
            # T3 = W * (U ⊙ V)^T
            # T3 is (dim_w, dim_u * dim_v)
            Z_uv = np.kron(U, V) # (dim_u*dim_v, R)
            try:
                W_new, _, _, _ = np.linalg.lstsq(Z_uv, T3.T, rcond=None)
                W = W_new.T
            except Exception:
                pass
            
            # Check convergence
            # Compute residual
            # T_approx = sum_r U[:,r] V[:,r] W[:,r]
            # Use einsum
            T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
            residual = np.linalg.norm(T_approx - T)
            
            if residual < tol:
                converged = True
                break
        
        if converged:
            # Verify exactness with high precision
            T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
            if np.allclose(T_approx, T, atol=1e-9):
                best_rank = R
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break # Found minimal rank
        
    # Fallback to schoolbook if no better rank found
    if best_result is None:
        R = n * m * p
        U = np.zeros((dim_u, R))
        V = np.zeros((dim_v, R))
        W = np.zeros((dim_w, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1.0
                    V[b, col] = 1.0
                    W[c, col] = 1.0
                    col += 1
        best_result = {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
        
    return best_result

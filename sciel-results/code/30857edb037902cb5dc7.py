import numpy as np
import time
import random

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 110  # seconds, safety margin
    
    def elapsed():
        return time.time() - start_time
    
    # Build target matrices T_ik
    # T_ik has shape (n*m, m*p)
    # T_ik[a, b] = 1 if a = i*m+j and b = j*p+k for some j, else 0
    T = np.zeros((n, p, n*m, m*p))
    for i in range(n):
        for k in range(p):
            for j in range(m):
                a = i * m + j
                b = j * p + k
                T[i, k, a, b] = 1.0
    
    def verify_decomposition(U, V, W, rank):
        """U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)"""
        # Test with random matrices
        for trial in range(3):
            A = np.random.rand(n, m)
            B = np.random.rand(m, p)
            C_true = A @ B
            C_comp = np.zeros((n, p))
            A_flat = A.flatten()
            B_flat = B.flatten()
            for c in range(rank):
                su = np.dot(U[:, c], A_flat)
                sv = np.dot(V[:, c], B_flat)
                w_vec = W[:, c]
                w_mat = w_vec.reshape(p, n).T  # (n, p)
                C_comp += w_mat * (su * sv)
            if not np.allclose(C_true, C_comp, atol=1e-8):
                return False
        return True
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    if verify_decomposition(U_sb, V_sb, W_sb, rank_sb):
        best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
        best_rank = rank_sb
    else:
        best_U, best_V, best_W = U_sb, V_sb, W_sb
        best_rank = rank_sb
    
    # Try to find lower rank using alternating minimization
    # For rank r, we need u_c in R^{n*m}, v_c in R^{m*p}
    # and W_ik,c such that T_ik = sum_c W_ik,c * outer(u_c, v_c)
    
    def try_rank(r, max_iters=50):
        """Try to find a decomposition of rank r"""
        if elapsed() > deadline:
            return None
        
        # Initialize
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.zeros((p * n, r))
        
        for it in range(max_iters):
            if elapsed() > deadline:
                return None
            
            # Update W: for each (i,k), solve least squares
            # T_ik = sum_c W_ik,c * outer(u_c, v_c)
            # Let M_c = outer(u_c, v_c).reshape(-1), then T_ik.flatten() = sum_c W_ik,c * M_c
            # This is a linear system: W_ik = (M^T M)^{-1} M^T T_ik.flatten()
            for i in range(n):
                for k in range(p):
                    # Build M: (r, n*m*m*p)
                    M = np.zeros((r, n * m * m * p))
                    for c in range(r):
                        M[c, :] = np.outer(U[:, c], V[:, c]).flatten()
                    t_flat = T[i, k].flatten()
                    # Solve M^T W_ik = t_flat
                    # W_ik = (M M^T)^{-1} M^T t_flat  (if r < n*m*m*p)
                    MTM = M @ M.T  # (r, r)
                    MTt = M @ t_flat  # (r,)
                    try:
                        W_ik = np.linalg.solve(MTM + 1e-10 * np.eye(r), MTt)
                    except np.linalg.LinAlgError:
                        W_ik = np.linalg.lstsq(MTM, MTt, rcond=None)[0]
                    W[k * n + i, :] = W_ik
            
            # Update U: for each c, solve
            # T_ik - sum_{c'!=c} W_ik,c' * outer(u_c', v_c') = W_ik,c * outer(u_c, v_c)
            # Let R_ik = T_ik - sum_{c'!=c} W_ik,c' * outer(u_c', v_c')
            # Then R_ik = W_ik,c * outer(u_c, v_c)
            # This is a rank-1 approximation: u_c ≈ R_ik.flatten() / (W_ik,c * ||v_c||^2) ... not quite
            # Actually, outer(u_c, v_c) = R_ik / W_ik,c
            # So u_c = (R_ik / W_ik,c) @ v_c^+  where v_c^+ is pseudo-inverse
            # Better: use SVD on the residual
            for c in range(r):
                # Compute residual for this c
                U_c = np.zeros(n * m)
                count = 0
                for i in range(n):
                    for k in range(p):
                        if abs(W[k * n + i, c]) < 1e-12:
                            continue
                        # Residual
                        R = T[i, k].copy()
                        for c2 in range(r):
                            if c2 != c:
                                R -= W[k * n + i, c2] * np.outer(U[:, c2], V[:, c2])
                        # R should ≈ W_ik,c * outer(u_c, v_c)
                        # So outer(u_c, v_c) ≈ R / W_ik,c
                        target = R / W[k * n + i, c]
                        # u_c ≈ target @ pinv(v_c)
                        # Use SVD: target = u_c * v_c^T, so u_c = target @ v_c^T (v_c^T v_c)^{-1}
                        vT = V[:, c]
                        vT_vT = np.dot(vT, vT)
                        if vT_vT < 1e-12:
                            continue
                        U_c += (target @ vT) / vT_vT
                        count += 1
                if count > 0:
                    U[:, c] = U_c / count
            
            # Update V similarly
            for c in range(r):
                V_c = np.zeros(m * p)
                count = 0
                for i in range(n):
                    for k in range(p):
                        if abs(W[k * n + i, c]) < 1e-12:
                            continue
                        R = T[i, k].copy()
                        for c2 in range(r):
                            if c2 != c:
                                R -= W[k * n + i, c2] * np.outer(U[:, c2], V[:, c2])
                        target = R / W[k * n + i, c]
                        uT = U[:, c]
                        uT_uT = np.dot(uT, uT)
                        if uT_uT < 1e-12:
                            continue
                        V_c += (target.T @ uT) / uT_uT
                        count += 1
                if count > 0:
                    V[:, c] = V_c / count
        
        # Check if we got a good solution
        # Verify by checking the reconstruction error
        max_err = 0
        for i in range(n):
            for k in range(p):
                recon = np.zeros((n * m, m * p))
                for c in range(r):
                    recon += W[k * n + i, c] * np.outer(U[:, c], V[:, c])
                err = np.linalg.norm(recon - T[i, k])
                max_err = max(max_err, err)
        
        if max_err < 1e-6:
            # Convert to the required format
            W_out = np.zeros((p * n, r))
            W_out[:] = W
            return U, V, W_out, max_err
        return None
    
    # Try ranks from best_rank-1 downward
    for r in range(best_rank - 1, max(best_rank - 20, 50), -1):
        if elapsed() > deadline:
            break
        result = try_rank(r, max_iters=30)
        if result is not None:
            U, V, W, err = result
            if verify_decomposition(U, V, W, r):
                best_U, best_V, best_W = U, V, W
                best_rank = r
                break
    
    # Return best found
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

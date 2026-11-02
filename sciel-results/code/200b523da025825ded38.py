import numpy as np
import time
import random

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    R_max = n * m * p  # 84
    
    deadline = time.time() + 115.0
    best_rank = R_max
    best_U = None
    best_V = None
    best_W = None
    
    # Trivial decomposition
    U_full = np.zeros((nm, R_max), dtype=float)
    V_full = np.zeros((mp, R_max), dtype=float)
    W_full = np.zeros((pn, R_max), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_full[i * m + j, col] = 1.0
                V_full[j * p + k, col] = 1.0
                W_full[k * n + i, col] = 1.0
                col += 1
                
    best_U = U_full.copy()
    best_V = V_full.copy()
    best_W = W_full.copy()
    
    def compute_error(U, V, W, R):
        U_t = U.reshape(n, m, R)
        V_t = V.reshape(m, p, R)
        W_t = W.reshape(p, n, R)
        T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        return np.max(np.abs(T - 1.0))

    def als_solve(U, V, W, R, max_iters=100, tol=1e-12):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        for it in range(max_iters):
            if time.time() > deadline:
                return U, V, W, False
                
            # Update U
            for i in range(n):
                for j in range(m):
                    # A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    A = np.zeros((p, R))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    # Solve A.T @ u = ones(p)
                    try:
                        u_sol, _, _, _ = np.linalg.lstsq(A.T, np.ones(p), rcond=None)
                        U[i*m+j, :] = u_sol
                    except:
                        pass

            # Update V
            for j in range(m):
                for k in range(p):
                    # B[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    B = np.zeros((n, R))
                    for i in range(n):
                        B[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        v_sol, _, _, _ = np.linalg.lstsq(B.T, np.ones(n), rcond=None)
                        V[j*p+k, :] = v_sol
                    except:
                        pass

            # Update W
            for k in range(p):
                for i in range(n):
                    # C[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    C = np.zeros((m, R))
                    for j in range(m):
                        C[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        w_sol, _, _, _ = np.linalg.lstsq(C.T, np.ones(m), rcond=None)
                        W[k*n+i, :] = w_sol
                    except:
                        pass
            
            err = compute_error(U, V, W, R)
            if err < tol:
                return U, V, W, True
        
        return U, V, W, False

    def try_rank(R, U_prev, V_prev, W_prev, max_restarts=10):
        if time.time() > deadline:
            return None, None, None, False
            
        best_err = np.inf
        best_U_sol = None
        best_V_sol = None
        best_W_sol = None
        success = False
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Initialize
            if restart == 0 and R < U_prev.shape[1]:
                # Try to inherit from previous solution
                # Select R columns from previous
                indices = random.sample(range(U_prev.shape[1]), R)
                U = U_prev[:, indices].copy()
                V = V_prev[:, indices].copy()
                W = W_prev[:, indices].copy()
                # Add small noise to break symmetry
                U += np.random.randn(nm, R) * 0.01
                V += np.random.randn(mp, R) * 0.01
                W += np.random.randn(pn, R) * 0.01
            else:
                # Random initialization
                # Scale matters. For rank R, typical value is ~ 1/sqrt(R) or similar?
                # Let's try random normal with small variance
                U = np.random.randn(nm, R) * 0.5
                V = np.random.randn(mp, R) * 0.5
                W = np.random.randn(pn, R) * 0.5
            
            U_sol, V_sol, W_sol, converged = als_solve(U, V, W, R, max_iters=200)
            
            if converged:
                return U_sol, V_sol, W_sol, True
                
            err = compute_error(U_sol, V_sol, W_sol, R)
            if err < best_err:
                best_err = err
                best_U_sol = U_sol
                best_V_sol = V_sol
                best_W_sol = W_sol
        
        if best_err < 1e-8:
            return best_U_sol, best_V_sol, best_W_sol, True
            
        return None, None, None, False

    # Start from R_max - 1 and go down
    current_U = U_full
    current_V = V_full
    current_W = W_full
    current_R = R_max
    
    for R in range(R_max - 1, 20, -1):
        if time.time() > deadline:
            break
            
        U_sol, V_sol, W_sol, found = try_rank(R, current_U, current_V, current_W, max_restarts=5)
        
        if found:
            current_U = U_sol
            current_V = V_sol
            current_W = W_sol
            current_R = R
            best_rank = R
            best_U = U_sol
            best_V = V_sol
            best_W = W_sol
            # Continue to try lower ranks
        else:
            # If we fail at R, we might try a few more times or break?
            # Let's try a couple more times with more restarts before giving up on this rank
            U_sol, V_sol, W_sol, found = try_rank(R, current_U, current_V, current_W, max_restarts=10)
            if found:
                current_U = U_sol
                current_V = V_sol
                current_W = W_sol
                current_R = R
                best_rank = R
                best_U = U_sol
                best_V = V_sol
                best_W = W_sol
            else:
                # If we can't reduce further, we might be stuck.
                # It's possible that R is the minimum for this path.
                # We can try to jump to a much lower rank with random init?
                # For now, break if we fail significantly.
                # Or try one more time with a completely random start for this R.
                U_sol, V_sol, W_sol, found = try_rank(R, None, None, None, max_restarts=20)
                if found:
                    current_U = U_sol
                    current_V = V_sol
                    current_W = W_sol
                    current_R = R
                    best_rank = R
                    best_U = U_sol
                    best_V = V_sol
                    best_W = W_sol
                else:
                    # Failed to reduce rank. Stop?
                    # Maybe we can try R-2? Unlikely to skip a rank.
                    # Let's break to save time for other potential improvements if any.
                    # But since we want minimal rank, and we failed at R,
                    # it's likely we won't succeed at R-1 easily from this state.
                    # However, a random restart at R-1 might work?
                    # Let's try one random restart at R-1.
                    if R - 1 > 20:
                        U_sol, V_sol, W_sol, found = try_rank(R-1, None, None, None, max_restarts=5)
                        if found:
                            best_rank = R-1
                            best_U = U_sol
                            best_V = V_sol
                            best_W = W_sol
                    break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

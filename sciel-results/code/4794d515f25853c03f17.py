import numpy as np
import time
import sys
from fractions import Fraction

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for 120s limit
    
    # Schoolbook solution
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb
    
    # Target tensor T_target[i, j, k] = 1 for all i, j, k
    T_target = np.ones((n, m, p))

    def get_tensor(U, V, W):
        """Compute the tensor T from decomposition U, V, W."""
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        T = np.zeros((n, m, p))
        R = U.shape[1]
        for r in range(R):
            u = U[:, r].reshape(n, m)
            v = V[:, r].reshape(m, p)
            w = W[:, r].reshape(p, n)
            # T[i,j,k] += u[i,j] * v[j,k] * w[k,i]
            # Efficiently: T[:,:,k] += (u * v[:,k]) * w[k,:]
            # Actually, let's do it element-wise or with broadcasting
            # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
            # This is a 3-index sum.
            # Loop over k is efficient enough for small dims.
            for k in range(p):
                # v[j,k] for all j
                v_col = v[:, k] # shape (m,)
                # w[k,i] for all i
                w_row = w[k, :] # shape (n,)
                # u is (n, m)
                # term[i,j] = u[i,j] * v_col[j] * w_row[i]
                # T[:,:,k] += term
                T[:, :, k] += u * np.outer(w_row, v_col)
        return T

    def residual_flat(U, V, W):
        """Return flattened residual vector for optimization."""
        T = get_tensor(U, V, W)
        return (T - T_target).flatten()

    def check_validity(U, V, W, tol=1e-9):
        """Check if decomposition is valid and entries are half-integers."""
        # Check half-integer constraint
        for M in [U, V, W]:
            M_arr = np.asarray(M)
            if not np.allclose(M_arr * 2, np.round(M_arr * 2), atol=tol):
                return False
        # Check exactness
        T = get_tensor(U, V, W)
        if not np.allclose(T, T_target, atol=tol):
            return False
        return True

    def snap_to_half(U, V, W):
        """Snap values to nearest half-integer."""
        U_snap = np.round(np.asarray(U) * 2) / 2.0
        V_snap = np.round(np.asarray(V) * 2) / 2.0
        W_snap = np.round(np.asarray(W) * 2) / 2.0
        return U_snap, V_snap, W_snap

    def try_rank_least_squares(R, num_restarts=10, max_nfev=200):
        """
        Attempt to find a rank-R decomposition using scipy.optimize.least_squares.
        Returns (U, V, W) if found, else None.
        """
        import scipy.optimize
        
        n_vars = (n * m + m * p + p * n) * R
        
        def residuals(x):
            U_flat = x[:n*m*R]
            V_flat = x[n*m*R : n*m*R + m*p*R]
            W_flat = x[n*m*R + m*p*R :]
            U_r = U_flat.reshape(n*m, R)
            V_r = V_flat.reshape(m*p, R)
            W_r = W_flat.reshape(p*n, R)
            return residual_flat(U_r, V_r, W_r)
        
        for _ in range(num_restarts):
            if time.time() > deadline - 5:
                return None
            
            # Random initialization
            # Scale matters. Schoolbook entries are 0 or 1.
            # Random normal with std 0.5 is a reasonable start.
            x0 = np.random.randn(n_vars) * 0.5
            
            try:
                result = scipy.optimize.least_squares(
                    residuals, x0, method='lm', max_nfev=max_nfev
                )
                
                if result.cost < 1e-8:
                    # Extract solution
                    U_sol = result.x[:n*m*R].reshape(n*m, R)
                    V_sol = result.x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                    W_sol = result.x[n*m*R + m*p*R :].reshape(p*n, R)
                    
                    # Snap to half-integers
                    U_snap, V_snap, W_snap = snap_to_half(U_sol, V_sol, W_sol)
                    
                    # Verify
                    if check_validity(U_snap, V_snap, W_snap):
                        return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
                        
            except Exception:
                pass
        
        return None

    # Try to find lower rank
    # Start from rank 35 down to 30
    # The theoretical lower bound is max(nm, mp, pn) = max(9, 12, 12) = 12.
    # But for 3x3x3, Strassen is 7. For 3x3x4, it's likely around 20-25?
    # Let's try a few ranks.
    
    # Note: Rank 36 is schoolbook.
    # Try 35, 34, 33, 32, 31, 30.
    
    for R in range(35, 29, -1):
        if time.time() > deadline - 10:
            break
            
        res = try_rank_least_squares(R, num_restarts=5, max_nfev=50)
        if res:
            U_sol, V_sol, W_sol = res
            best_U, best_V, best_W, best_rank = U_sol, V_sol, W_sol, R
            break
            
    return {"U": best_U, "V": best_V, "W": best_W}

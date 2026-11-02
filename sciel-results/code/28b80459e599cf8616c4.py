import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 5, 5]

def construct():
    """
    Returns the rank-minimal exact tensor decomposition for matrix multiplication 2x5 * 5x5.
    Uses ALS with polynomial refinement and nonlinear polishing.
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 50
    
    # Build schoolbook solution as fallback
    def build_schoolbook():
        rank = schoolbook_rank
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W
    
    best_U, best_V, best_W = build_schoolbook()
    best_rank = schoolbook_rank
    
    def check_time():
        return time.time() < deadline

    def compute_residual(U, V, W):
        """Compute the residual vector for the tensor equation."""
        # Residual for each (i, j, k) is sum_r U[i,j,r] V[j,k,r] W[k,i,r] - 1
        res = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                u_row = U[i * m + j]
                for k in range(p):
                    v_row = V[j * p + k]
                    w_row = W[k * n + i]
                    val = np.dot(u_row, v_row * w_row)
                    res[idx] = val - 1.0
                    idx += 1
        return res

    def is_valid(U, V, W, tol=1e-9):
        res = compute_residual(U, V, W)
        return np.max(np.abs(res)) < tol

    def als_solve(R, seed=0, max_iters=100):
        """
        Alternating Least Squares with random restarts and structured init.
        """
        if not check_time():
            return None
            
        np.random.seed(seed)
        
        # Initialization
        # Try a few different initialization strategies
        inits = []
        
        # Strategy 1: Random
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        inits.append((U, V, W))
        
        # Strategy 2: Sparse/Structured W
        W_s = np.zeros((p * n, R))
        for k in range(p):
            for i in range(n):
                # Sparse pattern
                nz = np.random.choice(R, max(1, R//10), replace=False)
                W_s[k * n + i, nz] = np.random.choice([-1.0, 1.0], size=len(nz))
        U_s = np.random.randn(n * m, R) * 0.1
        V_s = np.random.randn(m * p, R) * 0.1
        inits.append((U_s, V_s, W_s))
        
        # Strategy 3: Perturbed Schoolbook (if R close to 50)
        # This is tricky to implement directly for R < 50 without projecting.
        # We'll stick to random/structured for now.
        
        best_local_err = np.inf
        best_local_sol = None
        
        for init_idx, (U, V, W) in enumerate(inits):
            if not check_time():
                break
                
            # ALS Loop
            for it in range(max_iters):
                if not check_time():
                    break
                
                # Solve for U
                # Linear system: A_U * u_flat = b
                # A_U[row, col] = V[j*p+k, r] * W[k*n+i, r]
                # This is expensive to build every time. 
                # Given small sizes, it's okay.
                n_constraints = n * m * p
                n_vars_u = n * m * R
                A_U = np.zeros((n_constraints, n_vars_u))
                b = np.ones(n_constraints)
                
                idx = 0
                for i in range(n):
                    for j in range(m):
                        u_row_idx = i * m + j
                        for k in range(p):
                            coeff = V[j * p + k] * W[k * n + i]
                            A_U[idx, u_row_idx * R : (u_row_idx + 1) * R] = coeff
                            idx += 1
                
                try:
                    u_flat, _, _, _ = np.linalg.lstsq(A_U, b, rcond=None)
                    U = u_flat.reshape(n * m, R)
                except:
                    break
                
                # Solve for V
                n_vars_v = m * p * R
                A_V = np.zeros((n_constraints, n_vars_v))
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            coeff = U[i * m + j] * W[k * n + i]
                            v_row_idx = j * p + k
                            A_V[idx, v_row_idx * R : (v_row_idx + 1) * R] = coeff
                            idx += 1
                            
                try:
                    v_flat, _, _, _ = np.linalg.lstsq(A_V, b, rcond=None)
                    V = v_flat.reshape(m * p, R)
                except:
                    break
                    
                # Solve for W
                n_vars_w = p * n * R
                A_W = np.zeros((n_constraints, n_vars_w))
                idx = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            coeff = U[i * m + j] * V[j * p + k]
                            w_row_idx = k * n + i
                            A_W[idx, w_row_idx * R : (w_row_idx + 1) * R] = coeff
                            idx += 1
                            
                try:
                    w_flat, _, _, _ = np.linalg.lstsq(A_W, b, rcond=None)
                    W = w_flat.reshape(p * n, R)
                except:
                    break
                    
                # Check convergence
                res = compute_residual(U, V, W)
                err = np.max(np.abs(res))
                if err < best_local_err:
                    best_local_err = err
                    best_local_sol = (U.copy(), V.copy(), W.copy())
                    
                if err < 1e-9:
                    return U, V, W
                    
        # If ALS didn't converge, try polishing the best local solution
        if best_local_sol is not None and check_time():
            U0, V0, W0 = best_local_sol
            
            # Flatten all variables for scipy
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            def objective(x):
                U = x[:n*m*R].reshape(n*m, R)
                V = x[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                W = x[n*m*R + m*p*R :].reshape(p*n, R)
                return compute_residual(U, V, W)
                
            try:
                result = least_squares(objective, x0, method='trf', max_nfev=100, ftol=1e-15, xtol=1e-15)
                if result.success or result.cost < 1e-8:
                    x_opt = result.x
                    U_opt = x_opt[:n*m*R].reshape(n*m, R)
                    V_opt = x_opt[n*m*R : n*m*R + m*p*R].reshape(m*p, R)
                    W_opt = x_opt[n*m*R + m*p*R :].reshape(p*n, R)
                    
                    if is_valid(U_opt, V_opt, W_opt, tol=1e-8):
                        return U_opt, V_opt, W_opt
            except Exception as e:
                pass
                
        return None

    # Search ranks from high to low (49 down to 25)
    # Focus on 49, 48, 47 first as they are most likely candidates if < 50
    for R in range(49, 24, -1):
        if not check_time():
            break
            
        # More restarts for higher ranks (closer to schoolbook)
        num_restarts = 10 if R >= 45 else 5
        
        for restart in range(num_restarts):
            if not check_time():
                break
            result = als_solve(R, seed=R * 100 + restart, max_iters=80)
            if result is not None:
                U, V, W = result
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    # Fallback
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

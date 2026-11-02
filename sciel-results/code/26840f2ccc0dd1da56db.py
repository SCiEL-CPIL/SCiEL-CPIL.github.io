"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]

def _verify_exact(U, V, W, n, m, p):
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-12:
                    return False
    return True

def _try_rank_fixed_VW(n, m, p, rank, n_trials=1000):
    """Fix V and W randomly, solve for U, check if U is close to half-integers."""
    nm, mp, pn = n*m, m*p, p*n
    
    for trial in range(n_trials):
        if time.time() > 140:
            break
        np.random.seed(2000 + trial)
        V = np.random.randn(mp, rank)
        W = np.random.randn(pn, rank)
        
        # Build the linear system for U
        # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # This is a linear system in U
        
        # Let's build the matrix A and vector b for the linear system
        # Unknowns: U[0,0], U[0,1], ..., U[nm-1, rank-1] (nm*rank unknowns)
        # Equations: one for each (i,j,k) (n*m*p equations)
        
        n_eq = n * m * p
        A = np.zeros((n_eq, nm * rank))
        b = np.ones(n_eq)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        A[eq_idx, (i*m+j)*rank + r] = V[j*p+k, r] * W[k*n+i, r]
                    eq_idx += 1
        
        # Solve the linear system (least squares if overdetermined)
        U_flat, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
        U = U_flat.reshape(nm, rank)
        
        # Check if the solution is close to half-integers
        U_round = np.round(U * 2) / 2
        if np.max(np.abs(U - U_round)) < 1e-6:
            # Verify
            if _verify_exact(U_round, V, W, n, m, p):
                return U_round, V, W
        
        # Also try rounding to integers
        U_round_int = np.round(U)
        if np.max(np.abs(U - U_round_int)) < 1e-6:
            if _verify_exact(U_round_int, V, W, n, m, p):
                return U_round_int, V, W
    
    return None

def _try_rank_full(n, m, p, rank, n_restarts=50, max_iters=3000):
    """Full numerical optimization for rank-`rank` decomposition."""
    nm, mp, pn = n*m, m*p, p*n
    total_vars = (nm + mp + pn) * rank
    
    def objective(x):
        U = x[:nm*rank].reshape(nm, rank)
        V = x[nm*rank:(nm+mp)*rank].reshape(mp, rank)
        W = x[(nm+mp)*rank:].reshape(pn, rank)
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (s - 1.0)**2
        return err
    
    best_x = None
    best_obj = float('inf')
    
    for restart in range(n_restarts):
        if time.time() > 140:
            break
        np.random.seed(3000 + restart)
        x0 = np.random.randn(total_vars) * 1.0
        
        from scipy.optimize import minimize
        result = minimize(objective, x0, method='L-BFGS-B', 
                         options={'maxiter': max_iters, 'ftol': 1e-18, 'gtol': 1e-18})
        
        if result.fun < best_obj:
            best_obj = result.fun
            best_x = result.x
    
    if best_x is None or best_obj > 1e-8:
        return None
    
    U = best_x[:nm*rank].reshape(nm, rank)
    V = best_x[nm*rank:(nm+mp)*rank].reshape(mp, rank)
    W = best_x[(nm+mp)*rank:].reshape(pn, rank)
    
    # Try to snap to half-integers
    for scale in [1, 2]:
        U_round = np.round(U * scale) / scale
        V_round = np.round(V * scale) / scale
        W_round = np.round(W * scale) / scale
        
        if _verify_exact(U_round, V_round, W_round, n, m, p):
            return U_round, V_round, W_round
    
    return None

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130
    
    lower_bound = max(n*m, m*p, p*n)
    
    best_result = None
    
    for rank in range(lower_bound, n*m*p + 1):
        if time.time() > deadline - 10:
            break
        
        # Try fixed V,W approach first (faster)
        result = _try_rank_fixed_VW(n, m, p, rank, n_trials=200)
        if result is None:
            # Try full optimization
            result = _try_rank_full(n, m, p, rank, n_restarts=20, max_iters=2000)
        
        if result is not None:
            U, V, W = result
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            break
    
    if best_result is None:
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END

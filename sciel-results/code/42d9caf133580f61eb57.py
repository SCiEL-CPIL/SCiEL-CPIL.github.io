import numpy as np
from fractions import Fraction
import time
import itertools

def _verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0])
    U_arr = np.array([[float(x) for x in row] for row in U], dtype=float)
    V_arr = np.array([[float(x) for x in row] for row in V], dtype=float)
    W_arr = np.array([[float(x) for x in row] for row in W], dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                s = np.dot(U_arr[idx_u], V_arr[idx_v] * W_arr[idx_w])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank_lower(target_rank, n, m, p, base_U=None, base_V=None, base_W=None, max_tries=20, time_budget=45):
    """
    Attempt to find a rank-`target_rank` decomposition.
    If base_U, base_V, base_W are provided, try to initialize from them (e.g., by zeroing out extra terms).
    """
    nm, mp, pn = n * m, m * p, p * n
    n_params = target_rank * (nm + mp + pn)
    
    start_time = time.time()
    best_obj = np.inf
    best_x = None
    
    def compute_residual(x):
        U = x[:target_rank*nm].reshape(target_rank, nm).T
        V = x[target_rank*nm:target_rank*nm+target_rank*mp].reshape(target_rank, mp).T
        W = x[target_rank*nm+target_rank*mp:].reshape(target_rank, pn).T
        
        residual = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U[i*m+j]
                for k in range(p):
                    v_w = V[j*p+k] * W[k*n+i]
                    val = np.dot(u_row, v_w)
                    residual += (val - 1.0)**2
        return residual

    def _jac(x):
        U = x[:target_rank*nm].reshape(target_rank, nm).T
        V = x[target_rank*nm:target_rank*nm+target_rank*mp].reshape(target_rank, mp).T
        W = x[target_rank*nm+target_rank*mp:].reshape(target_rank, pn).T
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                err_vec = np.zeros(p)
                for k in range(p):
                    v_w = V[j*p+k] * W[k*n+i]
                    err_vec[k] = U[i*m+j] @ v_w - 1.0
                for k in range(p):
                    grad_U[i*m+j] += 2 * err_vec[k] * V[j*p+k] * W[k*n+i]
        
        for j in range(m):
            for k in range(p):
                err_vec = np.zeros(n)
                for i in range(n):
                    v_w = V[j*p+k] * W[k*n+i]
                    err_vec[i] = U[i*m+j] @ v_w - 1.0
                for i in range(n):
                    grad_V[j*p+k] += 2 * err_vec[i] * U[i*m+j] * W[k*n+i]
        
        for k in range(p):
            for i in range(n):
                err_vec = np.zeros(m)
                for j in range(m):
                    v_w = V[j*p+k] * W[k*n+i]
                    err_vec[j] = U[i*m+j] @ v_w - 1.0
                for j in range(m):
                    grad_W[k*n+i] += 2 * err_vec[j] * U[i*m+j] * V[j*p+k]
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    # Generate initial guesses
    initial_guesses = []
    
    # 1. Random initializations
    for t in range(5):
        np.random.seed(t * 100 + target_rank)
        x0 = np.random.randn(n_params) * 0.2
        initial_guesses.append(x0)
        
    # 2. Base solution initialization (if available and rank is close)
    if base_U is not None:
        base_rank = len(base_U[0])
        if base_rank > target_rank:
            # Take the first target_rank terms from the base solution
            U_base = np.array(base_U)[:, :target_rank]
            V_base = np.array(base_V)[:, :target_rank]
            W_base = np.array(base_W)[:, :target_rank]
            
            x0 = np.concatenate([U_base.flatten(), V_base.flatten(), W_base.flatten()])
            initial_guesses.append(x0)
            
            # Also try zeroing out one term from the base and re-optimizing
            # This is expensive, so we'll just use the truncated version for now
            
    for x0 in initial_guesses:
        if time.time() - start_time > time_budget:
            break
        
        try:
            from scipy.optimize import minimize
            result = minimize(compute_residual, x0, method='L-BFGS-B', jac=_jac,
                             options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'iprint': 0})
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_x = result.x.copy()
                
                if best_obj < 1e-10:
                    U = result.x[:target_rank*nm].reshape(target_rank, nm).T
                    V = result.x[target_rank*nm:target_rank*nm+target_rank*mp].reshape(target_rank, mp).T
                    W = result.x[target_rank*nm+target_rank*mp:].reshape(target_rank, pn).T
                    
                    # Round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    U_frac = [[Fraction(int(np.round(v * 2)), 2) for v in row] for row in U_r]
                    V_frac = [[Fraction(int(np.round(v * 2)), 2) for v in row] for row in V_r]
                    W_frac = [[Fraction(int(np.round(v * 2)), 2) for v in row] for row in W_r]
                    
                    if _verify_decomposition(U_frac, V_frac, W_frac, n, m, p):
                        return (U_frac, V_frac, W_frac)
        except Exception:
            pass
            
    return None

def _build_rank28():
    """Build the known rank-28 decomposition."""
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: j=2, all k (10 terms)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    # Part 2: j=0,1, k=4 (4 terms)
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1
            
    # Part 3: j=0,1, k=0,1 using Strassen (7 terms)
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 0][r] = 1
    W[0 * n + 1][r] = 1
    W[1 * n + 1][r] = -1
    r += 1
    U[0 * m + 0][r] = 1
    V[0 * p + 1][r] = 1
    V[1 * p + 1][r] = -1
    W[1 * n + 0][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    U[1 * m + 1][r] = 1
    V[1 * p + 0][r] = 1
    V[0 * p + 0][r] = -1
    W[0 * n + 0][r] = 1
    W[0 * n + 1][r] = 1
    r += 1
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = -1
    W[1 * n + 0][r] = 1
    r += 1
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 0][r] = 1
    V[0 * p + 1][r] = 1
    W[1 * n + 1][r] = 1
    r += 1
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 0][r] = 1
    V[1 * p + 1][r] = 1
    W[0 * n + 0][r] = 1
    r += 1
    
    # Part 4: j=0,1, k=2,3 using Strassen (7 terms)
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 2][r] = 1
    W[2 * n + 1][r] = 1
    W[3 * n + 1][r] = -1
    r += 1
    U[0 * m + 0][r] = 1
    V[0 * p + 3][r] = 1
    V[1 * p + 3][r] = -1
    W[3 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    U[1 * m + 1][r] = 1
    V[1 * p + 2][r] = 1
    V[0 * p + 2][r] = -1
    W[2 * n + 0][r] = 1
    W[2 * n + 1][r] = 1
    r += 1
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = -1
    W[3 * n + 0][r] = 1
    r += 1
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 2][r] = 1
    V[0 * p + 3][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 2][r] = 1
    V[1 * p + 3][r] = 1
    W[2 * n + 0][r] = 1
    r += 1
    
    return U, V, W

def _try_rank_lower(target_rank, n, m, p, base_U=None, base_V=None, base_W=None, max_tries=20, time_budget=45):
    """
    Attempt to find a rank-`target_rank` decomposition.
    If base_U, base_V, base_W are provided, try to initialize from them (e.g., by zeroing out extra terms).
    """
    nm, mp, pn = n * m, m * p, p * n
    n_params = target_rank * (nm + mp + pn)
    
    start_time = time.time()
    best_obj = np.inf
    best_x = None
    
    def compute_residual(x):
        U = x[:target_rank*nm].reshape(target_rank, nm).T
        V = x[target_rank*nm:target_rank*nm+target_rank*mp].reshape(target_rank, mp).T
        W = x[target_rank*nm+target_rank*mp:].reshape(target_rank, pn).T
        
        residual = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U[i*m+j]
                for k in range(p):
                    v_w = V[j*p+k] * W[k*n+i]
                    val = np.dot(u_row, v_w)
                    residual += (val - 1.0)**2
        return residual

    def _jac(x):
        U = x[:target_rank*nm].reshape(target_rank, nm).T
        V = x[target_rank*nm:target_rank*nm+target_rank*mp].reshape(target_rank, mp).T
        W = x[target_rank*nm+target_rank*mp:].reshape(target_rank, pn).T
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                err_vec = np.zeros(p)
                for k in range(p):
                    v_w = V[j*p+k] * W[k*n+i]
                    err_vec[k] = U[i*m+j] @ v_w - 1.0
                for k in range(p):
                    grad_U[i*m+j] += 2 * err_vec[k] * V[j*p+k] * W[k*n+i]
        
        for j in range(m):
            for k in range(p):
                err_vec = np.zeros(n)
                for i in range(n):
                    v_w = V[j*p+k] * W[k*n+i]
                    err_vec[i] = U[i*m+j] @ v_w - 1.0
                for i in range(n):
                    grad_V[j*p+k] += 2 * err_vec[i] * U[i*m+j] * W[k*n+i]
        
        for k in range(p):
            for i in range(n):
                err_vec = np.zeros(m)
                for j in range(m):
                    v_w = V[j*p+k] * W[k*n+i]
                    err_vec[j] = U[i*m+j] @ v_w - 1.0
                for j in range(m):
                    grad_W[k*n+i] += 2 * err_vec[j] * U[i*m+j] * V[j*p+k]
        
        return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    # Generate initial guesses
    initial_guesses = []
    
    # 1. Random initializations
    for t in range(5):
        np.random.seed(t * 100 + target_rank)
        x0 = np.random.randn(n_params) * 0.2
        initial_guesses.append(x0)
        
    # 2. Base solution initialization (if available and rank is close)
    if base_U is not None:
        base_rank = len(base_U[0])
        if base_rank > target_rank:
            # Take the first target_rank terms from the base solution
            U_base = np.array(base_U)[:, :target_rank]
            V_base = np.array(base_V)[:, :target_rank]
            W_base = np.array(base_W)[:, :target_rank]
            
            x0 = np.concatenate([U_base.flatten(), V_base.flatten(), W_base.flatten()])
            initial_guesses.append(x0)
            
            # Also try zeroing out one term from the base and re-optimizing
            # This is expensive, so we'll just use the truncated version for now
            
    for x0 in initial_guesses:
        if time.time() - start_time > time_budget:
            break
        
        try:
            from scipy.optimize import minimize
            result = minimize(compute_residual, x0, method='L-BFGS-B', jac=_jac,
                             options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'iprint': 0})
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_x = result.x.copy()
                
                if best_obj < 1e-10:
                    U = result.x[:target_rank*nm].reshape(target_rank, nm).T
                    V = result.x[target_rank*nm:target_rank*nm+target_rank*mp].reshape(target_rank, mp).T
                    W = result.x[target_rank*nm+target_rank*mp:].reshape(target_rank, pn).T
                    
                    # Round to half-integers
                    U_r = np.round(U * 2) / 2
                    V_r = np.round(V * 2) / 2
                    W_r = np.round(W * 2) / 2
                    
                    U_frac = [[Fraction(int(np.round(v * 2)), 2) for v in row] for row in U_r]
                    V_frac = [[Fraction(int(np.round(v * 2)), 2) for v in row] for row in V_r]
                    W_frac = [[Fraction(int(np.round(v * 2)), 2) for v in row] for row in W_r]
                    
                    if _verify_decomposition(U_frac, V_frac, W_frac, n, m, p):
                        return (U_frac, V_frac, W_frac)
        except Exception:
            pass
            
    return None

def construct():
    n, m, p = 2, 3, 5
    start_time = time.time()
    deadline = 140
    
    U28, V28, W28 = _build_rank28()
    if not _verify_decomposition(U28, V28, W28, n, m, p):
        raise ValueError("Rank 28 verification failed")
    
    best_U, best_V, best_W = U28, V28, W28
    best_rank = 28
    
    # Try rank 27 using rank 28 as base
    if time.time() - start_time < deadline - 60:
        result = _try_rank_lower(27, n, m, p, base_U=U28, base_V=V28, base_W=W28, max_tries=10, time_budget=30)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = 27
    
    # Try rank 26
    if time.time() - start_time < deadline - 45 and best_rank > 26:
        result = _try_rank_lower(26, n, m, p, base_U=best_U, base_V=best_V, base_W=best_W, max_tries=15, time_budget=35)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = 26
            
    return {"U": best_U, "V": best_V, "W": best_W}

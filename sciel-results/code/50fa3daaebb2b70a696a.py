import numpy as np
from fractions import Fraction
import time
import itertools

def _verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0])
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                u_val = U[i * m + j][r]
                if u_val == 0:
                    continue
                for k in range(p):
                    v_val = V[j * p + k][r]
                    if v_val == 0:
                        continue
                    w_val = W[k * n + i][r]
                    if w_val == 0:
                        continue
                    # We need sum of products = 1
                    # This verification approach is incorrect for general tensors
                    # Let's use the correct approach
                    pass
    
    # Correct verification: for each (i, j, k), sum over r of U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] should be 1
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

def _try_rank_lower(target_rank, n, m, p, max_tries=20, time_budget=45):
    """
    Attempt to find a rank-`target_rank` decomposition using numerical search.
    We use the structure: for each rank term r, we have U[:, r], V[:, r], W[:, r].
    The constraint is that the tensor T = sum_r U_r tensor V_r tensor W_r equals the
    matrix multiplication tensor for 2x3 x 3x5.
    
    We parameterize U, V, W as dense matrices and minimize the residual.
    """
    nm, mp, pn = n * m, m * p, p * n
    n_params = target_rank * (nm + mp + pn)
    
    start_time = time.time()
    best_obj = np.inf
    best_x = None
    
    def compute_residual_and_grad(x):
        U = x[:target_rank*nm].reshape(target_rank, nm).T
        V = x[target_rank*nm:target_rank*nm+target_rank*mp].reshape(target_rank, mp).T
        W = x[target_rank*nm+target_rank*mp:].reshape(target_rank, pn).T
        
        residual = 0.0
        # For each (i, j, k), the bilinear form is U[i*m+j] . (V[j*p+k] * W[k*n+i])
        # We want this to equal 1.
        for i in range(n):
            for j in range(m):
                u_row = U[i*m+j]
                for k in range(p):
                    v_w = V[j*p+k] * W[k*n+i]
                    val = np.dot(u_row, v_w)
                    residual += (val - 1.0)**2
        return residual
    
    def _objective(x):
        return compute_residual_and_grad(x)
    
    def _jac(x):
        U = x[:target_rank*nm].reshape(target_rank, nm).T
        V = x[target_rank*nm:target_rank*nm+target_rank*mp].reshape(target_rank, mp).T
        W = x[target_rank*nm+target_rank*mp:].reshape(target_rank, pn).T
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # d/dU[i*m+j] of (U[i*m+j].(V[j*p+k]*W[k*n+i]) - 1)^2
        # = 2 * (U[i*m+j].(V[j*p+k]*W[k*n+i]) - 1) * (V[j*p+k]*W[k*n+i])
        # This is only for the specific (i,j,k)
        
        # But U[i*m+j] appears in all terms where i,j is fixed but k varies
        # So grad_U[i*m+j] = 2 * sum_k (U[i*m+j].(V[j*p+k]*W[k*n+i]) - 1) * (V[j*p+k]*W[k*n+i])
        
        # Similarly for V[j*p+k] and W[k*n+i]
        
        for i in range(n):
            for j in range(m):
                s_k = 0.0
                # Vector of (V[j*p+k]*W[k*n+i]) for all k, weighted by error
                # Let's just do it directly for clarity, it's small
                err_vec = np.zeros(p)
                for k in range(p):
                    v_w = V[j*p+k] * W[k*n+i]
                    err_vec[k] = U[i*m+j] @ v_w - 1.0
                # grad_U[i*m+j] = 2 * sum_k err_vec[k] * V[j*p+k] * W[k*n+i]
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
    
    for trial in range(max_tries):
        if time.time() - start_time > time_budget:
            break
        
        np.random.seed(trial * 1000 + target_rank)
        # Initialize with values that might be close to a solution
        # Use small random values
        x0 = np.random.randn(n_params) * 0.2
        # Make sure the initial guess has some structure that might help
        # e.g. diagonal-like
        
        try:
            from scipy.optimize import minimize
            result = minimize(_objective, x0, method='L-BFGS-B', jac=_jac,
                             options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'iprint': 0})
            
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
                    
                    # Check if rounded works
                    U_frac = [[Fraction(int(np.round(v * 2)), 2) for v in row] for row in U_r]
                    V_frac = [[Fraction(int(np.round(v * 2)), 2) for v in row] for row in V_r]
                    W_frac = [[Fraction(int(np.round(v * 2)), 2) for v in row] for row in W_r]
                    
                    if _verify_decomposition(U_frac, V_frac, W_frac, n, m, p):
                        return (U_frac, V_frac, W_frac)
                    else:
                        # Try unrounded but converted to Fraction via continued fractions or similar
                        # For now, if unrounded works, we can't easily return it
                        # unless we can represent exactly
                        pass
        except Exception as e:
            pass
    
    return None

def construct():
    n, m, p = 2, 3, 5
    start_time = time.time()
    deadline = 140  # 150s limit, leave 10s margin
    
    # Start with rank 28 fallback
    U28, V28, W28 = _build_rank28()
    if not _verify_decomposition(U28, V28, W28, n, m, p):
        raise ValueError("Rank 28 verification failed")
    
    best_U, best_V, best_W = U28, V28, W28
    best_rank = 28
    
    # Try rank 27
    if time.time() - start_time < deadline - 60:
        result = _try_rank_lower(27, n, m, p, max_tries=10, time_budget=30)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = 27
    
    # Try rank 26
    if time.time() - start_time < deadline - 45 and best_rank > 26:
        result = _try_rank_lower(26, n, m, p, max_tries=15, time_budget=35)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = 26
    
    # Try rank 25
    if time.time() - start_time < deadline - 30 and best_rank > 25:
        result = _try_rank_lower(25, n, m, p, max_tries=10, time_budget=25)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = 25
    
    # Try rank 24
    if time.time() - start_time < deadline - 20 and best_rank > 24:
        result = _try_rank_lower(24, n, m, p, max_tries=5, time_budget=15)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = 24
    
    return {"U": best_U, "V": best_V, "W": best_W}

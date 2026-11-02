import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 9]

def _build_rank32():
    n, m, p = DIMENSIONS
    rank = 32
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen(k0, k1):
        nonlocal col
        U[0, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        U[2, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1
        W[k0*n + 1, col] = 1; W[k1*n + 1, col] = -1
        col += 1
        U[0, col] = 1
        V[0*p + k1, col] = 1; V[1*p + k1, col] = -1
        W[k1*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        U[3, col] = 1
        V[1*p + k0, col] = 1; V[0*p + k0, col] = -1
        W[k0*n + 0, col] = 1; W[k0*n + 1, col] = 1
        col += 1
        U[0, col] = 1; U[1, col] = 1
        V[1*p + k1, col] = 1
        W[k0*n + 0, col] = -1; W[k1*n + 0, col] = 1
        col += 1
        U[2, col] = 1; U[0, col] = -1
        V[0*p + k0, col] = 1; V[0*p + k1, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        U[1, col] = 1; U[3, col] = -1
        V[1*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        U[0, col] = 1; V[0*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[1, col] = 1; V[1*p + k, col] = 1; W[k*n + 0, col] = 1
        col += 1
        U[2, col] = 1; V[0*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1
        U[3, col] = 1; V[1*p + k, col] = 1; W[k*n + 1, col] = 1
        col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    add_trivial(8)
    
    return U, V, W

def _compute_residual(U, V, W, n, m, p):
    """Compute the maximum absolute residual of the tensor decomposition."""
    rank = U.shape[1]
    M = np.einsum('ir,jr,kr->ijk', U, V, W)
    
    # Now build the true tensor
    T_true = np.zeros((n * m, m * p, p * n))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T_true[i, j, k] = 1.0
    
    residual = M - T_true
    return np.max(np.abs(residual))

def _soft_max_abs(residual, beta=100.0):
    """Smooth approximation of max(|residual|) using log-sum-exp."""
    abs_res = np.abs(residual)
    max_val = np.max(abs_res)
    if max_val < 1e-15:
        return 0.0
    scaled = beta * abs_res
    shifted = scaled - np.max(scaled)
    log_sum_exp = np.max(scaled) + np.log(np.sum(np.exp(shifted)))
    return log_sum_exp / beta

def _build_rank31_init(n, m, p):
    """
    Build a rank-31 initialization by exploiting algebraic structure.
    
    For 2x2 * 2x9, we use 4 Strassen blocks (7 each = 28) for columns 0-7,
    then handle column 8 with 3 additional terms using a modified Strassen
    pattern that shares the U[3] structure.
    """
    nm, mp, pn = n * m, m * p, p * n
    rank = 31
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    
    def add_strassen(k0, k1):
        nonlocal col
        U[0, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        U[2, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1
        W[k0*n + 1, col] = 1; W[k1*n + 1, col] = -1
        col += 1
        U[0, col] = 1
        V[0*p + k1, col] = 1; V[1*p + k1, col] = -1
        W[k1*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        U[3, col] = 1
        V[1*p + k0, col] = 1; V[0*p + k0, col] = -1
        W[k0*n + 0, col] = 1; W[k0*n + 1, col] = 1
        col += 1
        U[0, col] = 1; U[1, col] = 1
        V[1*p + k1, col] = 1
        W[k0*n + 0, col] = -1; W[k1*n + 0, col] = 1
        col += 1
        U[2, col] = 1; U[0, col] = -1
        V[0*p + k0, col] = 1; V[0*p + k1, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        U[1, col] = 1; U[3, col] = -1
        V[1*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1
        col += 1
    
    # 4 Strassen blocks for columns 0-7
    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_strassen(6, 7)
    
    # For column 8 (single column, k0=k1=8), use 3 terms
    # These handle the 4 non-zero entries: T[0,8,8], T[1,17,9], T[2,8,10], T[3,17,11]
    # Wait, let me recheck: for c=8, a=0: i=0*2+b, j=b*9+8, k=8*2+0=16
    # a=0,b=0: i=0, j=8, k=16
    # a=0,b=1: i=1, j=17, k=16
    # a=1,b=0: i=2, j=8, k=17
    # a=1,b=1: i=3, j=17, k=17
    
    # Term 1: U=[1,0,0,1], V=[1,0,0,1] at positions 8,17, W at 16,17
    U[0, col] = 1; U[3, col] = 1
    V[0*p + 8, col] = 1; V[1*p + 8, col] = 1
    W[8*n + 0, col] = 1; W[8*n + 1, col] = 1
    col += 1
    
    # Term 2: U=[0,1,-1,0], V=[1,0,0,-1], W=[1,0,0,-1]
    U[1, col] = 1; U[2, col] = -1
    V[0*p + 8, col] = 1; V[1*p + 8, col] = -1
    W[8*n + 0, col] = 1; W[8*n + 1, col] = -1
    col += 1
    
    # Term 3: U=[0,0,1,1], V=[0,0,0,1], W=[0,0,0,1]
    U[2, col] = 1; U[3, col] = 1
    V[1*p + 8, col] = 1
    W[8*n + 1, col] = 1
    col += 1
    
    return U, V, W

def _try_rank31_search(U0, V0, W0, n, m, p, time_budget):
    """Try to find a rank-31 decomposition using direct rank-31 optimization."""
    start_time = time.time()
    
    nm, mp, pn = n * m, m * p, p * n
    rank = 31
    
    # Precompute true tensor once
    T_true = np.zeros((nm, mp, pn))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T_true[i, j, k] = 1.0
    
    # Start from a structured rank-31 initialization
    U31, V31, W31 = _build_rank31_init(n, m, p)
    
    # Also try starting from rank-32 with one column removed (try different columns)
    inits = [(U31, V31, W31)]
    for drop_col in [0, 7, 14, 21, 28, 31]:
        mask = np.ones(32, dtype=bool)
        mask[drop_col] = False
        inits.append((U0[:, mask].copy(), V0[:, mask].copy(), W0[:, mask].copy()))
    
    def objective(params, beta=30.0):
        U_flat = params[:nm * rank]
        V_flat = params[nm * rank : nm * rank + mp * rank]
        W_flat = params[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
        
        U = U_flat.reshape(nm, rank)
        V = V_flat.reshape(mp, rank)
        W = W_flat.reshape(pn, rank)
        
        M = np.einsum('ir,jr,kr->ijk', U, V, W)
        residual = M - T_true
        return _soft_max_abs(residual, beta=beta)
    
    best_U, best_V, best_W = U31.copy(), V31.copy(), W31.copy()
    best_score = _compute_residual(U31, V31, W31, n, m, p)
    
    # Try multiple starting points and beta values
    for U_init, V_init, W_init in inits:
        if time.time() - start_time > time_budget * 0.7:
            break
        
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        for beta in [15.0, 30.0, 60.0]:
            if time.time() - start_time > time_budget * 0.85:
                break
            
            try:
                result = minimize(objective, x0, args=(beta,), method='L-BFGS-B',
                                  options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-11})
            except Exception:
                continue
            
            U_flat = result.x[:nm * rank]
            V_flat = result.x[nm * rank : nm * rank + mp * rank]
            W_flat = result.x[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
            U = U_flat.reshape(nm, rank)
            V = V_flat.reshape(mp, rank)
            W = W_flat.reshape(pn, rank)
            
            score = _compute_residual(U, V, W, n, m, p)
            if score < best_score:
                best_score = score
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if score < 1e-8:
                return best_U, best_V, best_W, score
    
    if best_score < 1e-6:
        return best_U, best_V, best_W, best_score
    
    return None

def _polish_rank31(U31, V31, W31, n, m, p, time_budget):
    """High-precision polish pass on a near-exact rank-31 decomposition."""
    start_time = time.time()
    nm, mp, pn = n * m, m * p, p * n
    rank = 31
    
    T_true = np.zeros((nm, mp, pn))
    for a in range(n):
        for b in range(m):
            for c in range(p):
                i = a * m + b
                j = b * p + c
                k = c * n + a
                T_true[i, j, k] = 1.0
    
    def objective_polish(params):
        U_flat = params[:nm * rank]
        V_flat = params[nm * rank : nm * rank + mp * rank]
        W_flat = params[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
        
        U = U_flat.reshape(nm, rank)
        V = V_flat.reshape(mp, rank)
        W = W_flat.reshape(pn, rank)
        
        M = np.einsum('ir,jr,kr->ijk', U, V, W)
        residual = M - T_true
        return np.sum(residual**2)
    
    def gradient_polish(params):
        U_flat = params[:nm * rank]
        V_flat = params[nm * rank : nm * rank + mp * rank]
        W_flat = params[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
        
        U = U_flat.reshape(nm, rank)
        V = V_flat.reshape(mp, rank)
        W = W_flat.reshape(pn, rank)
        
        M = np.einsum('ir,jr,kr->ijk', U, V, W)
        residual = M - T_true
        
        # Gradient: 2 * sum over the other two dimensions
        dU = 2 * np.einsum('ijk,jk->ik', residual, V * W)
        dV = 2 * np.einsum('ijk,ik->jk', residual, U * W)
        dW = 2 * np.einsum('ijk,ij->jk', residual, U * V)
        
        return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
    
    x0 = np.concatenate([U31.flatten(), V31.flatten(), W31.flatten()])
    
    # First pass with L-BFGS-B
    try:
        result = minimize(objective_polish, x0, method='L-BFGS-B',
                          options={'maxiter': 400, 'ftol': 1e-16, 'gtol': 1e-13})
        x0 = result.x
    except Exception:
        pass
    
    # Second pass with tighter tolerance and gradient
    if time.time() - start_time < time_budget * 0.6:
        try:
            result = minimize(objective_polish, x0, jac=gradient_polish, method='L-BFGS-B',
                              options={'maxiter': 300, 'ftol': 1e-17, 'gtol': 1e-15})
            x0 = result.x
        except Exception:
            pass
    
    U_flat = x0[:nm * rank]
    V_flat = x0[nm * rank : nm * rank + mp * rank]
    W_flat = x0[nm * rank + mp * rank : nm * rank + mp * rank + pn * rank]
    U = U_flat.reshape(nm, rank)
    V = V_flat.reshape(mp, rank)
    W = W_flat.reshape(pn, rank)
    
    score = _compute_residual(U, V, W, n, m, p)
    if score < 1e-10:
        return U, V, W, score
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # First, build the rank-32 solution as fallback
    U32, V32, W32 = _build_rank32()
    
    # Convert to lists for return
    def to_list(arr):
        return arr.tolist()
    
    # Try to find rank-31 using direct optimization with time budget
    remaining_time = deadline - time.time()
    if remaining_time > 30:
        search_budget = min(remaining_time - 25, 90)  # Leave 25s safety margin
        result = _try_rank31_search(U32, V32, W32, n, m, p, search_budget)
        if result is not None:
            U31, V31, W31, score = result
            if score < 1e-6:
                # Try to polish for even higher precision
                remaining2 = deadline - time.time()
                if remaining2 > 15:
                    polished = _polish_rank31(U31, V31, W31, n, m, p, min(remaining2 - 10, 45))
                    if polished is not None:
                        U31p, V31p, W31p, scorep = polished
                        if time.time() < deadline - 5:
                            return {"U": to_list(U31p), "V": to_list(V31p), "W": to_list(W31p)}
                if time.time() < deadline - 5:
                    return {"U": to_list(U31), "V": to_list(V31), "W": to_list(W31)}
    
    # Fall back to rank-32
    return {"U": to_list(U32), "V": to_list(V32), "W": to_list(W32)}

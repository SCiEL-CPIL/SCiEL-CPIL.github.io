import numpy as np
from fractions import Fraction
import time

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

def _verify_decomposition_float(U, V, W, n, m, p, tol=1e-10):
    """Fast float verification."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                s = np.dot(U_arr[idx_u], V_arr[idx_v] * W_arr[idx_w])
                if abs(s - 1.0) > tol:
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
    U[0][r] = 1; U[4][r] = 1
    V[0][r] = 1; V[6][r] = 1
    W[0][r] = 1; W[3][r] = 1
    r += 1
    U[3][r] = 1; U[4][r] = 1
    V[0][r] = 1
    W[1][r] = 1; W[3][r] = -1
    r += 1
    U[0][r] = 1
    V[1][r] = 1; V[6][r] = -1
    W[2][r] = 1; W[3][r] = 1
    r += 1
    U[4][r] = 1
    V[5][r] = 1; V[0][r] = -1
    W[0][r] = 1; W[1][r] = 1
    r += 1
    U[0][r] = 1; U[1][r] = 1
    V[6][r] = 1
    W[0][r] = -1; W[2][r] = 1
    r += 1
    U[3][r] = 1; U[0][r] = -1
    V[0][r] = 1; V[1][r] = 1
    W[3][r] = 1
    r += 1
    U[1][r] = 1; U[4][r] = -1
    V[5][r] = 1; V[6][r] = 1
    W[0][r] = 1
    r += 1
    # Part 4: j=0,1, k=2,3 using Strassen (7 terms)
    U[0][r] = 1; U[4][r] = 1
    V[2][r] = 1; V[8][r] = 1
    W[4][r] = 1; W[7][r] = 1
    r += 1
    U[3][r] = 1; U[4][r] = 1
    V[2][r] = 1
    W[5][r] = 1; W[7][r] = -1
    r += 1
    U[0][r] = 1
    V[3][r] = 1; V[8][r] = -1
    W[6][r] = 1; W[7][r] = 1
    r += 1
    U[4][r] = 1
    V[7][r] = 1; V[2][r] = -1
    W[4][r] = 1; W[5][r] = 1
    r += 1
    U[0][r] = 1; U[1][r] = 1
    V[8][r] = 1
    W[4][r] = -1; W[6][r] = 1
    r += 1
    U[3][r] = 1; U[0][r] = -1
    V[2][r] = 1; V[3][r] = 1
    W[7][r] = 1
    r += 1
    U[1][r] = 1; U[4][r] = -1
    V[7][r] = 1; V[8][r] = 1
    W[4][r] = 1
    r += 1
    return U, V, W

def _residual_and_grad(x, rank, n, m, p):
    """Compute residual vector and its Jacobian for least_squares."""
    nm, mp, pn = n * m, m * p, p * n
    U = x[:rank*nm].reshape(nm, rank)
    V = x[rank*nm:rank*nm+rank*mp].reshape(mp, rank)
    W = x[rank*nm+rank*mp:].reshape(pn, rank)
    
    # Precompute index mappings
    # For each (i,j,k): u_idx=i*m+j, v_idx=j*p+k, w_idx=k*n+i
    u_indices = []
    v_indices = []
    w_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_indices.append(i*m+j)
                v_indices.append(j*p+k)
                w_indices.append(k*n+i)
    u_indices = np.array(u_indices)
    v_indices = np.array(v_indices)
    w_indices = np.array(w_indices)
    
    # Residual: res[idx] = sum_r U[u_idx,r]*V[v_idx,r]*W[w_idx,r] - 1
    U_sel = U[u_indices]  # (30, rank)
    V_sel = V[v_indices]  # (30, rank)
    W_sel = W[w_indices]  # (30, rank)
    res = np.einsum('ir,ir,ir->i', U_sel, V_sel, W_sel) - 1.0
    
    # Jacobian
    J = np.zeros((n*m*p, rank*(nm+mp+pn)))
    # d(res)/d(U[u_idx, r]) = V[v_idx, r]*W[w_idx, r]
    J[:, :rank] = V_sel * W_sel
    # d(res)/d(V[v_idx, r]) = U[u_idx, r]*W[w_idx, r]
    # Need to scatter: for each row idx, col (v_idx*rank + r)
    # J[idx, nm*rank + v_idx*rank + r] = U_sel[idx, r] * W_sel[idx, r]
    UW = U_sel * W_sel  # (30, rank)
    VU = V_sel * U_sel  # (30, rank)
    for idx in range(n*m*p):
        J[idx, nm*rank + v_indices[idx]*rank : nm*rank + v_indices[idx]*rank + rank] = UW[idx]
        J[idx, (nm+mp)*rank + w_indices[idx]*rank : (nm+mp)*rank + w_indices[idx]*rank + rank] = VU[idx]
    
    return res, J

def _try_rank_lower(target_rank, n, m, p, base_U=None, base_V=None, base_W=None, time_budget=40):
    """Attempt to find a rank-target_rank decomposition using least_squares with multiple starts."""
    nm, mp, pn = n * m, m * p, p * n
    n_params = target_rank * (nm + mp + pn)
    start_time = time.time()
    
    best_res = np.inf
    best_x = None
    
    def _make_x0(scale=0.1, seed=None):
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
        return rng.randn(n_params) * scale
    
    # Generate diverse initializations
    inits = []
    # Random with various scales
    for scale in [0.05, 0.1, 0.2, 0.5, 1.0]:
        for s in range(5):
            inits.append(_make_x0(scale, seed=s*1000 + int(scale*100) + target_rank))
    
    # Base solution truncation with perturbation
    if base_U is not None:
        base_rank = len(base_U[0])
        if base_rank > target_rank:
            U_base = np.array(base_U)[:, :target_rank]
            V_base = np.array(base_V)[:, :target_rank]
            W_base = np.array(base_W)[:, :target_rank]
            x0 = np.concatenate([U_base.flatten(), V_base.flatten(), W_base.flatten()])
            inits.append(x0)
            # Perturbed versions
            for pert in [0.01, 0.05, 0.1]:
                rng = np.random.RandomState(42)
                x0_pert = x0 + rng.randn(n_params) * pert
                inits.append(x0_pert)
    
    from scipy.optimize import least_squares
    
    for x0 in inits:
        if time.time() - start_time > time_budget:
            break
        try:
            result = least_squares(_residual_and_grad, x0, args=(target_rank, n, m, p),
                                   method='lm', max_nfev=5000)
            res_norm = np.linalg.norm(result.fun)
            if res_norm < best_res:
                best_res = res_norm
                best_x = result.x.copy()
                if best_res < 1e-10:
                    # Found a solution!
                    U = best_x[:target_rank*nm].reshape(nm, target_rank)
                    V = best_x[target_rank*nm:target_rank*nm+target_rank*mp].reshape(mp, target_rank)
                    W = best_x[target_rank*nm+target_rank*mp:].reshape(pn, target_rank)
                    # Try rounding to integers first, then half-integers
                    for mult in [1, 2, 4]:
                        U_r = np.round(U * mult) / mult
                        V_r = np.round(V * mult) / mult
                        W_r = np.round(W * mult) / mult
                        U_frac = [[Fraction(int(np.round(v * mult)), mult) for v in row] for row in U_r]
                        V_frac = [[Fraction(int(np.round(v * mult)), mult) for v in row] for row in V_r]
                        W_frac = [[Fraction(int(np.round(v * mult)), mult) for v in row] for row in W_r]
                        if _verify_decomposition(U_frac, V_frac, W_frac, n, m, p):
                            return (U_frac, V_frac, W_frac)
                    # If rounding fails, try the raw float values as fractions
                    U_frac = [[Fraction(v).limit_denominator(1000) for v in row] for row in U]
                    V_frac = [[Fraction(v).limit_denominator(1000) for v in row] for row in V]
                    W_frac = [[Fraction(v).limit_denominator(1000) for v in row] for row in W]
                    if _verify_decomposition(U_frac, V_frac, W_frac, n, m, p):
                        return (U_frac, V_frac, W_frac)
        except Exception:
            pass
    
    return None

def construct():
    n, m, p = 2, 3, 5
    start_time = time.time()
    deadline = 130  # Safety margin before 150s kill
    
    U28, V28, W28 = _build_rank28()
    if not _verify_decomposition(U28, V28, W28, n, m, p):
        raise ValueError("Rank 28 verification failed")
    
    best_U, best_V, best_W = U28, V28, W28
    best_rank = 28
    
    # Try rank 27
    if time.time() - start_time < deadline - 50:
        result = _try_rank_lower(27, n, m, p, base_U=U28, base_V=V28, base_W=W28, time_budget=40)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = 27
    
    # Try rank 26 if time permits
    if time.time() - start_time < deadline - 35 and best_rank > 26:
        result = _try_rank_lower(26, n, m, p, base_U=best_U, base_V=best_V, base_W=best_W, time_budget=30)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = 26
    
    return {"U": best_U, "V": best_V, "W": best_W}

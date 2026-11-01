import time
import random
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 2, 3]

def _verify(U, V, W, n, m, p, tol=1e-9):
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _round_half(mat):
    return [[round(x * 2) / 2.0 for x in row] for row in mat]

def _residual(U, V, W, n, m, p):
    r = len(U[0])
    res = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                res.append(s - 1.0)
    return np.array(res)

def _als_step_U(V, W, n, m, p, r):
    U = [[0.0]*r for _ in range(n*m)]
    for i in range(n):
        for j in range(m):
            A = np.zeros((p, r))
            for k in range(p):
                for c in range(r):
                    A[k, c] = V[j*p+k][c] * W[k*n+i][c]
            b = np.ones(p)
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            for c in range(r):
                U[i*m+j][c] = float(sol[c])
    return U

def _als_step_V(U, W, n, m, p, r):
    V = [[0.0]*r for _ in range(m*p)]
    for j in range(m):
        for k in range(p):
            A = np.zeros((n, r))
            for i in range(n):
                for c in range(r):
                    A[i, c] = U[i*m+j][c] * W[k*n+i][c]
            b = np.ones(n)
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            for c in range(r):
                V[j*p+k][c] = float(sol[c])
    return V

def _als_step_W(U, V, n, m, p, r):
    W = [[0.0]*r for _ in range(p*n)]
    for k in range(p):
        for i in range(n):
            A = np.zeros((m, r))
            for j in range(m):
                for c in range(r):
                    A[j, c] = U[i*m+j][c] * V[j*p+k][c]
            b = np.ones(m)
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            for c in range(r):
                W[k*n+i][c] = float(sol[c])
    return W

def _als_optimize(U0, V0, W0, n, m, p, r, iters, deadline):
    U, V, W = U0, V0, W0
    for _ in range(iters):
        if time.time() > deadline:
            break
        U = _als_step_U(V, W, n, m, p, r)
        V = _als_step_V(U, W, n, m, p, r)
        W = _als_step_W(U, V, n, m, p, r)
    return U, V, W

def _rank11_solution():
    n, m, p = 2, 2, 3
    rank = 11
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def set_u(i, j, r, val): U[i*m+j][r] = val
    def set_v(j, k, r, val): V[j*p+k][r] = val
    def set_w(k, i, r, val): W[k*n+i][r] = val

    set_u(0, 0, 0, 1); set_u(1, 1, 0, 1)
    set_v(0, 0, 0, 1); set_v(1, 1, 0, 1)
    set_w(0, 0, 0, 1); set_w(1, 1, 0, 1)

    set_u(1, 0, 1, 1); set_u(1, 1, 1, 1)
    set_v(0, 0, 1, 1)
    set_w(0, 1, 1, 1); set_w(1, 1, 1, -1)

    set_u(0, 0, 2, 1)
    set_v(0, 1, 2, 1); set_v(1, 1, 2, -1)
    set_w(1, 0, 2, 1); set_w(1, 1, 2, 1)

    set_u(1, 1, 3, 1)
    set_v(1, 0, 3, 1); set_v(0, 0, 3, -1)
    set_w(0, 0, 3, 1); set_w(0, 1, 3, 1)

    set_u(0, 0, 4, 1); set_u(0, 1, 4, 1)
    set_v(1, 1, 4, 1)
    set_w(0, 0, 4, -1); set_w(1, 0, 4, 1)

    set_u(1, 0, 5, 1); set_u(0, 0, 5, -1)
    set_v(0, 0, 5, 1); set_v(0, 1, 5, 1)
    set_w(1, 1, 5, 1)

    set_u(0, 1, 6, 1); set_u(1, 1, 6, -1)
    set_v(1, 0, 6, 1); set_v(1, 1, 6, 1)
    set_w(0, 0, 6, 1)

    set_u(0, 0, 7, 1)
    set_v(0, 2, 7, 1)
    set_w(2, 0, 7, 1)

    set_u(0, 1, 8, 1)
    set_v(1, 2, 8, 1)
    set_w(2, 0, 8, 1)

    set_u(1, 0, 9, 1)
    set_v(0, 2, 9, 1)
    set_w(2, 1, 9, 1)

    set_u(1, 1, 10, 1)
    set_v(1, 2, 10, 1)
    set_w(2, 1, 10, 1)

    return {"U": U, "V": V, "W": W}

def _svd_init_rank_r(n, m, p, r):
    """Use SVD on the flattened multiplication tensor to get a rank-r initialization."""
    # Build the full multiplication tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    
    # Reshape to a 2D matrix: (n*m) x (m*p*p*n)
    # Actually, for ALS-style init, we can just use random or structured init
    # Let's use a structured init based on the rank-11 solution truncated
    sol11 = _rank11_solution()
    U11 = np.array(sol11["U"], dtype=float)
    V11 = np.array(sol11["V"], dtype=float)
    W11 = np.array(sol11["W"], dtype=float)
    
    # Truncate to rank r
    U = U11[:, :r]
    V = V11[:, :r]
    W = W11[:, :r]
    
    return U.tolist(), V.tolist(), W.tolist()

def _try_rank10(deadline):
    """Try to find a rank-10 decomposition using multiple strategies."""
    n, m, p = 2, 2, 3
    r = 10
    
    # Strategy 1: Start from rank-11 truncated, run ALS
    U0, V0, W0 = _svd_init_rank_r(n, m, p, r)
    
    inits = [(U0, V0, W0)]
    
    # Strategy 2: Random initializations
    rng = np.random.RandomState(42)
    for _ in range(100):
        if time.time() > deadline:
            break
        U_rand = rng.uniform(-1, 1, (n*m, r)).tolist()
        V_rand = rng.uniform(-1, 1, (m*p, r)).tolist()
        W_rand = rng.uniform(-1, 1, (p*n, r)).tolist()
        inits.append((U_rand, V_rand, W_rand))
    
    # Strategy 3: Perturbed rank-11
    sol11 = _rank11_solution()
    U11 = np.array(sol11["U"], dtype=float)
    V11 = np.array(sol11["V"], dtype=float)
    W11 = np.array(sol11["W"], dtype=float)
    
    for _ in range(50):
        if time.time() > deadline:
            break
        U_pert = (U11[:, :r] + rng.normal(0, 0.1, (n*m, r))).tolist()
        V_pert = (V11[:, :r] + rng.normal(0, 0.1, (m*p, r))).tolist()
        W_pert = (W11[:, :r] + rng.normal(0, 0.1, (p*n, r))).tolist()
        inits.append((U_pert, V_pert, W_pert))
    
    best_residual = float('inf')
    best_U, best_V, best_W = None, None, None
    
    for U_init, V_init, W_init in inits:
        if time.time() > deadline:
            break
        U, V, W = _als_optimize(U_init, V_init, W_init, n, m, p, r, iters=80, deadline=deadline)
        
        # Round to half-integers
        Ur, Vr, Wr = _round_half(U), _round_half(V), _round_half(W)
        
        if _verify(Ur, Vr, Wr, n, m, p, tol=1e-9):
            return Ur, Vr, Wr
        
        # Track best residual
        res = _residual(U, V, W, n, m, p)
        res_norm = np.linalg.norm(res)
        if res_norm < best_residual:
            best_residual = res_norm
            best_U, best_V, best_W = U, V, W
    
    # Try scipy on the best ALS result
    if best_U is not None and time.time() < deadline:
        try:
            def _flatten(U, V, W):
                return np.concatenate([np.array(U).flatten(), np.array(V).flatten(), np.array(W).flatten()])
            
            def _unflatten(x):
                U = x[:n*m*r].reshape(n*m, r)
                V = x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                W = x[n*m*r+m*p*r:].reshape(p*n, r)
                return U, V, W
            
            def _objective(x):
                U, V, W = _unflatten(x)
                res = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(r))
                            res.append(s - 1.0)
                return res
            
            x0 = _flatten(best_U, best_V, best_W)
            result = least_squares(_objective, x0, method='lm', max_nfev=10000, xtol=1e-14, ftol=1e-14)
            U, V, W = _unflatten(result.x)
            Ur, Vr, Wr = _round_half(U.tolist()), _round_half(V.tolist()), _round_half(W.tolist())
            if _verify(Ur, Vr, Wr, n, m, p, tol=1e-9):
                return Ur, Vr, Wr
        except Exception:
            pass
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 150s kill
    n, m, p = DIMENSIONS
    
    best = _rank11_solution()
    
    # Try to find rank-10
    if time.time() < deadline:
        result10 = _try_rank10(deadline)
        if result10 is not None:
            best = {"U": result10[0], "V": result10[1], "W": result10[2]}
    
    return best

import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = 140

    def time_remaining():
        return DEADLINE - (time.time() - start_time)

    sol11 = _make_rank11()
    
    if time_remaining() > 40:
        sol10 = _search_rank10(n, m, p, time_remaining)
        if sol10 is not None:
            return sol10
    
    return sol11

def _make_rank11():
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

    set_u(0, 0, 7, 1); set_v(0, 2, 7, 1); set_w(2, 0, 7, 1)
    set_u(0, 1, 8, 1); set_v(1, 2, 8, 1); set_w(2, 0, 8, 1)
    set_u(1, 0, 9, 1); set_v(0, 2, 9, 1); set_w(2, 1, 9, 1)
    set_u(1, 1, 10, 1); set_v(1, 2, 10, 1); set_w(2, 1, 10, 1)

    return {"U": U, "V": V, "W": W}

def _search_rank10(n, m, p, time_remaining):
    rank = 10
    U_size, V_size, W_size = n*m, m*p, p*n
    
    sol11 = _make_rank11()
    U11 = np.array(sol11["U"], dtype=float)
    V11 = np.array(sol11["V"], dtype=float)
    W11 = np.array(sol11["W"], dtype=float)
    
    def solve_W(U, V):
        A_mat = np.zeros((n*m*p, W_size*rank))
        b = np.ones(n*m*p)
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        A_mat[eq_idx, (k*n+i)*rank + r] = U[i*m+j, r] * V[j*p+k, r]
                    eq_idx += 1
        W_flat, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b, rcond=None)
        W = W_flat.reshape(W_size, rank)
        residual = float(np.sum((A_mat @ W_flat - b)**2))
        return W, residual
    
    def objective_UV(x):
        U = x[:U_size*rank].reshape(U_size, rank)
        V = x[U_size*rank:].reshape(V_size, rank)
        W, residual = solve_W(U, V)
        return residual
    
    def verify(U, V, W, n_tests=5):
        for _ in range(n_tests):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_exp = A @ B
            C_comp = np.zeros((n, p))
            for r in range(rank):
                C_comp += W[:, r] * (np.dot(U[:, r], A.flatten()) * np.dot(V[:, r], B.flatten()))
            if not np.allclose(C_comp, C_exp, atol=1e-6):
                return False
        return True
    
    U0 = U11[:, :rank]
    V0 = V11[:, :rank]
    x0 = np.concatenate([U0.flatten(), V0.flatten()])
    
    best_obj = float('inf')
    best_x = None
    
    rng = np.random.RandomState(42)
    
    for attempt in range(20):
        if time_remaining() < 20:
            break
        
        scale = 0.15
        x = x0 + rng.randn(*x0.shape) * scale
        
        try:
            res = minimize(objective_UV, x, method='Nelder-Mead',
                          options={'maxiter': 150, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
        
        if best_obj < 1e-10:
            break
    
    if best_obj < 1e-8 and best_x is not None:
        U = best_x[:U_size*rank].reshape(U_size, rank)
        V = best_x[U_size*rank:].reshape(V_size, rank)
        W, _ = solve_W(U, V)
        
        if verify(U, V, W):
            U_r = np.round(U * 2) / 2
            V_r = np.round(V * 2) / 2
            W_r = np.round(W * 2) / 2
            if verify(U_r, V_r, W_r):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

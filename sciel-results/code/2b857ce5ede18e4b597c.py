import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 9]

def _build_schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def _solve_als(U, V, W, n, m, p, max_iter=100, tol=1e-9):
    r = U.shape[1]
    for it in range(max_iter):
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                idx_u = i * m + j
                A_U = np.zeros((p, r))
                for k in range(p):
                    idx_v = j * p + k
                    idx_w = k * n + i
                    A_U[k, :] = V[idx_v, :] * W[idx_w, :]
                u, residuals, rank, sv = np.linalg.lstsq(A_U, np.ones(p), rcond=None)
                U_new[idx_u, :] = u
        
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                idx_v = j * p + k
                A_V = np.zeros((n, r))
                for i in range(n):
                    idx_u = i * m + j
                    idx_w = k * n + i
                    A_V[i, :] = U_new[idx_u, :] * W[idx_w, :]
                v, residuals, rank, sv = np.linalg.lstsq(A_V, np.ones(n), rcond=None)
                V_new[idx_v, :] = v

        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                idx_w = k * n + i
                A_W = np.zeros((m, r))
                for j in range(m):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    A_W[j, :] = U_new[idx_u, :] * V_new[idx_v, :]
                w, residuals, rank, sv = np.linalg.lstsq(A_W, np.ones(m), rcond=None)
                W_new[idx_w, :] = w
        
        diff = np.max(np.abs(U_new - U) + np.abs(V_new - V) + np.abs(W_new - W))
        U, V, W = U_new, V_new, W_new
        if diff < tol:
            break
    return U, V, W

def _calculate_error(U, V, W, n, m, p):
    r = U.shape[1]
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                err += (val - 1.0) ** 2
    return np.sqrt(err)

def _try_rational(U, V, W, n, m, p):
    r = U.shape[1]
    U_rat = [[Fraction(0) for _ in range(r)] for _ in range(n*m)]
    V_rat = [[Fraction(0) for _ in range(r)] for _ in range(m*p)]
    W_rat = [[Fraction(0) for _ in range(r)] for _ in range(p*n)]
    
    def approx_frac(x):
        for d in [1, 2]:
            num = round(x * d)
            frac = Fraction(num, d)
            if abs(float(frac) - x) < 1e-4:
                return frac
        return None

    for i in range(n*m):
        for c in range(r):
            f = approx_frac(U[i, c])
            if f is None:
                return None
            U_rat[i][c] = f
            
    for i in range(m*p):
        for c in range(r):
            f = approx_frac(V[i, c])
            if f is None:
                return None
            V_rat[i][c] = f

    for i in range(p*n):
        for c in range(r):
            f = approx_frac(W[i, c])
            if f is None:
                return None
            W_rat[i][c] = f
            
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                s = Fraction(0)
                for c in range(r):
                    s += U_rat[idx_u][c] * V_rat[idx_v][c] * W_rat[idx_w][c]
                if s != 1:
                    return None
    return {"U": U_rat, "V": V_rat, "W": W_rat}

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Ranks to try
    ranks_to_try = [20, 24, 28, 32, 36, 40, 44, 48, 54]
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
            
        # Try multiple random restarts
        success = False
        for restart in range(10):
            if time.time() > deadline:
                break
                
            np.random.seed(restart * 100 + r)
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # Normalize
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            U_sol, V_sol, W_sol = _solve_als(U, V, W, n, m, p, max_iter=50)
            
            err = _calculate_error(U_sol, V_sol, W_sol, n, m, p)
            
            if err < 1e-5:
                rat_res = _try_rational(U_sol, V_sol, W_sol, n, m, p)
                if rat_res:
                    return rat_res
                # If rational fails, maybe try a bit more polish?
                # Or just continue.
                
    return _build_schoolbook()

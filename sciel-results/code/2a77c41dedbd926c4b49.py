"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    for R in range(15, 31):
        if time.time() > deadline:
            break
        sol = _find_decomposition(n, m, p, R, deadline)
        if sol is not None:
            U, V, W = sol
            return {"U": U, "V": V, "W": W}
    
    return _schoolbook(n, m, p)

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _find_decomposition(n, m, p, R, deadline):
    np.random.seed(999 + R)
    
    idx = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx.append((i*m+j, j*p+k, k*n+i))
    idx = np.array(idx)
    
    for trial in range(30):
        if time.time() > deadline:
            return None
        scale = 0.5 / (R ** 0.25)
        x = np.random.randn(n*m*R + m*p*R + p*n*R) * scale
        try:
            result = least_squares(_residuals_vec, x, args=(n, m, p, R, idx),
                                   method='lm', max_nfev=500, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        except Exception:
            continue
        if result.cost < 1e-10:
            U = result.x[:n*m*R].reshape(n*m, R)
            V = result.x[n*m*R:n*m*R+m*p*R].reshape(m*p, R)
            W = result.x[n*m*R+m*p*R:].reshape(p*n, R)
            if _verify(n, m, p, U, V, W, tol=1e-6):
                return (U.tolist(), V.tolist(), W.tolist())
    return None

def _residuals_vec(x, n, m, p, R, idx):
    U = x[:n*m*R].reshape(n*m, R)
    V = x[n*m*R:n*m*R+m*p*R].reshape(m*p, R)
    W = x[n*m*R+m*p*R:].reshape(p*n, R)
    U_sel = U[idx[:, 0], :]
    V_sel = V[idx[:, 1], :]
    W_sel = W[idx[:, 2], :]
    vals = np.sum(U_sel * V_sel * W_sel, axis=1)
    return vals - 1.0

def _verify(n, m, p, U, V, W, tol=1e-6):
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(len(U[0])))
                if abs(val - 1.0) > tol:
                    return False
    return True
# EVOLVE-BLOCK-END

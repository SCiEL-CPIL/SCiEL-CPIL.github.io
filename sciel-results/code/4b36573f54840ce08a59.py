import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 5]

def _verify(U, V, W, n, m, p, tol=1e-8):
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    val = float(np.dot(U[i*m+j] * W[k*n+i], V[jp*p+k]))
                    expected = 1.0 if j == jp else 0.0
                    if abs(val - expected) > tol:
                        return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 110

    # Rank-15 solution: U doesn't depend on i, so we save factor of n=2
    rank15 = m * p
    U15 = np.zeros((n*m, rank15))
    V15 = np.zeros((m*p, rank15))
    W15 = np.zeros((p*n, rank15))
    for j in range(m):
        for k in range(p):
            r = j * p + k
            for i in range(n):
                U15[i*m+j, r] = 1.0
            V15[j*p+k, r] = 1.0
            for i in range(n):
                W15[k*n+i, r] = 1.0

    assert _verify(U15, V15, W15, n, m, p)

    best_rank = rank15
    best_U, best_V, best_W = U15, V15, W15

    # Try to find lower rank via optimization
    for rank in range(10, rank15):
        if time.time() > deadline - 20:
            break

        found = False
        for trial in range(5):
            if time.time() > deadline - 20:
                break

            np.random.seed(trial * 1000 + rank * 7)
            U = np.random.randn(n*m, rank) * 0.5
            V = np.random.randn(m*p, rank) * 0.5
            W = np.random.randn(p*n, rank) * 0.5
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

            def make_residuals(rank_val):
                def residuals(params):
                    Uv = params[:n*m*rank_val].reshape(n*m, rank_val)
                    Vv = params[n*m*rank_val:n*m*rank_val+m*p*rank_val].reshape(m*p, rank_val)
                    Wv = params[n*m*rank_val+m*p*rank_val:].reshape(p*n, rank_val)
                    res = []
                    for i in range(n):
                        for j in range(m):
                            for jp in range(m):
                                for k in range(p):
                                    val = float(np.dot(Uv[i*m+j] * Wv[k*n+i], Vv[jp*p+k]))
                                    expected = 1.0 if j == jp else 0.0
                                    res.append(val - expected)
                    return np.array(res)
                return residuals

            try:
                result = least_squares(make_residuals(rank), params, max_nfev=2000, xtol=1e-14, ftol=1e-14, gtol=1e-14)
                U = result.x[:n*m*rank].reshape(n*m, rank)
                V = result.x[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
                W = result.x[n*m*rank+m*p*rank:].reshape(p*n, rank)
                if _verify(U, V, W, n, m, p, tol=1e-6):
                    found = True
                    break
            except Exception:
                pass

        if found:
            best_rank = rank
            best_U, best_V, best_W = U, V, W
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

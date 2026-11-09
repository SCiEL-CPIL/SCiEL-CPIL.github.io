import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 6]

def _schoolbook():
    n, m, p = DIMENSIONS
    r = n * m * p
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _mismatch_count(U, V, W, n, m, p, tol=1e-6):
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    T = np.einsum('it,jt,kt->ijk', Ua, Va, Wa)
    target = np.ones((n, m, p))
    return int(np.sum(np.abs(T - target) > tol))

def _verify(U, V, W, n, m, p, atol=1e-9):
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    T = np.einsum('it,jt,kt->ijk', Ua, Va, Wa)
    target = np.ones((n, m, p))
    return bool(np.allclose(T, target, atol=atol))

def _local_polish(U, V, W, n, m, p, max_iter=2000):
    """Try small half-integer moves to reduce mismatch count."""
    Ua = np.array(U, dtype=float).copy()
    Va = np.array(V, dtype=float).copy()
    Wa = np.array(W, dtype=float).copy()
    cur = _mismatch_count(Ua, Va, Wa, n, m, p)
    if cur == 0:
        return Ua.tolist(), Va.tolist(), Wa.tolist()
    r = Ua.shape[1]
    all_vars = [Ua, Va, Wa]
    for _ in range(max_iter):
        improved = False
        for mat in all_vars:
            for idx in range(mat.size):
                flat = mat.reshape(-1)
                base = flat[idx]
                for delta in (-0.5, 0.5):
                    new_val = base + delta
                    # snap to half-integer
                    new_val = np.round(new_val * 2) / 2
                    if new_val == base:
                        continue
                    flat[idx] = new_val
                    cnt = _mismatch_count(Ua, Va, Wa, n, m, p)
                    if cnt < cur:
                        cur = cnt
                        improved = True
                        if cur == 0:
                            return Ua.tolist(), Va.tolist(), Wa.tolist()
                        break
                    else:
                        flat[idx] = base
        if not improved:
            break
    if cur == 0:
        return Ua.tolist(), Va.tolist(), Wa.tolist()
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 110

    T = np.ones((n, m, p), dtype=float)

    def objective(params, r):
        U = params[:n * m * r].reshape(n * m, r)
        V = params[n * m * r : n * m * r + m * p * r].reshape(m * p, r)
        W = params[n * m * r + m * p * r :].reshape(p * n, r)
        Tc = np.einsum('it,jt,kt->ijk', U, V, W)
        diff = Tc - T
        return float(np.sum(diff * diff))

    def try_rank(r, n_restarts, maxiter=200, maxfun=1000):
        if time.time() - start_time > time_limit:
            return None
        for restart in range(n_restarts):
            if time.time() - start_time > time_limit:
                return None
            rng = np.random.RandomState(restart * 7919 + r * 131)
            U0 = rng.randint(-2, 3, size=(n * m, r)) * 0.5
            V0 = rng.randint(-2, 3, size=(m * p, r)) * 0.5
            W0 = rng.randint(-2, 3, size=(p * n, r)) * 0.5
            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            try:
                from scipy.optimize import minimize
                res = minimize(objective, x0, args=(r,), method='L-BFGS-B',
                               options={'maxiter': maxiter, 'maxfun': maxfun})
                if time.time() - start_time > time_limit:
                    return None
                if res.fun < 1e-4:
                    Uo = res.x[:n * m * r].reshape(n * m, r)
                    Vo = res.x[n * m * r : n * m * r + m * p * r].reshape(m * p, r)
                    Wo = res.x[n * m * r + m * p * r :].reshape(p * n, r)
                    Ui = np.round(Uo * 2) / 2
                    Vi = np.round(Vo * 2) / 2
                    Wi = np.round(Wo * 2) / 2
                    if _verify(Ui.tolist(), Vi.tolist(), Wi.tolist(), n, m, p):
                        return {"U": Ui.tolist(), "V": Vi.tolist(), "W": Wi.tolist()}
                    # Try local polish
                    polished = _local_polish(Ui.tolist(), Vi.tolist(), Wi.tolist(), n, m, p)
                    if polished is not None:
                        Ui2, Vi2, Wi2 = polished
                        if _verify(Ui2, Vi2, Wi2, n, m, p):
                            return {"U": Ui2, "V": Vi2, "W": Wi2}
            except Exception:
                pass
        return None

    for r in [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]:
        if time.time() - start_time > time_limit:
            break
        n_restarts = 50 if r == 21 else (30 if r <= 23 else 15)
        res = try_rank(r, n_restarts)
        if res is not None:
            return res

    return _schoolbook()

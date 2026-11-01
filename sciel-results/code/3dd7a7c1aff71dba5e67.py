import numpy as np
import time
import random
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = 2, 3, 3
    start = time.time()
    deadline = start + 130  # safety margin under 150s wall-clock

    # Precompute the 18 index triples (U-row, V-row, W-row)
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i * m + j, j * p + k, k * n + i))
    triples = np.array(triples)  # (18, 3)
    T18 = 18

    def residuals(x, R):
        U = x[0:6 * R].reshape(6, R)
        V = x[6 * R:6 * R + 9 * R].reshape(9, R)
        W = x[6 * R + 9 * R:6 * R + 15 * R].reshape(6, R)
        U_g = U[triples[:, 0], :]
        V_g = V[triples[:, 1], :]
        W_g = W[triples[:, 2], :]
        return (U_g * V_g * W_g).sum(axis=1) - 1.0

    def compute_s(Uh, Vh, Wh):
        return (Uh[triples[:, 0], :] * Vh[triples[:, 1], :] * Wh[triples[:, 2], :]).sum(axis=1)

    def coord_polish(Uh, Vh, Wh, R, max_iter=80):
        # Local exact search in half-unit integer space; target sum == 8.
        s = compute_s(Uh, Vh, Wh).astype(np.int64)
        err = int(((s - 8) ** 2).sum())
        if err == 0:
            return True
        mats = [(Uh, 0), (Vh, 1), (Wh, 2)]
        for _ in range(max_iter):
            improved = False
            for M, role in mats:
                rows, cols = M.shape
                for idx in range(rows * cols):
                    row = idx // cols
                    r = idx % cols
                    mask = triples[:, role] == row
                    if not mask.any():
                        continue
                    t_idx = np.where(mask)[0]
                    if role == 0:
                        c = Vh[triples[t_idx, 1], r] * Wh[triples[t_idx, 2], r]
                    elif role == 1:
                        c = Uh[triples[t_idx, 0], r] * Wh[triples[t_idx, 2], r]
                    else:
                        c = Uh[triples[t_idx, 0], r] * Vh[triples[t_idx, 1], r]
                    c = c.astype(np.int64)
                    base = M[idx]
                    best_delta = 0
                    best_err = err
                    for d in (-2, -1, 1, 2):
                        s_new = s.copy()
                        s_new[t_idx] += d * c
                        e = int(((s_new - 8) ** 2).sum())
                        if e < best_err:
                            best_err = e
                            best_delta = d
                    if best_delta != 0:
                        M[idx] = base + best_delta
                        s[t_idx] += best_delta * c
                        err = best_err
                        improved = True
                        if err == 0:
                            return True
            if not improved:
                break
        return err == 0

    def to_list(M):
        out = []
        for row in M:
            r2 = []
            for v in row:
                fv = v / 2.0
                if abs(fv - round(fv)) < 1e-9:
                    r2.append(int(round(fv)))
                else:
                    r2.append(fv)
            out.append(r2)
        return out

    def try_rank(R, budget):
        rng = random.Random(R * 7919 + 13)
        size = 21 * R
        end = min(time.time() + budget, deadline)
        best_cost = float('inf')
        attempts = 0
        while time.time() < end:
            attempts += 1
            init = np.array([rng.uniform(-1.0, 1.0) for _ in range(size)])
            try:
                res = least_squares(lambda x: residuals(x, R), init,
                                    method='lm', max_nfev=80)
            except Exception:
                continue
            if not np.all(np.isfinite(res.x)):
                continue
            best_cost = min(best_cost, float(res.cost))
            if res.cost < 1e-8:
                x = res.x
                U = x[0:6 * R].reshape(6, R)
                V = x[6 * R:6 * R + 9 * R].reshape(9, R)
                W = x[6 * R + 9 * R:6 * R + 15 * R].reshape(6, R)
                Uh = np.round(U * 2).astype(np.int64)
                Vh = np.round(V * 2).astype(np.int64)
                Wh = np.round(W * 2).astype(np.int64)
                ok = np.all(compute_s(Uh, Vh, Wh) == 8)
                if not ok:
                    ok = coord_polish(Uh, Vh, Wh, R)
                if ok:
                    return {"U": to_list(Uh), "V": to_list(Vh), "W": to_list(Wh)}
            # Early abort if clearly infeasible (residual stuck large)
            if attempts > 250 and best_cost > 1e-3:
                break
        return None

    best = None
    best_R = None
    # Try most-likely-feasible first, then attempt to improve (lower R).
    plan = [(16, 30), (15, 20), (14, 15), (17, 10)]
    for R, budget in plan:
        if time.time() >= deadline - 3:
            break
        sol = try_rank(R, budget)
        if sol is not None:
            if best is None or R < best_R:
                best, best_R = sol, R
            if R == 14:
                break

    if best is None:
        # Guaranteed fallback: schoolbook rank n*m*p = 18.
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    return best

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 595  # 5s safety margin before 600s kill

    n = 10
    ys = np.arange(n)
    xs = np.arange(n)
    X, Y = np.meshgrid(xs, ys)

    # Precompute bin indices
    XY_diff = (X - Y + (n - 1)).flatten()   # 0..18
    X_flat = X.flatten()                     # 0..9
    XY_sum = (X + Y).flatten()               # 0..18
    X2Y = (X + 2 * Y).flatten()              # 0..27

    def compute_ratio(P):
        P = np.maximum(P, 0.0)
        total = P.sum()
        if total <= 1e-9:
            return 0.0
        P = P / total
        Pf = P.flatten()

        def entropy(indices, minlength):
            dist = np.bincount(indices, weights=Pf, minlength=minlength)
            nz = dist[dist > 1e-15]
            if len(nz) == 0:
                return 0.0
            return -np.sum(nz * np.log2(nz))

        H_diff = entropy(XY_diff, 2 * n - 1)
        H_x = entropy(X_flat, n)
        H_sum = entropy(XY_sum, 2 * n - 1)
        H_2y = entropy(X2Y, 3 * n - 1)

        M = max(H_x, H_sum, H_2y)
        if M <= 1e-9:
            return 0.0
        return H_diff / M

    def objective(params):
        P = np.exp(params).reshape(n, n)
        total = P.sum()
        if total <= 1e-9:
            return 1e10
        P = P / total
        return -compute_ratio(P)

    # Generate diverse starting points
    def make_starting_points():
        starts = []
        # 1. Uniform on X-Y: P[y,x] ∝ 1/(n-|x-y|)
        P1 = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                P1[y, x] = 1.0 / (n - abs(x - y))
        P1 /= P1.sum()
        starts.append(np.log(np.maximum(P1, 1e-10)).flatten())

        # 2. Uniform
        P2 = np.ones((n, n)) / (n * n)
        starts.append(np.log(P2).flatten())

        # 3. Diagonal concentration (x=y)
        P3 = np.zeros((n, n))
        for d in range(n):
            for i in range(n - d):
                P3[i, i + d] = 1.0 / (n - d)
        P3 /= P3.sum()
        starts.append(np.log(np.maximum(P3, 1e-10)).flatten())

        # 4. Anti-diagonal concentration (x+y=9)
        P4 = np.zeros((n, n))
        for s in range(2 * n - 1):
            count = min(s + 1, 2 * n - 1 - s, n)
            for y in range(n):
                x = s - y
                if 0 <= x < n:
                    P4[y, x] = 1.0 / count
        P4 /= P4.sum()
        starts.append(np.log(np.maximum(P4, 1e-10)).flatten())

        # 5. Random Dirichlet starts
        np.random.seed(12345)
        for i in range(15):
            P_r = np.random.dirichlet(np.ones(n * n))
            P_r = P_r.reshape(n, n)
            starts.append(np.log(np.maximum(P_r, 1e-10)).flatten())

        # 6. Structured: weight by exp(-alpha * |x-y|) for various alpha
        for alpha in [0.5, 1.0, 2.0, 3.0]:
            P_a = np.zeros((n, n))
            for y in range(n):
                for x in range(n):
                    P_a[y, x] = np.exp(-alpha * abs(x - y))
            P_a /= P_a.sum()
            starts.append(np.log(np.maximum(P_a, 1e-10)).flatten())

        return starts

    starting_points = make_starting_points()

    best_ratio = 0.0
    P_best = np.ones((n, n)) / (n * n)

    # First pass: quick evaluation of all starting points
    for sp in starting_points:
        P = np.exp(sp).reshape(n, n)
        P = np.maximum(P, 0)
        P /= P.sum()
        r = compute_ratio(P)
        if r > best_ratio:
            best_ratio = r
            P_best = P.copy()

    # Second pass: gradient-based optimization from each starting point
    for idx, sp in enumerate(starting_points):
        if time.time() > deadline - 15:
            break

        try:
            result = minimize(objective, sp, method='L-BFGS-B',
                             options={'maxiter': 100, 'gtol': 1e-8})
            if result.fun < -best_ratio:
                P_opt = np.exp(result.x).reshape(n, n)
                P_opt = np.maximum(P_opt, 0)
                P_opt /= P_opt.sum()
                ratio = compute_ratio(P_opt)
                if ratio > best_ratio:
                    best_ratio = ratio
                    P_best = P_opt.copy()
        except Exception:
            pass

        # Also try Nelder-Mead from the L-BFGS-B result for refinement
        if time.time() > deadline - 10:
            break
        try:
            P_cur = np.exp(result.x).reshape(n, n) if hasattr(result, 'x') else None
            if P_cur is None:
                continue
            P_cur = np.maximum(P_cur, 0)
            P_cur /= P_cur.sum()
            params_cur = np.log(np.maximum(P_cur, 1e-10)).flatten()
            result2 = minimize(objective, params_cur, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10})
            if result2.fun < -best_ratio:
                P_opt = np.exp(result2.x).reshape(n, n)
                P_opt = np.maximum(P_opt, 0)
                P_opt /= P_opt.sum()
                ratio = compute_ratio(P_opt)
                if ratio > best_ratio:
                    best_ratio = ratio
                    P_best = P_opt.copy()
        except Exception:
            pass

    # Final refinement from the best solution
    if time.time() < deadline - 5:
        P_best_safe = np.maximum(P_best, 1e-10)
        params_best = np.log(P_best_safe).flatten()
        try:
            result = minimize(objective, params_best, method='L-BFGS-B',
                             options={'maxiter': 200, 'gtol': 1e-12})
            if result.fun < -best_ratio:
                P_opt = np.exp(result.x).reshape(n, n)
                P_opt = np.maximum(P_opt, 0)
                P_opt /= P_opt.sum()
                ratio = compute_ratio(P_opt)
                if ratio > best_ratio:
                    best_ratio = ratio
                    P_best = P_opt.copy()
        except Exception:
            pass

    # Final normalization and clipping
    P_best = np.maximum(P_best, 0)
    total = P_best.sum()
    if total <= 1e-9:
        P_best = np.ones((n, n)) / (n * n)
    else:
        P_best /= total

    return P_best
# EVOLVE-BLOCK-END

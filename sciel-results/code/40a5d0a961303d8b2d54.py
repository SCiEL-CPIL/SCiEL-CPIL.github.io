import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin before 600s kill (but evaluator kills at 630s)

    n = 10
    ys = np.arange(n)
    xs = np.arange(n)
    X, Y = np.meshgrid(xs, ys)

    # Precompute bin indices (flattened)
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
        for i in range(30):
            P_r = np.random.dirichlet(np.ones(n * n))
            P_r = P_r.reshape(n, n)
            starts.append(np.log(np.maximum(P_r, 1e-10)).flatten())

        # 6. Structured: weight by exp(-alpha * |x-y|) for various alpha
        for alpha in [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]:
            P_a = np.zeros((n, n))
            for y in range(n):
                for x in range(n):
                    P_a[y, x] = np.exp(-alpha * abs(x - y))
            P_a /= P_a.sum()
            starts.append(np.log(np.maximum(P_a, 1e-10)).flatten())

        # 7. Weighted by exp(-beta * (x+y)^2 / something) to spread anti-diagonals
        for beta in [0.1, 0.3, 0.5, 1.0, 2.0]:
            P_b = np.zeros((n, n))
            for y in range(n):
                for x in range(n):
                    P_b[y, x] = np.exp(-beta * (x + y - (n - 1)) ** 2 / (2 * (n - 1) ** 2))
            P_b /= P_b.sum()
            starts.append(np.log(np.maximum(P_b, 1e-10)).flatten())

        # 8. Mix of diagonal and anti-diagonal
        for t in [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]:
            P_mix = t * P1 + (1 - t) * P4
            P_mix /= P_mix.sum()
            starts.append(np.log(np.maximum(P_mix, 1e-10)).flatten())

        # 9. Checkerboard patterns
        for phase in [0, 1]:
            P_c = np.zeros((n, n))
            for y in range(n):
                for x in range(n):
                    if (x + y) % 2 == phase:
                        P_c[y, x] = 1.0
            P_c /= P_c.sum()
            starts.append(np.log(np.maximum(P_c, 1e-10)).flatten())

        # 10. Random Gaussian blobs
        for _ in range(15):
            cx = np.random.uniform(0, n-1)
            cy = np.random.uniform(0, n-1)
            sigma = np.random.uniform(0.5, 2.5)
            P_g = np.zeros((n, n))
            for y in range(n):
                for x in range(n):
                    P_g[y, x] = np.exp(-((x-cx)**2 + (y-cy)**2) / (2*sigma**2))
            P_g /= P_g.sum()
            starts.append(np.log(np.maximum(P_g, 1e-10)).flatten())

        # 11. Concentrate on specific x-y diagonals with varying intensity
        #     Try to maximize H_diff while keeping M small
        for target_diff in range(-4, 5):
            P_d = np.zeros((n, n))
            for y in range(n):
                for x in range(n):
                    d = x - y
                    # Weight more on cells where x-y is close to target_diff
                    P_d[y, x] = np.exp(-((d - target_diff) ** 2) / 4.0)
            P_d /= P_d.sum()
            starts.append(np.log(np.maximum(P_d, 1e-10)).flatten())

        # 12. Edge concentration
        for edge in ['top', 'bottom', 'left', 'right']:
            P_e = np.zeros((n, n))
            for y in range(n):
                for x in range(n):
                    if edge == 'top':
                        P_e[y, x] = np.exp(-y)
                    elif edge == 'bottom':
                        P_e[y, x] = np.exp(-(n-1-y))
                    elif edge == 'left':
                        P_e[y, x] = np.exp(-x)
                    elif edge == 'right':
                        P_e[y, x] = np.exp(-(n-1-x))
            P_e /= P_e.sum()
            starts.append(np.log(np.maximum(P_e, 1e-10)).flatten())

        # 13. Cross pattern (center row and column)
        P_cross = np.zeros((n, n))
        for i in range(n):
            P_cross[i, i] = 1.0  # diagonal
        for i in range(n):
            P_cross[i, n//2] = 0.5  # middle column
            P_cross[n//2, i] = 0.5  # middle row
        P_cross /= P_cross.sum()
        starts.append(np.log(np.maximum(P_cross, 1e-10)).flatten())

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
    # Use L-BFGS-B primarily, with Nelder-Mead for refinement
    for idx, sp in enumerate(starting_points):
        if time.time() > deadline - 30:
            break

        try:
            result = minimize(objective, sp, method='L-BFGS-B',
                             options={'maxiter': 200, 'gtol': 1e-12, 'maxls': 50})
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

        # Nelder-Mead refinement from the L-BFGS-B result
        if time.time() > deadline - 20:
            break
        try:
            P_cur = np.exp(result.x).reshape(n, n)
            P_cur = np.maximum(P_cur, 0)
            P_cur /= P_cur.sum()
            params_cur = np.log(np.maximum(P_cur, 1e-10)).flatten()
            result2 = minimize(objective, params_cur, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-14, 'fatol': 1e-14,
                                      'maxfev': 15000, 'adaptive': True})
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

    # Differential evolution for global exploration (if time permits)
    if time.time() < deadline - 80:
        try:
            bounds = [(-12, 12)] * (n * n)
            de_result = differential_evolution(objective, bounds, seed=42,
                                               maxiter=80, popsize=20,
                                               tol=1e-10, mutation=(0.5, 1.5),
                                               recombination=0.8,
                                               init='latinhypercube',
                                               polish=True,
                                               updating='immediate')
            if de_result.fun < -best_ratio:
                P_opt = np.exp(de_result.x).reshape(n, n)
                P_opt = np.maximum(P_opt, 0)
                P_opt /= P_opt.sum()
                ratio = compute_ratio(P_opt)
                if ratio > best_ratio:
                    best_ratio = ratio
                    P_best = P_opt.copy()
        except Exception:
            pass

    # Final refinement from the best solution
    if time.time() < deadline - 15:
        P_best_safe = np.maximum(P_best, 1e-10)
        params_best = np.log(P_best_safe).flatten()
        try:
            result = minimize(objective, params_best, method='L-BFGS-B',
                             options={'maxiter': 500, 'gtol': 1e-15, 'maxls': 100})
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

        # One more Nelder-Mead polish
        if time.time() < deadline - 8:
            try:
                P_cur = np.maximum(P_best, 1e-10)
                params_cur = np.log(P_cur).flatten()
                result3 = minimize(objective, params_cur, method='Nelder-Mead',
                                 options={'maxiter': 1000, 'xatol': 1e-16, 'fatol': 1e-16,
                                          'maxfev': 30000, 'adaptive': True})
                if result3.fun < -best_ratio:
                    P_opt = np.exp(result3.x).reshape(n, n)
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

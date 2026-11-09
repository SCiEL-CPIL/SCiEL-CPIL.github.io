import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # 40s safety margin before 630s kill

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

    def make_seeds():
        seeds = []
        
        # 1. Uniform
        P1 = np.ones((n, n)) / (n * n)
        seeds.append(np.log(P1).flatten())

        # 2. Anti-diagonal (x+y=9)
        P2 = np.zeros((n, n))
        for y in range(n):
            x = n - 1 - y
            P2[y, x] = 1.0
        P2 /= P2.sum()
        seeds.append(np.log(np.maximum(P2, 1e-10)).flatten())

        # 3. Two anti-diagonals (x+y=8, 9)
        P3 = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                if x + y in [8, 9]:
                    P3[y, x] = 1.0
        P3 /= P3.sum()
        seeds.append(np.log(np.maximum(P3, 1e-10)).flatten())

        # 4. Modular 4x4 structure
        # Concentrate on x % 4 == 0 and (x+y) % 4 == 0
        P4 = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                if x % 4 == 0 and (x + y) % 4 == 0:
                    P4[y, x] = 1.0
        if P4.sum() > 0:
            P4 /= P4.sum()
            seeds.append(np.log(np.maximum(P4, 1e-10)).flatten())
        
        # 5. Modular 3x3 structure
        P5 = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                if x % 3 == 0 and (x + y) % 3 == 0:
                    P5[y, x] = 1.0
        if P5.sum() > 0:
            P5 /= P5.sum()
            seeds.append(np.log(np.maximum(P5, 1e-10)).flatten())

        # 6. Parametric: exp(-a(x mod 4)^2 - b((x+y) mod 4)^2)
        for a in [0.5, 1.0, 2.0]:
            for b in [0.5, 1.0, 2.0]:
                P6 = np.zeros((n, n))
                for y in range(n):
                    for x in range(n):
                        P6[y, x] = np.exp(-a * (x % 4)**2 - b * ((x + y) % 4)**2)
                P6 /= P6.sum()
                seeds.append(np.log(np.maximum(P6, 1e-10)).flatten())

        # 7. Random Dirichlet
        np.random.seed(42)
        for _ in range(10):
            Pr = np.random.dirichlet(np.ones(n * n))
            Pr = Pr.reshape(n, n)
            seeds.append(np.log(np.maximum(Pr, 1e-10)).flatten())

        return seeds

    seeds = make_seeds()
    
    best_ratio = 0.0
    P_best = np.ones((n, n)) / (n * n)

    # Evaluate seeds
    for s in seeds:
        P = np.exp(s).reshape(n, n)
        P = np.maximum(P, 0)
        P /= P.sum()
        r = compute_ratio(P)
        if r > best_ratio:
            best_ratio = r
            P_best = P.copy()

    # Local optimization from best seeds
    # Sort seeds by their initial ratio to prioritize good ones
    seed_ratios = []
    for s in seeds:
        P = np.exp(s).reshape(n, n)
        P = np.maximum(P, 0)
        P /= P.sum()
        r = compute_ratio(P)
        seed_ratios.append((r, s))
    seed_ratios.sort(reverse=True)

    for r, s in seed_ratios:
        if time.time() > deadline - 30:
            break
        try:
            # L-BFGS-B
            res = minimize(objective, s, method='L-BFGS-B', options={'maxiter': 300, 'gtol': 1e-10})
            if res.fun < -best_ratio:
                P_opt = np.exp(res.x).reshape(n, n)
                P_opt = np.maximum(P_opt, 0)
                P_opt /= P_opt.sum()
                ratio = compute_ratio(P_opt)
                if ratio > best_ratio:
                    best_ratio = ratio
                    P_best = P_opt.copy()
        except Exception:
            pass

        # Nelder-Mead polish
        if time.time() > deadline - 20:
            break
        try:
            P_cur = np.exp(res.x).reshape(n, n)
            P_cur = np.maximum(P_cur, 0)
            P_cur /= P_cur.sum()
            params_cur = np.log(np.maximum(P_cur, 1e-10)).flatten()
            res2 = minimize(objective, params_cur, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12, 'maxfev': 20000})
            if res2.fun < -best_ratio:
                P_opt = np.exp(res2.x).reshape(n, n)
                P_opt = np.maximum(P_opt, 0)
                P_opt /= P_opt.sum()
                ratio = compute_ratio(P_opt)
                if ratio > best_ratio:
                    best_ratio = ratio
                    P_best = P_opt.copy()
        except Exception:
            pass

    # Final refinement from global best
    if time.time() < deadline - 10:
        P_best_safe = np.maximum(P_best, 1e-10)
        params_best = np.log(P_best_safe).flatten()
        try:
            res = minimize(objective, params_best, method='L-BFGS-B', options={'maxiter': 500, 'gtol': 1e-12})
            if res.fun < -best_ratio:
                P_opt = np.exp(res.x).reshape(n, n)
                P_opt = np.maximum(P_opt, 0)
                P_opt /= P_opt.sum()
                ratio = compute_ratio(P_opt)
                if ratio > best_ratio:
                    best_ratio = ratio
                    P_best = P_opt.copy()
        except Exception:
            pass

    P_best = np.maximum(P_best, 0)
    total = P_best.sum()
    if total <= 1e-9:
        P_best = np.ones((n, n)) / (n * n)
    else:
        P_best /= total

    return P_best

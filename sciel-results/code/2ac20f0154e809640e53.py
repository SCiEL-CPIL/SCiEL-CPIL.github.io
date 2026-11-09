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

    def fast_local_search(P_init, max_iters=2000, tol=1e-12):
        """Fast coordinate-free local search using Nelder-Mead with adaptive step size."""
        P_init = np.maximum(P_init, 1e-15)
        P_init = P_init / P_init.sum()
        params = np.log(P_init).flatten()
        
        best_P = P_init.copy()
        best_ratio = compute_ratio(P_init)
        
        # Use Nelder-Mead for local refinement
        try:
            res = minimize(
                objective, params, method='Nelder-Mead',
                options={'maxiter': max_iters, 'xatol': 1e-14, 'fatol': 1e-14, 'maxfev': max_iters * 2}
            )
            P_opt = np.exp(res.x).reshape(n, n)
            P_opt = np.maximum(P_opt, 0)
            P_opt /= P_opt.sum()
            ratio = compute_ratio(P_opt)
            if ratio > best_ratio:
                best_ratio = ratio
                best_P = P_opt.copy()
        except Exception:
            pass
        
        return best_P, best_ratio

    def make_seeds():
        seeds = []
        
        # 1. Uniform
        P1 = np.ones((n, n)) / (n * n)
        seeds.append(np.log(P1).flatten())

        # 2. Anti-diagonal (x+y=9) - maximizes H_diff concentration
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

        # 4. Three anti-diagonals (x+y=7, 8, 9)
        P4 = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                if x + y in [7, 8, 9]:
                    P4[y, x] = 1.0
        P4 /= P4.sum()
        seeds.append(np.log(np.maximum(P4, 1e-10)).flatten())

        # 5. Diagonal (x=y)
        P5 = np.zeros((n, n))
        for i in range(n):
            P5[i, i] = 1.0
        P5 /= P5.sum()
        seeds.append(np.log(np.maximum(P5, 1e-10)).flatten())

        # 6. Near-diagonal (|x-y|<=1)
        P6 = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                if abs(x - y) <= 1:
                    P6[y, x] = 1.0
        P6 /= P6.sum()
        seeds.append(np.log(np.maximum(P6, 1e-10)).flatten())

        # 7. Modular structure: x%k == y%k for various k
        for k in [2, 3, 4, 5]:
            Pk = np.zeros((n, n))
            for y in range(n):
                for x in range(n):
                    if x % k == y % k:
                        Pk[y, x] = 1.0
            if Pk.sum() > 0:
                Pk /= Pk.sum()
                seeds.append(np.log(np.maximum(Pk, 1e-10)).flatten())

        # 8. Parametric: exp(-a*(x-y)^2 - b*(x+y-c)^2)
        for a in [0.5, 1.0, 2.0, 4.0]:
            for b in [0.5, 1.0, 2.0, 4.0]:
                for c in [4, 5, 6, 7, 8, 9]:
                    Pa = np.zeros((n, n))
                    for y in range(n):
                        for x in range(n):
                            Pa[y, x] = np.exp(-a * (x - y)**2 - b * (x + y - c)**2)
                    if Pa.sum() > 1e-10:
                        Pa /= Pa.sum()
                        seeds.append(np.log(np.maximum(Pa, 1e-10)).flatten())

        # 9. Rank-based: weight by (x-y+9) to spread H_diff
        P9 = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                P9[y, x] = x - y + n - 1  # 0 to 18
        P9 /= P9.sum()
        seeds.append(np.log(np.maximum(P9, 1e-10)).flatten())

        # 10. Inverse rank: concentrate on middle diagonals
        P10 = np.zeros((n, n))
        for y in range(n):
            for x in range(n):
                d = abs(x - y)
                P10[y, x] = np.exp(-0.3 * d**2)
        P10 /= P10.sum()
        seeds.append(np.log(np.maximum(P10, 1e-10)).flatten())

        # 11. Random Dirichlet with different concentration parameters
        np.random.seed(42)
        for alpha in [0.5, 1.0, 2.0, 5.0]:
            for _ in range(5):
                Pr = np.random.dirichlet(np.ones(n * n) * alpha)
                Pr = Pr.reshape(n, n)
                seeds.append(np.log(np.maximum(Pr, 1e-10)).flatten())

        # 12. Structured random: random weights on diagonals
        for _ in range(10):
            np.random.seed(np.random.randint(10000))
            diag_weights = np.random.exponential(1.0, size=2 * n - 1)
            P12 = np.zeros((n, n))
            for y in range(n):
                for x in range(n):
                    d = x - y + n - 1
                    P12[y, x] = diag_weights[d]
            if P12.sum() > 1e-10:
                P12 /= P12.sum()
                seeds.append(np.log(np.maximum(P12, 1e-10)).flatten())

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

    # Sort seeds by their initial ratio to prioritize good ones
    seed_ratios = []
    for s in seeds:
        P = np.exp(s).reshape(n, n)
        P = np.maximum(P, 0)
        P /= P.sum()
        r = compute_ratio(P)
        seed_ratios.append((r, s))
    seed_ratios.sort(reverse=True)

    # Local optimization from top seeds
    top_k = min(15, len(seed_ratios))
    for i in range(top_k):
        if time.time() > deadline - 30:
            break
        r, s = seed_ratios[i]
        P_init = np.exp(s).reshape(n, n)
        P_init = np.maximum(P_init, 0)
        P_init /= P_init.sum()
        
        try:
            P_opt, ratio = fast_local_search(P_init, max_iters=1500)
            if ratio > best_ratio:
                best_ratio = ratio
                P_best = P_opt.copy()
        except Exception:
            pass

    # Global search with differential evolution if time permits
    if time.time() < deadline - 60:
        try:
            # Bound parameters to avoid extreme values
            bounds = [(-20, 20)] * (n * n)
            
            def de_objective(params):
                P = np.exp(params).reshape(n, n)
                total = P.sum()
                if total <= 1e-9:
                    return 1e10
                P = P / total
                return -compute_ratio(P)
            
            # Use multiple starting points for DE
            best_de_ratio = best_ratio
            best_de_P = P_best.copy()
            
            n_pop = 20
            max_iter = 30
            
            # Run DE with time check
            for restart in range(3):
                if time.time() > deadline - 30:
                    break
                
                # Random initial population
                np.random.seed(42 + restart * 100)
                x0 = np.random.uniform(-15, 15, size=n * n)
                
                try:
                    res = differential_evolution(
                        de_objective, bounds, x0=x0,
                        popsize=n_pop, maxiter=max_iter,
                        tol=1e-8, seed=42 + restart * 100,
                        polish=False, init='latinhypercube'
                    )
                    
                    P_opt = np.exp(res.x).reshape(n, n)
                    P_opt = np.maximum(P_opt, 0)
                    P_opt /= P_opt.sum()
                    ratio = compute_ratio(P_opt)
                    
                    if ratio > best_de_ratio:
                        best_de_ratio = ratio
                        best_de_P = P_opt.copy()
                except Exception:
                    pass
            
            if best_de_ratio > best_ratio:
                best_ratio = best_de_ratio
                P_best = best_de_P.copy()
        except Exception:
            pass

    # Final refinement from global best
    if time.time() < deadline - 10:
        try:
            P_opt, ratio = fast_local_search(P_best, max_iters=3000)
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

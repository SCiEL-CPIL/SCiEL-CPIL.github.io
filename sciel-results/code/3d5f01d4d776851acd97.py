"""Combined optimizer for BBOB-style f13 (Sharp ridge)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Sobol broad exploration + adaptive (1+1)-ES exploitation + BFGS polish.

    Structural idea (current program): deterministic low-discrepancy Sobol
    initialization for early, broad coverage (AOCC-friendly), plus a deadline
    guard and explicit budget tracking.
    Refinement machinery (inspiration 2): adaptive sigma with a 1/5-style
    success rule and bounded stall restarts, so exploitation keeps moving
    along the sharp ridge instead of stalling.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 275.0  # safety margin before the 300s kill
        n = self.dim
        budget = self.budget

        def ok():
            return time.time() < deadline

        rng = np.random.default_rng(0)

        # --- Phase 1: deterministic Sobol broad exploration for early AOCC ---
        try:
            from scipy.stats import qmc
            sobol = qmc.Sobol(d=n, scramble=True, seed=42)
            sobol.random(1)  # skip origin
            n_init = min(48, budget // 4)
            init = sobol.random(n_init) * 10.0 - 5.0
        except Exception:
            # Fallback: stratified grid-like deterministic points
            n_init = min(36, budget // 4)
            g = np.linspace(-4.0, 4.0, 3)  # 3^5 = 243 cells, sample 36
            init = np.array([[g[(i // (3 ** (n - 1))) % 3],
                              g[(i // (3 ** (n - 2))) % 3],
                              g[(i // (3 ** (n - 3))) % 3],
                              g[(i // (3 ** (n - 4))) % 3],
                              g[i % 3]] for i in range(n_init)])

        best_x = None
        best_f = np.inf
        calls = 0

        # Track a small set of local bests (one per Sobol "cell") to give
        # exploitation diverse, well-distributed starting points.
        local_best = []  # list of (f, x)
        for i in range(n_init):
            if calls >= budget or not ok():
                break
            x = init[i]
            f = float(func(x))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            # Keep up to 6 diverse local minima (dedup by rough region)
            if len(local_best) < 6:
                local_best.append((f, x.copy()))
            else:
                # replace the worst local best if this is much better
                worst = max(range(len(local_best)), key=lambda k: local_best[k][0])
                if f < local_best[worst][0] - 1e-6:
                    local_best[worst] = (f, x.copy())

        if best_x is None:
            best_x = np.zeros(n)
            best_f = float(func(best_x))
            calls += 1

        # --- Phase 2: adaptive (1+1)-ES exploitation with stall restarts ---
        # Start from the best point; sigma adapts via a success rule.
        x = best_x.copy()
        fx = best_f
        sigma = 1.5
        no_improve = 0
        max_no_improve = 25
        stall_restarts = 0
        max_stall_restarts = 8

        while calls < budget and ok():
            y = x + sigma * rng.standard_normal(n)
            y = np.clip(y, -5.0, 5.0)
            fy = float(func(y))
            calls += 1

            if fy < fx:
                x = y
                fx = fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                sigma *= 1.15
                no_improve = 0
            else:
                sigma *= 0.85
                no_improve += 1

            if sigma < 1e-7:
                sigma = 0.3

            # Stall handling: try a few bounded random restarts.
            if no_improve >= max_no_improve:
                improved = False
                for _ in range(min(3, budget - calls)):
                    if not ok() or calls >= budget:
                        break
                    # Restart: either fully random or a local best from phase 1
                    if rng.random() < 0.5 and local_best:
                        f0, x0 = local_best[int(rng.integers(len(local_best)))]
                        x_new = np.clip(x0 + 0.5 * rng.standard_normal(n), -5, 5)
                    else:
                        x_new = rng.uniform(-5, 5, n)
                    fx_new = float(func(x_new))
                    calls += 1
                    if fx_new < fx:
                        x = x_new
                        fx = fx_new
                        if fx_new < best_f:
                            best_f = fx_new
                            best_x = x.copy()
                        improved = True
                        break
                if improved:
                    no_improve = 0
                    sigma = 1.0
                else:
                    sigma *= 0.5
                    no_improve = 0
                    stall_restarts += 1
                    if stall_restarts >= max_stall_restarts:
                        break

        # --- Phase 3: BFGS polish from the overall best (memory-light) ---
        if calls < budget - 10 and ok():
            try:
                from scipy.optimize import minimize

                def objective(xx):
                    nonlocal calls
                    if calls >= budget:
                        return best_f
                    f = float(func(xx))
                    calls += 1
                    return f

                res = minimize(objective, best_x, method='BFGS',
                               options={'maxiter': 40, 'gtol': 1e-7})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.asarray(res.x, dtype=float)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END

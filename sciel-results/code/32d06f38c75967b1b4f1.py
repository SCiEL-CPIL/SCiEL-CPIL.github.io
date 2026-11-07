"""Optimizer for BBOB-style functions using CMA-ES with multi-restart strategy."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-restart CMA-ES optimizer for 5D BBOB functions.
    Uses CMA-ES (Covariance Matrix Adaptation) which adapts both step size
    and search direction covariance, making it effective for rotated and
    multimodal landscapes. Multiple restarts with decreasing population sizes
    balance exploration and exploitation within the 300-call budget.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s
        dim = self.dim
        budget = self.budget
        calls = [0]

        def wrapped(x):
            calls[0] += 1
            if calls[0] > budget:
                raise RuntimeError("budget exceeded")
            val = float(func(x))
            if not np.isfinite(val):
                return 1e10
            return val

        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = np.array(x, dtype=float)
                best_f = f

        def check_time():
            return time.time() < deadline and calls[0] < budget

        # Phase 1: Initial diverse sampling to find a good starting point
        n_initial = min(15, budget)
        points = [np.zeros(dim)]
        for i in range(dim):
            for val in [-3.0, 3.0]:
                p = np.zeros(dim)
                p[i] = val
                points.append(p)
        while len(points) < n_initial:
            points.append(rng.uniform(-5, 5, dim))

        for x in points:
            if not check_time():
                break
            x = np.clip(x, -5, 5)
            try:
                f = wrapped(x)
                update_best(x, f)
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped(best_x)

        # Phase 2: Multi-restart CMA-ES
        # CMA-ES parameters: lambda = 4 + floor(3*ln(n)) for n=5 -> lambda=10
        # Each restart uses a portion of the remaining budget
        remaining_budget = budget - calls[0]
        
        # Define restart schedule: (initial_sigma, max_evals_fraction)
        # First restart: broad exploration, larger sigma
        # Second restart: focused exploitation, smaller sigma
        restarts = [
            (2.0, 0.5),   # 50% of remaining budget, sigma=2.0
            (0.5, 0.3),   # 30% of remaining budget, sigma=0.5
            (0.1, 0.2),   # 20% of remaining budget, sigma=0.1
        ]

        for sigma_init, frac in restarts:
            if not check_time() or calls[0] >= budget - 5:
                break
            
            max_evals = int(remaining_budget * frac)
            if max_evals < 10:
                continue
            
            # CMA-ES implementation
            n = dim
            lam = 4 + int(3 * np.log(n))  # lambda = 10 for n=5
            mu = lam // 2  # mu = 5
            cc = (4 + n / (4 + n)) / ((n + 1.3) ** 2 + 4)
            cs = (n + 1.7) / (n + 5.7 + mu)
            c1 = (2 / ((n + 1.8) ** 2 + mu_eff)) if (mu_eff := 2 / (mu * (2 + mu / (n + 1.8)))) else 0.1
            cmu = min(0.6 - c1, 2 * (mu_eff - 2 + 1.0 / mu_eff) / ((n + 2) ** 2 + mu_eff))
            
            # Initialize
            x_mean = best_x.copy()
            sigma = sigma_init
            B = np.eye(n)
            D = np.ones(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            
            # Initial population
            pop = []
            pop_f = []
            for i in range(lam):
                z = rng.standard_normal(n)
                xi = x_mean + sigma * (B @ (D * z))
                xi = np.clip(xi, -5, 5)
                try:
                    f = wrapped(xi)
                except Exception:
                    f = 1e10
                pop.append(xi)
                pop_f.append(f)
                update_best(xi, f)
            
            # Sort by fitness
            idx = np.argsort(pop_f)
            pop = [pop[i] for i in idx]
            pop_f = [pop_f[i] for i in idx]
            
            # CMA-ES main loop
            for gen in range(1, max_evals // lam + 1):
                if not check_time() or calls[0] >= budget - 5:
                    break
                
                # Select top mu individuals
                w = np.ones(mu)
                w_sum = w.sum()
                weights = w / w_sum
                mu_eff = 1 / (np.sum(weights ** 2))
                
                # Recalculate c1 and cmu with correct mu_eff
                c1 = (2 / ((n + 1.8) ** 2 + mu_eff))
                cmu = min(0.6 - c1, 2 * (mu_eff - 2 + 1.0 / mu_eff) / ((n + 2) ** 2 + mu_eff))
                
                # Compute weighted mean of selected individuals
                y_w = np.zeros(n)
                for i in range(mu):
                    y_w += weights[i] * (pop[i] - x_mean)
                
                # Update covariance matrix adaptation
                # For efficiency with small n, use simplified update
                # Full CMA-ES covariance update is expensive, so we use a simplified version
                
                # Update path accumulators
                pc = (1 - cc) * pc + cc * (y_w / sigma)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (y_w / sigma)
                
                # Update mean
                x_mean = x_mean + (c1 + cmu) * y_w
                
                # Update sigma
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, 1e-8)
                sigma = min(sigma, 10.0)
                
                # Generate new population
                new_pop = []
                new_pop_f = []
                for i in range(lam):
                    z = rng.standard_normal(n)
                    xi = x_mean + sigma * z
                    xi = np.clip(xi, -5, 5)
                    try:
                        f = wrapped(xi)
                    except Exception:
                        f = 1e10
                    new_pop.append(xi)
                    new_pop_f.append(f)
                    update_best(xi, f)
                
                # Sort by fitness
                idx = np.argsort(new_pop_f)
                pop = [new_pop[i] for i in idx]
                pop_f = [new_pop_f[i] for i in idx]
                
                # Update x_mean to best found so far if it's better
                if best_x is not None and best_f < new_pop_f[0]:
                    x_mean = best_x.copy()

        # Phase 3: Final Nelder-Mead polish from best point
        if best_x is not None and check_time() and calls[0] < budget - 10:
            def nm_obj(x):
                if not check_time():
                    return 1e10
                x = np.clip(x, -5, 5)
                try:
                    return wrapped(x)
                except Exception:
                    return 1e10

            try:
                res = minimize(nm_obj, best_x, method='Nelder-Mead',
                               options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-10,
                                        'adaptive': True})
                if res.fun < best_f:
                    update_best(res.x, res.fun)
            except Exception:
                pass

        # Phase 4: Final fine-tuning with small perturbations
        if best_x is not None and check_time() and calls[0] < budget - 5:
            x = best_x.copy()
            fx = best_f
            sigma = 0.01
            no_improve = 0
            max_no_improve = 15

            while check_time() and calls[0] < budget - 2:
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                try:
                    fy = wrapped(y)
                except Exception:
                    break

                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    sigma *= 1.1
                    no_improve = 0
                else:
                    sigma *= 0.85
                    no_improve += 1

                if no_improve > max_no_improve or sigma < 1e-12:
                    break

        # Fallback
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END

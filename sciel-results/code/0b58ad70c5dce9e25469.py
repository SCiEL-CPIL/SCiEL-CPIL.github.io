"""Hybrid LHS, DE zoom, and Nelder-Mead polishing for BBOB f4."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


class Optimizer:
    """Hybrid optimizer: LHS initialization, DE zoom-in, and NM polishing."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 285.0  # Safety margin for 300s limit
        rng = np.random.default_rng(42)

        # Track best result
        x_best = None
        f_best = np.inf
        calls = 0

        def safe_func(x):
            nonlocal calls, x_best, f_best
            if calls >= self.budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x = np.asarray(x, dtype=np.float64).ravel()[:self.dim]
            x = np.clip(x, -5, 5)
            f = float(func(x))
            calls += 1
            if f < f_best:
                f_best = f
                x_best = x.copy()
            return f

        dim = self.dim

        # --- Phase 1: Latin Hypercube Sampling ---
        n_lhs = min(80, self.budget // 4)
        X = np.zeros((n_lhs, dim))
        for d in range(dim):
            perm = rng.permutation(n_lhs)
            X[:, d] = (perm + rng.random(n_lhs)) / n_lhs * 10.0 - 5.0

        # Evaluate LHS points
        y_obs = []
        X_obs = []
        for i in range(n_lhs):
            if calls >= self.budget or time.time() > deadline:
                break
            xi = X[i]
            fi = safe_func(xi)
            if fi < np.inf:
                X_obs.append(xi)
                y_obs.append(fi)

        if not X_obs:
            # Fallback: random search
            for _ in range(self.budget - calls):
                if time.time() > deadline:
                    break
                x = rng.uniform(-5, 5, dim)
                f = safe_func(x)
            return x_best, f_best

        X_obs = np.array(X_obs)
        y_obs = np.array(y_obs)

        # --- Phase 2: Differential Evolution Zoom-in on top basins ---
        sorted_indices = np.argsort(y_obs)
        top_k = min(5, len(sorted_indices))
        
        for k in range(top_k):
            if calls >= self.budget or time.time() > deadline:
                break
            
            x_start = X_obs[sorted_indices[k]]
            
            # Create local bounds around this promising point
            bounds_local = []
            for d in range(dim):
                lower = max(-5.0, x_start[d] - 2.5)
                upper = min(5.0, x_start[d] + 2.5)
                bounds_local.append((lower, upper))
            
            remaining_budget = self.budget - calls
            if remaining_budget < 50:
                break
            
            # Configure DE for local refinement
            popsize = 12
            maxiter = max(1, remaining_budget // popsize - 10)
            
            try:
                differential_evolution(
                    safe_func,
                    bounds_local,
                    maxiter=maxiter,
                    popsize=popsize,
                    tol=1e-10,
                    seed=42 + k,
                    init='latinhypercube',
                    polish=False
                )
            except Exception:
                pass

        if x_best is None:
            return np.zeros(dim), f_best

        # --- Phase 3: Nelder-Mead Polishing on the best point ---
        if calls < self.budget - 20 and time.time() < deadline:
            try:
                res = minimize(safe_func, x_best, method='Nelder-Mead', 
                               options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10})
                if res.fun < f_best:
                    f_best = res.fun
                    x_best = res.x.copy()
            except Exception:
                pass

        # --- Phase 4: Final exploration if budget remains ---
        while calls < self.budget and time.time() < deadline:
            # Mix of random and local perturbations
            if rng.random() < 0.3:
                # Random global
                x_new = rng.uniform(-5, 5, dim)
            else:
                # Local perturbation around best
                sigma = 0.3
                x_new = x_best + sigma * rng.standard_normal(dim)
                x_new = np.clip(x_new, -5, 5)
            
            f_new = safe_func(x_new)
            if f_new < f_best:
                f_best = f_new
                x_best = x_new.copy()
                
                # If we found a new best, do a quick local refinement
                if calls < self.budget - 30 and time.time() < deadline:
                    try:
                        res = minimize(safe_func, x_best, method='Nelder-Mead', 
                                       options={'maxiter': 30, 'xatol': 1e-8, 'fatol': 1e-10})
                        if res.fun < f_best:
                            f_best = res.fun
                            x_best = res.x.copy()
                    except Exception:
                        pass

        return x_best, f_best
# EVOLVE-BLOCK-END

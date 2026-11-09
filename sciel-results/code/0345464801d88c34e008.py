"""Multi-start Nelder-Mead for BBOB f21."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0  # Safety margin
        calls_made = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        best_x = np.zeros(dim)
        best_f = np.inf

        def safe_func(x):
            nonlocal calls_made
            # Ensure x is a valid finite vector
            x = np.asarray(x, dtype=float)
            if x.shape != (dim,):
                x = np.resize(x, dim)
            if not np.all(np.isfinite(x)):
                return 1e10
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            if not np.isfinite(f):
                f = 1e10
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Diverse initialization
        n_init = 15
        init_points = []
        
        # Use random points and a few grid points for coverage
        # Center
        init_points.append(np.zeros(dim))
        
        # Axis-aligned points
        for i in range(dim):
            for sign in [-1, 1]:
                p = np.zeros(dim)
                p[i] = sign * 3.0
                init_points.append(p)
        
        # Random points
        while len(init_points) < n_init:
            p = rng.uniform(-4, 4, size=dim)
            init_points.append(p)

        # Evaluate initial points
        for p in init_points:
            if time.time() > deadline or calls_made >= budget:
                break
            p = np.clip(p, -5.0, 5.0)
            safe_func(p)

        if calls_made >= budget:
            return best_x, best_f

        # Phase 2: Nelder-Mead from best start and top few
        # Sort init points by fitness? We don't store them, but we can re-evaluate or just use the best found.
        # Since safe_func updates best_x, we know the best.
        # Let's run Nelder-Mead from the best point found so far.
        
        # To improve success rate, we can run NM from a few top candidates.
        # However, we didn't store the top candidates. Let's just run from best_x.
        # If we have budget, we can do a second pass.
        
        # Run 1: From best_x
        if time.time() < deadline and calls_made < budget:
            try:
                res = minimize(safe_func, best_x, method='Nelder-Mead', 
                               options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-6})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Run 2: If we have more budget, try another random restart
        if time.time() < deadline and calls_made < budget - 20:
            # Pick a random new start point
            x_start = rng.uniform(-4, 4, size=dim)
            try:
                res = minimize(safe_func, x_start, method='Nelder-Mead', 
                               options={'maxiter': 30, 'xatol': 1e-4, 'fatol': 1e-6})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 3: Local polish with small steps
        if time.time() < deadline and calls_made < budget:
            x_polish = best_x.copy()
            f_polish = best_f
            sigma_p = 0.01
            max_polish = min(20, budget - calls_made)

            for _ in range(max_polish):
                if time.time() > deadline or calls_made >= budget:
                    break
                y = np.clip(x_polish + sigma_p * rng.standard_normal(dim), -5, 5)
                fy = safe_func(y)
                if fy < f_polish:
                    x_polish = y
                    f_polish = fy
                    sigma_p *= 1.2
                else:
                    sigma_p *= 0.8
                sigma_p = np.clip(sigma_p, 1e-5, 0.1)

            if f_polish < best_f:
                best_x = x_polish.copy()
                best_f = f_polish

        return best_x, best_f
# EVOLVE-BLOCK-END

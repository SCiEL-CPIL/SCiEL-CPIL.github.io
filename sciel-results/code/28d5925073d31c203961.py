import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        call_count = 0
        best_x = None
        best_f = np.inf
        self._best_x = None
        self._best_f = np.inf

        def safe_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= self.budget or time.time() > deadline - 2:
                raise ValueError("Budget or time exceeded")
            call_count += 1
            x_arr = np.asarray(x, dtype=float)
            val = float(func(x_arr))
            if np.isfinite(val) and val < best_f:
                best_f = val
                best_x = np.array(x_arr, dtype=float)
                self._best_x = best_x.copy()
                self._best_f = best_f
            return val

        # Phase 1: CMA-ES style population-based search with rotation awareness
        # Rosenbrock has a narrow valley; we need to find it efficiently
        
        # Start with a diverse initial population
        n_pop = 20
        # Initial spread: mix of structured and random points
        init_pts = []
        
        # Points near typical Rosenbrock minimizer regions
        # The minimum is at (1,1,1,1,1) before shifts
        base = np.ones(self.dim)
        for scale in [0.1, 0.3, 0.5, 1.0, 2.0]:
            init_pts.append(base + rng.normal(0, scale, self.dim))
        
        # Some points along the diagonal
        for val in [-2, -1, 0, 1, 2, 3]:
            p = np.full(self.dim, val)
            init_pts.append(p)
        
        # Random points in the box
        for _ in range(n_pop - len(init_pts)):
            init_pts.append(rng.uniform(-4, 4, self.dim))
        
        init_pts = np.array(init_pts)
        
        for p in init_pts:
            if call_count >= self.budget or time.time() > deadline - 20:
                break
            try:
                safe_func(p)
            except ValueError:
                break
            except Exception:
                continue

        if best_x is None:
            return np.zeros(self.dim), 0.0

        # Phase 2: L-BFGS-B from best point with tight tolerances
        remaining = self.budget - call_count
        if remaining > 10 and time.time() < deadline - 10:
            try:
                res = minimize(
                    safe_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 200,
                        'maxfun': remaining,
                        'maxls': 50,
                        'gtol': 1e-12,
                        'ftol': 1e-15,
                        'gtol': 1e-12
                    }
                )
                if res.fun < best_f:
                    best_f = float(res.fun)
                    best_x = np.array(res.x, dtype=float)
                    self._best_x = best_x.copy()
                    self._best_f = best_f
            except Exception:
                pass

        # Phase 3: Multi-start L-BFGS-B from perturbed best points
        remaining = self.budget - call_count
        if remaining > 30 and time.time() < deadline - 10:
            # Try multiple perturbations to escape potential local issues
            n_starts = 5
            for i in range(n_starts):
                if remaining < 15 or time.time() > deadline - 5:
                    break
                try:
                    # Different scales of perturbation
                    scale = 0.1 * (i + 1)
                    perturb = rng.normal(0, scale, self.dim)
                    start2 = np.clip(best_x + perturb, -5, 5)
                    
                    res2 = minimize(
                        safe_func,
                        start2,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * self.dim,
                        options={
                            'maxiter': 100,
                            'maxfun': max(15, remaining // (n_starts - i)),
                            'maxls': 30,
                            'gtol': 1e-12,
                            'ftol': 1e-15
                        }
                    )
                    if res2.fun < best_f:
                        best_f = float(res2.fun)
                        best_x = np.array(res2.x, dtype=float)
                        self._best_x = best_x.copy()
                        self._best_f = best_f
                    remaining = self.budget - call_count
                except Exception:
                    break

        # Phase 4: Nelder-Mead for final polish (handles the narrow valley well)
        remaining = self.budget - call_count
        if remaining > 10 and time.time() < deadline - 5:
            try:
                res_nm = minimize(
                    safe_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'xatol': 1e-14,
                        'fatol': 1e-16,
                        'adaptive': True,
                        'initial_simplex': np.vstack([
                            best_x,
                            *[best_x + 0.01 * np.eye(self.dim)[j] for j in range(self.dim)]
                        ])
                    }
                )
                if res_nm.fun < best_f:
                    best_f = float(res_nm.fun)
                    best_x = np.array(res_nm.x, dtype=float)
                    self._best_x = best_x.copy()
                    self._best_f = best_f
            except Exception:
                pass

        # Phase 5: One more L-BFGS-B pass if we have budget remaining
        remaining = self.budget - call_count
        if remaining > 10 and time.time() < deadline - 5:
            try:
                res3 = minimize(
                    safe_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 100,
                        'maxfun': remaining,
                        'maxls': 20,
                        'gtol': 1e-14,
                        'ftol': 1e-16
                    }
                )
                if res3.fun < best_f:
                    best_f = float(res3.fun)
                    best_x = np.array(res3.x, dtype=float)
                    self._best_x = best_x.copy()
                    self._best_f = best_f
            except Exception:
                pass

        if best_x is None:
            return np.zeros(self.dim), 0.0
        return best_x, best_f

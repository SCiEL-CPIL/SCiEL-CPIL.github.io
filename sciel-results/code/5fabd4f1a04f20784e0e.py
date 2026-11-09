"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid optimizer: LHS global scan + Powell + Nelder-Mead refinement with deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(0)

        calls = [0]
        best_x = None
        best_f = float('inf')

        def wrapped_func(x):
            nonlocal best_x, best_f
            if calls[0] >= self.budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            calls[0] += 1
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Latin Hypercube Sampling for global coverage
        n_lhs = 20
        if self.budget > n_lhs:
            lhs_points = []
            for i in range(n_lhs):
                point = []
                for d in range(self.dim):
                    low = -5 + (10 * i / n_lhs)
                    high = -5 + (10 * (i + 1) / n_lhs)
                    point.append(rng.uniform(low, high))
                lhs_points.append(point)

            lhs_points = np.array(lhs_points)

            for x in lhs_points:
                if calls[0] >= self.budget or time.time() > deadline:
                    break
                wrapped_func(x)

        if calls[0] >= self.budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: Powell's Method for coordinate-wise optimization
        try:
            res = minimize(wrapped_func, best_x, method='Powell',
                           bounds=[(-5, 5)] * self.dim,
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-12})
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

        if calls[0] >= self.budget or time.time() > deadline:
            return best_x, best_f

        # Phase 3: Adaptive Nelder-Mead based on gradient magnitude
        remaining_budget = self.budget - calls[0]
        if remaining_budget > 20 and time.time() < deadline:
            try:
                eps = 1e-6
                grad_est = np.zeros(self.dim)
                for d in range(self.dim):
                    x_plus = best_x.copy()
                    x_minus = best_x.copy()
                    x_plus[d] += eps
                    x_minus[d] -= eps
                    x_plus = np.clip(x_plus, -5, 5)
                    x_minus = np.clip(x_minus, -5, 5)

                    if calls[0] < self.budget and time.time() < deadline:
                        fp = wrapped_func(x_plus)
                    else:
                        fp = best_f + eps * 100
                    if calls[0] < self.budget and time.time() < deadline:
                        fm = wrapped_func(x_minus)
                    else:
                        fm = best_f + eps * 100

                    grad_est[d] = (fp - fm) / (2 * eps)

                grad_norm = np.linalg.norm(grad_est)

                if grad_norm > 10:
                    nm_init_simplex = 0.5
                    nm_maxiter = 50
                elif grad_norm > 1:
                    nm_init_simplex = 0.1
                    nm_maxiter = 80
                elif grad_norm > 0.1:
                    nm_init_simplex = 0.01
                    nm_maxiter = 100
                else:
                    nm_init_simplex = 0.001
                    nm_maxiter = 150

                nm_res = minimize(wrapped_func, best_x, method='Nelder-Mead',
                                 bounds=[(-5, 5)] * self.dim,
                                 options={'maxiter': nm_maxiter, 'xatol': 1e-10, 'fatol': 1e-14,
                                         'initial_simplex': nm_init_simplex})

                if nm_res.fun < best_f:
                    best_f = nm_res.fun
                    best_x = nm_res.x.copy()

            except Exception:
                pass

        if calls[0] >= self.budget or time.time() > deadline:
            return best_x, best_f

        # Phase 4: Final precision polish
        remaining_budget = self.budget - calls[0]
        if remaining_budget > 10 and time.time() < deadline:
            try:
                final_x = best_x.copy()
                for _ in range(min(5, remaining_budget // 2)):
                    if calls[0] >= self.budget or time.time() > deadline:
                        break
                    perturbation = rng.normal(0, 1e-4, self.dim)
                    candidate_x = np.clip(final_x + perturbation, -5, 5)
                    val = wrapped_func(candidate_x)
                    if val < best_f:
                        best_f = val
                        final_x = candidate_x.copy()

                if calls[0] < self.budget - 5 and time.time() < deadline:
                    nm_res2 = minimize(wrapped_func, best_x, method='Nelder-Mead',
                                       bounds=[(-5, 5)] * self.dim,
                                       options={'maxiter': 30, 'xatol': 1e-12, 'fatol': 1e-16,
                                               'initial_simplex': 1e-4})
                    if nm_res2.fun < best_f:
                        best_f = nm_res2.fun
                        best_x = nm_res2.x.copy()

            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END

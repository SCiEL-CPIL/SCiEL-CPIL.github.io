"""Combined LHS + Powell + adaptive Nelder-Mead optimizer for BBOB f1 (Sphere)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """LHS exploration -> Powell local refinement -> adaptive Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # explicit safety margin before 330s kill

        n = self.dim
        budget = self.budget
        calls = [0]
        best_x = None
        best_f = float('inf')

        def time_ok():
            return time.time() < deadline

        def wrapped_func(x):
            nonlocal best_x, best_f
            if calls[0] >= budget or not time_ok():
                return float('inf')
            calls[0] += 1
            val = func(np.clip(x, -5, 5))
            if val < best_f:
                best_f = val
                best_x = np.array(x, dtype=float)
            return val

        # ---- Phase 1: Latin Hypercube Sampling for broad exploration ----
        n_lhs = 20
        if budget > n_lhs and time_ok():
            lhs_points = []
            for i in range(n_lhs):
                point = []
                for d in range(n):
                    low = -5 + (10 * i / n_lhs)
                    high = -5 + (10 * (i + 1) / n_lhs)
                    point.append(np.random.uniform(low, high))
                lhs_points.append(point)
            lhs_points = np.array(lhs_points)

            for x in lhs_points:
                if calls[0] >= budget or not time_ok():
                    break
                wrapped_func(x)

        if best_x is None:
            # Fallback: evaluate origin
            best_x = np.zeros(n)
            if calls[0] < budget and time_ok():
                best_f = func(best_x)
                calls[0] = 1

        if calls[0] >= budget or not time_ok():
            return best_x, best_f

        # ---- Phase 2: Powell's Method for local refinement ----
        remaining = budget - calls[0]
        if remaining > 10 and time_ok():
            try:
                res = minimize(wrapped_func, best_x, method='Powell',
                               bounds=[(-5, 5)] * n,
                               options={'maxiter': min(500, remaining // 2),
                                        'xatol': 1e-8, 'fatol': 1e-12})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.array(res.x, dtype=float)
            except Exception:
                pass

        if calls[0] >= budget or not time_ok():
            return best_x, best_f

        # ---- Phase 3: Adaptive Nelder-Mead based on estimated gradient norm ----
        remaining = budget - calls[0]
        if remaining > 20 and time_ok():
            try:
                # Estimate local gradient magnitude via central finite differences
                eps = 1e-6
                grad_est = np.zeros(n)
                for d in range(n):
                    if calls[0] >= budget or not time_ok():
                        break
                    x_plus = best_x.copy()
                    x_minus = best_x.copy()
                    x_plus[d] += eps
                    x_minus[d] -= eps
                    x_plus = np.clip(x_plus, -5, 5)
                    x_minus = np.clip(x_minus, -5, 5)

                    if calls[0] < budget and time_ok():
                        fp = wrapped_func(x_plus)
                    else:
                        fp = best_f + eps * 100
                    if calls[0] < budget and time_ok():
                        fm = wrapped_func(x_minus)
                    else:
                        fm = best_f + eps * 100
                    grad_est[d] = (fp - fm) / (2 * eps)

                grad_norm = float(np.linalg.norm(grad_est))

                # Far from optimum -> large simplex (escape local convergence)
                # Close to optimum -> tiny simplex (maximize precision)
                if grad_norm > 10:
                    nm_simplex = 0.5
                    nm_maxiter = 60
                elif grad_norm > 1:
                    nm_simplex = 0.1
                    nm_maxiter = 90
                elif grad_norm > 0.1:
                    nm_simplex = 0.01
                    nm_maxiter = 120
                else:
                    nm_simplex = 0.001
                    nm_maxiter = 150

                nm_res = minimize(wrapped_func, best_x, method='Nelder-Mead',
                                 bounds=[(-5, 5)] * n,
                                 options={'maxiter': min(nm_maxiter, remaining),
                                          'xatol': 1e-10, 'fatol': 1e-14,
                                          'initial_simplex': nm_simplex,
                                          'adaptive': True})
                if nm_res.fun < best_f:
                    best_f = nm_res.fun
                    best_x = np.array(nm_res.x, dtype=float)
            except Exception:
                pass

        if calls[0] >= budget or not time_ok():
            return best_x, best_f

        # ---- Phase 4: Final micro-perturbation + ultra-tight NM polish ----
        remaining = budget - calls[0]
        if remaining > 10 and time_ok():
            try:
                # A few tiny random perturbations to escape any flat spot
                for _ in range(min(5, max(1, remaining // 4))):
                    if calls[0] >= budget or not time_ok():
                        break
                    perturbation = np.random.normal(0, 1e-4, n)
                    candidate_x = np.clip(best_x + perturbation, -5, 5)
                    val = wrapped_func(candidate_x)
                    if val < best_f:
                        best_f = val
                        best_x = candidate_x

                # One more NM pass with extremely tight settings
                if calls[0] < budget - 5 and time_ok():
                    nm_res2 = minimize(wrapped_func, best_x, method='Nelder-Mead',
                                       bounds=[(-5, 5)] * n,
                                       options={'maxiter': min(40, budget - calls[0]),
                                                'xatol': 1e-12, 'fatol': 1e-16,
                                                'initial_simplex': 1e-4,
                                                'adaptive': True})
                    if nm_res2.fun < best_f:
                        best_f = nm_res2.fun
                        best_x = np.array(nm_res2.x, dtype=float)
            except Exception:
                pass

        # Ensure best_x is within bounds
        best_x = np.clip(best_x, -5, 5)

        return best_x, best_f
# EVOLVE-BLOCK-END

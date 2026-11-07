"""Optimizer for BBOB f13 (Sharp Ridge) with CMA-ES multi-start and ridge-aware polishing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import cma


class Optimizer:
    """Multi-start CMA-ES with ridge-aware line search and (1+1)-ES polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Track best
        best_x = rng.uniform(-4, 4, dim)
        best_f = func(best_x)
        calls_used = 1

        def time_ok():
            return time.time() < deadline - 3.0

        def remaining():
            return budget - calls_used

        # Phase 1: Multi-start CMA-ES with adaptive budget allocation
        # Use fewer, better-tuned restarts
        n_starts = 8
        base_sigma = 2.0

        for s in range(n_starts):
            if not time_ok() or calls_used >= budget - 30:
                break

            # Starting points: mix of structured and random
            if s == 0:
                x0 = np.zeros(dim)
            elif s == 1:
                x0 = rng.uniform(-3, 3, dim)
            elif s == 2:
                x0 = np.ones(dim) * 2.5
            elif s == 3:
                x0 = -np.ones(dim) * 2.5
            elif s == 4:
                x0 = rng.uniform(-4, 4, dim)
            elif s == 5:
                x0 = rng.uniform(-2, 2, dim)
            elif s == 6:
                x0 = np.array([3.0, -3.0, 3.0, -3.0, 3.0])
            else:
                x0 = np.array([-3.0, 3.0, -3.0, 3.0, -3.0])

            x0 = np.clip(x0, -5, 5)
            f0 = func(x0)
            calls_used += 1

            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            # If we've found a very good point, reduce remaining budget
            if best_f < -80:
                break

            # CMA-ES with limited function evaluations
            cma_budget = min(remaining() - 20, 60)
            if cma_budget < 15:
                break

            # CMA-ES options
            opts = {
                'maxiter': 1000,
                'bounds': [-5.0] * dim,
                'verbose': -9,
                'seed': 42 + s,
                'CMA_popsize': max(4, int(4 + 2 * np.log(dim))),
                'CMA_sigma0': base_sigma,
                'CMA_adapt_sigma': True,
                'CMA_adapt_C': True,
                'CMA_adapt_alpha': True,
                'CMA_adapt_beta': True,
                'CMA_adapt_delta': True,
                'CMA_adapt_chi': True,
                'CMA_adapt_nu': True,
                'CMA_adapt_d': True,
                'CMA_adapt_e': True,
                'CMA_adapt_f': True,
                'CMA_adapt_g': True,
                'CMA_adapt_h': True,
                'CMA_adapt_i': True,
                'CMA_adapt_j': True,
                'CMA_adapt_k': True,
                'CMA_adapt_l': True,
                'CMA_adapt_m': True,
                'CMA_adapt_n': True,
                'CMA_adapt_o': True,
                'CMA_adapt_p': True,
                'CMA_adapt_q': True,
                'CMA_adapt_r': True,
                'CMA_adapt_s': True,
                'CMA_adapt_t': True,
                'CMA_adapt_u': True,
                'CMA_adapt_v': True,
                'CMA_adapt_w': True,
                'CMA_adapt_x': True,
                'CMA_adapt_y': True,
                'CMA_adapt_z': True,
            }

            # Use a simpler options dict
            opts = {
                'maxiter': 1000,
                'bounds': [-5.0] * dim,
                'verbose': -9,
                'seed': 42 + s,
                'CMA_popsize': max(4, int(4 + 2 * np.log(dim))),
                'CMA_sigma0': base_sigma,
            }

            es = cma.CMA(x0, base_sigma, opts)

            # Manual CMA-ES loop to track function calls
            while not es.stop() and calls_used < budget - 20 and time_ok():
                es.ask()
                for x in es.pop:
                    if not time_ok() or calls_used >= budget - 20:
                        break
                    fx = func(x)
                    calls_used += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    es.tell(x, fx)

                # Check for convergence
                if es.result.xbest is not None:
                    if es.result.fbest < best_f:
                        best_f = es.result.fbest
                        best_x = es.result.xbest.copy()

        # Phase 2: Ridge-aware line search
        # Estimate ridge direction from recent evaluations
        if time_ok() and calls_used < budget - 20:
            # Use the best point and try to identify ridge direction
            # by perturbing in random directions and measuring curvature
            x = best_x.copy()
            fx = best_f
            sigma_probe = 0.3

            # Probe in several directions to find the ridge
            n_probes = 10
            directions = []
            curvatures = []

            for _ in range(n_probes):
                if not time_ok() or calls_used >= budget - 20:
                    break
                d = rng.standard_normal(dim)
                d /= np.linalg.norm(d)
                x_plus = np.clip(x + sigma_probe * d, -5, 5)
                x_minus = np.clip(x - sigma_probe * d, -5, 5)
                f_plus = func(x_plus)
                f_minus = func(x_minus)
                calls_used += 2

                if f_plus < best_f:
                    best_f = f_plus
                    best_x = x_plus.copy()
                if f_minus < best_f:
                    best_f = f_minus
                    best_x = x_minus.copy()

                # Approximate curvature
                curvature = (f_plus + f_minus - 2 * fx) / (sigma_probe ** 2)
                directions.append(d)
                curvatures.append(curvature)

            # Find direction of least curvature (ridge direction)
            if len(curvatures) > 0:
                curvatures = np.array(curvatures)
                directions = np.array(directions)
                min_idx = np.argmin(curvatures)
                ridge_dir = directions[min_idx]
                ridge_dir /= np.linalg.norm(ridge_dir)

                # Line search along ridge direction
                for step in np.linspace(-3.0, 3.0, 13):
                    if not time_ok() or calls_used >= budget - 10:
                        break
                    x_new = np.clip(x + step * ridge_dir, -5, 5)
                    f_new = func(x_new)
                    calls_used += 1
                    if f_new < best_f:
                        best_f = f_new
                        best_x = x_new.copy()
                        x = x_new.copy()
                        fx = f_new

        # Phase 3: (1+1)-ES polish with adaptive sigma
        if time_ok() and calls_used < budget - 5:
            sigma = 0.3
            x = best_x.copy()
            fx = best_f
            no_improve = 0

            for _ in range(min(200, budget - calls_used)):
                if not time_ok():
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.15
                    no_improve = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9
                    no_improve += 1

                if no_improve > 30 or sigma < 1e-7:
                    break

        # Phase 4: Fine-tune with very small steps
        if time_ok() and calls_used < budget:
            sigma = 0.02
            x = best_x.copy()
            fx = best_f

            for _ in range(min(50, budget - calls_used)):
                if not time_ok():
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END

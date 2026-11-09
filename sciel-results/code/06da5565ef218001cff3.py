"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Hybrid optimizer: (1+1)-ES with success-ratio adaptation and L-BFGS-B polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Initial point: Center of the box is a strong heuristic for rotated Rosenbrock
        x = np.zeros(n)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls_made = 1

        sigma = 1.5
        success_ratio = 0.5
        no_improve = 0
        max_no_improve = 30
        restart_count = 0
        max_restarts = 8

        # Phase 1: (1+1)-ES with success-ratio-based sigma adaptation
        while calls_made < budget - 60:  # Reserve 60 calls for polish
            if time.time() > deadline:
                break

            # Generate candidate
            y = x + sigma * rng.standard_normal(n)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                # Accept step
                x = y
                fx = fy
                no_improve = 0
                success_ratio = 0.8 * success_ratio + 0.2 * 1.0

                # Update global best
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()

                # Increase sigma on success
                sigma *= 1.15
            else:
                # Reject step
                no_improve += 1
                success_ratio = 0.8 * success_ratio + 0.2 * 0.0
                # Decrease sigma on failure
                sigma *= 0.95

                # Restart logic if stuck
                if no_improve >= max_no_improve:
                    if restart_count >= max_restarts:
                        # If we've exhausted restarts, just keep going with smaller sigma
                        sigma *= 0.8
                        no_improve = 0
                        continue

                    # Try large jumps from the best point found so far
                    restarted = False
                    jump_sizes = [3.0, 2.0, 1.5, 1.0]
                    for jump in jump_sizes:
                        if calls_made >= budget - 60 or time.time() > deadline:
                            break
                        candidate = best_x + jump * rng.standard_normal(n)
                        candidate = np.clip(candidate, -5, 5)
                        cf = func(candidate)
                        calls_made += 1
                        if cf < best_fx:
                            best_fx = cf
                            best_x = candidate.copy()
                            x = best_x.copy()
                            fx = best_fx
                            no_improve = 0
                            sigma = 1.0
                            restart_count += 1
                            restarted = True
                            break

                    # If large jumps didn't help, try a full random restart
                    if not restarted:
                        x_new = rng.uniform(-4, 4, n)
                        fx_new = func(x_new)
                        calls_made += 1
                        if fx_new < best_fx:
                            best_fx = fx_new
                            best_x = x_new.copy()
                            x = x_new.copy()
                            fx = best_fx
                        no_improve = 0
                        sigma = 1.5
                        restart_count += 1

        # Phase 2: Local Refinement with L-BFGS-B
        # L-BFGS-B with finite differences is far more efficient than Nelder-Mead
        # for Rosenbrock because it estimates the gradient and follows the valley.
        if calls_made < budget and time.time() < deadline - 5:
            try:
                from scipy.optimize import minimize

                remaining_calls = budget - calls_made

                # L-BFGS-B with forward finite differences uses n+1 calls per gradient
                # In 5D, that's 6 calls per gradient. With maxiter, we can do ~remaining/6 iterations.
                # But we also need calls for the line search, so be conservative.
                max_iter = max(5, remaining_calls // 10)

                def lbfgsb_wrapper(z):
                    nonlocal calls_made
                    if calls_made >= budget:
                        return best_fx
                    if time.time() > deadline:
                        return best_fx
                    calls_made += 1
                    return func(np.clip(z, -5, 5))

                # Start L-BFGS-B from the best point found by ES
                bounds = [(-5, 5)] * n
                res = minimize(
                    lbfgsb_wrapper,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': max_iter,
                        'ftol': 1e-12,
                        'gtol': 1e-8,
                        'maxls': 20,
                    }
                )

                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = np.clip(res.x, -5, 5)

                # If we still have budget, do a second polish pass from the improved point
                if calls_made < budget - 10 and time.time() < deadline - 3:
                    remaining2 = budget - calls_made
                    max_iter2 = max(3, remaining2 // 12)
                    res2 = minimize(
                        lbfgsb_wrapper,
                        best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxiter': max_iter2,
                            'ftol': 1e-14,
                            'gtol': 1e-10,
                            'maxls': 15,
                        }
                    )
                    if res2.fun < best_fx:
                        best_fx = res2.fun
                        best_x = np.clip(res2.x, -5, 5)

            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END

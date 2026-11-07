"""Sobol global exploration + adaptive multi-scale hill climbing + FD gradient polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """
    Hybrid optimizer for Weierstrass f16:
    1. Deterministic Sobol sequence for uniform global exploration (first ~30% of budget).
    2. Adaptive per-dimension sigma hill-climbing with multi-scale Gaussian perturbation.
    3. Periodic finite-difference gradient polish for rapid local convergence.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        n = self.dim
        budget = self.budget

        # Phase 1: Sobol global exploration
        n_sobol = min(150, budget // 3)
        sobol = qmc.Sobol(d=n, scramble=True, seed=42)
        sobol_points = sobol.random(n_sobol)
        # Map [0,1]^n to [-5,5]^n
        sobol_points = sobol_points * 10.0 - 5.0

        best_x = sobol_points[0].copy()
        best_f = func(best_x)
        calls = 1

        for i in range(1, n_sobol):
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            x = sobol_points[i]
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 2: Adaptive per-dimension hill climbing with multi-scale sigma
        # Use remaining budget
        remaining = budget - calls
        if remaining <= 0 or time.time() > deadline:
            return best_x, best_f

        # Initialize per-dimension sigma
        sigma = np.full(n, 1.5)
        x = best_x.copy()
        fx = best_f

        # Weierstrass has self-similar structure; use multi-scale perturbation
        # Decay sigma over time, but allow occasional large jumps
        rng = np.random.default_rng(12345)

        # Track stagnation
        no_improve = 0
        stagnation_limit = 30

        for step in range(remaining):
            if time.time() > deadline:
                break

            # Multi-scale: occasionally use a larger sigma for escape
            # Base sigma decays geometrically
            scale_factor = max(0.01, 1.0 - step / (remaining * 0.8))
            base_sigma = sigma * scale_factor

            # Occasionally (every ~50 steps) do a larger jump
            if step % 50 == 0 and step > 0:
                jump_sigma = base_sigma * 3.0
            else:
                jump_sigma = base_sigma

            # Generate perturbation
            delta = rng.standard_normal(n) * jump_sigma
            y = np.clip(x + delta, -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                # Success: adapt sigma per dimension
                for d in range(n):
                    if abs(delta[d]) > 1e-10:
                        sigma[d] *= 1.2
                    else:
                        sigma[d] *= 1.05
                # Cap sigma
                sigma = np.clip(sigma, 0.01, 5.0)
                x, fx = y, fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                no_improve = 0
            else:
                # Failure: shrink sigma per dimension
                for d in range(n):
                    sigma[d] *= 0.85
                sigma = np.clip(sigma, 0.005, 5.0)
                no_improve += 1

            # Periodic FD gradient polish every 40 steps
            if step % 40 == 39:
                # Finite difference gradient estimate
                eps = 1e-4
                grad = np.zeros(n)
                for d in range(n):
                    x_plus = x.copy()
                    x_minus = x.copy()
                    x_plus[d] = min(5.0, x[d] + eps)
                    x_minus[d] = max(-5.0, x[d] - eps)
                    f_plus = func(x_plus)
                    f_minus = func(x_minus)
                    calls += 2
                    grad[d] = (f_plus - f_minus) / (2 * eps)

                # Gradient step with adaptive learning rate
                grad_norm = np.linalg.norm(grad)
                if grad_norm > 1e-8:
                    lr = 0.1 * min(1.0, 1.0 / grad_norm)
                    x_new = np.clip(x - lr * grad, -5, 5)
                    f_new = func(x_new)
                    calls += 1
                    if f_new < fx:
                        x, fx = x_new, f_new
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        no_improve = 0
                    else:
                        no_improve += 1

            # If stuck for too long, try a random restart from Sobol points
            if no_improve >= stagnation_limit:
                # Pick a random Sobol point and do a local search from there
                idx = rng.integers(0, n_sobol)
                x = sobol_points[idx].copy()
                fx = func(x)
                calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                sigma = np.full(n, 2.0)
                no_improve = 0

            # Check budget
            if calls >= budget:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END

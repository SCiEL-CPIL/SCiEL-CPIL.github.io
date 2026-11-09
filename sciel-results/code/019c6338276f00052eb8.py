"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with CMA-style covariance adaptation, adaptive sigma, and restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        best_x = None
        best_f = np.inf
        calls_made = 0
        budget = self.budget
        dim = self.dim

        # Restart strategy: 15 restarts with varying budget allocation
        # Early restarts get more calls to explore, later ones get fewer
        max_restarts = 15
        # Allocate budget: first 3 get 25% each, next 3 get 15%, next 3 get 10%, rest get remaining
        budget_alloc = []
        remaining = budget
        for i in range(max_restarts):
            if i < 3:
                alloc = int(budget * 0.25)
            elif i < 6:
                alloc = int(budget * 0.15)
            elif i < 9:
                alloc = int(budget * 0.10)
            else:
                alloc = max(5, remaining // (max_restarts - i))
            budget_alloc.append(alloc)
            remaining -= alloc
        # Ensure last restart gets all remaining
        budget_alloc[-1] = max(5, budget_alloc[-1] + remaining)
        remaining = budget - sum(budget_alloc)
        if remaining > 0:
            budget_alloc[0] += remaining

        # Stagnation threshold for early restart
        stagnation_threshold = 20

        for restart in range(max_restarts):
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break

            calls_this_restart = min(budget_alloc[restart], budget - calls_made)
            if calls_this_restart <= 0:
                break

            # Initialize with different strategies per restart
            if restart < 3:
                # Center-biased for early restarts
                x = rng.uniform(-2.5, 2.5, dim)
            elif restart < 6:
                x = rng.uniform(-3.5, 3.5, dim)
            elif restart < 9:
                x = rng.uniform(-4.5, 4.5, dim)
            else:
                # Late restarts: wider exploration
                x = rng.uniform(-5.0, 5.0, dim)

            fx = func(x)
            calls_made += 1

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # (1+1)-ES with covariance matrix adaptation
            sigma = 2.0
            no_improve = 0
            sigma_min = 0.001
            sigma_max = 10.0

            # Simple covariance tracking using recent successful moves
            n_samples = 5
            samples = []  # store (step_vector, success)
            C = np.eye(dim) * 0.5  # initial covariance
            d_sigma = 1.0

            for step in range(calls_this_restart):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break

                # Generate trial point using current covariance
                # Simple Cholesky-like sampling using eigen decomposition (dim=5, fast)
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-6)
                    noise = eigvecs @ (np.sqrt(eigvals) * rng.standard_normal(dim))
                except:
                    noise = rng.standard_normal(dim) * 0.5

                y = np.clip(x + sigma * noise, -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    # Success: move to y
                    step_vec = y - x
                    x, fx = y, fy
                    no_improve = 0

                    # Update covariance with successful step
                    samples.append(step_vec)
                    if len(samples) > n_samples:
                        samples.pop(0)
                    if len(samples) >= 3:
                        # Update C as outer product of recent successful steps
                        C_new = np.zeros((dim, dim))
                        for s in samples:
                            C_new += np.outer(s, s)
                        C_new /= len(samples)
                        # Blend with old C to avoid losing information
                        C = 0.7 * C_new + 0.3 * C
                        # Ensure symmetry
                        C = (C + C.T) / 2
                        # Regularize
                        C += np.eye(dim) * 1e-4

                    # Increase sigma on success
                    sigma = min(sigma * 1.1, sigma_max)
                else:
                    # Failure
                    no_improve += 1
                    # Decrease sigma on failure
                    sigma = max(sigma * 0.95, sigma_min)

                    # If many failures, reset covariance to identity
                    if no_improve > 10:
                        C = np.eye(dim) * 0.5
                        no_improve = 0

                # Update global best
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

                # Early restart if stagnating
                if no_improve > stagnation_threshold:
                    break

            # Stop if we found a very good solution
            if best_f < -90:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END

"""Optimizer for BBOB-style Rosenbrock (f8) on [-5,5]^5."""
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with restarts and adaptive sigma for Rosenbrock optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 285  # safety margin before 300s limit

        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def safe_call(x):
            nonlocal calls, best_x, best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            if calls >= budget:
                return np.inf
            f = func(x)
            if not np.isfinite(f):
                f = np.inf
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Multi-start (1+1)-ES with restarts
        # Each restart: run (1+1)-ES with adaptive sigma, restart if stuck
        max_restarts = 20
        min_calls_per_restart = 20

        for restart in range(max_restarts):
            if calls >= budget or time.time() > deadline:
                break

            # Choose starting point
            if restart == 0:
                # Try center first
                x = np.zeros(dim)
                f = safe_call(x)
            elif restart == 1:
                # Try a fixed shifted point
                x = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
                f = safe_call(x)
            else:
                # Random start
                x = rng.uniform(-5, 5, size=dim)
                f = safe_call(x)

            if f >= np.inf:
                continue

            # (1+1)-ES parameters
            sigma = 1.0
            success_count = 0
            no_improve = 0
            max_no_improve = 50
            min_sigma = 1e-4
            max_sigma = 5.0

            while calls < budget and time.time() < deadline:
                # Generate candidate
                direction = rng.standard_normal(dim)
                x_new = x + sigma * direction
                x_new = np.clip(x_new, -5, 5)
                f_new = safe_call(x_new)

                if f_new >= np.inf:
                    break

                if f_new < f:
                    # Success: move and adapt sigma
                    x = x_new
                    f = f_new
                    success_count += 1
                    no_improve = 0
                    # 1/5 rule: increase sigma if success rate > 1/5
                    if success_count >= 10:
                        if success_count / 10 > 0.2:
                            sigma = min(sigma * 1.25, max_sigma)
                        else:
                            sigma = max(sigma * 0.8, min_sigma)
                        success_count = 0
                else:
                    # Failure: keep x, adapt sigma
                    no_improve += 1
                    if no_improve >= max_no_improve:
                        break

            # Check if we should restart (stuck or converged)
            if no_improve >= max_no_improve or sigma < min_sigma * 2:
                continue

            # If we've used most of our budget, stop
            if calls > budget * 0.8:
                break

        return best_x, best_f

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive sigma, multi-start, and local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 330s kill

        # Track best
        best_x = None
        best_f = float('inf')
        calls = 0

        def eval_func(x):
            nonlocal best_x, best_f, calls
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Multi-start (1+1)-ES with adaptive sigma
        # Use 5 restarts with varying budgets
        restart_configs = [
            (0, 150, 1.0),   # (start_type, budget, initial_sigma)
            (1, 100, 2.0),   # random start
            (1, 80, 1.5),
            (1, 60, 2.5),
            (1, 50, 1.0),
        ]

        for start_type, per_restart, sigma0 in restart_configs:
            if calls >= budget or time.time() > deadline:
                break

            # Start point
            if start_type == 0:
                x = np.zeros(dim)
            else:
                x = rng.uniform(-4.5, 4.5, dim)

            fx = eval_func(x)
            sigma = sigma0

            # Success rate tracking for sigma adaptation
            window = 8
            recent = []

            for _ in range(per_restart):
                if calls >= budget or time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = eval_func(y)

                if fy < fx:
                    x, fx = y, fy
                    recent.append(True)
                else:
                    recent.append(False)

                # Adapt sigma based on success rate in recent window
                if len(recent) >= window:
                    sr = sum(recent[-window:]) / window
                    if sr > 0.45:
                        sigma *= 1.15
                    elif sr < 0.15:
                        sigma *= 0.85
                    else:
                        sigma *= 1.0

                # Keep sigma in reasonable bounds
                sigma = np.clip(sigma, 0.05, 5.0)

        # Phase 2: Local refinement using Nelder-Mead-like search on best point
        if best_x is not None and calls < budget and time.time() < deadline:
            x_local = best_x.copy()
            f_local = best_f
            remaining = budget - calls
            
            # Use a simple gradient-free local search with decreasing step
            step = 0.3
            min_step = 0.01
            no_improve_count = 0
            
            while calls < budget and time.time() < deadline and step >= min_step:
                improved = False
                for d in range(dim):
                    if calls >= budget or time.time() > deadline:
                        break
                    for sign in [1, -1]:
                        if calls >= budget or time.time() > deadline:
                            break
                        y = x_local.copy()
                        y[d] += sign * step
                        y = np.clip(y, -5, 5)
                        fy = eval_func(y)
                        if fy < f_local:
                            x_local, f_local = y, fy
                            improved = True
                            no_improve_count = 0
                            break
                if not improved:
                    step *= 0.7
                    no_improve_count += 1
                    if no_improve_count > 3:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END

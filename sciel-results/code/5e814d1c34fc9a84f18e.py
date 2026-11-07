"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive sigma, restarts, and a dedicated polish phase."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 330s kill

        rng = np.random.default_rng(0)

        best_x = None
        best_f = np.inf
        num_calls = 0

        def check_time():
            return time.time() > deadline

        def make_eval():
            nonlocal num_calls, best_x, best_f
            def evaluate(x):
                nonlocal num_calls, best_x, best_f
                x = np.clip(x, -5, 5)
                f = func(x)
                num_calls += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                return f
            return evaluate

        evaluate = make_eval()

        # Phase 1: Global search with restarts
        # Use a smaller effective budget for global search, save some for polish
        global_budget = int(budget * 0.7)
        polish_budget = budget - global_budget

        # Multiple restarts for global exploration
        num_restarts = 3
        calls_per_restart = global_budget // num_restarts
        if calls_per_restart < 20:
            num_restarts = 1
            calls_per_restart = global_budget

        for r in range(num_restarts):
            if num_calls >= global_budget or check_time():
                break

            if r == 0:
                x = rng.uniform(-5, 5, dim)
            else:
                # Restart near best point but with some perturbation
                perturbation = 2.0 / (r + 1)
                x = np.clip(best_x + perturbation * rng.standard_normal(dim), -5, 5)

            fx = evaluate(x)
            sigma = 2.0 / (r + 1)
            min_sigma = 1e-6
            max_sigma = 3.0

            for _ in range(calls_per_restart):
                if num_calls >= global_budget or check_time():
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.3, max_sigma)
                else:
                    sigma = max(sigma * 0.8, min_sigma)

        # Phase 2: Polish phase - fine-grained local refinement
        # Use remaining budget for high-precision convergence around best point
        if best_x is not None and num_calls < budget and not check_time():
            x = best_x.copy()
            fx = best_f
            sigma = 0.1  # Start with small sigma for refinement
            min_sigma = 1e-8
            max_sigma = 0.5

            for _ in range(polish_budget):
                if num_calls >= budget or check_time():
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.3, max_sigma)
                else:
                    sigma = max(sigma * 0.7, min_sigma)

                # If sigma gets very small, we've likely converged
                if sigma < 1e-7:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END

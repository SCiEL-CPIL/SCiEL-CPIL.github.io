"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coordinate descent with adaptive steps for Rosenbrock valley,
    with deadline guard and budget tracking."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.perf_counter()
        deadline = start_time + 280.0  # 300s limit, 20s safety margin
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Track best solution and call count
        best_x = None
        best_f = np.inf
        call_count = [0]

        def safe_call(x):
            if call_count[0] >= budget:
                return np.inf
            if time.perf_counter() > deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            try:
                fx = float(func(x))
            except Exception:
                fx = np.inf
            call_count[0] += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Initial random start
        x = rng.uniform(-5, 5, n)
        fx = safe_call(x)

        # Adaptive coordinate descent for Rosenbrock valley
        # Rosenbrock: f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1-x_i)^2)
        # Valley follows x_{i+1} ≈ x_i^2, so we use larger steps in that direction
        step_sizes = np.ones(n) * 1.0
        patience = 30  # calls without improvement before reducing steps
        no_improve = 0
        min_step = 1e-6

        for _ in range(budget - 1):
            if call_count[0] >= budget:
                break
            if time.perf_counter() > deadline:
                break

            # Cycle through coordinates with adaptive step sizes
            improved = False
            for i in range(n):
                if call_count[0] >= budget:
                    break
                if time.perf_counter() > deadline:
                    break

                # Try positive and negative steps
                for direction in [1, -1]:
                    if call_count[0] >= budget:
                        break
                    if time.perf_counter() > deadline:
                        break

                    x_trial = x.copy()
                    x_trial[i] += direction * step_sizes[i]
                    x_trial = np.clip(x_trial, -5, 5)
                    f_trial = safe_call(x_trial)

                    if f_trial < fx:
                        x = x_trial
                        fx = f_trial
                        improved = True
                        # Increase step size slightly on success
                        step_sizes[i] = min(step_sizes[i] * 1.2, 3.0)
                        break  # Move to next coordinate
                    else:
                        # Reduce step size on failure
                        step_sizes[i] = max(step_sizes[i] * 0.8, min_step)

            if improved:
                no_improve = 0
            else:
                no_improve += 1

            # If stuck for too long, try a random perturbation
            if no_improve >= patience:
                # Random restart in a small neighborhood
                x_new = x + rng.standard_normal(n) * 0.5
                x_new = np.clip(x_new, -5, 5)
                f_new = safe_call(x_new)
                if f_new < fx:
                    x = x_new
                    fx = f_new
                    no_improve = 0
                else:
                    # Reduce all step sizes
                    step_sizes *= 0.5
                    if np.min(step_sizes) < min_step:
                        # Try a larger random jump
                        x_new = x + rng.standard_normal(n) * 2.0
                        x_new = np.clip(x_new, -5, 5)
                        f_new = safe_call(x_new)
                        if f_new < fx:
                            x = x_new
                            fx = f_new
                        no_improve = 0

        # Final return
        if best_x is None:
            best_x = np.zeros(n)
            best_f = float('inf')

        return best_x, best_f
# EVOLVE-BLOCK-END

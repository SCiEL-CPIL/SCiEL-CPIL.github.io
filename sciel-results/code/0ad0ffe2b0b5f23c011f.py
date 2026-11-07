"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Stochastic gradient descent with finite differences for BBOB Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        deadline = time.time() + 280.0

        calls_made = [0]
        best_x = [None]
        best_f = [float('inf')]

        def safe_call(x):
            if calls_made[0] >= budget:
                return None
            if time.time() > deadline:
                return None
            x_clipped = np.clip(x, -5.0, 5.0)
            val = func(x_clipped)
            calls_made[0] += 1
            return val

        def estimate_gradient(x, fx, eps=1e-4):
            """Estimate gradient using central finite differences."""
            grad = np.zeros(dim)
            for i in range(dim):
                x_plus = x.copy()
                x_plus[i] += eps
                x_minus = x.copy()
                x_minus[i] -= eps

                f_plus = safe_call(x_plus)
                f_minus = safe_call(x_minus)

                if f_plus is None or f_minus is None:
                    return None

                grad[i] = (f_plus - f_minus) / (2.0 * eps)
            return grad

        def run_sgd(x0, fx0, n_steps, lr0=0.1, lr_decay=0.995, momentum=0.9):
            """Run SGD with momentum for n_steps."""
            nonlocal calls_made, best_x, best_f

            x = x0.copy()
            fx = fx0
            lr = lr0
            velocity = np.zeros(dim)

            for step in range(n_steps):
                if calls_made[0] >= budget:
                    break
                if time.time() > deadline:
                    break

                # Estimate gradient
                grad = estimate_gradient(x, fx)
                if grad is None:
                    break

                # Update with momentum
                velocity = momentum * velocity - lr * grad
                x_new = np.clip(x + velocity, -5.0, 5.0)

                # Evaluate new point
                fx_new = safe_call(x_new)
                if fx_new is None:
                    break

                # Accept if improved or with some probability for exploration
                if fx_new < fx:
                    x = x_new
                    fx = fx_new
                    if fx < best_f[0]:
                        best_f[0] = fx
                        best_x[0] = x.copy()
                else:
                    # Occasionally accept worse solutions to escape local minima
                    if rng.random() < 0.1:
                        x = x_new
                        fx = fx_new

                # Decay learning rate
                lr *= lr_decay

            return x, fx

        # Phase 1: Multi-start with SGD
        n_starts = 5
        calls_per_start = budget // (n_starts + 2)  # Reserve some for polish

        for s in range(n_starts):
            if calls_made[0] >= budget or time.time() > deadline:
                break

            # Generate starting point
            if s == 0:
                x0 = np.random.uniform(-4.0, 4.0, dim)
            else:
                # Mix of random and perturbed best
                if best_x[0] is not None and rng.random() < 0.5:
                    x0 = np.clip(best_x[0] + rng.randn(dim) * 1.5, -5.0, 5.0)
                else:
                    x0 = np.random.uniform(-4.0, 4.0, dim)

            fx0 = safe_call(x0)
            if fx0 is None:
                break

            if fx0 < best_f[0]:
                best_f[0] = fx0
                best_x[0] = x0.copy()

            # Run SGD from this start
            remaining_for_sgd = min(calls_per_start - 1, budget - calls_made[0])
            if remaining_for_sgd > 20:
                run_sgd(x0, fx0, remaining_for_sgd, lr0=0.5, lr_decay=0.99, momentum=0.8)

        # Phase 2: Fine-tune around the best point found
        if best_x[0] is not None and calls_made[0] < budget and time.time() < deadline:
            remaining2 = budget - calls_made[0]
            if remaining2 > 30:
                run_sgd(best_x[0].copy(), best_f[0], remaining2, lr0=0.05, lr_decay=0.995, momentum=0.7)

        # Phase 3: Very fine tuning
        if best_x[0] is not None and calls_made[0] < budget and time.time() < deadline:
            remaining3 = budget - calls_made[0]
            if remaining3 > 20:
                run_sgd(best_x[0].copy(), best_f[0], remaining3, lr0=0.01, lr_decay=0.998, momentum=0.6)

        return best_x[0], best_f[0]
# EVOLVE-BLOCK-END

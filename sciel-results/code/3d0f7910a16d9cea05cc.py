"""Combined program: Gradient-sign corner jump with coordinate-wise refinement for Linear Slope."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer exploiting the linear structure of f5 (Linear slope).
    Estimates gradient signs, jumps to predicted optimal corner,
    then refines using coordinate-wise search.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # 1. Initial point at origin
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # 2. Estimate gradient signs using forward differences with h=1.0
        #    This gives strong signal for sign determination and
        #    intermediate points may already improve the objective.
        h = 1.0
        grad = np.zeros(dim)

        for i in range(dim):
            if calls >= budget or time.time() > deadline:
                break
            x_test = x.copy()
            x_test[i] += h
            x_test = np.clip(x_test, -5, 5)
            fy_test = func(x_test)
            calls += 1

            # Update best if this intermediate point is better
            if fy_test < best_fx:
                best_fx = fy_test
                best_x = x_test.copy()

            # Record gradient component (sign is what matters for corner)
            grad[i] = (fy_test - fx) / h

        # 3. Predict optimal corner based on gradient signs
        #    For linear f: min is at corner where each coord is
        #    -5 if grad[i] > 0, +5 if grad[i] < 0
        x_pred = np.zeros(dim)
        for i in range(dim):
            if grad[i] > 0:
                x_pred[i] = -5.0
            elif grad[i] < 0:
                x_pred[i] = 5.0
            else:
                x_pred[i] = 0.0

        # Evaluate predicted corner
        if calls < budget and time.time() < deadline:
            fx_pred = func(x_pred)
            calls += 1
            if fx_pred < best_fx:
                best_fx = fx_pred
                best_x = x_pred.copy()

        # 4. Coordinate-wise refinement phase
        #    Since f is linear, if we're slightly off the true optimal corner,
        #    we can try flipping individual coordinates to see if it helps.
        #    Also try intermediate positions for fine-tuning.
        x = best_x.copy()
        fx = best_fx

        # First pass: try flipping each coordinate to opposite extreme
        for i in range(dim):
            if calls >= budget or time.time() > deadline:
                break
            x_test = x.copy()
            # Try the opposite extreme
            if x_test[i] > 0:
                x_test[i] = -5.0
            else:
                x_test[i] = 5.0
            x_test = np.clip(x_test, -5, 5)
            fy_test = func(x_test)
            calls += 1
            if fy_test < fx:
                x = x_test
                fx = fy_test
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()

        # Second pass: fine-tune each coordinate with a few probe points
        # Try positions: current, and a few fractions toward center
        probe_positions = [0.0, 1.0, 2.0, 3.0, 4.0]
        for i in range(dim):
            if calls >= budget or time.time() > deadline:
                break
            for pp in probe_positions:
                if calls >= budget or time.time() > deadline:
                    break
                x_test = x.copy()
                # Try both positive and negative probe positions
                for sign in [1, -1]:
                    if calls >= budget or time.time() > deadline:
                        break
                    x_test[i] = sign * pp
                    x_test = np.clip(x_test, -5, 5)
                    fy_test = func(x_test)
                    calls += 1
                    if fy_test < fx:
                        x = x_test
                        fx = fy_test
                        if fx < best_fx:
                            best_fx = fx
                            best_x = x.copy()

        # 5. Final (1+1)-ES polish with small sigma for any remaining improvement
        if calls < budget and time.time() < deadline:
            sigma = 0.1
            success_counter = 0
            total_steps = 0

            while calls < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1

                if fy < fx:
                    x = y
                    fx = fy
                    if fy < best_fx:
                        best_fx = fy
                        best_x = x.copy()
                    sigma *= 1.1
                    success_counter += 1
                else:
                    sigma *= 0.9
                    success_counter = 0

                total_steps += 1

                # Restart if stuck
                if total_steps > 30 and success_counter == 0:
                    x = best_x.copy()
                    fx = best_fx
                    sigma = 0.1
                    total_steps = 0
                    success_counter = 0

        return best_x, best_fx
# EVOLVE-BLOCK-END

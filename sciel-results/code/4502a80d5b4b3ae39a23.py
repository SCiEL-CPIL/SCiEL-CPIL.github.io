"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, time-bounded, plus deterministic coordinate-ascent polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        best_x, best_fx = x.copy(), fx
        sigma = 2.0
        calls = 1

        # Use 70% of budget for (1+1)-ES
        es_budget = int(0.7 * (self.budget - 1))

        for i in range(es_budget):
            if time.time() > deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
            else:
                sigma *= 0.87

            if fx < best_fx:
                best_x, best_fx = x.copy(), fx

        # Deterministic coordinate-ascent polish on the best point found
        if time.time() < deadline - 10.0:
            x_curr = best_x.copy()
            f_curr = best_fx
            remaining_calls = self.budget - calls

            # Multiple passes of coordinate ascent with golden-section line search
            max_passes = 3
            for p in range(max_passes):
                if time.time() > deadline or calls >= self.budget:
                    break
                improved = False
                for d in range(self.dim):
                    if time.time() > deadline or calls >= self.budget:
                        break
                    # Golden-section search along dimension d
                    lo = -5.0
                    hi = 5.0
                    # Try to narrow the initial bracket around current value
                    center = x_curr[d]
                    span = max(0.5, sigma * 2)
                    lo = max(-5.0, center - span)
                    hi = min(5.0, center + span)
                    if hi - lo < 1e-10:
                        continue

                    # Golden-section parameters
                    gr = (np.sqrt(5.0) - 1.0) / 2.0
                    x1 = hi - gr * (hi - lo)
                    x2 = lo + gr * (hi - lo)

                    def eval_dim(val):
                        nonlocal calls
                        if calls >= self.budget:
                            return np.inf
                        test_x = x_curr.copy()
                        test_x[d] = val
                        val_f = func(test_x)
                        calls += 1
                        return val_f

                    f1 = eval_dim(x1)
                    f2 = eval_dim(x2)

                    # If either evaluation failed or exceeded budget, skip
                    if np.isinf(f1) or np.isinf(f2):
                        continue

                    # Golden-section iterations
                    max_gs_iter = 20
                    for _ in range(max_gs_iter):
                        if time.time() > deadline or calls >= self.budget:
                            break
                        if f1 < f2:
                            hi = x2
                            x2 = x1
                            f2 = f1
                            x1 = hi - gr * (hi - lo)
                            if hi - lo < 1e-10:
                                break
                            f1 = eval_dim(x1)
                            if np.isinf(f1):
                                break
                        else:
                            lo = x1
                            x1 = x2
                            f1 = f2
                            x2 = lo + gr * (hi - lo)
                            if hi - lo < 1e-10:
                                break
                            f2 = eval_dim(x2)
                            if np.isinf(f2):
                                break

                    # Take the best point found in this dimension search
                    best_val = x_curr[d]
                    best_f = f_curr
                    if f1 < best_f:
                        best_val, best_f = x1, f1
                    if f2 < best_f:
                        best_val, best_f = x2, f2

                    if best_f < f_curr:
                        x_curr[d] = best_val
                        f_curr = best_f
                        improved = True

                if not improved:
                    break

            if f_curr < best_fx:
                best_x, best_fx = x_curr.copy(), f_curr

        return best_x, best_fx
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Pattern search + adaptive (1+1)-ES hybrid with wall-clock deadline."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        t0 = time.time()
        DEADLINE = 280.0
        dim = self.dim
        budget = self.budget
        L, U = -5.0, 5.0

        rng = np.random.default_rng(20240517)

        # ---- Phase 1: Multi-start to find a good initial basin ----
        x = rng.uniform(L, U, dim)
        fx = float(func(x))
        n_calls = 1

        for _ in range(8):
            if n_calls >= budget or time.time() - t0 > DEADLINE:
                break
            xc = rng.uniform(L, U, dim)
            fc = float(func(xc))
            n_calls += 1
            if fc < fx:
                x, fx = xc, fc

        # ---- Phase 2: Coordinate-wise pattern search for fast early progress ----
        # This gives strong early AOCC gains by making steady progress along axes.
        step = 1.0
        no_improve = 0
        max_no_improve = 30
        
        while n_calls < budget and time.time() - t0 < DEADLINE:
            improved = False
            for d in range(dim):
                if n_calls >= budget or time.time() - t0 > DEADLINE:
                    break
                # Try positive step
                y = x.copy()
                y[d] += step
                y = np.clip(y, L, U)
                fy = float(func(y))
                n_calls += 1
                if fy < fx:
                    x, fx = y, fy
                    improved = True
                    no_improve = 0
                    # Try to take another step in same direction
                    y2 = x.copy()
                    y2[d] += step
                    y2 = np.clip(y2, L, U)
                    if n_calls < budget and time.time() - t0 < DEADLINE:
                        fy2 = float(func(y2))
                        n_calls += 1
                        if fy2 < fx:
                            x, fx = y2, fy2
                        else:
                            # Shrink step since we overshot
                            step *= 0.5
                else:
                    # Try negative step
                    y = x.copy()
                    y[d] -= step
                    y = np.clip(y, L, U)
                    fy = float(func(y))
                    n_calls += 1
                    if fy < fx:
                        x, fx = y, fy
                        improved = True
                        no_improve = 0
                        # Try to take another step in same direction
                        y2 = x.copy()
                        y2[d] -= step
                        y2 = np.clip(y2, L, U)
                        if n_calls < budget and time.time() - t0 < DEADLINE:
                            fy2 = float(func(y2))
                            n_calls += 1
                            if fy2 < fx:
                                x, fx = y2, fy2
                            else:
                                step *= 0.5
                    else:
                        pass  # No improvement in this dimension

            if not improved:
                no_improve += 1
                step *= 0.5
                if no_improve >= max_no_improve:
                    break
            else:
                step *= 1.1  # Slight expansion when improving

            if step < 1e-8:
                break

        # ---- Phase 3: Adaptive (1+1)-ES for fine-grained refinement ----
        sigma = max(step * 0.5, 0.01)
        while n_calls < budget and time.time() - t0 < DEADLINE:
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, L, U)
            fy = float(func(y))
            n_calls += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
            else:
                sigma *= 0.8
            if sigma < 1e-10:
                break

        return (x, fx)
# EVOLVE-BLOCK-END

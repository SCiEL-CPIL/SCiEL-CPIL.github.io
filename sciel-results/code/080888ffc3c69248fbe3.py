"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Adaptive (1+1)-ES with 1/5 success rule, deterministic restarts and a
    wall-clock deadline guard. Tuned for early progress (AOCC) on Schaffers F7
    in 5-D, where the minimizer lies in [-4, 4]^5."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        deadline = t_start + 250.0  # safety margin well under the 330 s kill

        # Deterministic, reproducible RNG.
        rng = np.random.default_rng(12345)

        # --- Early diverse probes: establish a low best value quickly. ---
        # The origin is a strong prior (minimizer in [-4,4]^5); a few probes
        # around it cover the central region cheaply for AOCC.
        init_points = [np.zeros(dim)]
        init_points.append(rng.uniform(-1.0, 1.0, dim))
        init_points.append(rng.uniform(-2.0, 2.0, dim))
        init_points.append(rng.uniform(-3.0, 3.0, dim))

        best_x = None
        best_f = np.inf
        used = 0
        for p in init_points:
            if used >= budget:
                break
            fp = float(func(np.asarray(p, dtype=float)))
            used += 1
            if fp < best_f:
                best_f = fp
                best_x = np.asarray(p, dtype=float).copy()

        # --- Adaptive (1+1)-ES with 1/5 success rule and restarts. ---
        sigma = 0.5
        cur_x = best_x.copy()
        cur_f = best_f
        since_improve = 0
        max_no_improve = 40

        while used < budget:
            if time.time() > deadline:
                break

            y = cur_x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5.0, 5.0)
            fy = float(func(y))
            used += 1

            if fy < cur_f:
                # Accept: move and grow the step (1/5 success rule).
                cur_x = y
                cur_f = fy
                since_improve = 0
                sigma *= 1.5
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            else:
                since_improve += 1
                sigma *= 0.87

            # Restart on prolonged stagnation: jump to a fresh point in the
            # central region and reset the step. Capped by the budget loop.
            if since_improve >= max_no_improve:
                since_improve = 0
                sigma = 0.5
                new_p = rng.uniform(-1.5, 1.5, dim)
                fnp = float(func(np.clip(new_p, -5.0, 5.0)))
                used += 1
                if fnp < cur_f:
                    cur_x = np.clip(new_p, -5.0, 5.0)
                    cur_f = fnp
                    if fnp < best_f:
                        best_f = fnp
                        best_x = cur_x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END

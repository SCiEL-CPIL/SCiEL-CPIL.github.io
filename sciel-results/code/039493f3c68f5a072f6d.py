"""Combined optimizer for BBOB f24 (Lunacek bi-Rastrigin):
structured basin scan + (1+1)-ES with adaptive sigma, periodic re-exploration,
and stagnation restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coarse scan to locate basins, then (1+1)-ES with adaptive sigma and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 295.0  # safety margin before the 300s kill

        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(7)

        calls = 0
        best_x = None
        best_f = np.inf

        def record(x, f):
            nonlocal calls, best_x, best_f
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def eval_pt(x):
            x = np.clip(x, -5.0, 5.0)
            return float(func(x))

        # ---- Phase 1: coarse structured scan to find promising basins ----
        # Lunacek f24 has two basins; structured points help locate both.
        candidates = [np.zeros(n)]
        # Axis-aligned offsets
        for i in range(n):
            for s in (-3.0, -1.5, 1.5, 3.0):
                p = np.zeros(n)
                p[i] = s
                candidates.append(p)
        # Diagonal points
        for s in (-2.0, -1.0, 1.0, 2.0):
            candidates.append(np.full(n, s))
        # Random points in the box
        for _ in range(30):
            candidates.append(rng.uniform(-4.5, 4.5, n))
        
        # Sort candidates by distance to origin (center is often a good starting point)
        candidates.sort(key=lambda x: np.linalg.norm(x))

        for x in candidates:
            if calls >= budget or time.time() > deadline:
                break
            record(x, eval_pt(x))

        if best_x is None:
            best_x = np.zeros(n)
            best_f = eval_pt(best_x)
            record(best_x, best_f)

        if time.time() > deadline:
            return best_x, best_f

        # ---- Phase 2: (1+1)-ES with adaptive sigma + periodic re-exploration ----
        x = best_x.copy()
        fx = best_f
        sigma = 2.0
        no_improve = 0
        max_no_improve = 30
        exploration_every = 20  # every 20 steps, try a random re-exploration
        steps_since_exploration = 0

        while calls < budget:
            if time.time() > deadline:
                break

            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(n), -5.0, 5.0)
            fy = eval_pt(y)
            record(y, fy)
            steps_since_exploration += 1

            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.1       # increase on accept (faster convergence)
                sigma = min(sigma, 5.0)
                no_improve = 0
            else:
                sigma *= 0.85      # decrease on reject
                sigma = max(sigma, 0.01)
                no_improve += 1

            # Periodic re-exploration: try a few random points to find other basins
            if steps_since_exploration >= exploration_every and no_improve > 5:
                steps_since_exploration = 0
                found_better = False
                for _ in range(3):
                    if calls >= budget or time.time() > deadline:
                        break
                    xr = rng.uniform(-4.5, 4.5, n)
                    fr = eval_pt(xr)
                    record(xr, fr)
                    if fr < fx:
                        x = xr
                        fx = fr
                        sigma = 2.0
                        no_improve = 0
                        found_better = True
                        break
                if found_better:
                    # Reset exploration counter since we found something
                    steps_since_exploration = 0

            # Stagnation restart: if stuck for too long, restart from best with fresh sigma
            if no_improve > max_no_improve:
                x = best_x.copy()
                fx = best_f
                sigma = 2.0
                no_improve = 0
                steps_since_exploration = 0
                max_no_improve = max(20, max_no_improve - 5)  # tighten after restart

        return best_x, best_f
# EVOLVE-BLOCK-END

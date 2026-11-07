"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start coordinate descent exploiting Rastrigin periodicity."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin

        rng = np.random.default_rng(0)

        best_x = np.zeros(dim)
        best_f = np.inf
        evals = 0

        # Multi-start strategy:
        # 1. Evaluate a few random starts.
        # 2. For each start, perform a coordinate descent refinement.
        #    Since Rastrigin has period 2, we can try shifting coordinates
        #    by +/- 1.0 to jump to the next local minimum if stuck.

        num_starts = 10
        points = rng.uniform(-5, 5, (num_starts, dim))

        for p in points:
            if evals >= budget or time.time() > deadline:
                break

            x = p.copy()
            fx = func(x)
            evals += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Coordinate descent refinement
            # We try to improve each coordinate.
            # For Rastrigin, the local minima are at integers.
            # So we check x, x+1, x-1 for each coordinate.
            improved = True
            while improved and evals < budget and time.time() < deadline:
                improved = False
                for d in range(dim):
                    # Try original, +1, -1
                    candidates = [
                        x.copy(),
                        np.clip(x + np.eye(dim)[d], -5, 5),
                        np.clip(x - np.eye(dim)[d], -5, 5)
                    ]
                    # Also try a small local step if not near integer?
                    # Actually, Rastrigin minima are at integers.
                    # If we are at 3.2, moving to 3.0 or 4.0 is good.
                    # But we don't know the integers.
                    # However, the period is 2. So x, x+1, x-1 covers neighbors.
                    
                    min_f = np.inf
                    min_x = None
                    for cand in candidates:
                        if evals >= budget or time.time() > deadline:
                            break
                        f_cand = func(cand)
                        evals += 1
                        if f_cand < min_f:
                            min_f = f_cand
                            min_x = cand

                    if min_x is None:
                        continue

                    if min_f < fx:
                        x = min_x
                        fx = min_f
                        improved = True
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    else:
                        # If no improvement from +-1, maybe we are close to a minimum?
                        # Or we are stuck in a plateau?
                        # Let's try a small random perturbation to escape?
                        # Or just break out of this coordinate loop for this start?
                        # If we didn't improve at all in a full pass, break.
                        pass

        # If we have budget left, do some random restarts
        while evals < budget and time.time() < deadline:
            x_try = rng.uniform(-5, 5, dim)
            f_try = func(x_try)
            evals += 1
            if f_try < best_f:
                best_f = f_try
                best_x = x_try.copy()
                # Quick refinement
                improved = True
                while improved and evals < budget and time.time() < deadline:
                    improved = False
                    for d in range(dim):
                        candidates = [
                            best_x.copy(),
                            np.clip(best_x + np.eye(dim)[d], -5, 5),
                            np.clip(best_x - np.eye(dim)[d], -5, 5)
                        ]
                        min_f = np.inf
                        min_x = None
                        for cand in candidates:
                            if evals >= budget or time.time() > deadline:
                                break
                            f_cand = func(cand)
                            evals += 1
                            if f_cand < min_f:
                                min_f = f_cand
                                min_x = cand
                        if min_x is None:
                            continue
                        if min_f < best_f:
                            best_x = min_x
                            best_f = min_f
                            improved = True

        return best_x, best_f
# EVOLVE-BLOCK-END

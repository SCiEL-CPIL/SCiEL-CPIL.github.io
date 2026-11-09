"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: multi-start CMA-ES-like with coordinate descent refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(12345)

        dim = self.dim
        budget = self.budget

        # Track best
        best_x = None
        best_fx = np.inf
        calls_made = 0

        def safe_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            return fx

        def update_best(x, fx):
            nonlocal best_x, best_fx
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        # Phase 1: Multi-start exploration with (1+1)-ES
        # Try several starting points to find good basins
        n_starts = 5
        for s in range(n_starts):
            if calls_made >= budget or time.time() > deadline:
                break

            if s == 0:
                x = np.zeros(dim)
            else:
                x = rng.uniform(-4, 4, dim)

            fx = safe_func(x)
            update_best(x, fx)

            if calls_made >= budget or time.time() > deadline:
                break

            # (1+1)-ES from this start
            sigma = 1.0
            no_improve = 0
            max_no_improve = 30

            while calls_made < budget and time.time() < deadline:
                y = x + sigma * rng.standard_normal(dim)
                fy = safe_func(y)

                if fy < fx:
                    x = y
                    fx = fy
                    no_improve = 0
                    update_best(x, fx)
                    sigma *= 1.15
                else:
                    no_improve += 1
                    sigma *= 0.85

                    if no_improve >= max_no_improve:
                        break

        if calls_made >= budget or time.time() > deadline:
            return best_x, best_fx

        # Phase 2: Coordinate descent refinement from best point
        if calls_made < budget and time.time() < deadline - 10:
            x_cd = best_x.copy()
            fx_cd = best_fx
            step_sizes = [2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]

            for step in step_sizes:
                if calls_made >= budget or time.time() > deadline - 5:
                    break

                improved = True
                while improved and calls_made < budget and time.time() < deadline - 5:
                    improved = False
                    coord_order = rng.permutation(dim)

                    for coord in coord_order:
                        if calls_made >= budget or time.time() > deadline - 5:
                            break

                        for direction in [1.0, -1.0]:
                            if calls_made >= budget or time.time() > deadline - 5:
                                break

                            candidate = x_cd.copy()
                            candidate[coord] += direction * step
                            candidate = np.clip(candidate, -5, 5)

                            if np.allclose(candidate, x_cd, atol=1e-10):
                                continue

                            fcandidate = safe_func(candidate)

                            if fcandidate < fx_cd:
                                x_cd = candidate
                                fx_cd = fcandidate
                                update_best(x_cd, fx_cd)
                                improved = True
                                break

        if calls_made >= budget or time.time() > deadline - 5:
            return best_x, best_fx

        # Phase 3: Fine-tuning with small perturbations
        if calls_made < budget and time.time() < deadline - 2:
            x_fine = best_x.copy()
            fx_fine = best_fx
            fine_steps = [0.001, 0.0005, 0.0002, 0.0001]

            for step in fine_steps:
                if calls_made >= budget or time.time() > deadline - 2:
                    break

                improved = True
                while improved and calls_made < budget and time.time() > deadline - 2:
                    improved = False
                    coord_order = rng.permutation(dim)

                    for coord in coord_order:
                        if calls_made >= budget or time.time() > deadline - 2:
                            break

                        for direction in [1.0, -1.0]:
                            if calls_made >= budget or time.time() > deadline - 2:
                                break

                            candidate = x_fine.copy()
                            candidate[coord] += direction * step
                            candidate = np.clip(candidate, -5, 5)

                            if np.allclose(candidate, x_fine, atol=1e-12):
                                continue

                            fcandidate = safe_func(candidate)

                            if fcandidate < fx_fine:
                                x_fine = candidate
                                fx_fine = fcandidate
                                update_best(x_fine, fx_fine)
                                improved = True
                                break

        return best_x, best_fx
# EVOLVE-BLOCK-END

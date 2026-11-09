"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Pure cyclic coordinate descent with adaptive steps and periodic perturbations."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Multi-start initialization to find good basins
        # Lunacek bi-Rastrigin has two main basins; try several starting points
        candidates = []
        for i in range(min(5, budget // 100)):
            x_init = rng.uniform(-5, 5, dim)
            fx_init = func(x_init)
            candidates.append((fx_init, x_init))
            if not candidates or fx_init < candidates[0][0]:
                candidates.sort(key=lambda t: t[0])
        
        calls = len(candidates)
        if not candidates:
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls = 1
        else:
            fx, x = candidates[0]

        # Phase 2: Cyclic coordinate descent with per-dimension adaptive steps
        # Initialize step sizes per dimension
        steps = np.full(dim, 1.0)
        step_min = 1e-8
        step_max = 3.0

        # Global decay factor for when no improvement occurs
        global_decay = 0.85
        no_improve_count = 0
        best_fx = fx
        max_cd_iterations = budget - calls

        cd_iter = 0
        while cd_iter < max_cd_iterations and calls < budget:
            if time.time() >= deadline:
                break

            improved_this_cycle = False
            for d in range(dim):
                if calls >= budget:
                    break
                if time.time() >= deadline:
                    break

                # Try positive step
                y = x.copy()
                y[d] = np.clip(y[d] + steps[d], -5, 5)
                fy = func(y)
                calls += 1

                if fy < fx:
                    x, fx = y, fy
                    improved_this_cycle = True
                    no_improve_count = 0
                    # Increase step slightly on success
                    steps[d] = min(steps[d] * 1.05, step_max)
                else:
                    # Try negative step
                    y = x.copy()
                    y[d] = np.clip(y[d] - steps[d], -5, 5)
                    fy = func(y)
                    calls += 1

                    if fy < fx:
                        x, fx = y, fy
                        improved_this_cycle = True
                        no_improve_count = 0
                        # Increase step slightly on success
                        steps[d] = min(steps[d] * 1.05, step_max)
                    else:
                        # No improvement in either direction; decrease step
                        steps[d] = max(steps[d] * 0.9, step_min)
                        no_improve_count += 1

                # Track best
                if fx < best_fx:
                    best_fx = fx

            cd_iter += 1

            # If no improvement in a full cycle, apply global decay and periodic perturbation
            if not improved_this_cycle:
                steps *= global_decay
                steps = np.clip(steps, step_min, step_max)

                # Periodic random perturbation to escape local minima
                # Every 50 iterations without improvement, perturb
                if no_improve_count % 50 == 0 and calls < budget - 20:
                    # Small random perturbation
                    perturbation = 0.1 * rng.standard_normal(dim)
                    x_perturbed = np.clip(x + perturbation, -5, 5)
                    fx_perturbed = func(x_perturbed)
                    calls += 1

                    if fx_perturbed < fx:
                        x, fx = x_perturbed, fx_perturbed
                        # Reset some steps to explore from new location
                        steps = np.full(dim, 0.5)
                        no_improve_count = 0

        # Phase 3: Final fine-grained polish using very small steps
        if calls < budget and time.time() < deadline - 2:
            polish_calls = budget - calls
            sigma_polish = 1e-4
            for _ in range(min(polish_calls, 50)):
                if time.time() >= deadline:
                    break
                y = np.clip(x + sigma_polish * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma_polish = min(sigma_polish * 1.02, 1e-3)
                else:
                    sigma_polish = max(sigma_polish * 0.95, 1e-8)

        return x, fx
# EVOLVE-BLOCK-END

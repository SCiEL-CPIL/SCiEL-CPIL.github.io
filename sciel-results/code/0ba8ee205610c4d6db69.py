"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic lattice search followed by (1+1)-ES refinement for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 270.0  # safety margin before 300s kill

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 1: Deterministic 3^5 lattice search for global coverage
        # 3 levels per dimension: -3, 0, +3 (covers most of [-5,5] with good spacing)
        # 3^5 = 243 evaluations
        lattice_levels = np.array([-3.0, 0.0, 3.0])
        lattice_calls = 0
        for i in range(3 ** dim):
            if time.time() > deadline:
                break
            # Convert index to multi-dimensional coordinate
            coords = []
            tmp = i
            for d in range(dim):
                coords.append(tmp % 3)
                tmp //= 3
            x = np.array([lattice_levels[c] for c in coords])
            f = func(x)
            update_best(x, f)
            lattice_calls += 1

        remaining = budget - lattice_calls
        if remaining <= 0 or best_x is None:
            return best_x, best_f

        # Phase 2: (1+1)-ES refinement around the best lattice point
        # Adaptive step size with restarts on stagnation
        x = best_x.copy()
        fx = best_f
        sigma = 0.5  # initial step size
        stagnation_counter = 0
        max_stagnation = 15  # restart if no improvement for this many steps
        restart_count = 0
        max_restarts = 3

        while remaining > 0 and time.time() < deadline:
            # Generate one offspring
            child = x + sigma * rng.standard_normal(dim)
            child = np.clip(child, -5, 5)
            fc = func(child)
            remaining -= 1

            if fc < fx:
                x = child
                fx = fc
                update_best(x, fx)
                stagnation_counter = 0
                # Slightly increase sigma on success
                sigma = min(sigma * 1.1, 3.0)
            else:
                stagnation_counter += 1
                # Decrease sigma on failure
                sigma = max(sigma * 0.85, 0.01)

            # Check for stagnation and restart
            if stagnation_counter >= max_stagnation:
                restart_count += 1
                if restart_count > max_restarts:
                    break
                # Restart from a random point near the best or from a new random location
                if restart_count <= max_restarts:
                    if rng.random() < 0.5:
                        # Local restart: perturb best
                        x = best_x + rng.normal(0, 1.0, dim)
                    else:
                        # Global restart: new random point
                        x = rng.uniform(-5, 5, dim)
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    remaining -= 1
                    update_best(x, fx)
                    sigma = 1.0
                    stagnation_counter = 0

        # Phase 3: Final fine-tuning with very small steps
        if remaining > 0 and best_x is not None and time.time() < deadline:
            fine_sigma = 0.05
            for _ in range(min(remaining, 50)):
                if time.time() > deadline:
                    break
                child = best_x + fine_sigma * rng.standard_normal(dim)
                child = np.clip(child, -5, 5)
                f = func(child)
                if f < best_f:
                    best_f = f
                    best_x = child
                    fine_sigma = min(fine_sigma * 1.2, 0.5)
                else:
                    fine_sigma = max(fine_sigma * 0.8, 0.005)
                remaining -= 1

        return best_x, best_f
# EVOLVE-BLOCK-END

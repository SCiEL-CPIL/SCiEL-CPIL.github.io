"""Lattice-based random walk exploiting Rastrigin's periodic structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Lattice-based random walk for Rastrigin rotated (f15)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Global best
        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Coarse grid search to find a good starting region
        # Rastrigin has global minimum at 0 in each dimension (before rotation/shift)
        # The shifted minimum is in [-4, 4]^5
        # Use a coarse grid with step 1.0 to find approximate basin

        # Start with a few random points to get a sense of the landscape
        n_init = 10
        for _ in range(n_init):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            evaluate(x)

        # Phase 2: Lattice-based local search
        # Use discrete steps that align with Rastrigin's period of 2π
        # But since the function is rotated, we don't know the axes.
        # Instead, use adaptive step sizes that start large and shrink.

        # Multi-start local search with lattice moves
        # Each local search: try moves in all 2*dim directions plus diagonal moves
        # Step size adapts based on success rate

        n_restarts = 50
        for restart in range(n_restarts):
            if calls >= budget or time.time() > deadline:
                break

            # Choose starting point: either best so far or random
            if restart == 0 or best_x is None:
                x = rng.uniform(-5, 5, dim)
            elif restart % 3 == 0:
                x = best_x.copy() + rng.normal(0, 1.0, dim)
                x = np.clip(x, -5, 5)
            else:
                x = rng.uniform(-5, 5, dim)

            f = evaluate(x)
            if calls >= budget or time.time() > deadline:
                break

            # Local search with adaptive step size
            step = 1.0  # initial step size
            no_improve = 0
            max_no_improve = 20

            # Directions: axis-aligned and some diagonal
            # For dim=5, use axis-aligned moves
            while no_improve < max_no_improve and calls < budget and time.time() < deadline:
                improved = False

                # Try axis-aligned moves
                for d in range(dim):
                    if calls >= budget or time.time() > deadline:
                        break
                    for direction in [1, -1]:
                        if calls >= budget or time.time() > deadline:
                            break
                        x_try = x.copy()
                        x_try[d] += direction * step
                        x_try = np.clip(x_try, -5, 5)
                        f_try = evaluate(x_try)

                        if f_try < f:
                            x = x_try
                            f = f_try
                            improved = True
                            no_improve = 0
                            # Try to continue in same direction
                            # (greedy: keep moving in this direction if it helps)
                            while calls < budget and time.time() < deadline:
                                x_next = x.copy()
                                x_next[d] += direction * step
                                x_next = np.clip(x_next, -5, 5)
                                f_next = evaluate(x_next)
                                if f_next < f:
                                    x = x_next
                                    f = f_next
                                else:
                                    break
                            break

                if not improved:
                    no_improve += 1
                    # Shrink step size
                    step *= 0.7
                    if step < 0.01:
                        break

                # Occasionally try a larger jump to escape local minima
                if no_improve % 5 == 0 and step > 0.1:
                    x_jump = x + rng.normal(0, 2.0, dim)
                    x_jump = np.clip(x_jump, -5, 5)
                    f_jump = evaluate(x_jump)
                    if f_jump < f:
                        x = x_jump
                        f = f_jump
                        step = 1.0
                        no_improve = 0

        return best_x, best_f
# EVOLVE-BLOCK-END

import numpy as np
import time

class Optimizer:
    """Deterministic multi-scale optimizer with structured restarts for BBOB f24."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # Safety margin before 330s kill

        best_x = None
        best_f = float('inf')
        calls = 0

        def evaluate(x):
            nonlocal calls, best_f, best_x
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coarse grid-like exploration with deterministic offsets
        # Use structured sampling to cover the space efficiently
        n_coarse = min(50, budget // 10)
        if n_coarse > 0:
            # Create deterministic points using Latin hypercube-like stratification
            for i in range(n_coarse):
                if calls >= budget or time.time() > deadline:
                    break
                # Deterministic pseudo-random but structured
                seed = 42 + i * 7
                rng = np.random.default_rng(seed)
                x = rng.uniform(-5, 5, dim)
                evaluate(x)

        # Phase 2: Multi-scale local search with geometric step decay
        # Start from best point found, use deterministic perturbation directions
        if best_x is not None and calls < budget:
            x = best_x.copy()
            fx = best_f

            # Precompute deterministic perturbation directions
            rng = np.random.default_rng(123)
            directions = rng.standard_normal((20, dim))
            directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

            # Geometric sequence of step sizes
            sigma_start = 3.0
            sigma_end = 1e-3
            n_scales = 8
            sigmas = np.geomspace(sigma_start, sigma_end, n_scales)

            for sigma in sigmas:
                if calls >= budget or time.time() > deadline:
                    break

                improved = False
                # Try all directions at this scale
                for d in directions:
                    if calls >= budget or time.time() > deadline:
                        break

                    # Try both positive and negative directions
                    for sign in [1, -1]:
                        if calls >= budget or time.time() > deadline:
                            break

                        y = x + sign * sigma * d
                        fy = evaluate(y)

                        if fy < fx - 1e-10:
                            x, fx = y, fy
                            improved = True
                            break  # Move to next direction

                # If no improvement at this scale, continue to smaller scales
                if not improved:
                    # Check for plateau: if we've been stuck, do a larger jump
                    if sigma > sigma_end * 10:
                        # Jump to a new location based on current best
                        rng_jump = np.random.default_rng(int(time.time() * 1000) % 100000)
                        jump_size = sigma * 2
                        y = x + rng_jump.standard_normal(dim) * jump_size
                        fy = evaluate(y)
                        if fy < fx:
                            x, fx = y, fy

        # Phase 3: Final high-precision polish pass
        if best_x is not None and calls < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f

            # Very fine step sizes for polish
            polish_sigmas = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
            rng_polish = np.random.default_rng(999)

            for sigma in polish_sigmas:
                if calls >= budget or time.time() > deadline:
                    break

                # Multiple iterations at each polish scale
                for _ in range(5):
                    if calls >= budget or time.time() > deadline:
                        break

                    # Try several random directions
                    for _ in range(3):
                        if calls >= budget or time.time() > deadline:
                            break

                        d = rng_polish.standard_normal(dim)
                        d = d / np.linalg.norm(d)

                        y = x + sigma * d
                        fy = evaluate(y)

                        if fy < fx - 1e-12:
                            x, fx = y, fy
                            break

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls += 1

        return best_x, best_f

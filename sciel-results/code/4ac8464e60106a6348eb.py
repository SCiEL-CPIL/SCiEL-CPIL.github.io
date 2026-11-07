"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Sobol quasi-random global search with adaptive centering and
    Rastrigin-aware coordinate refinement.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _sobol_sequence(self, n, dim, rng_seed=42):
        """Generate a Sobol quasi-random sequence in [0,1)^dim."""
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=dim, scramble=True, seed=rng_seed)
        # Need to generate at least n points; Sobol requires power-of-2 for best quality
        # but we can just generate n points
        points = sampler.random(n)
        return points

    def __call__(self, func):
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 330s kill

        def time_ok():
            return time.time() < deadline

        calls = 0
        best_x = None
        best_fx = np.inf

        def evaluate(x):
            nonlocal calls, best_x, best_fx
            fx = func(x)
            calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 1: Sobol quasi-random sampling for uniform basin coverage
        # Use a good chunk of budget for exploration
        n_sobol = min(200, budget - 100)
        sobol_points = self._sobol_sequence(n_sobol, dim, rng_seed=42)

        # Map from [0,1)^dim to [-5,5]^dim
        for i in range(n_sobol):
            if not time_ok() or calls >= budget - 50:
                break
            x = sobol_points[i] * 10.0 - 5.0
            x = np.clip(x, -5, 5)
            evaluate(x)

        # Phase 2: Adaptive variance reduction - shift sampling center to best point
        # and scale step size to exploit Rastrigin periodicity
        if best_x is not None and time_ok() and calls < budget - 50:
            # Continue with Sobol but centered on best point with shrinking radius
            remaining_phase2 = min(150, budget - calls - 50)
            # Use multiple radii to cover different scales
            radii = [2.0, 1.0, 0.5, 0.25, 0.1]
            points_per_radius = max(1, remaining_phase2 // len(radii))

            for radius in radii:
                if not time_ok() or calls >= budget - 50:
                    break
                n_points = min(points_per_radius, budget - calls - 50)
                if n_points <= 0:
                    continue

                # Generate Sobol points in a box around best_x
                sobol_local = self._sobol_sequence(n_points, dim, rng_seed=42 + int(radius * 100))
                for i in range(n_points):
                    if not time_ok() or calls >= budget - 50:
                        break
                    # Map Sobol [0,1)^dim to [-radius, radius)^dim
                    offset = sobol_local[i] * 2.0 * radius - radius
                    x = best_x + offset
                    x = np.clip(x, -5, 5)
                    evaluate(x)

        # Phase 3: Rastrigin-aware refinement
        # Rastrigin has local minima at integer points; try snapping coordinates
        if best_x is not None and time_ok() and calls < budget - 10:
            # Try integer snapping for each coordinate
            for coord in range(dim):
                if not time_ok() or calls >= budget - 5:
                    break
                # Try rounding current coordinate to nearest integer
                x_int = best_x.copy()
                x_int[coord] = np.round(best_x[coord])
                x_int = np.clip(x_int, -5, 5)
                evaluate(x_int)

            # Also try nearby integers (floor and ceil)
            for coord in range(dim):
                if not time_ok() or calls >= budget - 5:
                    break
                for delta in [-1, 1]:
                    x_near = best_x.copy()
                    x_near[coord] = np.round(best_x[coord]) + delta
                    x_near = np.clip(x_near, -5, 5)
                    evaluate(x_near)

        # Phase 4: Fine-grained local search around the best point
        if best_x is not None and time_ok() and calls < budget - 2:
            sigma_small = 0.05
            max_local_steps = min(50, budget - calls - 2)
            for _ in range(max_local_steps):
                if not time_ok() or calls >= budget - 2:
                    break
                y = np.clip(best_x + sigma_small * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()
                    sigma_small *= 1.05
                else:
                    sigma_small *= 0.85
                    if sigma_small < 1e-10:
                        break

        # Ensure we return something valid
        if best_x is None:
            # Fallback: return a default point
            best_x = np.zeros(dim)
            best_fx = func(best_x)
            calls += 1

        return best_x, best_fx
# EVOLVE-BLOCK-END

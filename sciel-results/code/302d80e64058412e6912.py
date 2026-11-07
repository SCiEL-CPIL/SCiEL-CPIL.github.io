import numpy as np
import time


class Optimizer:
    """Hybrid structured search + CMA-ES style local refinement for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self._calls = 0
        self._deadline = time.time() + 280.0  # safety margin before 300s
        self._best_x = None
        self._best_f = np.inf

        rng = np.random.default_rng(0)
        dim = self.dim

        def track(x, f):
            self._calls += 1
            if f < self._best_f:
                self._best_f = f
                self._best_x = x.copy()
            return f

        # ---- Phase 1: Structured lattice search exploiting Rastrigin periodicity ----
        # Rastrigin has local minima near integer points (after shift/rotation,
        # the structure is preserved but shifted). We probe a grid of candidate
        # minima at multi-scale resolutions.

        # First, try a coarse grid to find which basin we're in
        # The shift is in [-4,4]^5, so the global minimum is somewhere in that box.
        # Try a grid of step 1.0 over [-4, 4] per dimension (but that's 9^5=59049, too many)
        # Instead, use a smarter approach: try several scales.

        # Scale 1: Very coarse - 3 points per dim over [-5,5]
        grid_coarse = np.linspace(-4, 4, 3)  # [-4, 0, 4]
        # That's only 3^5 = 241 points. Good.

        for i1 in range(3):
            for i2 in range(3):
                for i3 in range(3):
                    for i4 in range(3):
                        for i5 in range(3):
                            if self._calls >= self.budget:
                                return self._best_x, self._best_f
                            if time.time() > self._deadline:
                                return self._best_x, self._best_f
                            x = np.array([grid_coarse[i1], grid_coarse[i2], grid_coarse[i3],
                                          grid_coarse[i4], grid_coarse[i5]], dtype=float)
                            f = track(x, func(x))

        # Scale 2: Medium grid - 5 points per dim over [-4,4]
        # 5^5 = 3125 points. With ~241 already used, ~2884 remaining from this phase.
        grid_med = np.linspace(-4, 4, 5)  # [-4, -2, 0, 2, 4]
        remaining = self.budget - self._calls
        # We have limited budget. Let's do a smarter selection.
        # Actually, let's use the best from coarse grid to guide medium grid.
        # Do a local refinement around the best coarse point.

        if self._best_x is not None:
            # Phase 2: Multi-scale local search around best point
            # Try offsets at decreasing scales
            scales = [1.0, 0.5, 0.25, 0.1, 0.05]
            for scale in scales:
                if self._calls >= self.budget:
                    return self._best_x, self._best_f
                if time.time() > self._deadline:
                    return self._best_x, self._best_f

                # Generate candidate points on a small grid around best_x
                # Use ±scale offsets in each dimension (3^5 = 243 candidates)
                # But that might be too many. Let's be selective.
                # Try the best few points from previous scale as centers.

                # Simple: do a (1+1)-ES style search from best_x
                x = self._best_x.copy()
                fx = self._best_f

                # Adaptive step size
                step = scale
                improved = False
                for _ in range(20):
                    if self._calls >= self.budget:
                        break
                    if time.time() > self._deadline:
                        break
                    y = x + step * rng.standard_normal(dim)
                    y = np.clip(y, -5, 5)
                    fy = track(y, func(y))
                    if fy < fx:
                        x, fx = y, fy
                        step *= 1.3
                        improved = True
                    else:
                        step *= 0.75

        # Phase 3: Final refinement with smaller steps
        if self._calls < self.budget and self._best_x is not None:
            x = self._best_x.copy()
            fx = self._best_f
            step = 0.01
            for _ in range(self.budget - self._calls):
                if time.time() > self._deadline:
                    break
                y = x + step * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = track(y, func(y))
                if fy < fx:
                    x, fx = y, fy
                    step *= 1.1
                else:
                    step *= 0.9

        return self._best_x, self._best_f

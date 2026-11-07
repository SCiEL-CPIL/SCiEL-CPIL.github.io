"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Grid-search + (1+1)-ES hybrid targeting Büche-Rastrigin periodicity."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        # Phase 1: Coarse grid search to find good basin centers
        # Büche-Rastrigin has periodic structure with period 2 in each dimension
        # The function is roughly: sum of (x_i - shift_i)^2 * something + 10*(1-cos(2*pi*x_i))
        # So minima are near integer/2-integer points after accounting for shift/rotation
        
        best_x = None
        best_f = np.inf
        calls_used = 0

        def try_point(x):
            nonlocal best_x, best_f, calls_used
            if calls_used >= self.budget:
                return
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Coarse grid: sample at period-2 intervals with some offset
        # Since we don't know the shift, sample multiple offsets
        # Grid spacing of 2.0 (the period) with offsets 0, 0.5, 1.0, 1.5
        grid_points = []
        for offset in [0.0, 0.5, 1.0, 1.5]:
            for i in range(-2, 3):  # -4 to 4 roughly
                pt = np.array([i * 2.0 + offset] * self.dim)
                if np.all(pt >= -5) and np.all(pt <= 5):
                    grid_points.append(pt)
        
        # Also add some random points
        for _ in range(50):
            grid_points.append(rng.uniform(-5, 5, self.dim))
        
        # Shuffle and evaluate
        rng.shuffle(grid_points)
        for pt in grid_points:
            if calls_used >= self.budget or time.time() > deadline:
                break
            try_point(pt)
        
        if best_x is None:
            # Fallback: random point
            x = rng.uniform(-5, 5, self.dim)
            try_point(x)
            if best_x is None:
                return np.zeros(self.dim), np.inf
        
        # Phase 2: Local refinement with (1+1)-ES from the best grid point
        x = best_x.copy()
        fx = best_f
        sigma = 0.5  # start with smaller step since we're near a minimum
        
        # Adaptive sigma with 1/5 success rule
        successes = 0
        window = 10
        
        for _ in range(self.budget - calls_used):
            if time.time() > deadline:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls_used += 1
            
            if fy < fx:
                x, fx = y, fy
                successes += 1
                sigma *= 1.1  # gentle increase
            else:
                sigma *= 0.95  # gentle decrease
            
            # Reset success counter periodically
            if calls_used % window == 0:
                if successes / window < 0.2:
                    sigma *= 0.5  # too many failures, shrink
                elif successes / window > 0.8:
                    sigma *= 1.5  # too many successes, expand
                successes = 0
            
            # Early termination if sigma becomes very small
            if sigma < 1e-6:
                break
        
        return x, fx
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc


class Optimizer:
    """Sobol space-filling search followed by local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit with safety margin

        n = self.dim
        budget = self.budget

        # Phase 1: Sobol sequence for broad exploration
        # Use a Sobol sequence to cover the space systematically
        # We'll use about 60% of budget for exploration
        n_explore = min(300, budget // 2)
        
        # Create Sobol sampler
        sampler = qmc.Sobol(d=n, scramble=True, seed=42)
        # Generate points
        n_points = n_explore
        # Sobol requires power of 2, so round up
        n_pow2 = 1
        while n_pow2 < n_points:
            n_pow2 *= 2
        sobol_samples = sampler.random(n_pow2)
        
        # Map from [0,1]^n to [-5,5]^n
        sobol_samples = sobol_samples * 10.0 - 5.0
        # Take only the first n_points
        sobol_samples = sobol_samples[:n_points]

        best_x = None
        best_f = np.inf
        call_count = 0

        # Evaluate Sobol points
        for i in range(n_points):
            if time.time() > deadline:
                break
            x = sobol_samples[i]
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Local refinement around best point
        # Use a local search with decreasing step sizes
        remaining = budget - call_count
        if remaining <= 0 or best_x is None:
            return best_x, best_f

        # Try several refinement strategies
        # Strategy 1: Nelder-Mead style simplex
        # Strategy 2: Coordinate descent with adaptive step
        # Strategy 3: Random local search with shrinking radius

        # Let's use a combination: coordinate descent + random local search
        x = best_x.copy()
        f = best_f

        # Adaptive coordinate descent
        step = 1.0
        no_improve_count = 0
        max_no_improve = 20

        while call_count < budget and time.time() < deadline:
            improved = False
            for d in range(n):
                if call_count >= budget or time.time() > deadline:
                    break
                # Try positive direction
                x_trial = x.copy()
                x_trial[d] = np.clip(x_trial[d] + step, -5, 5)
                f_trial = func(x_trial)
                call_count += 1
                if f_trial < f:
                    x = x_trial
                    f = f_trial
                    improved = True
                    continue
                # Try negative direction
                x_trial = x.copy()
                x_trial[d] = np.clip(x_trial[d] - step, -5, 5)
                f_trial = func(x_trial)
                call_count += 1
                if f_trial < f:
                    x = x_trial
                    f = f_trial
                    improved = True
            if not improved:
                step *= 0.5
                no_improve_count += 1
                if step < 1e-6 or no_improve_count > max_no_improve:
                    break
            else:
                no_improve_count = 0

        # Phase 3: If still have budget, do random local search around best
        if call_count < budget and time.time() < deadline:
            rng = np.random.default_rng(123)
            radius = 0.5
            for _ in range(budget - call_count):
                if time.time() > deadline:
                    break
                direction = rng.standard_normal(n)
                direction /= (np.linalg.norm(direction) + 1e-10)
                x_trial = np.clip(x + radius * direction, -5, 5)
                f_trial = func(x_trial)
                call_count += 1
                if f_trial < f:
                    x = x_trial
                    f = f_trial
                    radius *= 1.2
                else:
                    radius *= 0.9
                    radius = max(radius, 1e-5)

        # Update best
        if f < best_f:
            best_f = f
            best_x = x

        return best_x, best_f
# EVOLVE-BLOCK-END

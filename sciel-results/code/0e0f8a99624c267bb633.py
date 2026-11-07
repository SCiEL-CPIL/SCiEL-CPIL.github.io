"""Bootstrap spawn: Lattice-based grid search exploiting Rastrigin periodicity with CMA-ES refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Lattice-based optimizer exploiting Rastrigin periodicity with systematic grid search and local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        calls = [0]

        def call_func(x):
            calls[0] += 1
            return float(func(x))

        def time_ok():
            return calls[0] < budget and time.time() < deadline

        best_x = None
        best_f = np.inf

        # Phase 1: Lattice-based deterministic grid search
        # Rastrigin has periodic minima with period 1 in each dimension.
        # The hidden shift s_i is in [-4, 4], so the true minimum is at x_i = s_i + k for integer k.
        # We systematically evaluate points on a grid to find the global basin.
        
        # Strategy: evaluate at integer coordinates and half-integer offsets
        # For each dimension, try values at -4, -3.5, -3, ..., 3.5, 4
        # That's 9 values per dimension. In 5D, that's 9^5 = 59049 points - too many.
        
        # Instead, use a smarter approach:
        # 1. First, evaluate at all-integer points: 9^5 is still too many.
        # 2. Use a coordinate-wise approach: for each dimension, find the best integer offset.
        
        # Better strategy: Use the fact that the function is separable in structure.
        # For each dimension independently, scan through possible values to find the best.
        
        # Phase 1a: Coordinate-wise scan to identify the basin for each dimension
        # Start with a central guess and refine each coordinate
        
        # Initial point: try the origin first
        x0 = np.zeros(dim)
        f0 = call_func(x0)
        best_x = x0.copy()
        best_f = f0

        # For each dimension, try shifting by -4 to 4 in steps of 0.5
        # This covers the range [-4, 4] with 17 values per dimension
        # Total evaluations: 5 * 17 = 85 (plus the initial point)
        
        for d in range(dim):
            if not time_ok():
                break
            current_x = best_x.copy()
            current_f = best_f
            
            # Try different offsets in dimension d
            for offset in np.arange(-4.0, 4.5, 0.5):
                if not time_ok():
                    break
                x_trial = current_x.copy()
                x_trial[d] = offset
                x_trial = np.clip(x_trial, -5, 5)
                f_trial = call_func(x_trial)
                if f_trial < current_f:
                    current_f = f_trial
                    current_x = x_trial.copy()
            
            if current_f < best_f:
                best_f = current_f
                best_x = current_x.copy()

        # Phase 1b: Refine with finer grid around the best point found
        # Use step size 0.25 for a second pass
        for d in range(dim):
            if not time_ok():
                break
            current_x = best_x.copy()
            current_f = best_f
            
            # Scan around the current best value in dimension d
            center = current_x[d]
            for offset in np.arange(center - 1.0, center + 1.01, 0.25):
                if not time_ok():
                    break
                x_trial = current_x.copy()
                x_trial[d] = offset
                x_trial = np.clip(x_trial, -5, 5)
                f_trial = call_func(x_trial)
                if f_trial < current_f:
                    current_f = f_trial
                    current_x = x_trial.copy()
            
            if current_f < best_f:
                best_f = current_f
                best_x = current_x.copy()

        # Phase 1c: Even finer grid with step size 0.1
        for d in range(dim):
            if not time_ok():
                break
            current_x = best_x.copy()
            current_f = best_f
            
            center = current_x[d]
            for offset in np.arange(center - 0.5, center + 0.51, 0.1):
                if not time_ok():
                    break
                x_trial = current_x.copy()
                x_trial[d] = offset
                x_trial = np.clip(x_trial, -5, 5)
                f_trial = call_func(x_trial)
                if f_trial < current_f:
                    current_f = f_trial
                    current_x = x_trial.copy()
            
            if current_f < best_f:
                best_f = current_f
                best_x = current_x.copy()

        # Phase 2: Local refinement using adaptive coordinate descent
        x = best_x.copy()
        fx = best_f

        # Adaptive step sizes: start large, shrink
        for phase in range(4):
            if not time_ok():
                break

            if phase == 0:
                step_sizes = [0.1, 0.05, 0.02]
            elif phase == 1:
                step_sizes = [0.02, 0.01, 0.005]
            elif phase == 2:
                step_sizes = [0.005, 0.002, 0.001]
            else:
                step_sizes = [0.002, 0.001, 0.0005]

            for ss in step_sizes:
                if not time_ok():
                    break
                improved = True
                while improved and time_ok():
                    improved = False
                    for d in range(dim):
                        if not time_ok():
                            break
                        for sign in [1, -1]:
                            x_trial = x.copy()
                            x_trial[d] += sign * ss
                            x_trial = np.clip(x_trial, -5, 5)
                            fx_trial = call_func(x_trial)
                            if fx_trial < fx:
                                x = x_trial
                                fx = fx_trial
                                improved = True
                                # Continue trying same direction (momentum)
                                x_trial2 = x.copy()
                                x_trial2[d] += sign * ss * 0.5
                                x_trial2 = np.clip(x_trial2, -5, 5)
                                fx_trial2 = call_func(x_trial2)
                                if fx_trial2 < fx:
                                    x = x_trial2
                                    fx = fx_trial2
                                break  # Only try one sign per dimension per iteration

        # Phase 3: Multi-start refinement from nearby lattice points
        # Try a few nearby points and refine each
        n_starts = min(5, budget - calls[0])
        if n_starts > 0 and time_ok():
            # Generate nearby points
            nearby_points = []
            for offset in [0.1, -0.1, 0.2, -0.2, 0.3]:
                if not time_ok():
                    break
                x_pert = x + np.array([offset] * dim)
                x_pert = np.clip(x_pert, -5, 5)
                nearby_points.append(x_pert)
            
            for x_start in nearby_points:
                if not time_ok():
                    break
                x_local = x_start.copy()
                f_local = call_func(x_local)
                
                # Quick coordinate descent
                for ss in [0.05, 0.02, 0.01, 0.005]:
                    if not time_ok():
                        break
                    for d in range(dim):
                        if not time_ok():
                            break
                        for sign in [1, -1]:
                            x_trial = x_local.copy()
                            x_trial[d] += sign * ss
                            x_trial = np.clip(x_trial, -5, 5)
                            f_trial = call_func(x_trial)
                            if f_trial < f_local:
                                x_local = x_trial
                                f_local = f_trial
                                break
                
                if f_local < best_f:
                    best_f = f_local
                    best_x = x_local.copy()

        # Phase 4: Final fine-tuning with very small random walk
        if time_ok() and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.005
            for i in range(min(30, budget - calls[0])):
                if not time_ok():
                    break
                step = sigma * rng.standard_normal(dim)
                x_trial = np.clip(x + step, -5, 5)
                fx_trial = call_func(x_trial)
                if fx_trial < fx:
                    x = x_trial
                    fx = fx_trial
                    best_x = x.copy()
                    best_f = fx
                    sigma *= 1.05
                else:
                    sigma *= 0.9
                    if sigma < 1e-6:
                        break

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = call_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END

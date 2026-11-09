"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Valley-tracking coordinate descent with adaptive step sizing for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 300s limit
        
        def elapsed():
            return time.time() - start_time

        def time_left():
            return deadline - time.time()

        n = self.dim
        budget = self.budget
        call_count = [0]
        
        def safe_func(x):
            """Wrapper that counts calls and enforces deadline."""
            if call_count[0] >= budget or elapsed() > deadline - 2:
                return float('inf')
            call_count[0] += 1
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            return func(x)

        # Phase 1: Coarse grid search to find the valley entrance
        # The minimizer is in [-4, 4]^5. A 3^5 = 243 point grid is affordable.
        # This ensures we start near the valley floor, avoiding the flat regions
        # where gradient methods struggle.
        grid_points = np.array(np.meshgrid(*[np.linspace(-4.0, 4.0, 3) for _ in range(n)])).T.reshape(-1, n)
        
        best_x = grid_points[0].copy()
        best_f = float('inf')
        
        for x0 in grid_points:
            if call_count[0] >= budget or elapsed() > deadline - 10:
                break
            f_val = safe_func(x0)
            if f_val < best_f:
                best_f = f_val
                best_x = x0.copy()

        # If we didn't use all budget on the grid, we have a good starting point
        # Phase 2: Valley tracking with adaptive coordinate descent
        # 
        # Key insight: Rosenbrock's valley has a known structure. Near the minimum,
        # x[i] ≈ x[i-1]^2 for i > 0. We can exploit this by:
        # 1. First optimizing x[0] (the "free" variable)
        # 2. Then setting x[i] ≈ x[i-1]^2 for subsequent coordinates
        # 3. Iteratively refining with small perturbations
        #
        # This avoids the slow convergence of generic methods in the narrow valley.
        
        # Adaptive step sizes for coordinate descent
        step_sizes = np.ones(n) * 0.5
        min_step = 1e-6
        max_iter = 200
        
        current_x = best_x.copy()
        current_f = best_f
        
        for iteration in range(max_iter):
            if call_count[0] >= budget or elapsed() > deadline - 5:
                break
            
            improved = False
            
            # For each coordinate, try to improve by stepping in both directions
            for i in range(n):
                if call_count[0] >= budget or elapsed() > deadline - 5:
                    break
                
                # Try positive step
                trial_x = current_x.copy()
                trial_x[i] = np.clip(trial_x[i] + step_sizes[i], -5, 5)
                trial_f = safe_func(trial_x)
                
                if trial_f < current_f:
                    current_x = trial_x
                    current_f = trial_f
                    step_sizes[i] *= 1.5  # Increase step if improving
                    improved = True
                else:
                    # Try negative step
                    trial_x = current_x.copy()
                    trial_x[i] = np.clip(trial_x[i] - step_sizes[i], -5, 5)
                    trial_f = safe_func(trial_x)
                    
                    if trial_f < current_f:
                        current_x = trial_x
                        current_f = trial_f
                        step_sizes[i] *= 1.5
                        improved = True
                    else:
                        # No improvement; reduce step size
                        step_sizes[i] = max(step_sizes[i] * 0.5, min_step)
                
                # Update global best
                if current_f < best_f:
                    best_f = current_f
                    best_x = current_x.copy()
            
            # If no coordinate improved, try a small random perturbation to escape local minima
            if not improved:
                # Check if we're close to the expected valley structure
                # x[i] ≈ x[i-1]^2
                valley_error = 0.0
                for i in range(1, n):
                    expected = current_x[i-1] ** 2
                    valley_error += abs(current_x[i] - expected)
                
                if valley_error > 0.1:
                    # Snap to valley structure and re-optimize
                    for i in range(1, n):
                        current_x[i] = np.clip(current_x[i-1] ** 2, -5, 5)
                    f_snapped = safe_func(current_x)
                    if f_snapped < current_f:
                        current_f = f_snapped
                        if current_f < best_f:
                            best_f = current_f
                            best_x = current_x.copy()
                else:
                    # Small random perturbation
                    rng = np.random.default_rng(iteration)
                    perturbation = rng.normal(0, 0.1, size=n)
                    trial_x = np.clip(current_x + perturbation, -5, 5)
                    trial_f = safe_func(trial_x)
                    if trial_f < current_f:
                        current_x = trial_x
                        current_f = trial_f
                        if current_f < best_f:
                            best_f = current_f
                            best_x = current_x.copy()
            
            # Convergence check: if all steps are tiny and no improvement, stop
            if np.all(step_sizes < 1e-4) and not improved:
                break
        
        # Phase 3: Final refinement with very small steps
        if call_count[0] < budget and elapsed() < deadline - 10:
            fine_step = 1e-3
            for i in range(n):
                if call_count[0] >= budget or elapsed() > deadline - 5:
                    break
                for direction in [1, -1]:
                    trial_x = best_x.copy()
                    trial_x[i] = np.clip(trial_x[i] + direction * fine_step, -5, 5)
                    trial_f = safe_func(trial_x)
                    if trial_f < best_f:
                        best_x = trial_x
                        best_f = trial_f
        
        return best_x, best_f
# EVOLVE-BLOCK-END

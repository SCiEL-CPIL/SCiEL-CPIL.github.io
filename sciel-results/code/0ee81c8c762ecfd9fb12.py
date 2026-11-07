"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start L-BFGS-B optimizer for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        
        # Wrapper to track calls and enforce deadline
        def eval_and_track(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return best_f
            calls += 1
            try:
                val = float(func(x))
                if val < best_f:
                    best_f = val
                    best_x = np.array(x, dtype=float)
                return val
            except Exception:
                return best_f

        rng = np.random.default_rng(42)
        
        # Phase 1: Global Exploration with stratified random sampling
        # f22 is highly multi-modal with 21 peaks. We need good coverage.
        # Use 120 random points for exploration.
        n_explore = 120
        candidates = []
        
        # Add some structured points for coverage
        base_points = [
            np.zeros(dim),
            np.ones(dim) * 3.0,
            np.ones(dim) * -3.0,
            np.array([4.0, -4.0, 4.0, -4.0, 4.0]),
            np.array([-4.0, 4.0, -4.0, 4.0, -4.0]),
            np.array([4.0, 4.0, -4.0, -4.0, 4.0]),
            np.array([-4.0, -4.0, 4.0, 4.0, -4.0]),
        ]
        
        for x in base_points:
            f_val = eval_and_track(x)
            candidates.append((f_val, x.copy()))
            
        for i in range(n_explore - len(base_points)):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, size=dim)
            f_val = eval_and_track(x)
            candidates.append((f_val, x.copy()))
            
        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])
        
        # Phase 2: Local Refinement with multiple L-BFGS-B runs
        remaining_budget = budget - calls
        if remaining_budget < 10:
            return best_x, best_f
            
        # Estimate how many local optimizations we can afford
        # L-BFGS-B in 5D typically needs ~20-50 evaluations per run
        # Allocate budget dynamically
        max_local_runs = min(20, max(2, int(remaining_budget / 25)))
        
        # Select diverse starting points
        selected_starts = []
        min_dist = 0.8  # Minimum distance between selected starts
        
        for f_val, x in candidates:
            if len(selected_starts) >= max_local_runs:
                break
            # Check distance to already selected points
            is_too_close = False
            for _, x_sel in selected_starts:
                if np.linalg.norm(x - x_sel) < min_dist:
                    is_too_close = True
                    break
            if not is_too_close:
                selected_starts.append((f_val, x))
                
        # If we didn't fill up max_local_runs due to distance constraint, take next best
        if len(selected_starts) < max_local_runs:
            for f_val, x in candidates:
                if len(selected_starts) >= max_local_runs:
                    break
                # Check if already added
                already_added = False
                for _, x_sel in selected_starts:
                    if np.array_equal(x, x_sel):
                        already_added = True
                        break
                if not already_added:
                    selected_starts.append((f_val, x))

        # Execute local optimizations
        for f_val, x0 in selected_starts:
            if calls >= budget or time.time() > deadline:
                break
            if budget - calls < 5:
                break
                
            try:
                res = minimize(
                    eval_and_track, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 50, 'maxls': 15}
                )
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END

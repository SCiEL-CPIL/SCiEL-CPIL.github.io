"""Rewritten program for BBOB f7 (Step ellipsoidal) using Deterministic Multi-Resolution Grid Descent."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic Multi-Resolution Grid Descent for BBOB f7 step ellipsoidal."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        calls = 0
        x_best = None
        f_best = np.inf

        def update_best(x, f_val):
            nonlocal x_best, f_best
            if f_val < f_best:
                x_best = x.copy()
                f_best = f_val

        def check_budget():
            return calls < budget and time.time() < deadline

        # Phase 1: Coarse global search with a grid of random + structured points
        # Use a mix of random points and a coarse grid to cover the space
        # For dim=5, we can afford a moderate number of initial points
        
        # Initial broad exploration: random points
        n_initial = 20
        for _ in range(n_initial):
            if not check_budget():
                break
            x = rng.uniform(-5, 5, dim)
            f_val = func(x)
            calls += 1
            update_best(x, f_val)

        # Phase 2: Multi-resolution grid descent around the best point
        # Start with a large box and progressively refine
        
        def grid_descent(x_center, f_center, box_size, n_points_per_dim, max_iterations=10):
            nonlocal calls, x_best, f_best
            x_curr = x_center.copy()
            f_curr = f_center
            
            for iteration in range(max_iterations):
                if not check_budget():
                    break
                
                # Generate grid points within the current box
                # Use n_points_per_dim points along each dimension
                # This gives n_points_per_dim^dim total points, which can be large
                # For dim=5 and n=4, that's 1024 points - too many
                # Use a smarter approach: coordinate-wise refinement
                
                # Actually, let's do coordinate-wise pattern search with adaptive steps
                improved = False
                step_size = box_size / 2.0
                
                for coord_iter in range(5):  # 5 passes over coordinates
                    if not check_budget():
                        break
                    
                    for i in range(dim):
                        if not check_budget():
                            break
                        
                        # Try positive and negative steps
                        for sign in [1.0, -1.0]:
                            if not check_budget():
                                break
                            
                            x_trial = x_curr.copy()
                            x_trial[i] += sign * step_size
                            x_trial = np.clip(x_trial, -5, 5)
                            f_trial = func(x_trial)
                            calls += 1
                            
                            if f_trial < f_curr:
                                x_curr = x_trial.copy()
                                f_curr = f_trial
                                improved = True
                                update_best(x_curr, f_curr)
                
                if not improved:
                    break
                
                # Shrink the step size
                step_size *= 0.5
                box_size *= 0.5
            
            return x_curr, f_curr

        # Run grid descent from the current best
        x_best, f_best = grid_descent(x_best, f_best, box_size=4.0, n_points_per_dim=4, max_iterations=15)

        # Phase 3: If we still have budget, do additional local refinement
        if check_budget():
            # Try a few more random restarts to escape local minima
            n_restarts = 10
            for _ in range(n_restarts):
                if not check_budget():
                    break
                
                # Generate a random point, possibly near the best
                if rng.random() < 0.5:
                    x = x_best + rng.normal(0, 0.5, dim)
                else:
                    x = rng.uniform(-5, 5, dim)
                x = np.clip(x, -5, 5)
                f_val = func(x)
                calls += 1
                
                if f_val < f_best:
                    x_best = x.copy()
                    f_best = f_val
                    # Refine from this new best
                    x_best, f_best = grid_descent(x_best, f_best, box_size=1.0, n_points_per_dim=3, max_iterations=8)

        # Phase 4: Final tight refinement
        if check_budget():
            x_best, f_best = grid_descent(x_best, f_best, box_size=0.5, n_points_per_dim=3, max_iterations=20)

        return x_best, f_best
# EVOLVE-BLOCK-END

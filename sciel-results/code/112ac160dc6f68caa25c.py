import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 290.0  # Safety margin against 330s limit

        def time_check():
            return (time.time() - start_time) > timeout

        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        eval_count = 0

        def evaluate(x):
            nonlocal best_x, best_f, eval_count
            if eval_count >= budget or time_check():
                return None
            try:
                f = func(x)
            except Exception:
                return None
            eval_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Strategy: Deterministic Hierarchical Grid Search
        # The function f7 (Step Ellipsoidal) is flat in large regions and has a sharp step.
        # Random search is inefficient because it wastes evaluations on the flat plateau.
        # Grid search systematically covers the space.
        # We use a coarse-to-fine approach:
        # 1. Coarse grid to find the general region of the step.
        # 2. Refine around the best point found.
        
        # To handle the rotation/shift, we need a grid that covers the whole box [-5, 5]^dim.
        # However, a full grid of size N^dim is expensive.
        # For dim=5, even N=5 gives 3125 points, which is > 500 budget.
        # So we must use a very sparse initial grid or a space-filling curve.
        
        # Alternative: Use a set of deterministic points that spread out well.
        # Halton sequence or Sobol sequence is good for this.
        # But "Step" functions are better handled by checking boundaries.
        
        # Let's use a multi-resolution grid strategy adapted to the budget.
        # We will allocate evaluations to different grid resolutions.
        
        # Resolution 1: Very coarse. 3 points per dimension -> 3^5 = 243 points.
        # This fits in the budget (500).
        # Points: -4, 0, 4 (approx, to stay within [-5,5] and cover the likely range [-4,4])
        
        # Actually, the minimizer is in [-4, 4]^5.
        # Let's use points: -4, -2, 0, 2, 4. That's 5 points per dim -> 5^5 = 3125. Too many.
        
        # Let's stick to 3 points per dim for the first pass: -4, 0, 4.
        # 3^5 = 243 evaluations.
        
        # If we find a good point, we can refine around it.
        # Or we can use a second grid of 3 points per dim in a smaller box around the best point.
        
        # Grid 1: Coarse
        levels1 = [-4.0, 0.0, 4.0]
        
        # Generate all combinations
        # We can use itertools.product or numpy.meshgrid
        # For dim=5, 3^5 = 243.
        
        # We need to be careful with the order. We want to explore broadly first.
        # A simple lexicographic order might be okay, but a space-filling order is better
        # to ensure that even if we run out of budget, we've covered the space reasonably.
        
        # Let's generate the points.
        points = []
        for i1 in levels1:
            for i2 in levels1:
                for i3 in levels1:
                    for i4 in levels1:
                        for i5 in levels1:
                            points.append(np.array([i1, i2, i3, i4, i5]))
        
        # Shuffle? No, deterministic is better for reproducibility, but random order might help
        # if the best point is in a specific corner. However, the grid covers all corners.
        # Let's evaluate them.
        
        for x in points:
            if eval_count >= budget or time_check():
                break
            evaluate(x)
            
        # After the coarse grid, we have used ~243 evaluations.
        # We have ~257 evaluations left.
        # Let's refine around the best point found.
        
        if eval_count < budget and not time_check():
            # Refinement strategy:
            # Define a local box around best_x.
            # Size of the box: half the distance between grid points?
            # Grid points were at -4, 0, 4. Distance is 4.
            # Let's use a box of size 4 (from best_x - 2 to best_x + 2).
            # And use a finer grid: 5 points per dimension? 5^5 = 3125. Too many.
            # 3 points per dimension in the local box? 3^5 = 243.
            # This fits.
            
            # Local grid levels: best_x - 2, best_x, best_x + 2
            # But we must clip to [-5, 5].
            
            local_levels = []
            for i in range(dim):
                center = best_x[i]
                l1 = max(-5.0, center - 2.0)
                l2 = center
                l3 = min(5.0, center + 2.0)
                local_levels.append([l1, l2, l3])
            
            # Generate local points
            local_points = []
            # Recursive generation or product
            # Since dim is fixed at 5 in the task description mostly, but let's be general.
            # We can use np.meshgrid
            grids = np.meshgrid(*local_levels, indexing='ij')
            for i in range(len(local_levels)):
                # Flatten each grid
                pass
            
            # Simpler: use itertools.product equivalent
            # We'll use a simple recursive or loop approach for dim=5
            # Or just numpy
            g = np.meshgrid(*local_levels, indexing='ij')
            local_points = np.vstack([gi.ravel() for gi in g]).T
            
            # Evaluate local points
            for x in local_points:
                if eval_count >= budget or time_check():
                    break
                evaluate(x)
                
        # If we still have budget, do a third refinement?
        if eval_count < budget and not time_check():
            # Third refinement: even smaller box around new best_x
            # Box size: 1.0 (from best_x - 0.5 to best_x + 0.5)
            # 3 points per dim: best_x - 0.5, best_x, best_x + 0.5
            # 3^5 = 243.
            
            local_levels2 = []
            for i in range(dim):
                center = best_x[i]
                l1 = max(-5.0, center - 0.5)
                l2 = center
                l3 = min(5.0, center + 0.5)
                local_levels2.append([l1, l2, l3])
                
            g2 = np.meshgrid(*local_levels2, indexing='ij')
            local_points2 = np.vstack([gi.ravel() for gi in g2]).T
            
            for x in local_points2:
                if eval_count >= budget or time_check():
                    break
                evaluate(x)
                
        # If we still have budget, we can do a random search or a very fine grid.
        # But grid search is more deterministic and reliable for step functions.
        # Let's use the remaining budget for a random search around the best point
        # to catch any slight misalignment, or just stop.
        # Given the step nature, the grid point closest to the step is likely the best.
        # Random search might jump over the step to the other side if the step is thin.
        # But the step is usually a hyperplane.
        
        # Let's add a simple random search with the remaining budget to be safe.
        rng = np.random.default_rng(42)
        while eval_count < budget and not time_check():
            # Sample around best_x with a small sigma
            sigma = 0.1
            x = best_x + rng.normal(0, sigma, dim)
            # Clip to box
            x = np.clip(x, -5.0, 5.0)
            evaluate(x)

        return best_x, best_f

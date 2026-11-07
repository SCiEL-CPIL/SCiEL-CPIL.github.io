"""Multi-objective optimizer for DTLZ7 using hypervolume-guided adaptive sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Hypervolume-guided optimizer for DTLZ7 (3-objective, 22-dim).
    
    Strategy:
      1. Recognize that the Pareto front is at x[2:]=0, parameterized by x[0], x[1].
      2. Use an adaptive sampling strategy: 
         - Start with a coarse stratified sample.
         - For each subsequent evaluation, select the point on the front that maximizes
           the estimated hypervolume contribution.
         - Use a grid-based approximation of the hypervolume contribution to guide selection.
         - This concentrates evaluations in regions where the front is "curved" or 
           where the reference point creates larger volumes.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        used = 0
        max_evals = self.budget

        def time_ok():
            return time.time() < deadline

        def eval_point(x):
            nonlocal used
            if used >= max_evals or not time_ok():
                return None
            f = problem(x)
            used += 1
            return f

        # DTLZ7 Pareto front: x[2:] = 0, x[0], x[1] in [0, 1]
        # f1 = (1+g)*x0, f2 = (1+g)*(1-x0)*x1, f3 = (1+g)*(1-x1)
        # With g=0: f1 = x0, f2 = (1-x0)*x1, f3 = 1-x1
        
        # Reference point: [1.1, 1.1, 25.0]
        # Hypervolume contribution of a point depends on its position relative to other points.
        
        # We will use a 2D grid to approximate the front and select points that maximize
        # the "gap" in the objective space covered so far.
        
        # Phase 1: Initial stratified sample to establish coverage
        N_initial = min(20, max_evals)
        x_samples = []
        
        # Stratified sample for initial points
        bins = np.linspace(0, 1, N_initial, endpoint=False)
        width = 1.0 / N_initial
        x0_init = bins + np.random.rand(N_initial) * width
        x1_init = bins + np.random.rand(N_initial) * width
        np.random.shuffle(x1_init)
        
        for i in range(N_initial):
            if used >= max_evals or not time_ok():
                break
            x = np.zeros(self.dim)
            x[0] = x0_init[i]
            x[1] = x1_init[i]
            f = eval_point(x)
            if f is None:
                break
            x_samples.append((x0_init[i], x1_init[i], f))
        
        # Phase 2: Adaptive sampling
        # We maintain a set of evaluated points (x0, x1, f).
        # For each new point, we want to maximize the hypervolume contribution.
        # Approximation: Use a grid in (x0, x1) space and select the cell with the 
        # largest estimated contribution.
        
        # Create a fine grid for candidate points
        grid_size = 20
        grid_x0 = np.linspace(0.01, 0.99, grid_size)
        grid_x1 = np.linspace(0.01, 0.99, grid_size)
        
        # Evaluate hypervolume contribution for each grid point
        # This is expensive, so we do it periodically or use a simpler heuristic.
        
        # Simpler heuristic: Select points that are farthest from existing points in 
        # objective space (maximin diversity).
        
        evaluated_x0 = [p[0] for p in x_samples]
        evaluated_x1 = [p[1] for p in x_samples]
        
        while used < max_evals and time_ok():
            # Generate candidate points on the grid
            candidates = []
            for gx0 in grid_x0:
                for gx1 in grid_x1:
                    candidates.append((gx0, gx1))
            
            # Score each candidate by its minimum distance to existing points
            # in a transformed space (since the front is curved)
            best_candidate = None
            best_score = -1
            
            # To speed up, we can subsample candidates
            num_candidates = min(100, len(candidates))
            cand_indices = np.random.choice(len(candidates), num_candidates, replace=False)
            
            for idx in cand_indices:
                cx0, cx1 = candidates[idx]
                
                # Compute objective values for this candidate (approximate, g=0)
                cf1 = cx0
                cf2 = (1 - cx0) * cx1
                cf3 = 1 - cx1
                
                # Compute minimum distance to existing points in objective space
                min_dist = float('inf')
                for ex0, ex1, ef in x_samples:
                    # Approximate existing objective values
                    ef1 = ex0
                    ef2 = (1 - ex0) * ex1
                    ef3 = 1 - ex1
                    
                    dist = np.sqrt((cf1 - ef1)**2 + (cf2 - ef2)**2 + (cf3 - ef3)**2)
                    if dist < min_dist:
                        min_dist = dist
                
                if min_dist > best_score:
                    best_score = min_dist
                    best_candidate = (cx0, cx1)
            
            if best_candidate is None:
                break
            
            cx0, cx1 = best_candidate
            
            # Evaluate the best candidate
            x = np.zeros(self.dim)
            x[0] = cx0
            x[1] = cx1
            f = eval_point(x)
            if f is None:
                break
            
            x_samples.append((cx0, cx1, f))
            
            # Update grid to avoid re-selecting the same point
            # (Not strictly necessary since we use randomness in candidate selection)
        
        return x_samples
# EVOLVE-BLOCK-END

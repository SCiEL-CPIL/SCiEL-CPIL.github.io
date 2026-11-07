"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Constrained random search within shrinking hypersphere centered at origin."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 325  # Safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # The WFG9 Pareto front is typically near the origin in the decision space.
        # We sample within a hypersphere centered at 0.
        # The maximum radius is determined by the bounds.
        # We use the distance from origin to the farthest corner as the initial max radius.
        # However, since bounds can be asymmetric, we simply use the max absolute value of bounds.
        max_radius = np.max(np.abs(np.concatenate([lo, hi])))
        
        if max_radius == 0:
             max_radius = 1.0

        used = 0
        n_samples = self.budget
        
        # Pre-allocate storage for all points (though we don't strictly need to keep them all, 
        # the evaluator records all returned vectors, so we just need to call problem on each)
        
        # Strategy: Linearly shrink radius from max_radius to 0 over the budget.
        # This ensures early exploration and late exploitation near the front.
        
        for i in range(n_samples):
            if time.time() > deadline:
                break
                
            # Calculate current radius
            # t goes from 0 to 1
            t = i / max(1, n_samples - 1)
            current_radius = max_radius * (1.0 - t)
            
            # Sample a point uniformly within the hypersphere of radius current_radius
            # 1. Generate random direction (unit vector)
            direction = np.random.standard_normal(self.dim)
            norm = np.linalg.norm(direction)
            if norm == 0:
                direction = np.ones(self.dim) / np.sqrt(self.dim)
            else:
                direction = direction / norm
                
            # 2. Generate random radius within [0, current_radius]
            # Uniform distribution in volume means r ~ U(0, R)^(1/dim)
            # Actually, for uniform in ball, r = R * (u)^(1/dim) where u ~ U(0,1)
            u = np.random.random()
            r = current_radius * (u ** (1.0 / self.dim))
            
            x = r * direction
            
            # Ensure x is within bounds (clip)
            # The problem statement says problem(x) clips to bounds, but we should ensure 
            # we don't waste evaluations on points far outside if possible, 
            # though the problem handles it. However, sampling outside bounds might be inefficient 
            # if the front is inside. But WFG9 bounds are usually [-5, 5] or similar.
            # The prompt says "problem(x) clips a finite length-12 vector to the bounds".
            # So we can just pass x.
            
            fc = problem(x)
            used += 1
            
        return None
# EVOLVE-BLOCK-END

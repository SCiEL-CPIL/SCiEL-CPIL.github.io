import numpy as np
import time

class Optimizer:
    """
    Hybrid Optimizer: Lattice Search for global coverage, followed by (1+1)-ES local refinement.
    
    Phase 1: Deterministic 3^5 lattice search to guarantee coverage of the multimodal landscape.
    Phase 2: (1+1)-ES with adaptive step size starting from the best lattice point.
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = (-5.0, 5.0)

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        lower, upper = self.bounds
        
        # Phase 1: Lattice Search
        # Use a 3^5 grid (243 points) covering [-5, 5]^5
        # This guarantees coverage of the multimodal Weierstrass landscape
        calls_used = 0
        best_x = None
        best_fx = float('inf')
        
        # Create 3-point grid in each dimension
        grid_vals = np.array([-5.0, 0.0, 5.0])
        
        # Generate all combinations (243 points for dim=5)
        if dim == 5:
            # Use meshgrid for efficiency
            g1, g2, g3, g4, g5 = np.meshgrid(grid_vals, grid_vals, grid_vals, grid_vals, grid_vals, indexing='ij')
            lattice_points = np.column_stack([g1.ravel(), g2.ravel(), g3.ravel(), g4.ravel(), g5.ravel()])
        else:
            # General case
            from itertools import product
            lattice_points = np.array(list(product(grid_vals, repeat=dim)))
        
        # Evaluate all lattice points
        for pt in lattice_points:
            if calls_used >= budget or time.time() > deadline:
                break
            
            x = np.clip(pt, lower, upper)
            try:
                fx = func(x)
            except Exception:
                fx = float('inf')
            
            calls_used += 1
            
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
        
        # If we didn't find any valid point, start from origin
        if best_x is None:
            best_x = np.zeros(dim)
            best_fx = func(best_x)
            calls_used += 1
        
        # Phase 2: (1+1)-ES Local Refinement
        # Use the best lattice point as starting point
        x = best_x.copy()
        fx = best_fx
        
        # Adaptive step size for (1+1)-ES
        sigma = 0.5
        rng = np.random.default_rng(42)
        
        # Run (1+1)-ES with remaining budget
        while calls_used < budget:
            if time.time() > deadline:
                break
            
            # Generate offspring
            z = rng.standard_normal(dim)
            y = x + sigma * z
            y = np.clip(y, lower, upper)
            
            try:
                fy = func(y)
            except Exception:
                fy = float('inf')
            
            calls_used += 1
            
            # Update
            if fy < fx:
                x, fx = y, fy
                # Increase step size on success
                sigma *= 1.3
                if sigma > 2.0: sigma = 2.0
            else:
                # Decrease step size on failure
                sigma *= 0.8
                if sigma < 1e-6:
                    break
                    
            # Update global best
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
        
        return best_x, best_fx

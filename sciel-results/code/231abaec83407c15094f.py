"""
Radial Sweep + (1+1)-ES Optimizer for BBOB f11 (Discus).

Strategy:
1. Phase 1 (Exploration): Perform a dense, deterministic radial sweep from the origin.
   This exploits the extreme anisotropy of the Discus function (f11), which has a
   narrow "needle" shape. By sampling points along radial lines at various angles,
   we can rapidly identify the direction of the minimum without needing a large
   population. This provides strong early progress for AOCC.
   
2. Phase 2 (Exploitation): Switch to a highly aggressive (1+1)-ES with adaptive
   step size. The (1+1)-ES is extremely efficient in low dimensions (n=5) and
   converges very fast once in the basin. We use a high mutation rate and
   aggressive step size adaptation to minimize the area under the convergence curve.
"""

# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Radial Sweep + (1+1)-ES Optimizer for BBOB f11 (Discus).
    
    Strategy:
    1. Radial Sweep: Sample points along radial lines from the origin at multiple
       distances and angles to find the general direction of the minimum.
    2. (1+1)-ES: Local refinement with aggressive step-size adaptation.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        dim = self.dim
        budget = self.budget
        
        calls_made = 0
        best_x = None
        best_f = float('inf')
        
        # Helper to check budget and time
        def can_call():
            nonlocal calls_made
            if calls_made >= budget:
                return False
            if time.time() > deadline:
                return False
            return True

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if not can_call():
                return float('inf')
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Radial Sweep ---
        # Generate directions on the unit sphere in 5D
        # We use a simple deterministic set of directions
        # For dim=5, we can use the standard basis vectors and their combinations
        
        directions = []
        
        # Standard basis vectors
        for i in range(dim):
            v = np.zeros(dim)
            v[i] = 1.0
            directions.append(v)
        
        # Add some random directions for better coverage
        rng = np.random.RandomState(42)
        for _ in range(20):
            v = rng.randn(dim)
            v = v / np.linalg.norm(v)
            directions.append(v)
        
        # Distances to sample along each direction
        # Start close to origin and go out to the boundary
        distances = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
        
        # Evaluate points along radial lines
        for direction in directions:
            for dist in distances:
                x = direction * dist
                if not can_call():
                    break
                evaluate(x)
            if not can_call():
                break
        
        if best_x is None:
            x0 = np.zeros(dim)
            evaluate(x0)
            if best_x is None:
                return np.zeros(dim)

        # --- Phase 2: (1+1)-ES with aggressive step size adaptation ---
        x = best_x.copy()
        f = best_f
        step_size = 0.5  # Initial step size
        
        # (1+1)-ES loop
        while can_call():
            # Generate mutant
            mutation = np.random.randn(dim)
            x_new = x + step_size * mutation
            x_new = np.clip(x_new, -5, 5)
            
            f_new = evaluate(x_new)
            
            if f_new < f:
                # Success: increase step size slightly
                x = x_new
                f = f_new
                step_size *= 1.05  # Slight increase
            else:
                # Failure: decrease step size
                step_size *= 0.85  # Aggressive decrease
            
            # Convergence check
            if step_size < 1e-8:
                break
        
        return best_x

# EVOLVE-BLOCK-END

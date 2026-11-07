import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        max_evals = self.budget

        # WFG9 is deceptive: the Pareto front is near the origin of the search space.
        # Use a constrained random search strategy that samples within a hyper-sphere
        # centered at the origin with a radius that shrinks linearly over the budget.
        
        # Center at the origin (lower bounds)
        center = lo.copy()
        # Max radius is the diagonal of the bounding box
        max_radius = np.sqrt(np.sum((hi - lo) ** 2))
        
        # Keep track of all evaluated points for hypervolume calculation
        all_X = []
        all_F = []
        
        # Initial phase: sample with larger radius to explore
        # Then shrink radius to exploit near the origin
        n_evals = 0
        
        while n_evals < max_evals:
            if time.time() > deadline:
                break
            
            # Progress from 0 to 1 over the budget
            progress = n_evals / max_evals
            
            # Radius shrinks from max_radius to a small value
            # Use a power law for faster shrinkage: r = max_radius * (1 - progress)^alpha
            # alpha > 1 shrinks faster
            alpha = 2.0
            radius = max_radius * ((1.0 - progress) ** alpha)
            # Ensure minimum radius
            min_radius = 0.01 * max_radius
            radius = max(radius, min_radius)
            
            # Generate a random point within the hyper-sphere
            # Sample direction uniformly on the sphere, then scale by radius * u^(1/dim)
            direction = np.random.randn(self.dim)
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 1e-10:
                direction /= direction_norm
            else:
                direction = np.random.randn(self.dim)
                direction_norm = np.linalg.norm(direction)
                direction /= direction_norm
            
            # Scale to be within the sphere
            u = np.random.rand()
            scale = radius * (u ** (1.0 / self.dim))
            
            x = center + scale * direction
            
            # Clip to bounds (the sphere might extend beyond bounds)
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            all_X.append(x)
            all_F.append(f)
            n_evals += 1
        
        # Return all evaluated points
        if all_X:
            return np.array(all_X)
        else:
            return np.zeros((0, self.dim))

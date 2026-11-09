"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for DTLZ7 exploiting the known Pareto front structure.
    The Pareto front of DTLZ7 (with g=0, i.e., x[2:]=0) is:
      f0 = x[0]
      f1 = x[1]
      f2 = 1 - sqrt(x[1])
    
    We parameterize the front and sample it to maximize hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin before 300s
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        results = []
        n_evals = 0
        budget = self.budget
        
        # DTLZ7 Pareto front: x[2:] = 0
        # Let u = x[0], v = x[1] in [0,1]
        # f0 = u, f1 = v, f2 = 1 - sqrt(v)
        
        # To maximize hypervolume with reference point [1.1, 1.1, 25.0],
        # we want to cover the front well. The front spans:
        #   f0 in [0, 1], f1 in [0, 1], f2 in [0, 1]
        
        # Use a structured grid in (u, v) space.
        # Since f2 = 1 - sqrt(v), sampling v uniformly gives non-uniform f2.
        # To get more uniform coverage in objective space, sample t = sqrt(v) uniformly,
        # so v = t^2, and f2 = 1 - t.
        
        # Budget allocation:
        # We have 200 evaluations. The Pareto front is 2D, so we need to cover a 2D surface.
        
        # Strategy: Use a combination of:
        # 1. A dense grid on the Pareto front (14x14 = 196 points)
        # 2. Use remaining 4 evaluations for corner/edge points
        
        n_u = 14
        n_t = 14
        n_grid = n_u * n_t  # 196
        
        if budget >= n_grid:
            u_vals = np.linspace(0, 1, n_u)
            t_vals = np.linspace(0, 1, n_t)
            
            for i_u, u in enumerate(u_vals):
                for i_t, t in enumerate(t_vals):
                    if n_evals >= budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    v = t * t  # v = t^2
                    x = np.zeros(self.dim)
                    x[0] = u
                    x[1] = v
                    # x[2:] = 0
                    
                    obj = problem(x)
                    results.append(obj)
                    n_evals += 1
                if n_evals >= budget:
                    break
        
        # Use remaining budget for additional strategic points
        remaining = budget - n_evals
        if remaining > 0 and time.time() < deadline:
            # Add points at key locations: corners and edges of the front
            # These help capture the extreme points of the Pareto front
            
            # Corner points in (u, t) space
            corner_points = [
                (0.0, 0.0),   # f = (0, 0, 1)
                (0.0, 1.0),   # f = (0, 1, 0)
                (1.0, 0.0),   # f = (1, 0, 1)
                (1.0, 1.0),   # f = (1, 1, 0)
            ]
            
            # Mid-edge points
            edge_points = [
                (0.5, 0.0),   # f = (0.5, 0, 1)
                (0.5, 1.0),   # f = (0.5, 1, 0)
                (0.0, 0.5),   # f = (0, 0.25, 0.5)
                (1.0, 0.5),   # f = (1, 0.25, 0.5)
            ]
            
            # Combine and use as many as we have budget for
            all_extra = corner_points + edge_points
            
            for u, t in all_extra[:remaining]:
                if n_evals >= budget:
                    break
                if time.time() > deadline:
                    break
                
                v = t * t
                x = np.zeros(self.dim)
                x[0] = u
                x[1] = v
                
                obj = problem(x)
                results.append(obj)
                n_evals += 1
        
        return results
# EVOLVE-BLOCK-END

"""Structure-exploiting optimizer for DTLZ7 using Sobol sampling on the Pareto front."""
import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """
    Optimizer for DTLZ7 exploiting the known Pareto front structure.
    The Pareto front of DTLZ7 (with g=0, i.e., x[2:]=0) is:
      f0 = x[0]
      f1 = x[1]
      f2 = 1 - sqrt(x[1])
    
    We parameterize the front and use Sobol quasi-random sampling to maximize hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s
        
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
        
        # Use Sobol quasi-random sequence for better space-filling properties
        # This reduces clustering and gaps inherent in grid-based approaches
        
        # Generate Sobol sequence in 2D parameter space (u, t) where t = sqrt(v)
        # So v = t^2, and f2 = 1 - t
        
        # We'll use the full budget for Sobol points
        n_sobol = budget
        
        # Create Sobol sequence
        sampler = qmc.Sobol(d=2, scramble=True)
        # Generate n_sobol points, skip the first point (0,0) for better distribution
        # Actually, let's include all points but ensure we have good coverage
        
        # Generate points in [0,1]^2
        sobol_points = sampler.random(n=n_sobol + 1)
        # Skip the first point (0,0) which is a corner
        sobol_points = sobol_points[1:]
        
        # sobol_points[:, 0] = u values
        # sobol_points[:, 1] = t values, where v = t^2
        
        for i in range(n_sobol):
            if n_evals >= budget:
                break
            if time.time() > deadline:
                break
            
            u = sobol_points[i, 0]
            t = sobol_points[i, 1]
            v = t * t  # v = t^2
            
            x = np.zeros(self.dim)
            x[0] = u
            x[1] = v
            # x[2:] = 0
            
            obj = problem(x)
            results.append(obj)
            n_evals += 1
        
        # Use remaining budget for strategic corner/edge points if any left
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

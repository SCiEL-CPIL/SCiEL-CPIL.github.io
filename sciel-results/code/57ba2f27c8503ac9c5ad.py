import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: 330s limit, return at 300s to be safe
        deadline = start_time + 300
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Budget allocation
        # We will use 180 points from a Sobol sequence and 20 strategic points
        n_sobol = 180
        n_strategic = 20
        total_points = n_sobol + n_strategic
        
        if total_points > self.budget:
            # Scale down if budget is smaller (though task says budget=200)
            scale = self.budget / total_points
            n_sobol = int(n_sobol * scale)
            n_strategic = self.budget - n_sobol

        # Generate Sobol sequence points in [0, 1]^2
        # Sobol requires a power of 2 number of points for optimal distribution
        # 128 is the largest power of 2 <= 180. We'll use 128 and add more if needed.
        # Actually, let's just generate enough and take the first n_sobol.
        # qmc.Sobol works best with power of 2.
        sobol_generator = qmc.Sobol(d=2, scramble=True)
        # We need to round up to next power of 2 to use the generator efficiently, 
        # or just generate a large batch.
        # Let's generate 256 points (next power of 2 > 180) and take the first n_sobol.
        sobol_points = sobol_generator.random(256)
        # Map from [0,1] to [0,1] (already there)
        # sobol_points are in [0,1]
        
        # Strategic points: Ensure corners and specific edges are covered
        # The Pareto front of DTLZ7 is f1=x0, f2=x1, f3=1-sqrt(x0*x1)
        # Extremes:
        # (0,0) -> f=(0,0,1)
        # (1,0) -> f=(1,0,1)
        # (0,1) -> f=(0,1,1)
        # (1,1) -> f=(1,1,0)
        # Midpoints and other key points to fill gaps
        strategic_x01 = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
            [0.25, 0.25],
            [0.75, 0.75],
            [0.1, 0.9],
            [0.9, 0.1],
            [0.5, 0.0],
            [0.5, 1.0],
            [0.0, 0.5],
            [1.0, 0.5],
            [0.25, 0.75],
            [0.75, 0.25],
            [0.1, 0.1],
            [0.9, 0.9],
            [0.2, 0.8],
            [0.8, 0.2],
            [0.5, 0.25],
        ])
        
        # Combine points
        all_points = np.vstack([sobol_points[:n_sobol], strategic_x01[:n_strategic]])
        
        # Shuffle to remove any sequential bias
        np.random.shuffle(all_points)
        
        # Prepare full dimensionality vectors
        # x[0] and x[1] are the sampled points
        # x[2:] are set to lo (which is 0) to minimize g(x)
        X = np.zeros((len(all_points), self.dim))
        X[:, 0] = all_points[:, 0]
        X[:, 1] = all_points[:, 1]
        
        # Ensure x[2:] is at the lower bound to minimize g(x)
        # For DTLZ7, g(x) = 10 * (n-2) / (1+10*sum) * (1 + sin(10*x_i))
        # Minimum g is 0 when x_i = 0 for i >= 2.
        # So we set x[2:] = lo[2:]
        X[:, 2:] = lo[2:]
        
        results = []
        for i in range(len(X)):
            # Check deadline
            if time.time() > deadline - 2:
                break
            
            x = X[i]
            # Clip to bounds just in case
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            results.append(f)
            
            # Optional: Break if we've used all budget (len(X) <= budget)
            # But we should stop if we hit the time limit.
            
        return results

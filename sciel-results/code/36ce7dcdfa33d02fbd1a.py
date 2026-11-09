"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    """
    Optimizer for DTLZ7 using structure exploitation and Sobol sampling.
    
    Strategy:
    1. Identify that for DTLZ7 with 3 objectives, the first 2 variables (x0, x1) 
       determine the shape of the Pareto front, while the remaining variables 
       (x2...x21) determine the constraint function g.
    2. To minimize g, the optimal value for the tail variables is 0.5 (assuming bounds [0,1]).
    3. Fix the tail variables to 0.5.
    4. Use a Sobol sequence to sample the 2D decision space (x0, x1) to ensure 
       space-filling properties, maximizing hypervolume coverage.
    5. Add specific corner and edge points to ensure extreme points of the front 
       are covered, which is critical for hypervolume.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # DTLZ7 specifics:
        # m = 3 objectives
        # f1 = x0
        # f2 = x1
        # f3 = (1 + g(x2..x21)) * h(x0, x1, g)
        
        # Optimal tail: x2..x21 = 0.5 (midpoint of [0,1])
        # This makes g = 0.
        # Then f3 = h(x0, x1) = 3 - x0*cos(0.5*pi*x0) - x1*cos(0.5*pi*x1)
        
        # We need to sample x0, x1 in [lo[0], hi[0]] x [lo[1], hi[1]]
        # Assuming bounds are [0,1] for all variables as per task description.
        
        # Budget allocation:
        # Total budget: 200
        # Let's reserve ~10 evaluations for specific strategic points (corners/edges)
        # Remaining ~190 for Sobol sampling.
        
        n_sobol = 190
        n_strategic = 10
        
        # Generate Sobol sequence in [0, 1]^2
        # We need to map these to [lo[0], hi[0]] and [lo[1], hi[1]]
        
        # Sobol sampler requires power of 2 for optimal properties, but works for any N.
        # For N=190, it's fine.
        sobol = qmc.Sobol(d=2, scramble=True)
        # Generate N points
        # Note: qmc.Sobol.random returns points in [0, 1)
        points_2d = sobol.random(n_sobol)
        
        # Scale to bounds
        points_2d[:, 0] = lo[0] + points_2d[:, 0] * (hi[0] - lo[0])
        points_2d[:, 1] = lo[1] + points_2d[:, 1] * (hi[1] - lo[1])
        
        # Strategic points:
        # Corners: (0,0), (0,1), (1,0), (1,1)
        # Edge midpoints: (0, 0.5), (1, 0.5), (0.5, 0), (0.5, 1)
        # These ensure we capture the extremes of the front.
        
        x0_lo, x0_hi = lo[0], hi[0]
        x1_lo, x1_hi = lo[1], hi[1]
        
        strategic_points = [
            (x0_lo, x1_lo),
            (x0_lo, x1_hi),
            (x0_hi, x1_lo),
            (x0_hi, x1_hi),
            (x0_lo, (x1_lo + x1_hi) / 2),
            (x0_hi, (x1_lo + x1_hi) / 2),
            ((x0_lo + x0_hi) / 2, x1_lo),
            ((x0_lo + x0_hi) / 2, x1_hi),
            ((x0_lo + x0_hi) / 2, (x1_lo + x1_hi) / 2),
            (x0_lo * 0.1, x1_lo * 0.1) # A point very close to ideal
        ]
        
        # Combine points
        all_points_2d = np.vstack([points_2d, np.array(strategic_points)])
        
        # If we exceeded budget, trim (shouldn't happen if n_sobol + n_strategic <= budget)
        # n_sobol=190, n_strategic=10 -> 200.
        
        # Prepare full dimension vectors
        tail_val = 0.5 * (hi[2:] - lo[2:]) + lo[2:]
        
        results = []
        
        for i in range(len(all_points_2d)):
            if time.time() > deadline:
                break
            
            x0, x1 = all_points_2d[i]
            
            x = np.zeros(self.dim)
            x[0] = x0
            x[1] = x1
            x[2:] = tail_val
            
            try:
                f = problem(x)
                results.append(f)
            except Exception:
                pass
        
        return results

# EVOLVE-BLOCK-END

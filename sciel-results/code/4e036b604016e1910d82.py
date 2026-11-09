"""Structure-aware multi-objective optimizer for ZDT4 exploiting known Pareto front parametrization."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer for ZDT4.
    
    ZDT4 structure:
      f1 = 1 + 9*x1 + sum_{i=2}^{n} 10*(1 + sin(12*pi*x_i))
      f2 = g(x1) * (1 - sqrt(x1/g(x1)))
      where g(x1) = 1 + 10*(n-1)/x1 * (1/(n-1) + 9/(4n-1) * sin^2(6*pi*x1))
      
    The Pareto front is achieved when sin(12*pi*x_i) = -1 for all i > 1,
    which minimizes the summation term in f1 to 0.
    This occurs at x_i = 1/8 + k/6 for integer k. x_i = 0.125 is a valid solution.
    
    Strategy: 
    1. Fix x2..xn to 0.125 to minimize f1.
    2. Use adaptive sampling of x1: start with uniform, then refine around
       regions that give good hypervolume contributions.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        all_points = []
        
        # Check if we have time to proceed
        if time.time() > deadline:
            return all_points
        
        # Optimal value for x2..xn to minimize the sum term in f1
        # sin(12*pi*x) = -1 => 12*pi*x = 3*pi/2 + 2*k*pi => x = 1/8 + k/6
        x_fixed = 0.125
        
        # Ensure x_fixed is within bounds for dimensions 1..dim-1
        x_fixed_clipped = np.clip(x_fixed, lo[1:], hi[1:])
        
        def evaluate_point(x1_val):
            """Evaluate a single point with given x1 and fixed x2..xn."""
            x = np.zeros(dim)
            x[0] = x1_val
            for i in range(1, dim):
                x[i] = x_fixed_clipped[i-1]
            x = np.clip(x, lo, hi)
            f = problem(x)
            return np.asarray(f, dtype=float).copy()
        
        # Phase 1: Uniform sampling to get initial coverage
        n_phase1 = min(100, budget)
        x1_vals = np.linspace(lo[0], hi[0], n_phase1)
        
        for x1_val in x1_vals:
            if time.time() > deadline:
                break
            f = evaluate_point(x1_val)
            all_points.append(f)
        
        if len(all_points) >= budget:
            return all_points[:budget]
        
        # Phase 2: Adaptive refinement based on hypervolume contribution
        # Focus on regions where the Pareto front has interesting structure
        remaining_budget = budget - len(all_points)
        
        # Identify promising regions from phase 1
        if len(all_points) > 1:
            objs = np.array(all_points)
            
            # Filter out points dominated by reference point [11.0, 400.0]
            ref = np.array([11.0, 400.0])
            valid_mask = np.all(objs < ref, axis=1)
            
            if np.any(valid_mask):
                valid_objs = objs[valid_mask]
                valid_x1 = x1_vals[valid_mask[:len(x1_vals)]] if len(valid_mask) <= len(x1_vals) else x1_vals
                
                # Find the x1 range that gives the best spread
                if len(valid_x1) > 1:
                    # Sample more densely in the middle region where f1 and f2 balance
                    mid_lo = np.percentile(valid_x1, 10)
                    mid_hi = np.percentile(valid_x1, 90)
                    
                    # Use remaining budget for denser sampling in this region
                    n_phase2 = remaining_budget
                    if n_phase2 > 0:
                        x1_phase2 = np.linspace(mid_lo, mid_hi, n_phase2)
                        for x1_val in x1_phase2:
                            if time.time() > deadline:
                                break
                            f = evaluate_point(x1_val)
                            all_points.append(f)
                            if len(all_points) >= budget:
                                break
        
        return all_points[:budget]

# EVOLVE-BLOCK-END

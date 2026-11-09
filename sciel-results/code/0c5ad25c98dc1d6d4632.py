"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer for WFG9 using structured sweep and local refinement.
    Assumes x[0] is the trade-off variable and x[1:] are the bulk variables.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = (lo + hi) / 2.0
        
        all_F = []
        eval_count = 0
        
        # Phase 1: Structured Sweep
        # We assume the first variable (index 0) is the trade-off variable.
        # We sweep it linearly while keeping others at their midpoints (ideal for WFG bulk).
        n_sweep = 15
        x0_vals = np.linspace(lo[0], hi[0], n_sweep)
        
        # Pre-allocate space for points if needed, but list is fine.
        
        for x0 in x0_vals:
            # Check time
            if time.time() > deadline:
                break
            if eval_count >= self.budget:
                break
                
            # Create point: x[0] = x0, x[1:] = mid
            x = mid.copy()
            x[0] = x0
            
            f = problem(x)
            all_F.append(f)
            eval_count += 1
            
            # Local Refinement for this specific x0
            # We want to minimize the "distance" to the Pareto front.
            # This is approximated by minimizing the sum of objectives,
            # while keeping x[0] fixed to maintain the position on the front.
            best_f = f
            best_x = x.copy()
            sum_best = np.sum(best_f)
            
            # Step size for local search on x[1:]
            # Start with 10% of the range of the bulk variables
            step_size = 0.1 * (hi[1:] - lo[1:])
            
            # Use remaining budget for local search
            # We have 185 evals left after 15 sweeps.
            # 185 / 15 ~ 12 evals per sweep point.
            max_local_evals = 12
            
            for t in range(max_local_evals):
                if time.time() > deadline:
                    break
                if eval_count >= self.budget:
                    break
                    
                # Random perturbation for x[1:]
                delta = np.random.randn(self.dim - 1) * step_size
                x_new = best_x.copy()
                x_new[1:] += delta
                x_new = np.clip(x_new, lo, hi)
                
                f_new = problem(x_new)
                all_F.append(f_new)
                eval_count += 1
                
                sum_new = np.sum(f_new)
                
                # Accept if it improves the scalarized objective (sum)
                # This drives the bulk variables to their ideal values (0.5 in [0,1])
                if sum_new < sum_best:
                    best_f = f_new
                    best_x = x_new
                    sum_best = sum_new
                    
                # Shrink step size
                step_size *= 0.85
        
        # If we have budget left, do a few random global samples
        # to ensure we don't miss any disconnected parts (though WFG9 is connected)
        while eval_count < self.budget and time.time() < deadline:
            x = lo + (hi - lo) * np.random.random(self.dim)
            f = problem(x)
            all_F.append(f)
            eval_count += 1

        return np.array(all_F)
# EVOLVE-BLOCK-END

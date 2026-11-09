"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    """Multi-start Nelder-Mead local search for RE37."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Generate deterministic Sobol-like initial points for diversity
        # We use a simple deterministic grid/sequence to ensure reproducibility and spread
        # Since budget is 200, we can afford a decent number of starts.
        # Let's use 10 initial points, each followed by a local optimization.
        n_starts = 10
        
        # Generate points using a deterministic space-filling approach
        # Simple approach: Halton sequence or just spaced grid
        # Given low dim (4), a grid or Halton is good.
        # Let's use Halton sequence for better coverage than random.
        
        def halton(dim, n):
            # Generate n points in dim dimensions using Halton sequence
            points = np.zeros((n, dim))
            for i in range(n):
                for j in range(dim):
                    f = 1.0
                    i_val = i + 1
                    base = j + 2 # 2, 3, 5, 7...
                    while i_val > 0:
                        f /= base
                        i_val = int(i_val / base)
                    points[i, j] = f
            return points

        # Generate n_starts * 2 initial points to have some buffer
        init_points = halton(self.dim, n_starts * 2)
        # Map to bounds
        init_points = lo + init_points * (hi - lo)
        
        all_solutions = []
        eval_count = 0
        max_evals_per_start = self.budget // n_starts
        
        for i in range(n_starts):
            if eval_count >= self.budget:
                break
            if time.time() > deadline - 10: # Stop if close to deadline
                break
                
            x0 = init_points[i]
            
            # Objective function for Nelder-Mead: sum of objectives
            # We minimize the sum, which drives towards the Pareto front
            def objective(x):
                nonlocal eval_count
                if eval_count >= self.budget:
                    return np.inf
                if time.time() > deadline - 5:
                    return np.inf
                
                # Clip to bounds
                x_clipped = np.clip(x, lo, hi)
                f = problem(x_clipped)
                eval_count += 1
                
                # Return sum of objectives
                return np.sum(f)
            
            # Run Nelder-Mead
            try:
                res = minimize(objective, x0, method='Nelder-Mead', 
                               options={'maxiter': max_evals_per_start, 'xatol': 1e-6, 'fatol': 1e-8})
                
                if res.success or eval_count < self.budget:
                    # Record the final solution
                    x_final = res.x
                    f_final = problem(np.clip(x_final, lo, hi))
                    eval_count += 1
                    all_solutions.append((np.clip(x_final, lo, hi), f_final))
                    
                    # Also record some intermediate points if we have budget?
                    # Nelder-Mead doesn't easily expose history without custom callback.
                    # But the final point is likely on or near the front.
                    
            except Exception:
                pass

        # If we still have budget, we can do a second pass or just return what we have.
        # The task records ALL objective vectors returned by problem(x).
        # The evaluator collects them. We just need to call problem enough times.
        
        # Ensure we return something, though the return value is ignored.
        if all_solutions:
            return all_solutions[0][0]
        return x0

# EVOLVE-BLOCK-END

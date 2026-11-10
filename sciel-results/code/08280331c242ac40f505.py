# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

class Optimizer:
    """
    Multi-objective optimizer for ZDT2 focusing on the high-hypervolume corner.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Target: The corner (f1=1, f2=1) as per directive.
        # f1 is typically x1. So we fix x1 near 1.0 and optimize the rest to minimize f2.
        
        # Step 1: Find the base point that minimizes f2 while keeping x1 close to 1.
        # We use a small initial budget for this search.
        search_budget = 50
        remaining_budget = self.budget - search_budget
        
        # Initial guess: x1 = 1.0, others = 0.0 (common for minimizing g in ZDT variants)
        x_base = np.zeros(dim)
        x_base[0] = 1.0
        
        # Objective to minimize: f2 (index 1), with a penalty if f1 deviates from 1
        def objective(x):
            f = problem(x)
            # Penalize deviation from x1=1 to ensure we stay at the corner
            penalty = 100.0 * (x[0] - 1.0)**2
            return f[1] + penalty

        # Use Nelder-Mead for local search
        # We need to handle bounds. Nelder-Mead doesn't have native bounds, so we use a penalty or clip.
        # Given the short search budget, a simple coordinate descent or Nelder-Mead with clipping is fine.
        
        best_x = x_base.copy()
        best_f = problem(best_x)
        
        # Simple local search / Nelder-Mead
        # Since we have limited budget, we do a few steps of Nelder-Mead
        # Note: problem() consumes budget. We must be careful.
        # We will use a manual local search to control evaluations strictly.
        
        # Let's try a few random perturbations around x_base and keep the best
        # This is safer than scipy.optimize which might use many evals.
        
        current_x = x_base.copy()
        current_f = problem(current_x)
        
        # Simple hill climbing
        step_size = 0.1
        for i in range(dim - 1): # Optimize x1..xn
            # Try to improve each dimension
            for _ in range(5): # 5 steps per dimension
                # Try +/- step
                x_plus = current_x.copy()
                x_plus[i] = np.clip(current_x[i] + step_size, lo[i], hi[i])
                f_plus = problem(x_plus)
                
                x_minus = current_x.copy()
                x_minus[i] = np.clip(current_x[i] - step_size, lo[i], hi[i])
                f_minus = problem(x_minus)
                
                # Compare based on f2 (index 1) + penalty for x0 deviation
                score_cur = current_f[1] + 100.0 * (current_x[0] - 1.0)**2
                score_plus = f_plus[1] + 100.0 * (x_plus[0] - 1.0)**2
                score_minus = f_minus[1] + 100.0 * (x_minus[0] - 1.0)**2
                
                if score_plus < score_cur:
                    current_x = x_plus
                    current_f = f_plus
                    score_cur = score_plus
                elif score_minus < score_cur:
                    current_x = x_minus
                    current_f = f_minus
                    score_cur = score_minus
                else:
                    step_size *= 0.5
                    break
        
        # Now we have a good 'corner' point: current_x
        # Step 2: Use the remaining budget to densely sample around this point.
        # This creates a "cluster" that maximizes the hypervolume contribution from this region.
        
        # Determine a small radius for the cluster
        # The objective values change slowly, so a small radius in x-space is good.
        radius = 0.01
        rng = np.random.RandomState(42)
        
        archive_F = [current_f]
        
        # Generate points in a hypercube around current_x
        # Number of points = remaining_budget
        n_points = remaining_budget
        
        # We can use a Latin Hypercube or just random sampling within the box
        # To ensure we don't go out of bounds, we clip.
        for _ in range(n_points):
            # Random offset
            offset = rng.uniform(-radius, radius, dim)
            x_new = np.clip(current_x + offset, lo, hi)
            f_new = problem(x_new)
            archive_F.append(f_new)
            
        return np.array(archive_F)
# EVOLVE-BLOCK-END

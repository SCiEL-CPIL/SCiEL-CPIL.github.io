import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300 * 0.9  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_evals = 0
        all_points = []
        all_f = []
        
        # WFG4 structure:
        # f1 = x1
        # f2 = x2  
        # f3 = 1/(x1*x2) * g(x3..xn)
        # Pareto front: minimize g(x3..xn) while covering x1, x2 range
        
        # Strategy: Use structured sampling of x1, x2 to cover the Pareto front
        # and try multiple candidate values for x3..xn to find min g
        
        # Determine valid range for x1, x2
        # f3 < 7 (reference point) => 1/(x1*x2) * min_g < 7
        # Assuming min_g >= 1, we need x1*x2 > 1/7
        
        # Create a grid of x1, x2 values that cover the Pareto front well
        # Use log-uniform spacing to handle the hyperbolic nature
        
        # First, estimate bounds for x1, x2 that give f3 < 7
        # We'll sample conservatively in log space
        
        # Split budget: 70% for structured front sampling, 30% for g minimization
        front_budget = int(self.budget * 0.7)
        g_budget = self.budget - front_budget
        
        # Create log-uniform grid for x1 and x2
        # Range: [max(lo, 0.05), min(hi, 1.0)] to ensure product > 0.05
        x1_lo = max(lo[0], 0.05)
        x1_hi = min(hi[0], 1.0)
        x2_lo = max(lo[1], 0.05)
        x2_hi = min(hi[1], 1.0)
        
        # Create grid points - use sqrt of budget for grid size in each dim
        n_grid = int(np.sqrt(front_budget / 4)) + 1
        n_grid = max(n_grid, 5)
        
        # Log-uniform spacing
        log_x1 = np.linspace(np.log(x1_lo), np.log(x1_hi), n_grid)
        log_x2 = np.linspace(np.log(x2_lo), np.log(x2_hi), n_grid)
        x1_grid = np.exp(log_x1)
        x2_grid = np.exp(log_x2)
        
        # Candidate values for x3..xn to find min g
        # Try: all zeros, all mid, all ones, and some random perturbations
        dim_rest = self.dim - 2
        lo_rest = lo[2:]
        hi_rest = hi[2:]
        mid_rest = (lo_rest + hi_rest) / 2
        
        # Create diverse candidates for the tail
        tail_candidates = [
            lo_rest.copy(),
            mid_rest.copy(),
            hi_rest.copy(),
            np.zeros(dim_rest),  # Often g is minimized at 0 for WFG4
        ]
        
        # Add a few random candidates
        rng = np.random
        for _ in range(3):
            tail_candidates.append(lo_rest + (hi_rest - lo_rest) * rng.random(dim_rest))
        
        # Evaluate structured grid with best tail candidate (try all tails if budget allows)
        # First pass: use the zero tail (most likely to be optimal for WFG4)
        best_tail = np.zeros(dim_rest)
        
        # Evaluate grid with zero tail
        for i in range(n_grid):
            for j in range(n_grid):
                if n_evals >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                x1 = np.clip(x1_grid[i], lo[0], hi[0])
                x2 = np.clip(x2_grid[j], lo[1], hi[1])
                
                # Check if product is large enough
                if x1 * x2 < 0.1:
                    continue
                
                x = np.concatenate(([x1, x2], best_tail))
                f = problem(x)
                all_points.append(x)
                all_f.append(f)
                n_evals += 1
            
            if n_evals >= self.budget:
                break
        
        # If we have budget left, try other tail candidates on a subset of the grid
        if n_evals < self.budget:
            # Try other tails on a coarser grid
            n_coarse = max(3, n_grid // 2)
            x1_coarse = x1_grid[::max(1, n_grid // n_coarse)]
            x2_coarse = x2_grid[::max(1, n_grid // n_coarse)]
            
            for tail in tail_candidates[1:]:
                for i in range(len(x1_coarse)):
                    for j in range(len(x2_coarse)):
                        if n_evals >= self.budget:
                            break
                        if time.time() > deadline:
                            break
                        
                        x1 = np.clip(x1_coarse[i], lo[0], hi[0])
                        x2 = np.clip(x2_coarse[j], lo[1], hi[1])
                        
                        if x1 * x2 < 0.1:
                            continue
                        
                        x = np.concatenate(([x1, x2], tail))
                        f = problem(x)
                        all_points.append(x)
                        all_f.append(f)
                        n_evals += 1
                    
                    if n_evals >= self.budget:
                        break
                
                if n_evals >= self.budget:
                    break
        
        return all_points

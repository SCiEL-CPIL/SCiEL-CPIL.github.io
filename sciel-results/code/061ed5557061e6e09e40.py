# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer exploiting WFG9 structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget
        
        # Strategy: Exploit WFG9 structure
        # Assume first 3 variables define the Pareto front shape.
        # Fix remaining variables to midpoints.
        
        # Midpoints for fixed variables
        midpoints = (lo + hi) / 2.0
        
        # We will search over the first 3 variables.
        # Budget is 200.
        # Use a denser grid: 15 x 15 x 1 = 225 (too many), so 14 x 14 x 1 = 196
        # Or 12 x 12 x 1 = 144, leaving 56 for refinement.
        # Let's try 14 x 14 x 1 = 196, leaving 4 for refinement.
        # Actually, let's try 12 x 12 x 2 = 288 (too many).
        # 10 x 10 x 2 = 200 is the current.
        # Let's try 14 x 14 x 1 = 196. This gives denser coverage in 2D.
        
        n1, n2, n3 = 14, 14, 1
        total_grid_points = n1 * n2 * n3
        
        if total_grid_points > budget:
             scale = (budget ** (1.0/3.0)) / (total_grid_points ** (1.0/3.0))
             n1 = max(2, int(n1 * scale))
             n2 = max(2, int(n2 * scale))
             n3 = max(1, int(n3 * scale))
             total_grid_points = n1 * n2 * n3

        points = []
        
        # Generate grid points for first 3 variables
        x0_vals = np.linspace(lo[0], hi[0], n1)
        x1_vals = np.linspace(lo[1], hi[1], n2)
        x2_vals = np.linspace(lo[2], hi[2], n3)
        
        for x0 in x0_vals:
            for x1 in x1_vals:
                for x2 in x2_vals:
                    x = np.copy(midpoints)
                    x[0] = x0
                    x[1] = x1
                    x[2] = x2
                    points.append(x)
                    
        # Evaluate grid points
        F = np.array([problem(p) for p in points])
        eval_count = len(points)
        
        # Refinement phase: Use remaining budget to refine non-dominated points
        # Find non-dominated points in the grid
        def is_dominated(f, F):
            # Check if f is dominated by any point in F
            # f is dominated if there exists g in F such that g <= f and g != f
            for g in F:
                if np.all(g <= f) and np.any(g < f):
                    return True
            return False

        # Identify non-dominated indices
        nd_indices = []
        for i in range(len(F)):
            if not is_dominated(F[i], F):
                nd_indices.append(i)
                
        # If we have budget left, refine these points
        remaining_budget = budget - eval_count
        if remaining_budget > 0 and nd_indices:
            # Select a subset of non-dominated points to refine
            # We can refine all if budget allows, or just the best few
            # Let's refine all non-dominated points with small random perturbations
            # Number of refinements per point
            n_refine_per_point = max(1, remaining_budget // len(nd_indices))
            
            for idx in nd_indices:
                if eval_count >= budget:
                    break
                base_point = points[idx]
                for _ in range(n_refine_per_point):
                    if eval_count >= budget or time.time() > deadline:
                        break
                    # Random perturbation
                    # Scale perturbation by 10% of range
                    perturbation = np.random.uniform(-0.1, 0.1, size=dim) * (hi - lo)
                    new_point = base_point + perturbation
                    new_point = np.clip(new_point, lo, hi)
                    
                    f_new = problem(new_point)
                    points.append(new_point)
                    F = np.vstack([F, [f_new]])
                    eval_count += 1
                    
        return points
# EVOLVE-BLOCK-END

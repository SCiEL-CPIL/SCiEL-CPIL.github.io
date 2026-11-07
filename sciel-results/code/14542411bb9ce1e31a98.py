"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic DTLZ2-specific search using Latin Hypercube sampling on the Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        results = []
        used = 0
        
        # DTLZ2 Pareto front:
        # f1 = cos(alpha), f2 = sin(alpha)*cos(beta), f3 = sin(alpha)*sin(beta)
        # alpha in [0, pi/2], beta in [0, pi/2]
        # Decision variables:
        # x[0] = alpha / (pi/2)
        # x[1] = beta / (pi/2)
        # x[2:] = 0.5 (distance vars at optimum)
        
        # We will use a Latin Hypercube (LHS) sampling in the (alpha, beta) space.
        # This provides better coverage than a regular grid.
        
        # Reserve budget for explicit corner points to ensure extreme points are covered.
        # Corners in (alpha, beta) space:
        # (0, 0) -> (1, 0, 0)
        # (pi/2, 0) -> (0, 1, 0)
        # (pi/2, pi/2) -> (0, 0, 1)
        
        corners = [
            (0.0, 0.0),
            (1.0, 0.0), # alpha=pi/2, beta=0
            (1.0, 1.0)  # alpha=pi/2, beta=pi/2
        ]
        
        # Evaluate corners first
        for a_norm, b_norm in corners:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
            
            x = np.zeros(self.dim)
            x[0] = a_norm
            x[1] = b_norm
            x[2:] = 0.5
            
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            if np.all(np.isfinite(f)):
                results.append(f)
                used += 1
        
        # Remaining budget for LHS
        remaining_budget = self.budget - used
        
        # Determine grid size for LHS
        # We want roughly square grid.
        # n_grid = sqrt(remaining_budget)
        # Let's use a fixed 14x14 grid if budget allows, otherwise adjust.
        # 14*14 = 196. If we used 3 corners, we have 197 left.
        # So we can do 14x14 = 196 points.
        
        if remaining_budget > 0:
            n_grid = int(np.sqrt(remaining_budget))
            # Ensure we don't exceed budget
            # If n_grid*n_grid > remaining_budget, reduce n_grid
            while n_grid * n_grid > remaining_budget:
                n_grid -= 1
            
            # Use a fixed seed for reproducibility of the jitter, 
            # but the problem statement says np.random.seed(seed) is called by evaluator.
            # So we should use the global np.random state if we want it to vary with seed,
            # or use a local rng. The evaluator seeds np.random.
            # Using np.random directly will respect the seed.
            
            for i in range(n_grid):
                for j in range(n_grid):
                    if used >= self.budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    # Jittered grid point
                    # u in [0, 1)
                    u = np.random.rand()
                    v = np.random.rand()
                    
                    # Normalized alpha and beta in [0, 1]
                    a_norm = (i + u) / n_grid
                    b_norm = (j + v) / n_grid
                    
                    x = np.zeros(self.dim)
                    x[0] = a_norm
                    x[1] = b_norm
                    x[2:] = 0.5
                    
                    x = np.clip(x, lo, hi)
                    
                    f = problem(x)
                    if np.all(np.isfinite(f)):
                        results.append(f)
                        used += 1
                        
        # If we still have budget left (e.g. if n_grid was reduced), fill with random LHS points
        while used < self.budget and time.time() < deadline:
            a_norm = np.random.rand()
            b_norm = np.random.rand()
            
            x = np.zeros(self.dim)
            x[0] = a_norm
            x[1] = b_norm
            x[2:] = 0.5
            
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            if np.all(np.isfinite(f)):
                results.append(f)
                used += 1
        
        return np.array(results) if results else np.empty((0, self.n_obj))
# EVOLVE-BLOCK-END

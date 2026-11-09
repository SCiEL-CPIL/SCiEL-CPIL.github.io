"""Rewritten optimizer using a deterministic multi-start local search with adaptive step sizes."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-start CMA-ES-like optimizer with restarts and adaptive step sizes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        
        # Phase 1: Global exploration with multiple random starts and short local searches
        # We'll use a combination of random sampling and local refinement
        
        # First, do a coarse random search to find a good region
        num_coarse = min(30, budget // 10)
        candidates = []
        for _ in range(num_coarse):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            candidates.append((fx, x))
        
        # Sort by fitness and pick the best few to refine
        candidates.sort(key=lambda t: t[0])
        
        # Local search from the best candidates
        # Use a simple adaptive step size local search
        used_budget = num_coarse
        
        # Take the top 3 candidates for local refinement
        num_refine = min(3, len(candidates))
        for i in range(num_refine):
            fx, x = candidates[i]
            if used_budget >= budget:
                break
            
            # Adaptive step size local search
            sigma = 1.0
            x_curr = x.copy()
            f_curr = fx
            
            # Perform local search until budget is exhausted or no progress
            while used_budget < budget:
                # Try a few steps in random directions
                for _ in range(min(5, budget - used_budget)):
                    if used_budget >= budget:
                        break
                    direction = rng.standard_normal(dim)
                    direction /= (np.linalg.norm(direction) + 1e-10)
                    
                    # Try positive and negative steps
                    for sign in [1.0, -1.0]:
                        if used_budget >= budget:
                            break
                        x_try = np.clip(x_curr + sign * sigma * direction, -5, 5)
                        f_try = func(x_try)
                        used_budget += 1
                        
                        if f_try < f_curr:
                            x_curr = x_try
                            f_curr = f_try
                            sigma *= 1.2  # Increase step size on success
                            if f_curr < best_f:
                                best_f = f_curr
                                best_x = x_curr.copy()
                            break
                    else:
                        # No improvement in this direction
                        sigma *= 0.85  # Decrease step size on failure
                
                # If step size becomes too small, stop local search
                if sigma < 1e-4:
                    break
            
            # If we've used too much budget, stop
            if used_budget >= budget:
                break
        
        # If we still have budget, do a final random search to see if we can find something better
        if used_budget < budget:
            # Use remaining budget for random sampling
            while used_budget < budget:
                x = rng.uniform(-5, 5, dim)
                fx = func(x)
                used_budget += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END

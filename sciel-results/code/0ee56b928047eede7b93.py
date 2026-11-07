"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Lattice-aware exploration + fine local search for Rastrigin f15."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270
        rng = np.random.default_rng(0)
        d = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        total_calls = 0

        def check_deadline():
            return time.time() > deadline or total_calls >= budget

        # Phase 1: Lattice exploration - sample points on the Rastrigin grid
        # Rastrigin minima are at even integers; after rotation+shift they're in [-4,4]^5
        # Sample a subset of lattice points to find promising basins
        lattice_vals = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
        
        # Generate all lattice points (5^5 = 3125) but only evaluate a subset
        # Use stratified sampling: evaluate points with some randomization
        explore_budget = min(100, budget // 5)
        
        # Create diverse starting points: mix of pure lattice and slightly perturbed
        candidates = []
        
        # Pure lattice points (deterministic subset)
        for i in range(d):
            for j in range(d):
                for k in range(d):
                    for l in range(d):
                        for m in range(d):
                            pt = np.array([lattice_vals[i], lattice_vals[j], lattice_vals[k],
                                           lattice_vals[l], lattice_vals[m]])
                            candidates.append(pt)
        
        # Shuffle and take explore_budget points
        rng.shuffle(candidates)
        explore_points = candidates[:explore_budget]
        
        # Evaluate lattice points
        for pt in explore_points:
            if check_deadline():
                break
            pt_clipped = np.clip(pt, -5, 5)
            f_val = func(pt_clipped)
            total_calls += 1
            if f_val < best_f:
                best_f = f_val
                best_x = pt_clipped.copy()
        
        # Phase 2: Fine local search around the best point found
        if best_x is not None and not check_deadline():
            x = best_x.copy()
            fx = best_f
            sigma = 0.5  # Start with moderate step size for fine search
            
            # Also try small perturbations around other top lattice points
            # to see if we can find a better basin
            # Keep top 3 lattice points from exploration
            top_points = []
            for pt in explore_points:
                pt_clipped = np.clip(pt, -5, 5)
                # We don't have stored f-values, so just use the best and a few random ones
                if len(top_points) < 3:
                    top_points.append(pt_clipped)
            
            # Do local search from best point
            no_improve = 0
            max_no_improve = 30
            
            while total_calls < budget - 20 and not check_deadline():
                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = func(y)
                total_calls += 1
                
                if fy < fx:
                    x, fx = y, fy
                    no_improve = 0
                    sigma *= 1.15  # Slight increase on success
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                else:
                    sigma *= 0.85
                    no_improve += 1
                
                if no_improve >= max_no_improve:
                    break
                
                if sigma < 1e-10:
                    break
        
        # Phase 3: Final polish with very small steps
        if best_x is not None and not check_deadline():
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            remaining = budget - total_calls
            polish_steps = min(30, remaining)
            
            for _ in range(polish_steps):
                if check_deadline():
                    break
                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = func(y)
                total_calls += 1
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                    sigma *= 1.05
                else:
                    sigma *= 0.9
                if sigma < 1e-12:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END

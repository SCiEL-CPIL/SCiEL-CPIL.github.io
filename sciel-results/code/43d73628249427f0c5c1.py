import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s margin before 300s limit
        
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Multi-Start Randomized Coordinate Descent
        # 1. Generate a set of diverse starting points
        # 2. For each, perform a series of 1D line searches along random directions
        #    (or coordinate axes, but random directions are more robust to rotation)
        
        # We will use a mix of random directions and coordinate axes.
        # Since the function is Sphere (separable in some basis), random directions
        # might not align perfectly, but with enough restarts and steps, it should find the minimum.
        
        # Strategy:
        # - Split budget among K restarts.
        # - For each restart, do N steps of 1D search.
        # - Use a golden section search or simple parabolic interpolation for the 1D search.
        # - Given the budget is 500, and dim=5, we can afford many restarts.
        
        # Let's try 10 restarts, 50 function evals each.
        # Or 5 restarts, 100 evals each.
        # Let's go with 5 restarts.
        
        K = 5
        evals_per_restart = budget // K
        # Ensure we don't exceed budget
        total_evals = 0
        best_f = np.inf
        best_x = None
        
        # Pre-generate random directions to save time? No, generate on fly.
        
        for k in range(K):
            if time.time() > deadline:
                break
            
            # Initialize point
            # Use a stratified sampling approach for initial points?
            # Or just uniform random.
            x = rng.uniform(-4, 4, dim)  # Slightly narrower than [-5,5] to avoid boundary issues initially
            
            f = func(x)
            total_evals += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            
            # Perform 1D searches
            # We will alternate between random directions and coordinate axes
            # Actually, for Sphere, coordinate axes in the rotated frame are ideal.
            # But we don't know the rotation.
            # Random directions are a good compromise.
            
            for step in range(evals_per_restart):
                if time.time() > deadline:
                    break
                if total_evals >= budget:
                    break
                
                # Choose a direction
                # 50% chance random direction, 50% chance coordinate axis
                if rng.random() < 0.5:
                    d = rng.normal(size=dim)
                    d /= np.linalg.norm(d)
                else:
                    axis = rng.integers(0, dim)
                    d = np.zeros(dim)
                    d[axis] = 1.0
                
                # 1D search along d
                # Find optimal step size alpha such that f(x + alpha * d) is minimized
                # Use a simple bracketing and golden section search.
                # Initial bracket: try small steps, expand if needed.
                
                # Try a few candidate steps to find a good one quickly
                # This is a heuristic approach to avoid expensive bracketing.
                # Candidates: 0.1, 0.5, 1.0, 2.0, 5.0
                # Evaluate f at x + alpha * d for these alphas.
                # Pick the best one.
                # Then refine with a smaller step around the best one.
                
                candidates = [0.1, 0.5, 1.0, 2.0, 5.0]
                best_alpha = 0.0
                best_f_candidate = f
                x_candidate = x
                
                for alpha in candidates:
                    x_new = x + alpha * d
                    # Clip to bounds
                    x_new = np.clip(x_new, -5, 5)
                    f_new = func(x_new)
                    total_evals += 1
                    if f_new < best_f_candidate:
                        best_f_candidate = f_new
                        best_alpha = alpha
                        x_candidate = x_new
                    
                    if total_evals >= budget:
                        break
                
                if best_alpha != 0.0:
                    x = x_candidate
                    f = best_f_candidate
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                    
                    # Refine the step around best_alpha
                    # Try smaller steps around best_alpha
                    refine_step = best_alpha / 5.0
                    if refine_step < 1e-4:
                        refine_step = 1e-4
                    
                    # Try 3 points: best_alpha - refine_step, best_alpha, best_alpha + refine_step
                    # Actually, just try a few nearby points to see if we can improve further
                    # This is a simple local search refinement
                    for delta in [-refine_step, refine_step]:
                        if total_evals >= budget:
                            break
                        alpha_new = best_alpha + delta
                        if alpha_new < 0:
                            continue
                        x_new = x + (alpha_new - best_alpha) * d  # Wait, x is already x + best_alpha * d
                        # Correction: x is the current best point.
                        # We want to move from x by (alpha_new - best_alpha) * d ?
                        # No, x is x_old + best_alpha * d.
                        # We want to try x_old + alpha_new * d.
                        # So the move from x is (alpha_new - best_alpha) * d.
                        x_new = x + (alpha_new - best_alpha) * d
                        x_new = np.clip(x_new, -5, 5)
                        f_new = func(x_new)
                        total_evals += 1
                        if f_new < f:
                            x = x_new
                            f = f_new
                            if f < best_f:
                                best_f = f
                                best_x = x.copy()

        return best_x, best_f

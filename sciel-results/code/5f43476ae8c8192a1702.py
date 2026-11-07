"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Adaptive Basin Hopping with L-BFGS-B for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0
        
        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)
        
        # Wrapper to track calls and enforce deadline
        def eval_and_track(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return best_f
            calls += 1
            try:
                val = float(func(x))
                if val < best_f:
                    best_f = val
                    best_x = np.array(x, dtype=float)
                return val
            except Exception:
                return best_f

        rng = np.random.default_rng(12345)
        
        # Phase 1: Initial Exploration and Deep Dive
        # We want to find a good local minimum quickly.
        # Start with a few random points and refine the best one thoroughly.
        
        # Try 5 random starts, keep the best
        candidates = []
        for i in range(5):
            if calls >= budget or time.time() > deadline:
                break
            x0 = rng.uniform(-5, 5, size=dim)
            f_val = eval_and_track(x0)
            candidates.append((f_val, x0))
            
        if not candidates:
            return best_x, best_f
            
        candidates.sort(key=lambda t: t[0])
        current_x = candidates[0][1].copy()
        current_f = candidates[0][0]
        
        # Deep dive on the best candidate using L-BFGS-B
        # Allocate ~50% of budget to this first deep dive to ensure a strong early score
        # But we must leave enough for basin hopping.
        # Let's say we allow 250 calls for the first deep dive.
        
        # L-BFGS-B configuration
        bounds = [(-5, 5)] * dim
        
        # We will run L-BFGS-B until it converges or we run out of time/budget.
        # To implement "Basin Hopping", we will:
        # 1. Minimize current_x -> x_min
        # 2. If x_min is better, update best.
        # 3. Perturb x_min -> x_new
        # 4. Set current_x = x_new
        # 5. Repeat.
        
        # Adaptive perturbation scale
        # Start with a scale that covers a good portion of the box, e.g., 1.0
        # Decrease it as we find better solutions to refine.
        
        scale = 1.5
        min_scale = 0.01
        
        # We will run this loop until budget is exhausted.
        # Each iteration consists of:
        # - Local minimization (variable cost)
        # - Perturbation (1 call? No, perturbation is just generating a point, 
        #   the next minimization will evaluate it. 
        #   Actually, we should evaluate the perturbed point? 
        #   No, L-BFGS-B will evaluate the starting point first.
        
        # Optimization: 
        # 1. Run L-BFGS-B from current_x.
        # 2. Get result x_opt, f_opt.
        # 3. If f_opt < best_f, update best.
        # 4. If we have budget left, perturb x_opt and repeat.
        
        while calls < budget and time.time() < deadline:
            # Run L-BFGS-B
            try:
                res = minimize(
                    eval_and_track, 
                    current_x, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={'maxiter': 100, 'maxls': 20}
                )
            except Exception:
                break
                
            x_opt = res.x
            f_opt = res.fun
            
            # Update best
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()
                
            # Check if we should stop
            if calls >= budget or time.time() > deadline:
                break
            
            # If we are stuck (f_opt is not improving significantly), we might want to 
            # increase perturbation? Or just keep going.
            # Basin Hopping logic:
            # Perturb the current optimum to escape the basin.
            
            # Adaptive scale: 
            # If we found a very good solution, reduce scale to refine.
            # If we are stuck, keep scale or increase? 
            # Simple approach: geometric decay.
            # But if we decay too fast, we might get stuck in the first good basin.
            # Let's use a constant scale for a while, then decay.
            
            # Let's try: scale = max(0.1, scale * 0.95)
            # Or based on progress: if best_f improved, reduce scale.
            
            # For now, let's use a fixed schedule:
            # Start with 1.5, multiply by 0.95 each step.
            
            noise = rng.normal(0, scale, size=dim)
            current_x = x_opt + noise
            # Clip to bounds
            current_x = np.clip(current_x, -5, 5)
            
            # Decay scale
            scale = max(0.05, scale * 0.95)
            
            # If scale becomes very small, we are effectively doing local refinement.
            # This is good for the end of the run.

        return best_x, best_f
# EVOLVE-BLOCK-END

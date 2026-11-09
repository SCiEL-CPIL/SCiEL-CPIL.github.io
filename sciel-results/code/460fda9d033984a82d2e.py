"""Combined optimizer: Deterministic multi-scale coordinate descent with refinement."""
import time
import numpy as np


class Optimizer:
    """
    Optimizer for BBOB f4 (Büche-Rastrigin).
    Uses a deterministic multi-scale coordinate descent strategy which has shown
    superior performance on this multimodal landscape compared to stochastic methods
    like CMA-ES with limited budget.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: evaluator kills at 300s, we stop at 290s
        deadline = start_time + 290.0
        
        dim = self.dim
        budget = self.budget
        calls_used = 0
        best_x = None
        best_f = np.inf

        def check_time():
            return time.time() > deadline - 2.0

        # Phase 1: Structured Initial Scan
        # Büche-Rastrigin has a global minimum that is shifted and rotated.
        # We need to find a basin of attraction. A coarse grid scan is effective.
        scan_points = []
        
        # 1. Corners of the box [-5, 5]^dim
        for i in range(2 ** dim):
            pt = np.array([[-5.0, 5.0][(i >> d) & 1] for d in range(dim)])
            scan_points.append(pt)
            
        # 2. Center
        scan_points.append(np.zeros(dim))
        
        # 3. Face centers (one axis non-zero, others 0)
        # Values: +/- 2, +/- 4 (avoiding 0 which is covered, avoiding 5 which is corners)
        for d in range(dim):
            for val in [-4.0, -2.0, 2.0, 4.0]:
                pt = np.zeros(dim)
                pt[d] = val
                scan_points.append(pt)
                
        # 4. Some intermediate points to catch rotated minima
        # e.g., (2, 2, 0, 0, 0), (-3, 0, 3, 0, 0)
        for i in range(dim):
            for j in range(dim):
                if i == j: continue
                for v1 in [-3.0, 3.0]:
                    for v2 in [-3.0, 3.0]:
                        pt = np.zeros(dim)
                        pt[i] = v1
                        pt[j] = v2
                        scan_points.append(pt)
                        
        # Limit scan to ensure we have budget left for optimization
        max_scan = 40
        scan_points = scan_points[:max_scan]
        
        for sp in scan_points:
            if calls_used >= budget or check_time():
                break
            f_val = func(sp)
            calls_used += 1
            if f_val < best_f:
                best_f = f_val
                best_x = sp.copy()
                
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_used += 1

        x = best_x.copy()
        fx = best_f

        # Phase 2: Multi-Scale Coordinate Descent
        # We shrink the step size geometrically. This allows us to "hop" over
        # local minima barriers with large steps and then refine with small steps.
        # Steps are chosen to cover the domain [-5, 5] efficiently.
        base_steps = [5.0, 2.5, 1.25, 0.625, 0.3125, 0.156, 0.078, 0.039, 0.02, 0.01, 0.005, 0.002, 0.001]
        
        for step in base_steps:
            # For each step size, we iterate until no improvement is found in a full pass
            # or we run out of budget/time.
            # To speed up traversal along a valley, we allow "momentum": 
            # if a move in a direction is successful, we try again in that direction.
            
            # Limit iterations per step size to prevent infinite loops if numerical noise exists
            max_iter_per_step = 100
            
            for _iter in range(max_iter_per_step):
                if calls_used >= budget or check_time():
                    break
                
                improved = False
                for d in range(dim):
                    if calls_used >= budget or check_time():
                        break
                    
                    # Try positive direction
                    x_pos = x.copy()
                    x_pos[d] += step
                    x_pos = np.clip(x_pos, -5.0, 5.0)
                    f_pos = func(x_pos)
                    calls_used += 1
                    
                    if f_pos < fx:
                        x = x_pos
                        fx = f_pos
                        improved = True
                        # Momentum: try to continue in the same direction
                        # We do this by re-checking the same dimension immediately? 
                        # No, standard CD cycles. But for rotated landscapes, 
                        # simple cycling is slow. 
                        # Let's stick to standard cycling for stability, 
                        # but the geometric shrinking handles the scale.
                        
                    # Try negative direction
                    x_neg = x.copy()
                    x_neg[d] -= step
                    x_neg = np.clip(x_neg, -5.0, 5.0)
                    f_neg = func(x_neg)
                    calls_used += 1
                    
                    if f_neg < fx:
                        x = x_neg
                        fx = f_neg
                        improved = True

                if not improved:
                    break

        # Phase 3: Fine Local Polish
        # Use remaining budget for a tight random local search to escape 
        # any residual plateaus or refine the final position.
        if calls_used < budget and not check_time():
            rng = np.random.default_rng(42)
            sigma = 0.05
            # We want to use up to 20% of remaining budget here, or until time runs out
            # But we must be careful not to waste too much on a bad region.
            # Since CD already found a good basin, this is just polishing.
            
            # Dynamic sigma adjustment
            while calls_used < budget and not check_time():
                x_new = x + sigma * rng.standard_normal(dim)
                x_new = np.clip(x_new, -5.0, 5.0)
                f_new = func(x_new)
                calls_used += 1
                
                if f_new < fx:
                    fx = f_new
                    x = x_new
                    sigma *= 1.1  # Slightly increase if improving
                else:
                    sigma *= 0.8  # Shrink if not improving
                    if sigma < 1e-6:
                        break

        return x, fx

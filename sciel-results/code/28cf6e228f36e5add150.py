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
        # Safety margin: evaluator kills at 330s, we stop at 310s
        deadline = start_time + 310.0
        
        dim = self.dim
        budget = self.budget
        calls_used = 0
        best_x = None
        best_f = np.inf

        def check_time():
            return time.time() > deadline - 5.0

        def update_best(x, fx):
            nonlocal best_x, best_f
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

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
            update_best(sp, f_val)
                
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
        # Using a finer geometric sequence for better resolution.
        base_steps = [5.0, 2.5, 1.25, 0.625, 0.3125, 0.15625, 0.078125, 0.0390625, 
                      0.01953125, 0.009765625, 0.0048828125, 0.00244140625, 
                      0.001220703125, 0.0006103515625, 0.00030517578125, 0.000152587890625]
        
        for step in base_steps:
            if check_time():
                break
            # For each step size, we iterate until no improvement is found in a full pass
            # or we run out of budget/time.
            
            # Limit iterations per step size to prevent infinite loops if numerical noise exists
            max_iter_per_step = 50
            
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
                        update_best(x, fx)

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
                        update_best(x, fx)

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
                    update_best(x, fx)
                    sigma *= 1.1  # Slightly increase if improving
                else:
                    sigma *= 0.8  # Shrink if not improving
                    if sigma < 1e-6:
                        break

        return best_x, best_f

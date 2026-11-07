import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        d = self.dim
        budget = self.budget
        best_x = np.zeros(d)
        best_f = np.inf

        def _time_left():
            return deadline - time.time()

        try:
            # Phase 1: Multi-resolution Sobol scan to find the step boundary
            # The step ellipsoidal function has a flat plateau outside and smooth inside.
            # We need to cross the boundary early to maximize AOCC.
            
            # Generate Sobol sequence points covering [-5, 5]^d
            # Use multiple resolutions: coarse first, then refine
            # Total budget for scanning: ~150 points, leaving ~350 for local optimization
            
            scan_budget = min(150, budget)
            
            # Generate Sobol points
            sampler = qmc.Sobol(d=d, scramble=True, seed=42)
            
            # Phase 1a: Coarse scan with 32 points
            n_coarse = 32
            if n_coarse > scan_budget:
                n_coarse = scan_budget
            
            points_coarse = sampler.random(n_coarse)
            # Map from [0,1]^d to [-5,5]^d
            points_coarse = points_coarse * 10.0 - 5.0
            
            for x in points_coarse:
                if budget <= 0 or _time_left() < 5.0:
                    break
                x = np.clip(x, -5, 5)
                fx = func(x)
                budget -= 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            
            # Phase 1b: Medium resolution scan with 64 points
            n_medium = 64
            if n_medium > scan_budget - n_coarse:
                n_medium = max(0, scan_budget - n_coarse)
            
            if n_medium > 0:
                sampler_medium = qmc.Sobol(d=d, scramble=True, seed=123)
                points_medium = sampler_medium.random(n_medium)
                points_medium = points_medium * 10.0 - 5.0
                
                for x in points_medium:
                    if budget <= 0 or _time_left() < 5.0:
                        break
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    budget -= 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
            
            # Phase 1c: Fine resolution scan with remaining scan budget
            n_fine = scan_budget - n_coarse - n_medium
            if n_fine > 0:
                sampler_fine = qmc.Sobol(d=d, scramble=True, seed=456)
                points_fine = sampler_fine.random(n_fine)
                points_fine = points_fine * 10.0 - 5.0
                
                for x in points_fine:
                    if budget <= 0 or _time_left() < 5.0:
                        break
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    budget -= 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
            
            # Phase 2: Local coordinate descent refinement
            # Start from the best point found
            x = best_x.copy()
            f = best_f
            
            # Coordinate descent with adaptive step size
            step_sizes = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]
            rng = np.random.default_rng(789)
            
            for step_size in step_sizes:
                if budget <= 0 or _time_left() < 2.0:
                    break
                
                improved = True
                while improved and budget > 0 and _time_left() > 2.0:
                    improved = False
                    for i in range(d):
                        if budget <= 0 or _time_left() < 2.0:
                            break
                        
                        # Try positive direction
                        y_pos = x.copy()
                        y_pos[i] += step_size
                        y_pos = np.clip(y_pos, -5, 5)
                        fy_pos = func(y_pos)
                        budget -= 1
                        
                        if fy_pos < f:
                            x = y_pos.copy()
                            f = fy_pos
                            best_f = f
                            best_x = x.copy()
                            improved = True
                            continue
                        
                        # Try negative direction
                        if budget <= 0 or _time_left() < 2.0:
                            break
                        y_neg = x.copy()
                        y_neg[i] -= step_size
                        y_neg = np.clip(y_neg, -5, 5)
                        fy_neg = func(y_neg)
                        budget -= 1
                        
                        if fy_neg < f:
                            x = y_neg.copy()
                            f = fy_neg
                            best_f = f
                            best_x = x.copy()
                            improved = True
                            continue
            
            # Phase 3: Final local refinement with small random perturbations
            if budget > 0 and _time_left() > 2.0:
                sigma = 0.05
                lam = 5
                no_improve = 0
                max_no_improve = 20
                rng2 = np.random.default_rng(999)
                
                while budget > 0 and _time_left() > 2.0:
                    best_in_pop = f
                    best_y = x.copy()
                    
                    for i in range(lam):
                        if budget <= 0 or _time_left() < 2.0:
                            break
                        y = x + sigma * rng2.standard_normal(d)
                        y = np.clip(y, -5, 5)
                        fy = func(y)
                        budget -= 1
                        if fy < best_in_pop:
                            best_in_pop = fy
                            best_y = y.copy()
                    
                    if best_in_pop < f:
                        x = best_y.copy()
                        f = best_in_pop
                        best_f = f
                        best_x = x.copy()
                        no_improve = 0
                        sigma *= 1.2
                    else:
                        no_improve += 1
                        sigma *= 0.8
                        if no_improve > max_no_improve:
                            break
            
        except Exception as e:
            pass
        
        return best_x, best_f

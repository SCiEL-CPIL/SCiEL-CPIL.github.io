import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        eval_count = 0
        
        def counting_func(x):
            nonlocal eval_count, best_f, best_x
            if eval_count >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            x_clipped = np.clip(np.asarray(x, dtype=float), -5, 5)
            val = float(func(x_clipped))
            eval_count += 1
            if val < best_f:
                best_f = val
                best_x = x_clipped.copy()
            return val
        
        # Phase 1: Coarse grid search to find promising region
        # Katsuura function is periodic with period 2, so we can use a grid
        # Start with a coarse grid over the domain
        n_coarse = 3  # 3 points per dimension
        coarse_points = []
        for i in range(n_coarse):
            for j in range(n_coarse):
                for k in range(n_coarse):
                    for l in range(n_coarse):
                        for m in range(n_coarse):
                            x = np.array([
                                -5 + 10 * i / (n_coarse - 1),
                                -5 + 10 * j / (n_coarse - 1),
                                -5 + 10 * k / (n_coarse - 1),
                                -5 + 10 * l / (n_coarse - 1),
                                -5 + 10 * m / (n_coarse - 1)
                            ])
                            coarse_points.append(x)
        
        # Evaluate coarse grid
        for pt in coarse_points:
            if eval_count >= budget or time.time() > deadline:
                break
            counting_func(pt)
        
        # Phase 2: Hierarchical refinement around best point
        if best_x is not None and eval_count < budget and time.time() < deadline:
            # Refine around the best point with progressively finer grids
            center = best_x.copy()
            for level in range(3):
                if eval_count >= budget or time.time() > deadline:
                    break
                
                # Grid size increases with each level
                n_refine = [5, 7, 9][level]
                # Search radius decreases with each level
                radius = [2.0, 1.0, 0.5][level]
                
                # Generate grid points around center
                refine_points = []
                for i in range(n_refine):
                    for j in range(n_refine):
                        for k in range(n_refine):
                            for l in range(n_refine):
                                for m in range(n_refine):
                                    x = center + radius * (
                                        np.array([i, j, k, l, m]) / (n_refine - 1) - 0.5
                                    ) * 2
                                    x = np.clip(x, -5, 5)
                                    refine_points.append(x)
                
                # Evaluate refinement grid
                for pt in refine_points:
                    if eval_count >= budget or time.time() > deadline:
                        break
                    counting_func(pt)
                
                # Update center to best point found
                if best_x is not None:
                    center = best_x.copy()
        
        # Phase 3: CMA-ES from the best point for fine-tuning
        remaining = budget - eval_count
        if remaining > 50 and best_x is not None:
            try:
                popsize = 15
                max_iter = max(5, remaining // popsize - 5)
                
                res = scipy.optimize.cma(
                    counting_func,
                    best_x,
                    sigma0=0.3,
                    options={
                        'maxiter': max_iter,
                        'popsize': popsize,
                        'tol': 1e-10,
                        'verbose': 0,
                        'bounds': [(-5.0, 5.0)] * dim
                    }
                )
            except Exception:
                pass
        
        # Phase 4: Polish with Nelder-Mead
        remaining = budget - eval_count
        if remaining > 20 and best_x is not None:
            try:
                res_nm = scipy.optimize.minimize(
                    counting_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': remaining // 3,
                        'xatol': 1e-12,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
            except Exception:
                pass
        
        # Ensure we have a valid result
        if best_x is None:
            best_x = np.zeros(dim)
            try:
                best_f = float(func(best_x))
                eval_count += 1
            except Exception:
                best_f = 0.0
        
        return best_x, best_f

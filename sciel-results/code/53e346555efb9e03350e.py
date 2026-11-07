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
        
        # Phase 1: Structured initial sampling (Halton sequence)
        # Increased to 50 points for better global coverage of the multimodal landscape
        n_initial = 50
        
        def halton_sequence(n, dim):
            """Generate n points in dim dimensions using Halton sequence."""
            points = []
            for i in range(1, n + 1):
                pt = []
                for d in range(dim):
                    # Find the next prime for this dimension
                    bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
                    base = bases[d % len(bases)]
                    f = 1.0
                    r = 0.0
                    idx = i
                    while idx > 0:
                        f /= base
                        r += f * (idx % base)
                        idx //= base
                    pt.append(r)
                points.append(pt)
            return np.array(points)
        
        # Generate initial points
        init_points = halton_sequence(n_initial, dim)
        # Map from [0,1]^dim to [-5,5]^dim
        init_points = init_points * 10.0 - 5.0
        
        # Evaluate all initial points
        for pt in init_points:
            if eval_count >= budget or time.time() > deadline:
                break
            counting_func(pt)
        
        # Phase 2: CMA-ES from the best point
        # Larger sigma0 to allow exploration out of local minima
        # Smaller popsize to allow more generations within the budget
        remaining = budget - eval_count
        if remaining > 30 and best_x is not None:
            try:
                popsize = 10
                # CMA-ES needs at least popsize evals per generation
                max_iter = max(5, remaining // popsize - 5)
                
                res = scipy.optimize.cma(
                    counting_func,
                    best_x,
                    sigma0=2.0,
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
        
        # Phase 3: Polish with Nelder-Mead
        remaining = budget - eval_count
        if remaining > 10 and best_x is not None:
            try:
                res_nm = scipy.optimize.minimize(
                    counting_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': remaining // 2,
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

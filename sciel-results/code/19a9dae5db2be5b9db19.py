import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        n_evals = 0

        def evaluate(x):
            nonlocal n_evals, best_x, best_f
            if n_evals >= budget:
                return float('inf')
            x = np.clip(x, -5, 5)
            f = func(x)
            n_evals += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def line_search(func_eval, x, direction, step_size=1.0, max_iter=100):
            # Golden section search or parabolic interpolation
            # We will use a simple parabolic interpolation line search
            # which is efficient for smooth unimodal functions
            
            def f_line(t):
                return func_eval(x + t * direction)
            
            # Initial bracket
            t0 = 0.0
            t1 = step_size
            f0 = f_line(t0)
            f1 = f_line(t1)
            
            if f1 > f0:
                # Try negative direction
                t1 = -step_size
                f1 = f_line(t1)
                if f1 > f0:
                    # We are at a local minimum in both directions?
                    # Just return t0
                    return t0, f0
            
            # Ensure f0 < f1 for bracket [t0, t1] where min is inside
            # If f1 < f0, expand t1
            while f1 < f0:
                t0, f0 = t1, f1
                t1 = t1 * 2 + 0.1
                if t1 > 10.0 or n_evals >= budget:
                    break
                f1 = f_line(t1)
            
            # Now we have a bracket [t0, t1] with f0 < f1? 
            # Wait, if we started with t0=0, t1=step, and f1 > f0, we flipped.
            # If f1 < f0, we expanded. 
            # We need a bracket [ta, tb] such that f(ta) > f(t_min) > f(tb) or similar.
            # Actually, simpler: use scipy.optimize.minimize_scalar if available? 
            # "Python standard library, NumPy and SciPy are available."
            # Using scipy is safer and faster.
            
            from scipy.optimize import minimize_scalar
            res = minimize_scalar(f_line, bounds=(-5.0, 5.0), method='bounded', options={'maxiter': 50})
            return res.x, res.fun

        def powell_method(func_eval, x0, max_iter=100):
            x = x0.copy()
            f = func_eval(x)
            directions = [np.zeros(dim) for _ in range(dim)]
            for i in range(dim):
                directions[i][i] = 1.0
            
            for _ in range(max_iter):
                if time.time() > deadline:
                    break
                if n_evals >= budget - 10:
                    break
                    
                x_old = x.copy()
                f_old = f
                min_delta = 0.0
                min_dir_idx = -1
                
                # 1. Minimize along each direction
                for i in range(dim):
                    t_opt, f_opt = line_search(func_eval, x, directions[i], step_size=0.5)
                    if n_evals >= budget:
                        break
                    
                    x_new = x + t_opt * directions[i]
                    f_new = f_opt
                    
                    delta = f - f_new
                    if delta > min_delta:
                        min_delta = delta
                        min_dir_idx = i
                    
                    x = x_new
                    f = f_new
                
                if n_evals >= budget:
                    break

                # 2. Update direction
                if min_dir_idx != -1:
                    # Replace the direction that gave the largest improvement
                    directions[min_dir_idx] = x - x_old
                
                # 3. Check convergence
                if abs(f_old - f) < 1e-10:
                    break

            return x, f

        # Phase 1: Global Search / Diverse Starting Points
        # Use a small number of diverse starts to find a good basin
        n_starts = 20
        rng = np.random.RandomState(42)
        
        # Generate points: center, vertices, and random
        candidates = []
        candidates.append(np.zeros(dim))
        
        # Vertices of [-4, 4]^dim
        for mask in range(1 << dim):
            p = np.zeros(dim)
            for i in range(dim):
                if mask & (1 << i):
                    p[i] = 4.0
                else:
                    p[i] = -4.0
            candidates.append(p)
            
        # Random points
        while len(candidates) < n_starts:
            candidates.append(rng.uniform(-4, 4, dim))
            
        best_start_x = None
        best_start_f = float('inf')
        
        for p in candidates:
            if time.time() > deadline:
                break
            if n_evals >= budget:
                break
            f = evaluate(p)
            if f < best_start_f:
                best_start_f = f
                best_start_x = p.copy()
                
        if best_start_x is None:
            best_start_x = np.zeros(dim)
            evaluate(best_start_x)

        # Phase 2: Intensive Local Optimization with Powell
        # Allocate remaining budget to Powell
        remaining = budget - n_evals
        if remaining > 0:
            # Powell iterations depend on line searches.
            # Each Powell iteration does dim line searches.
            # Each line search does ~20-50 evals.
            # So one Powell iteration ~ 100-250 evals.
            # We can afford 1-2 Powell iterations if remaining is small.
            # Let's just let it run until budget/time limit.
            powell_method(evaluate, best_start_x, max_iter=100)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f

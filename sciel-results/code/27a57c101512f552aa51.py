import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0
        n = self.dim
        budget = self.budget
        
        best_x = np.zeros(n)
        best_f = np.inf
        calls = 0

        def try_func(x):
            nonlocal best_f, best_x, calls
            if calls >= budget or time.time() > deadline:
                return best_f
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = float(func(x))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Initial sampling to find a good starting point
        # Use a mix of deterministic and random points
        rng = np.random.default_rng(42)
        
        # Start with origin and some axis-aligned points
        init_points = [np.zeros(n)]
        
        # Add points along axes
        for i in range(n):
            for sign in [-1, 1]:
                pt = np.zeros(n)
                pt[i] = sign * 4.0
                init_points.append(pt)
        
        # Add random points
        n_random = 20
        random_pts = rng.uniform(-5.0, 5.0, (n_random, n))
        for pt in random_pts:
            init_points.append(pt)
        
        # Evaluate initial points
        init_results = []
        for x in init_points:
            if calls >= budget or time.time() > deadline:
                break
            f = try_func(x)
            init_results.append((f, x))
            
        if not init_results:
            return best_x, best_f

        # Sort by function value
        init_results.sort(key=lambda x: x[0])
        
        # Phase 2: Adaptive coordinate descent with ridge exploitation
        # Use the best few initial points as starting candidates
        n_restarts = min(8, len(init_results))
        
        for k in range(n_restarts):
            if calls >= budget or time.time() > deadline:
                break
                
            x = init_results[k][1].copy()
            f = init_results[k][0]
            
            # Adaptive step sizes with exponential moving average
            step_sizes = np.ones(n) * 1.0
            step_history = np.ones(n) * 1.0  # EMA of successful steps
            
            # Coordinate descent iterations with adaptive step sizing
            max_cd_iters = 50
            for cd_iter in range(max_cd_iters):
                if calls >= budget or time.time() > deadline:
                    break
                    
                improved = False
                for i in range(n):
                    if calls >= budget or time.time() > deadline:
                        break
                    
                    best_local_f = f
                    best_local_x = x.copy()
                    best_step = 0.0
                    
                    # Try a range of step sizes: current, larger, smaller
                    step_candidates = [
                        step_sizes[i] * 2.0,
                        step_sizes[i] * 1.5,
                        step_sizes[i],
                        step_sizes[i] * 0.5,
                        step_sizes[i] * 0.25,
                    ]
                    
                    for step in step_candidates:
                        if step < 1e-6:
                            continue
                            
                        # Positive direction
                        x_plus = x.copy()
                        x_plus[i] += step
                        if np.all(x_plus >= -5.0) and np.all(x_plus <= 5.0):
                            f_plus = try_func(x_plus)
                            if f_plus < best_local_f:
                                best_local_f = f_plus
                                best_local_x = x_plus.copy()
                                best_step = step
                                improved = True
                        
                        # Negative direction
                        x_minus = x.copy()
                        x_minus[i] -= step
                        if np.all(x_minus >= -5.0) and np.all(x_minus <= 5.0):
                            f_minus = try_func(x_minus)
                            if f_minus < best_local_f:
                                best_local_f = f_minus
                                best_local_x = x_minus.copy()
                                best_step = -step
                                improved = True
                    
                    if improved:
                        x = best_local_x
                        f = best_local_f
                        # Update step size using EMA: increase if we moved, keep direction
                        abs_step = abs(best_step)
                        # Blend: 70% new, 30% old (favors recent successful steps)
                        step_history[i] = 0.7 * abs_step + 0.3 * step_history[i]
                        step_sizes[i] = step_history[i]
                    else:
                        # No improvement, shrink step size
                        step_sizes[i] *= 0.5
                        step_history[i] *= 0.5
                        if step_sizes[i] < 1e-5:
                            step_sizes[i] = 1e-5
                            step_history[i] = 1e-5
                
                if not improved:
                    # No dimension improved, try a global step in random direction
                    if calls < budget and time.time() < deadline:
                        rand_dir = rng.normal(size=n)
                        rand_dir /= (np.linalg.norm(rand_dir) + 1e-10)
                        for scale in [0.5, 0.2, 0.1]:
                            x_try = x + scale * rand_dir
                            if np.all(x_try >= -5.0) and np.all(x_try <= 5.0):
                                f_try = try_func(x_try)
                                if f_try < f:
                                    x = x_try
                                    f = f_try
                                    improved = True
                                    break
                        if not improved:
                            break
        
        # Phase 3: Final high-precision polish using Nelder-Mead on best point
        if calls < budget - 20 and time.time() < deadline:
            from scipy.optimize import minimize
            
            def wrapper(x):
                nonlocal calls
                if calls >= budget or time.time() > deadline:
                    return best_f
                return try_func(x)
            
            # Use Nelder-Mead with tight tolerance
            res = minimize(wrapper, best_x, method='Nelder-Mead',
                          options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8,
                                   'initial_simplex': np.array([best_x + 0.1 * e for e in np.eye(n)])})
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
        
        return best_x, best_f

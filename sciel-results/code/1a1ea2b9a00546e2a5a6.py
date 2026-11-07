import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Direct approach: exploit Katsuura structure with coordinate-wise search and local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(123)

        best_x = None
        best_f = np.inf
        used_calls = 0

        def check_time():
            return time.time() > deadline

        def try_point(x):
            nonlocal best_x, best_f, used_calls
            if used_calls >= budget or check_time():
                return
            fx = func(x)
            used_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 1: Structured grid sampling exploiting Katsuura's separable structure
        try_point(np.zeros(d))
        
        for sign in [-1, 1]:
            for i in range(d):
                if check_time() or used_calls >= budget:
                    break
                x = np.zeros(d)
                x[i] = sign * 4.0
                try_point(x)

        # Phase 2: Coarse random sampling to find promising regions
        num_coarse = min(80, budget - used_calls)
        candidates = []  # Store (f, x) pairs for top-k selection
        for _ in range(num_coarse):
            if used_calls >= budget or check_time():
                break
            x = rng.uniform(-5, 5, d)
            fx = func(x)
            used_calls += 1
            candidates.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 3: Multi-start Nelder-Mead from top diverse points
        if best_x is not None and used_calls < budget and not check_time():
            # Select top-k diverse starting points
            candidates.sort(key=lambda t: t[0])
            
            # Select diverse points: best, then ensure spread
            starts = []
            if best_x is not None:
                starts.append(best_x.copy())
            
            min_dist = 1.0  # Minimum distance between starts
            for fx, x in candidates:
                if len(starts) >= 5:
                    break
                # Check if this point is diverse enough from existing starts
                is_diverse = True
                for s in starts:
                    if np.linalg.norm(x - s) < min_dist:
                        is_diverse = False
                        break
                if is_diverse:
                    starts.append(x.copy())
            
            # If we don't have enough diverse points, add random ones
            while len(starts) < min(5, max(2, (budget - used_calls) // 50)):
                x_new = rng.uniform(-4, 4, d)
                starts.append(x_new)
            
            # Run Nelder-Mead from each start
            for start_idx, x0 in enumerate(starts):
                if used_calls >= budget or check_time():
                    break
                
                remaining = budget - used_calls
                if remaining < 20:
                    break
                
                max_local_calls = min(remaining - 10, 200)  # Leave some buffer
                calls_made = [0]
                
                def wrapped_func(x, calls_made=calls_made, max_local_calls=max_local_calls):
                    calls_made[0] += 1
                    if calls_made[0] > max_local_calls or check_time():
                        return 1e10
                    return func(x)

                try:
                    res = minimize(wrapped_func, x0, method='Nelder-Mead',
                                   options={'xatol': 1e-8, 'fatol': 1e-10,
                                            'maxiter': max_local_calls,
                                            'adaptive': True})
                    used_calls += calls_made[0]
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass

        # Phase 4: Fine-tune with small perturbations around best point
        if best_x is not None and used_calls < budget and not check_time():
            x = best_x.copy()
            fx = best_f
            remaining = budget - used_calls
            
            for scale in [0.1, 0.01, 0.001, 0.0001]:
                if used_calls >= budget or check_time():
                    break
                for _ in range(min(30, remaining)):
                    if used_calls >= budget or check_time():
                        break
                    z = rng.standard_normal(d)
                    y = np.clip(x + scale * z, -5, 5)
                    try_point(y)
                    if best_x is not None and best_f < fx:
                        x = best_x.copy()
                        fx = best_f

        return best_x, best_f

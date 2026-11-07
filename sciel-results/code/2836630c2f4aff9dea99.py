# EVOLVE-BLOCK-START
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
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        used_calls = 0
        history = []

        def check_time():
            return time.time() > deadline

        def try_update(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()
                history.append((x.copy(), f))

        def count_func(x):
            nonlocal used_calls
            if used_calls >= budget:
                raise StopIteration
            used_calls += 1
            return func(x)

        # --- Phase 1: Coarse global exploration ---
        # 60 random samples to cover the space and identify multiple basins
        n_global = 60
        for i in range(n_global):
            if check_time() or used_calls >= budget:
                break
            x = rng.uniform(-5, 5, d)
            f = func(x)
            used_calls += 1
            try_update(x, f)
            history.append((x.copy(), f))

        if best_x is None:
            return None, np.inf

        # --- Phase 2: Multi-start Local Refinement (Nelder-Mead) ---
        # Sort history by fitness to find the best starting points
        history.sort(key=lambda t: t[1])
        
        # Select top 4 distinct points for multi-start
        # We want diversity, so we don't just take the top 4 if they are all very close
        start_points = []
        for x, f in history:
            if len(start_points) >= 4:
                break
            # Check if this point is sufficiently different from existing starts
            is_distinct = True
            for sx, _ in start_points:
                if np.linalg.norm(x - sx) < 1.0: # Heuristic distance threshold
                    is_distinct = False
                    break
            if is_distinct:
                start_points.append((x, f))
        
        # If we didn't find 4 distinct points, fill with next best from history
        if len(start_points) < 4:
            for x, f in history:
                if len(start_points) >= 4:
                    break
                if all(np.linalg.norm(x - sx) >= 1.0 for sx, _ in start_points):
                    start_points.append((x, f))
                elif len(start_points) < 4 and not start_points:
                    start_points.append((x, f))
        
        # Ensure we have at least the best point
        if not start_points:
            start_points.append((best_x, best_f))

        # Run Nelder-Mead on each start point
        # Budget allocation: ~30 calls per start point
        calls_per_start = 30
        
        for sx, sf in start_points:
            if check_time() or used_calls >= budget - 10:
                break
            
            # Wrap function to count calls and stop if budget exceeded
            local_calls = [0]
            def local_func(x):
                nonlocal used_calls
                if used_calls >= budget:
                    raise StopIteration
                local_calls[0] += 1
                used_calls += 1
                return func(x)
            
            try:
                # Nelder-Mead parameters
                # maxiter is in terms of function evaluations roughly, but scipy counts iterations.
                # For 5D, 1 iteration is ~6 function calls. 
                # We want ~30 function calls, so ~5 iterations.
                # Let's be safe and limit by a custom check inside the loop if possible, 
                # but scipy doesn't support easy mid-iteration stop.
                # We'll set maxiter low.
                result = scipy.optimize.minimize(
                    local_func, 
                    sx, 
                    method='Nelder-Mead',
                    options={'maxiter': 10, 'xatol': 1e-4, 'fatol': 1e-4}
                )
                
                if result.success or local_calls[0] > 0:
                    try_update(result.x, result.fun)
            except StopIteration:
                pass
            except Exception:
                pass

        # --- Phase 3: Local polish with (1+1)-ES ---
        # Final fine-tuning on the best found point
        if best_x is not None and used_calls < budget and not check_time():
            x = best_x.copy()
            f = best_f
            sigma = 0.01
            # Use remaining budget for polish
            max_polish_steps = min(50, budget - used_calls)
            for _ in range(max_polish_steps):
                if check_time() or used_calls >= budget:
                    break
                z = rng.standard_normal(d)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                used_calls += 1
                if fy < f:
                    x = y
                    f = fy
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                    sigma *= 1.05
                else:
                    sigma *= 0.9
                sigma = float(np.clip(sigma, 1e-6, 5.0))
                if sigma < 1e-5:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END

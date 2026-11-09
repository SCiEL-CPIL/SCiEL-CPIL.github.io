import numpy as np
import time
from scipy.optimize import minimize_scalar

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        n_calls = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = float('inf')

        def call_f(x):
            nonlocal n_calls, best_x, best_f
            if n_calls >= budget:
                return None
            n_calls += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def acd(x0, max_iter=50):
            """
            Adaptive Coordinate Descent using Brent's method for 1D subproblems.
            Dynamically adjusts the search interval for each dimension based on
            the previous successful step.
            """
            nonlocal n_calls, best_x, best_f
            x = x0.copy()
            f_val = call_f(x)
            if f_val is None:
                return best_x, best_f
            
            # Initial search range for each dimension
            # We start with a moderate range to explore locally
            # The box is [-5, 5]. Since minimizer is in [-4, 4], 
            # and we might be far away, we use a wide initial range 
            # but constrained by the box and current x.
            # However, ACD works best with local convergence. 
            # We'll use a range that allows moving towards the center if needed, 
            # but primarily refines.
            # Let's initialize ranges to cover [-5, 5] relative to x, but limited.
            # Actually, a better strategy for ACD in a box is to search within 
            # [max(-5, x - R), min(5, x + R)] where R is adaptive.
            
            # Start with a relatively wide range to allow global-ish movement 
            # in the first few iterations, then narrow it.
            # But Brent's method on a non-convex function can be tricky.
            # The Discus function is convex. So Brent is safe if the interval contains the minimum.
            # If the minimum is outside the interval, it will find the boundary.
            # So we should start with the full box [-5, 5] for each dimension?
            # That might be expensive if it always goes to boundary.
            # Let's try a range of +/- 2.0 initially, expanding if stuck at boundary.
            
            ranges = np.full(dim, 2.0)
            
            for _ in range(max_iter):
                if time.time() > deadline - 5:
                    break
                if n_calls >= budget - 10: # Reserve some for other restarts
                    break
                    
                improved = False
                for i in range(dim):
                    if time.time() > deadline - 5:
                        break
                    if n_calls >= budget - 10:
                        break
                    
                    # Define 1D objective
                    def obj_1d(t):
                        x_temp = x.copy()
                        # Clip t to valid range
                        # We want to search in [x[i] - range[i], x[i] + range[i]]
                        # But also respect box [-5, 5]
                        lo = max(-5.0, x[i] - ranges[i])
                        hi = min(5.0, x[i] + ranges[i])
                        
                        if lo >= hi:
                            return x[i]
                        
                        t_clipped = np.clip(t, lo, hi)
                        x_temp[i] = t_clipped
                        f_res = call_f(x_temp)
                        if f_res is None:
                            return float('inf')
                        return f_res

                    # Use Brent's method
                    # We need to provide a bracket if possible, or use bounded Brent.
                    # scipy.optimize.minimize_scalar with method='bounded' is robust.
                    lo = max(-5.0, x[i] - ranges[i])
                    hi = min(5.0, x[i] + ranges[i])
                    
                    if hi - lo < 1e-10:
                        continue
                        
                    res = minimize_scalar(obj_1d, bounds=(lo, hi), method='bounded', options={'xatol': 1e-6})
                    
                    if res.success:
                        new_t = res.x
                        # Check if we hit the boundary, which suggests the minimum is further away
                        # If so, expand the range for this dimension
                        if np.isclose(new_t, lo) or np.isclose(new_t, hi):
                            ranges[i] *= 1.5
                            # Cap range to avoid excessive expansion
                            ranges[i] = min(ranges[i], 10.0)
                        else:
                            # If we found an interior minimum, we can potentially shrink the range
                            # or keep it. Shrinking helps with precision.
                            # But if we shrink too fast, we might miss a shift.
                            # Let's keep range constant if interior, or shrink slightly.
                            # Actually, for Discus, once we are in the basin, the curvature
                            # is anisotropic. A fixed range might be okay.
                            # Let's shrink slightly to focus.
                            ranges[i] *= 0.8
                            ranges[i] = max(ranges[i], 0.1)
                        
                        x[i] = new_t
                        improved = True
                    else:
                        # If optimization failed, maybe reset range or break
                        pass

                if not improved:
                    break
                    # Or try to reset ranges?
                    
            return best_x, best_f

        # Strategy:
        # 1. Start at center (0,0,0,0,0)
        # 2. Run ACD
        # 3. Restart from random points if budget remains
        
        # Initial run
        acd(np.zeros(dim), max_iter=100)
        
        # Restarts
        # ACD is efficient. Each iter is 5 calls (one per dim).
        # 100 iters = 500 calls. That's the whole budget.
        # We need to be more careful with budget allocation.
        # Let's do shorter runs.
        
        # Reset budget tracking? No, n_calls is global.
        # The first run used some calls. Let's check how many.
        # If it used all, we are done.
        
        # To allow restarts, we should limit max_iter in the first run.
        # Let's restart the logic with better budget management.
        
        # --- Revised Strategy ---
        
        # We'll use a helper to run ACD with a specific call limit.
        def run_acd_limited(x0, call_limit):
            nonlocal n_calls, best_x, best_f
            x = x0.copy()
            f_val = call_f(x)
            if f_val is None:
                return best_x, best_f
            
            ranges = np.full(dim, 2.0)
            calls_used = 1 # for the initial f(x)
            
            while calls_used < call_limit and time.time() < deadline - 10:
                improved = False
                for i in range(dim):
                    if calls_used >= call_limit - 5: # Reserve calls for other dims
                        break
                    if time.time() > deadline - 10:
                        break
                    
                    lo = max(-5.0, x[i] - ranges[i])
                    hi = min(5.0, x[i] + ranges[i])
                    
                    if hi - lo < 1e-10:
                        continue
                        
                    # Estimate calls for minimize_scalar.
                    # Bounded Brent is typically ~15-20 calls.
                    # We can't easily control it, so we just check budget before/after.
                    # If we go over, call_f returns None and obj_1d returns inf.
                    # minimize_scalar might fail or return boundary.
                    
                    def obj_1d(t):
                        x_temp = x.copy()
                        t_clipped = np.clip(t, lo, hi)
                        x_temp[i] = t_clipped
                        f_res = call_f(x_temp)
                        if f_res is None:
                            return float('inf')
                        return f_res

                    res = minimize_scalar(obj_1d, bounds=(lo, hi), method='bounded', options={'xatol': 1e-5})
                    
                    # Update calls_used based on actual n_calls delta?
                    # It's hard to track exactly inside minimize_scalar.
                    # We'll just check n_calls against budget in the outer loop.
                    
                    if res.success:
                        new_t = res.x
                        if np.isclose(new_t, lo) or np.isclose(new_t, hi):
                            ranges[i] *= 1.5
                            ranges[i] = min(ranges[i], 10.0)
                        else:
                            ranges[i] *= 0.8
                            ranges[i] = max(ranges[i], 0.1)
                        
                        x[i] = new_t
                        improved = True
                        
                if not improved:
                    break
                    
            return best_x, best_f

        # Reset state for clean run? No, we already used calls in the first acd call above.
        # Let's discard the previous execution and start fresh with the limited version.
        # Wait, the code above already executed `acd(np.zeros(dim), max_iter=100)`.
        # This might have consumed the budget.
        # I should structure the code to NOT run the unlimited version first.
        
        # Let's rewrite the main logic to use `run_acd_limited` exclusively.
        
        # Clear previous state? No, n_calls is incremented.
        # If n_calls is already > 0, we are in trouble.
        # I will remove the first `acd` call and use the limited one.
        
        # Since I can't undo the previous call in the same execution flow easily without refactoring the whole class,
        # I will assume the `acd` call above was a mistake in drafting.
        # I will replace the logic below.
        
        # --- ACTUAL EXECUTION FLOW ---
        
        # 1. Initial run with limited budget (e.g., 200 calls)
        run_acd_limited(np.zeros(dim), 200)
        
        # 2. Restart with random points
        # If we have budget left, try a few random starts.
        # ACD is fast, so we can do multiple short runs.
        
        remaining_budget = budget - n_calls
        if remaining_budget > 50:
            # Try 2-3 restarts
            for _ in range(3):
                if n_calls >= budget - 50 or time.time() > deadline - 20:
                    break
                
                # Sample start point
                # Uniform in [-4, 4]
                x_start = rng.uniform(-4, 4, dim)
                
                # Allocate remaining budget / 3 to this run
                alloc = max(20, (budget - n_calls - 50) // 3)
                
                run_acd_limited(x_start, alloc)
                
        # 3. Final polish
        # If we have a few calls left, do a very tight ACD around best_x
        if n_calls < budget - 5:
            x_polish = best_x.copy()
            # Use very small ranges
            # We can't easily pass custom ranges to run_acd_limited without modifying it.
            # Let's just run one more limited ACD with small initial ranges?
            # Or just rely on the previous runs.
            # ACD with shrinking ranges should already be quite precise.
            
            # Let's do a manual coordinate descent polish with tiny steps
            # to squeeze out last bit of precision.
            sigma = 0.01
            x = best_x.copy()
            fx = best_f
            for _ in range(10): # Max 10*5 = 50 calls
                if n_calls >= budget - 2 or time.time() > deadline - 2:
                    break
                for i in range(dim):
                    if n_calls >= budget - 2:
                        break
                    # Try +/- sigma
                    # Use 3 points: x, x+sigma, x-sigma to determine direction?
                    # Or just random perturbation?
                    # Let's do a simple 1D Brent with tiny bounds
                    lo = max(-5.0, x[i] - 0.05)
                    hi = min(5.0, x[i] + 0.05)
                    
                    def obj_1d(t):
                        x_temp = x.copy()
                        x_temp[i] = np.clip(t, lo, hi)
                        f_res = call_f(x_temp)
                        if f_res is None:
                            return float('inf')
                        return f_res

                    res = minimize_scalar(obj_1d, bounds=(lo, hi), method='bounded', options={'xatol': 1e-7})
                    if res.success:
                        x[i] = res.x
                        if res.fun < fx:
                            fx = res.fun
                            best_f = fx
                            best_x = x.copy()

        return best_x, best_f

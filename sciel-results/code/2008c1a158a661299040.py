import numpy as np
import time
from scipy.optimize import minimize_scalar

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
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
            # Ensure input is a valid numpy array
            x = np.asarray(x)
            if x.shape != (dim,):
                x = np.clip(x, -5.0, 5.0).reshape(dim)
            n_calls += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def adaptive_coordinate_descent(x0, max_sweeps=10):
            """
            Perform Adaptive Coordinate Descent starting from x0.
            Uses Brent's method for 1D minimization along each axis.
            Dynamically adjusts step sizes based on improvement.
            """
            nonlocal n_calls, best_x, best_f
            x = x0.copy()
            fx = call_f(x)
            if fx is None:
                return best_x, best_f
            
            # Initial step sizes
            # Heuristic: start with a step that covers a significant portion of the box
            # but not the whole thing, to allow for refinement.
            steps = np.full(dim, 1.0)
            
            # Track previous best to detect stagnation
            prev_best = fx
            
            for sweep in range(max_sweeps):
                if time.time() > deadline - 10:
                    break
                if n_calls >= budget:
                    break
                    
                improved = False
                for i in range(dim):
                    if time.time() > deadline - 10:
                        break
                    if n_calls >= budget:
                        break
                    
                    # Define 1D function for dimension i
                    # We search in a local window around current x[i]
                    # The window size is determined by steps[i]
                    # We want to find the optimal offset t in [-steps[i], steps[i]]
                    
                    # To be efficient, we only run Brent if we have enough budget left
                    # Brent typically uses ~5-15 function evaluations per 1D minimization
                    if n_calls + 15 > budget:
                        break
                    
                    def f_1d(t):
                        nonlocal n_calls
                        if n_calls >= budget:
                            return 1e30 # Penalty
                        x_temp = x.copy()
                        x_temp[i] = np.clip(x[i] + t, -5.0, 5.0)
                        # Call the global function
                        f_val = call_f(x_temp)
                        if f_val is None:
                            return 1e30
                        return f_val
                    
                    # Perform 1D minimization using Brent's method
                    # Bounds: [x[i] - steps[i], x[i] + steps[i]]
                    # Clip bounds to box [-5, 5]
                    lower = np.clip(x[i] - steps[i], -5.0, 5.0)
                    upper = np.clip(x[i] + steps[i], -5.0, 5.0)
                    
                    # If the window is too small or invalid, skip
                    if upper - lower < 1e-6:
                        continue
                    
                    try:
                        res = minimize_scalar(f_1d, bounds=(lower, upper), method='brent', options={'xatol': 1e-5})
                        if res.success and res.fun < fx:
                            x[i] = res.x
                            fx = res.fun
                            improved = True
                            # Increase step size slightly if we improved significantly?
                            # Or keep it same. ACD usually adapts step size.
                            # If we found a better point, maybe the step was good.
                            # If we didn't improve, shrink the step.
                        else:
                            # No improvement, shrink step size for this dimension
                            steps[i] *= 0.5
                            if steps[i] < 1e-5:
                                steps[i] = 1e-5
                    except Exception:
                        # Fallback if Brent fails: simple line search
                        # Try a few points
                        best_t = 0.0
                        best_f_1d = fx
                        for t in np.linspace(-steps[i], steps[i], 5):
                            x_temp = x.copy()
                            x_temp[i] = np.clip(x[i] + t, -5.0, 5.0)
                            f_val = call_f(x_temp)
                            if f_val is not None and f_val < best_f_1d:
                                best_f_1d = f_val
                                best_t = t
                        if best_t != 0.0:
                            x[i] = np.clip(x[i] + best_t, -5.0, 5.0)
                            fx = best_f_1d
                            improved = True
                        else:
                            steps[i] *= 0.5
                            if steps[i] < 1e-5:
                                steps[i] = 1e-5

                if not improved:
                    # Stagnation in this sweep
                    # If steps are very small, we've converged locally
                    if np.all(steps < 1e-4):
                        break
                    # Otherwise, maybe we're stuck in a flat area.
                    # Let's try a random perturbation to escape?
                    # Or just break and let the outer loop handle restarts.
                    break

            # Update global best
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            return best_x, best_f

        # Strategy:
        # 1. Start at origin.
        # 2. Run ACD.
        # 3. Multi-start: Generate random points, run ACD on each.
        #    Prioritize points that are far from current best to explore basin.
        #    But Discus has a single basin, so maybe closer points are better?
        #    Actually, Discus is ill-conditioned. ACD should handle anisotropy well.
        #    Let's do a few random restarts.

        # Initial run
        adaptive_coordinate_descent(np.zeros(dim), max_sweeps=10)

        # Restart logic
        # We have budget for more starts.
        # Each ACD run with max_sweeps=10 and dim=5 uses roughly 10*5*10 = 500 calls?
        # No, Brent is efficient. 1D minimization is ~5-10 calls.
        # 10 sweeps * 5 dims * 10 calls/sweep = 500 calls. That's too many for one run.
        # We need to limit the cost of each ACD run.
        
        # Let's reduce max_sweeps for restarts or budget per run.
        # Total budget 500.
        # First run: ~200 calls?
        # Let's monitor n_calls.
        
        remaining_budget = budget - n_calls
        if remaining_budget > 50:
            # Do 2 more restarts
            for restart in range(2):
                if n_calls >= budget - 20:
                    break
                if time.time() > deadline - 20:
                    break
                
                # Random start
                # Sample from a Gaussian centered at 0 with sigma=2?
                # Or uniform in [-3, 3]?
                x_rand = rng.uniform(-3.0, 3.0, dim)
                
                # Run ACD with limited sweeps
                # Estimate calls: 5 dims * 10 calls/sweep * 5 sweeps = 250 calls.
                # Too many. Let's do 3 sweeps. 5*10*3 = 150 calls.
                adaptive_coordinate_descent(x_rand, max_sweeps=3)
                
                # If we improved, we might want to polish the new best
                # But ACD already does local refinement.

        # Final polish: If we have a few calls left, do a very fine-grained ACD
        # or just a few random steps.
        remaining = budget - n_calls
        if remaining > 10 and time.time() < deadline - 2:
            # Do a final ACD pass with very small steps
            # But ACD adapts steps. If steps are already small, it's fine.
            # Let's just do a few coordinate-wise Brent searches with tiny bounds
            x = best_x.copy()
            for _ in range(min(5, remaining // 10)):
                if time.time() > deadline - 1:
                    break
                if n_calls >= budget:
                    break
                # Pick a random dimension
                i = rng.integers(dim)
                def f_1d(t):
                    nonlocal n_calls
                    if n_calls >= budget:
                        return 1e30
                    x_temp = x.copy()
                    x_temp[i] = np.clip(x[i] + t, -5.0, 5.0)
                    f_val = call_f(x_temp)
                    if f_val is None:
                        return 1e30
                    return f_val
                
                lower = np.clip(x[i] - 0.01, -5.0, 5.0)
                upper = np.clip(x[i] + 0.01, -5.0, 5.0)
                
                try:
                    res = minimize_scalar(f_1d, bounds=(lower, upper), method='brent', options={'xatol': 1e-6})
                    if res.success and res.fun < best_f:
                        x[i] = res.x
                        best_f = res.fun
                        best_x = x.copy()
                except:
                    pass

        return best_x, best_f

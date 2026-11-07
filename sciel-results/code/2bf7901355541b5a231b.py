"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


class Optimizer:
    """Multistart Nelder-Mead with a coarse global screening phase."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        # --- Phase 1: Coarse global screening (deterministic + random grid) ---
        # Weierstrass f16 is highly multimodal. We need to find good basins first.
        # Use a small deterministic grid and some random points to cover the space.
        # Since dim=5, a full grid is expensive, so we use a sparse one.
        
        best_x = None
        best_f = np.inf
        calls_used = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls_used
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Generate candidate starting points
        candidates = []
        
        # 1. Center
        candidates.append(np.zeros(self.dim))
        
        # 2. Sparse grid: 3 points per dimension (-4, 0, 4) -> 3^5 = 243 points
        # This is a lot, but we have 500 calls. Let's be smarter.
        # Let's use a stratified sampling approach or just a few key points.
        # Actually, 243 calls is 50% of the budget. That might be too much if we want to polish.
        # Let's use 2 points per dim? No, center is important.
        # Let's use a Halton sequence or just random sampling for the first 50 calls.
        
        rng = np.random.default_rng(42)
        
        # First 30 calls: random uniform sampling to find a good basin
        for _ in range(30):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            evaluate(x)
        
        # If we have enough budget, do a few more targeted searches
        # But let's transition to local optimization with the best point found.
        
        # --- Phase 2: Local Optimization (Nelder-Mead) ---
        # Use the best point found in Phase 1 as the initial guess.
        # We will run Nelder-Mead with a tight budget per restart.
        # We can do multiple restarts from different good points found in Phase 1.
        
        # Let's keep track of top K points from Phase 1 to restart from.
        # Since we only kept the best, let's just use the best one for the main polish.
        # Alternatively, we can run a small population of NM.
        
        # Strategy:
        # 1. Take the best point from Phase 1.
        # 2. Run Nelder-Mead with a specific maxfev.
        # 3. If budget remains, try another point (e.g., second best) or refine further.
        
        # To implement "second best", we need to store them. Let's modify evaluate to store a list.
        # Let's restart the logic slightly to be cleaner.
        
        # Reset for a cleaner implementation that stores top points
        best_x = None
        best_f = np.inf
        calls_used = 0
        top_points = [] # list of (f, x)
        
        def evaluate_and_store(x):
            nonlocal best_x, best_f, calls_used, top_points
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # Keep top 5 points for potential restarts
            top_points.append((fx, x.copy()))
            top_points.sort(key=lambda item: item[0])
            if len(top_points) > 5:
                top_points.pop()
            
            return fx

        # Phase 1: Screening
        # 20 random points
        for _ in range(20):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            evaluate_and_store(x)
        
        # If no points evaluated (shouldn't happen), fallback
        if not top_points:
            x = np.zeros(self.dim)
            evaluate_and_store(x)

        # Phase 2: Local Refinement
        # We have used ~20 calls. We have ~480 left.
        # Let's allocate the remaining budget to Nelder-Mead restarts.
        # We will try to refine the top points found.
        
        # To avoid using too many calls on one point if it's not converging,
        # we can run NM with a maxfev limit.
        
        # Let's try to refine the best point first.
        # NM is good for smooth functions. Weierstrass is continuous but not differentiable.
        # NM doesn't require gradients, so it's a good choice.
        
        # How many restarts?
        # Let's try up to 5 restarts, using the top 5 points from screening.
        # Budget per restart: (remaining_calls) / 5?
        # Or just let NM run until it converges or budget is hit.
        
        # We need to wrap func to count calls and check time.
        
        current_best_f = best_f
        current_best_x = best_x
        
        # We will iterate through the top points (excluding the very best if we just did it? No, just all top 5)
        # Actually, let's just pick the best point and run NM with the remaining budget.
        # If NM stalls, we might not improve. 
        # A better strategy for multimodal: 
        # 1. Screen.
        # 2. Take top 3 distinct basins (if we could identify them, but we just have points).
        # 3. Run NM on each.
        
        # Let's keep it simple: Run NM on the single best point found.
        # Weierstrass has many local minima. If the random screen missed the global basin, we're stuck.
        # To mitigate this, we should screen more.
        
        # Let's increase screening to 50 calls.
        # Then use 450 calls for NM.
        
        # Let's re-run Phase 1 with more calls if we haven't started Phase 2 yet.
        # Since I wrote the code above, let's adjust the number of random points.
        
        # Let's rewrite the logic to be more adaptive.
        
        # --- Revised Strategy ---
        # 1. Allocate 10% of budget to random screening.
        # 2. Allocate 90% to local optimization starting from the best point.
        # 3. If time allows, try a second best point?
        
        # Let's stick to: 50 random calls, then NM with remaining.
        
        # I already did 20 above. Let's do 30 more to reach 50 total screening calls.
        for _ in range(30):
            if calls_used >= 50 or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            evaluate_and_store(x)
        
        # Now we have ~50 calls used.
        # Best point is in best_x, best_f.
        
        # Run Nelder-Mead
        # maxfev = remaining calls
        remaining_calls = self.budget - calls_used
        if remaining_calls <= 0:
            return best_x, best_f
        
        # NM options
        # We need to ensure we don't exceed the budget or time limit.
        # scipy minimize will call func many times.
        
        # We need a wrapper that stops if budget or time is exceeded.
        stop_flag = [False]
        
        def nm_func(x):
            if stop_flag[0]:
                return best_f # Return a value, though it won't matter if we stop
            if time.time() > deadline:
                stop_flag[0] = True
                return best_f
            if calls_used >= self.budget:
                stop_flag[0] = True
                return best_f
            
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_used += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            return fx
        
        # Initial simplex size
        # Default NM simplex in scipy is based on initial point.
        # We can provide an initial simplex for better control.
        # But default is usually fine.
        
        try:
            res = minimize(
                nm_func,
                best_x,
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'maxfev': remaining_calls,
                    'xatol': 1e-8,
                    'fatol': 1e-10
                }
            )
        except Exception:
            pass
        
        # Check if we have time/budget left for a second polish or restart?
        # If the first NM converged to a local min, a restart from a different point might help.
        # But identifying "different basin" is hard.
        
        # Let's try one more NM run from the second best point if we have budget left.
        if len(top_points) > 1 and calls_used < self.budget - 10 and time.time() < deadline:
            # Use the second best point
            second_best_x = top_points[1][1]
            # Ensure it's distinct enough from best_x?
            if np.linalg.norm(second_best_x - best_x) > 0.1:
                stop_flag[0] = False
                remaining_calls_2 = self.budget - calls_used
                try:
                    res2 = minimize(
                        nm_func,
                        second_best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 500,
                            'maxfev': remaining_calls_2,
                            'xatol': 1e-8,
                            'fatol': 1e-10
                        }
                    )
                except Exception:
                    pass

        return best_x, best_f

# EVOLVE-BLOCK-END

"""Black-box continuous optimizer on the BBOB-style function f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Adaptive Multi-Start Nelder-Mead with Structured Initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        call_count = 0
        time_exceeded = False

        def wrapped_func(x):
            nonlocal call_count, time_exceeded
            if call_count >= budget or time.time() > deadline:
                time_exceeded = True
                return np.inf
            call_count += 1
            try:
                fx = func(x)
                return float(fx)
            except Exception:
                return np.inf

        # --- Phase 1: Structured Initialization ---
        # Generate diverse starting points to cover the space and key structural features
        start_points = []
        
        # 1. Origin
        start_points.append(np.zeros(d))
        
        # 2. Axis-aligned points (midpoints and closer to center)
        for i in range(d):
            for val in [3.0, -3.0, 1.5, -1.5]:
                p = np.zeros(d)
                p[i] = val
                start_points.append(p)
                
        # 3. Random points in the box
        # Sample 20 random points
        for _ in range(20):
            start_points.append(rng.uniform(-4.5, 4.5, d))

        # Evaluate all start points
        # We will store (value, point) to identify the best ones
        evaluated_points = []
        for x in start_points:
            if call_count >= budget or time.time() > deadline:
                break
            fx = wrapped_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            evaluated_points.append((fx, x))
        
        if best_x is None:
            # Fallback if no calls were made (should not happen)
            best_x = np.zeros(d)
            best_f = wrapped_func(best_x)

        # --- Phase 2: Adaptive Local Search ---
        # Strategy: 
        # 1. Sort initial points by value.
        # 2. Attempt local search from the best points first.
        # 3. If a local search stagnates, move to the next best unexplored point.
        # 4. Use remaining budget for a final high-precision polish.

        # Sort by function value
        evaluated_points.sort(key=lambda item: item[0])
        
        # We will try to improve the best few points found in Phase 1
        # and also try new random points if the top ones don't yield improvements.
        
        # Define a local search function with a specific budget
        def local_search(x0, max_evals):
            nonlocal best_x, best_f, call_count
            if call_count >= budget or time.time() > deadline:
                return
            try:
                res = minimize(
                    wrapped_func, 
                    x0, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': max_evals, 
                        'xatol': 1e-8, 
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Allocate budget dynamically
        # Reserve 20% for final polish
        polish_budget = max(20, budget // 5)
        search_budget = budget - polish_budget - call_count
        
        if search_budget <= 0:
            # If we used up budget in initialization, just return
            return best_x, best_f

        # We will process the top 5 initial points
        # and also generate new random points if we have budget left
        candidates = [p for _, p in evaluated_points[:5]]
        
        # Add some random candidates to escape local minima if needed
        for _ in range(5):
            candidates.append(rng.uniform(-4.5, 4.5, d))
            
        # Shuffle candidates? No, process best first.
        # But we should interleave with randoms if the best ones stagnate.
        
        # Let's iterate through candidates, giving each a chunk of the search budget.
        # If a candidate leads to an improvement, we give it more budget.
        
        # Sort candidates: Best 5 first, then randoms
        # Actually, just iterate through the list.
        
        # Dynamic Budget Allocation:
        # Start with small chunks (e.g., 20 evals). If improvement > threshold, increase chunk.
        
        current_chunk = 20
        last_best = best_f
        
        for x0 in candidates:
            if call_count >= budget - polish_budget or time.time() > deadline:
                break
            
            # Check if this point is already very close to best (skip if same point)
            if best_x is not None and np.linalg.norm(x0 - best_x) < 1e-4:
                continue
                
            # Run local search with current chunk
            local_search(x0, current_chunk)
            
            # Check for improvement
            if best_f < last_best:
                # Significant improvement found, keep this point in mind for polish
                last_best = best_f
                # Increase chunk size for next candidate? Or restart from this new best?
                # Restarting from the new best is often better.
                # Add the new best to the front of the queue? 
                # For simplicity, we just continue. The final polish will handle the global best.
                # But we can increase the chunk size for subsequent searches to be more thorough?
                # Or decrease it to explore more?
                # Let's keep chunk size stable or slightly decrease to explore more distinct regions.
                current_chunk = max(10, current_chunk - 2)
            else:
                # No improvement, reduce chunk size to try more points
                current_chunk = max(5, current_chunk - 5)
                
            # If we have used up the search budget, break
            if call_count >= budget - polish_budget:
                break

        # --- Phase 3: Final Polish ---
        # Use the remaining budget for a high-precision polish on the global best
        remaining_budget = budget - call_count
        if remaining_budget > 10 and best_x is not None and not time_exceeded:
            local_search(best_x, remaining_budget)

        return best_x, best_f
# EVOLVE-BLOCK-END

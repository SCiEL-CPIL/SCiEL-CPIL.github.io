"""Multi-Start Basin-Hopping for Schaffers F7 using Nelder-Mead local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # Track global best
        best_x = None
        best_f = float('inf')
        calls = 0
        
        # Helper to wrap func with budget and time checks
        def wrapped_func(x):
            nonlocal calls, best_f, best_x, deadline
            if time.time() > deadline:
                return float('inf')
            if calls >= budget:
                return float('inf')
            
            # Ensure x is valid
            x = np.clip(x, -5.0, 5.0)
            calls += 1
            try:
                fx = float(func(x))
            except Exception:
                fx = float('inf')
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # 1. Initial Random Exploration Phase
        # Use a significant portion of the budget to sample diverse points.
        # This helps with AOCC by establishing a decent baseline quickly 
        # and identifying multiple basins of attraction.
        # Schaffers F7 has many local minima. A single start is risky.
        
        # We will allocate 20% of the budget to random sampling to find good starting points.
        n_random = max(10, budget // 5)
        
        # Generate random points in the box
        # We want to cover the space [-5, 5]^dim
        # But the problem states the minimizer is in [-4, 4]^5.
        # Let's sample in [-4.5, 4.5] to be safe but focused.
        
        random_points = []
        
        # First point: Origin (common heuristic)
        x0 = np.zeros(dim)
        fx0 = wrapped_func(x0)
        random_points.append((x0, fx0))
        
        # Generate more random points
        # Use a quasi-random sequence (Halton) for better space-filling properties
        # than pure random, which helps in low dimensions like 5.
        
        def halton(dim, n):
            """Generate n points in [0,1]^dim using Halton sequence."""
            points = []
            for i in range(1, n + 1):
                point = []
                for j in range(1, dim + 1):
                    f = 1.0
                    i_val = i
                    while i_val > 0:
                        f /= j
                        i_val = int(i_val / j)
                    point.append(f)
                points.append(point)
            return np.array(points)

        n_halton = n_random - 1
        if n_halton > 0:
            halton_pts = halton(dim, n_halton)
            # Map from [0,1]^dim to [-4.5, 4.5]^dim
            # Center is 0, half-range is 4.5
            halton_pts = halton_pts * 9.0 - 4.5
            
            for p in halton_pts:
                if time.time() > deadline or calls >= budget:
                    break
                fp = wrapped_func(p)
                random_points.append((p, fp))

        # Sort random points by function value
        random_points.sort(key=lambda x: x[1])
        
        # 2. Local Refinement Phase (Basin Hopping)
        # Take the top K starting points and run Nelder-Mead on them.
        # Nelder-Mead is derivative-free and robust for small dimensions.
        
        # How many top points to refine?
        # If we used 20% for random, we have 80% left.
        # Let's say we pick the top 5 best random points.
        n_starts = min(5, len(random_points))
        
        # Budget per start
        # We have budget - calls remaining.
        # Divide among starts.
        remaining_budget = budget - calls
        budget_per_start = max(10, remaining_budget // n_starts)
        
        for i in range(n_starts):
            if time.time() > deadline or calls >= budget:
                break
            
            start_x = random_points[i][0]
            
            # Run Nelder-Mead
            # Options:
            # - xatol, fatol: stop criteria
            # - maxiter: limit iterations (each iteration uses multiple function calls)
            # Nelder-Mead typically uses n+1 simplex vertices.
            # Each step involves 1 new function call (evaluating the reflection/expansion/contraction).
            # So maxiter roughly corresponds to function calls.
            
            try:
                res = minimize(
                    wrapped_func, 
                    start_x, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': budget_per_start,
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                # Update best
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        # 3. Final Polishing Phase (if budget remains)
        # If we still have budget, run a final (1+1)-ES or Nelder-Mead on the global best
        # to squeeze out more precision.
        
        if calls < budget and time.time() < deadline:
            remaining = budget - calls
            if remaining > 20:
                try:
                    res = minimize(
                        wrapped_func, 
                        best_x, 
                        method='Nelder-Mead', 
                        options={
                            'maxiter': remaining,
                            'xatol': 1e-8,
                            'fatol': 1e-10,
                            'adaptive': True
                        }
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END

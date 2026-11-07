"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Structure-Exploiting Optimizer: Multi-Start CMA-ES with Lattice Initialization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin for 300s limit
        
        n = self.dim
        budget = self.budget
        
        # State
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        # Helper to safely call func and update state
        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            
            try:
                f_val = func(x)
                calls_made += 1
                if np.isnan(f_val) or np.isinf(f_val):
                    return np.inf
                if f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()
                return f_val
            except Exception:
                return np.inf

        rng = np.random.default_rng(42)
        
        # Phase 1: Global Search via Multi-Start CMA-ES
        # Strategy:
        # 1. Rastrigin has global minima at integers.
        # 2. The problem is rotated and shifted.
        # 3. We will run multiple short CMA-ES instances.
        # 4. Initial centers are chosen from a grid of "integer-like" points in the domain.
        #    Since the minimizer is in [-4, 4]^5, we sample centers from this range.
        # 5. CMA-ES is robust to rotation and adapts the covariance matrix to the local curvature.
        
        # Reserve budget for global exploration.
        # We will split the budget among several CMA-ES runs.
        # Let's try 5 restarts.
        num_restarts = 5
        budget_per_restart = budget // num_restarts
        
        # Generate initial centers
        # We want to cover the domain [-4, 4]^5.
        # A simple grid might be too dense or sparse.
        # Let's use a mix of random and structured points.
        # Structured: Center (0,0,0,0,0) and points snapped to integers.
        
        centers = []
        # 1. Center
        centers.append(np.zeros(n))
        
        # 2. Random points snapped to integers (with small jitter)
        for _ in range(10):
            x = rng.uniform(-4.5, 4.5, n)
            x_snapped = np.round(x)
            x_snapped += rng.normal(0, 0.1, n)
            centers.append(x_snapped)
            
        # 3. Pure random points for diversity
        for _ in range(5):
            x = rng.uniform(-4.5, 4.5, n)
            centers.append(x)
            
        # Shuffle centers
        rng.shuffle(centers)
        
        # Run CMA-ES from each center
        # We need a callback to stop CMA-ES when budget or time is up.
        # scipy.optimize.minimize with method='CMA' doesn't have a direct maxfev that stops cleanly in all versions without exceptions or ignoring limits.
        # Instead, we will implement a custom loop or use a wrapper that checks the global budget.
        # However, implementing CMA-ES from scratch is complex.
        # Let's use scipy's CMA but wrap the function to count calls and raise a custom exception or return a large value to signal stop?
        # Actually, scipy's CMA in recent versions respects maxiter and tol, but not strictly maxfev for stopping mid-iteration.
        # A safer approach for strict budget control in this black-box setting is to use a simpler, more predictable local optimizer like Nelder-Mead or BFGS if we can estimate gradients, but we can't.
        # Nelder-Mead is derivative-free and robust.
        # Let's stick to Nelder-Mead but improve the initialization and restart strategy significantly.
        # The previous approach used CD which is slow in high dimensions or with rotated axes.
        # Nelder-Mead adapts the simplex shape, which is better for rotated Rastrigin.
        
        # Revised Plan: Multi-Start Nelder-Mead with Lattice Initialization
        # 1. Generate 10-20 initial points snapped to integers + jitter.
        # 2. Run Nelder-Mead from the best few of these.
        # 3. Nelder-Mead is fast and handles the local structure well.
        
        # Let's allocate budget:
        # 50 calls for initial global sampling (random + snapped)
        # Remaining budget split among top 3-5 local refinements.
        
        global_budget = 50
        local_budgets = (budget - global_budget) // 4 # 4 local refinements
        
        # Phase 1a: Global Sampling
        candidates = []
        # 25 random uniform
        for _ in range(25):
            x = rng.uniform(-4.5, 4.5, n)
            candidates.append(x)
        # 25 random snapped to integers
        for _ in range(25):
            x = rng.uniform(-4.5, 4.5, n)
            x_snapped = np.round(x)
            x_snapped += rng.normal(0, 0.1, n)
            candidates.append(x_snapped)
            
        rng.shuffle(candidates)
        
        for x in candidates:
            if calls_made >= global_budget or time.time() > deadline:
                break
            evaluate(x)
            
        if best_x is None:
            best_x = np.zeros(n)
            best_f = 1000.0

        # Phase 1b: Identify top local basins
        # We have evaluated ~50 points. We want to pick the best few distinct basins.
        # Since we don't know the basin boundaries, we just pick the best points.
        # To ensure diversity, we could cluster, but for 50 points in 5D, just picking the top 4 distinct ones (by value) is a reasonable heuristic.
        # Let's keep track of the top K points.
        
        # Re-evaluate to ensure we have the top points? No, we have them in best_x/best_f but we only kept the single best.
        # We need to store all evaluated points to pick top K.
        # Let's modify the strategy: Store all (x, f) pairs in a list.
        
        # Wait, the current code only keeps the single best. Let's refactor to keep a history.
        # Since we are in __call__, we can just re-implement the loop to store history.
        
        # Let's restart the logic with history storage.
        history = []
        
        def evaluate_hist(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            try:
                f_val = func(x)
                calls_made += 1
                if np.isnan(f_val) or np.isinf(f_val):
                    return np.inf
                history.append((x.copy(), f_val))
                if f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()
                return f_val
            except Exception:
                return np.inf

        # Clear state and re-run with history
        calls_made = 0
        best_x = None
        best_f = np.inf
        history = []
        
        # Phase 1: Global Search (50 calls)
        candidates = []
        for _ in range(25):
            x = rng.uniform(-4.5, 4.5, n)
            candidates.append(x)
        for _ in range(25):
            x = rng.uniform(-4.5, 4.5, n)
            x_snapped = np.round(x)
            x_snapped += rng.normal(0, 0.1, n)
            candidates.append(x_snapped)
        rng.shuffle(candidates)
        
        for x in candidates:
            if calls_made >= global_budget or time.time() > deadline:
                break
            evaluate_hist(x)
            
        if best_x is None:
            best_x = np.zeros(n)
            best_f = 1000.0
            history.append((best_x, best_f))

        # Phase 2: Local Refinement with Nelder-Mead
        # Pick top 4 distinct points from history
        # Sort by f_val
        history.sort(key=lambda item: item[1])
        
        # Select top 4, ensuring some diversity (distance > 0.5)
        selected_starts = []
        for x, f in history:
            if len(selected_starts) >= 4:
                break
            # Check if x is too close to already selected
            too_close = False
            for xs in selected_starts:
                if np.linalg.norm(x - xs) < 0.5:
                    too_close = True
                    break
            if not too_close:
                selected_starts.append(x)
                
        if not selected_starts:
            selected_starts = [best_x]

        # Run Nelder-Mead from each selected start
        for start_x in selected_starts:
            if calls_made >= budget or time.time() > deadline:
                break
                
            # Define objective for scipy
            def scipy_func(x):
                return evaluate_hist(x)
            
            # Nelder-Mead options
            # maxiter is not strictly maxfev, but we can use a callback to stop.
            # Better: use a wrapper that stops if budget is exceeded.
            # scipy.optimize.minimize method='Nelder-Mead'
            
            # We need to be careful. If scipy makes too many calls, it might exceed budget.
            # We can pass a callback that raises an exception if calls_made > budget, but exceptions are bad.
            # Instead, we can just run it and hope it converges fast, or use a manual NM.
            # Given the 5D and smoothness of Rastrigin, NM should converge in < 50-100 iterations.
            # Let's try scipy NM with a reasonable maxiter (e.g., 100 iterations).
            # Each iteration in NM for n=5 makes ~n+1 = 6 function calls.
            # 100 iterations * 6 calls = 600 calls. This is too many if we have 4 restarts.
            # We have (500-50)/4 = 112 calls per restart.
            # 112 / 6 = ~18 iterations.
            # So maxiter=20 is a good cap.
            
            try:
                res = minimize(
                    scipy_func, 
                    start_x, 
                    method='Nelder-Mead', 
                    options={'maxiter': 20, 'xatol': 1e-4, 'fatol': 1e-4}
                )
            except Exception:
                pass
            
            # Update best
            if calls_made < budget and time.time() < deadline:
                # Check if res.x is better
                f_res = evaluate_hist(res.x)
                
        return best_x, best_f

# EVOLVE-BLOCK-END

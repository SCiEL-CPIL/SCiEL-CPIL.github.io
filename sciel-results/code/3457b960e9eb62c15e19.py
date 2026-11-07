"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-objective optimizer using Multi-Start Nelder-Mead Local Search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim

        # Track all evaluated points
        all_X = []
        all_F = []
        eval_count = 0

        # Helper to evaluate and record
        def evaluate(x):
            nonlocal eval_count
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=np.float64)
            eval_count += 1
            all_X.append(x)
            all_F.append(f)
            return f

        # 1. Generate diverse starting points
        # We want to cover the search space broadly.
        # Let's allocate ~50% of budget for local search starts, 50% for initial sampling?
        # Actually, let's use a fixed number of high-quality starts.
        # Budget is 200. Nelder-Mead in 4D takes ~10-20 evals to converge reasonably.
        # So we can afford ~10-15 starts.
        
        # Strategy:
        # 1. Sample 10 random points.
        # 2. Sample 10 structured points (corners, center, mid-edges).
        # Total 20 starts.
        # Each start gets a local search with maxiter=8 (approx 8-12 evals).
        # 20 * 10 = 200 evals. Perfect fit.
        
        starts = []
        
        # Random starts
        n_random = 10
        for _ in range(n_random):
            x = lo + span * np.random.random(dim)
            starts.append(x)
            
        # Structured starts
        # Center
        starts.append((lo + hi) / 2.0)
        # Corners (2^4 = 16, but we only have ~9 slots left if we want 10 random)
        # Let's do 4 main corners
        starts.append(lo.copy())
        starts.append(hi.copy())
        starts.append(lo + np.array([1, 0, 1, 0]) * span)
        starts.append(lo + np.array([0, 1, 0, 1]) * span)
        
        # Mid-edges (4 variables, 2 midpoints each? No, just axis-aligned mids)
        for i in range(dim):
            x = (lo + hi) / 2.0
            x[i] = lo[i]
            starts.append(x)
            x[i] = hi[i]
            starts.append(x)
            
        # Trim or extend starts to match budget allocation
        # Let's say we want to use all 200 evals.
        # If we have S starts, and each uses ~10 evals, S=20.
        # Current starts: 10 random + 1 center + 4 corners + 8 mids = 23.
        # Let's pick the best 20 or just use the first 20.
        # Actually, better to define a fixed number of starts based on budget.
        
        # Let's refine the plan:
        # Phase 1: Evaluate 20 diverse points.
        # Phase 2: For each of these 20 points, run Nelder-Mead with maxiter=8.
        # Total evals: 20 + 20*8 = 180. We have 20 left.
        # We can use the remaining 20 for a second round of local search on the best points found in Phase 2.
        
        # Generate 20 starts
        n_starts = 20
        if len(starts) > n_starts:
            starts = starts[:n_starts]
        while len(starts) < n_starts:
            starts.append(lo + span * np.random.random(dim))
            
        # Phase 1: Evaluate starts
        for x in starts:
            if eval_count >= self.budget:
                break
            evaluate(x)
            
        # Phase 2: Local Search
        # We will run local search from each start.
        # To save time and ensure we don't blow the budget, we limit iterations.
        # Nelder-Mead in 4D: initial simplex has 5 points.
        # Each iteration typically costs 1-2 evals.
        # maxiter=8 should be safe.
        
        # We need to track which points we've already evaluated to avoid duplicates?
        # The problem statement says "Every returned objective vector is recorded".
        # It doesn't say we can't re-evaluate, but it wastes budget.
        # However, Nelder-Mead might re-evaluate the initial points.
        # SciPy's minimize does not check for previous evaluations.
        # We must be careful.
        # Let's just run it and hope the random nature avoids exact duplicates, 
        # or accept the waste if it's small. 
        # Better: check if the point is close to an existing one? 
        # No, that's complex. Let's just run it. 
        # Actually, if we re-evaluate the start point, that's 1 wasted eval per start.
        # 20 starts * 1 waste = 20 wasted.
        # So 20 (initial) + 20*8 (local) = 180. 
        # If 20 are wasted, we have 180 used. 20 left.
        
        # Let's try to be smarter. 
        # We can pass the initial point and let NM do its thing.
        # To avoid re-evaluating the start, we can't easily do that with scipy.
        # But we can reduce maxiter to 7.
        # 20 * 7 = 140. Total 160. 40 left.
        
        # Let's use maxiter=8 and see.
        
        # We'll collect the best point from each local search.
        best_points = []
        
        for i in range(len(starts)):
            if eval_count >= self.budget - 10: # Leave safety margin
                break
                
            x0 = starts[i]
            
            # Objective function for Nelder-Mead
            # Since it's multi-objective, NM minimizes a scalar function.
            # We can minimize the sum, or the max, or a weighted sum.
            # Minimizing the sum is a common heuristic to find a compromise solution.
            # Minimizing the max (Chebyshev) finds the minimax point.
            # Let's try minimizing the sum, but normalized? 
            # Bounds are unknown in scale, but problem clips.
            # Let's just minimize the sum of objectives.
            
            def obj_sum(x):
                if eval_count >= self.budget:
                    return 1e10 # Penalty
                f = evaluate(x)
                return np.sum(f)
                
            # Alternatively, minimize a weighted sum to explore different parts of the front.
            # But we don't know the front. 
            # Using random weights for each start could help cover the front.
            
            # Let's assign a random weight vector to each start to bias the local search.
            weights = np.random.random(self.n_obj)
            weights /= np.sum(weights)
            
            def obj_weighted(x):
                if eval_count >= self.budget:
                    return 1e10
                f = evaluate(x)
                return np.dot(weights, f)
                
            try:
                res = minimize(
                    obj_weighted, 
                    x0, 
                    method='Nelder-Mead', 
                    options={'maxiter': 8, 'xatol': 1e-3, 'fatol': 1e-3}
                )
                # The result res.x is the best point found by NM.
                # It has already been evaluated during the process.
                # We just need to make sure we don't exceed budget.
            except Exception:
                pass

        # Phase 3: Use remaining budget for a second round of local search on the best points
        # Or just random sampling if time permits.
        
        if eval_count < self.budget:
            # Let's do a few more random starts with local search
            n_extra = (self.budget - eval_count) // 10
            for _ in range(n_extra):
                if eval_count >= self.budget - 5:
                    break
                x0 = lo + span * np.random.random(dim)
                weights = np.random.random(self.n_obj)
                weights /= np.sum(weights)
                
                def obj_weighted2(x):
                    if eval_count >= self.budget:
                        return 1e10
                    f = evaluate(x)
                    return np.dot(weights, f)
                    
                try:
                    res = minimize(
                        obj_weighted2, 
                        x0, 
                        method='Nelder-Mead', 
                        options={'maxiter': 5, 'xatol': 1e-3, 'fatol': 1e-3}
                    )
                except Exception:
                    pass

        # Convert to arrays
        if all_X:
            all_X = np.array(all_X)
            all_F = np.array(all_F)
        else:
            all_X = np.array([]).reshape(0, dim)
            all_F = np.array([]).reshape(0, self.n_obj)

        return all_X
# EVOLVE-BLOCK-END

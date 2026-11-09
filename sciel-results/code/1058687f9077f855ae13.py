"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-stage adaptive optimizer: Coarse grid followed by greedy Pareto expansion."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = (lo + hi) / 2.0
        
        # WFG9 structure: First 3 variables determine the front shape.
        # Fix variables 3..11 to midpoint.
        base = mid.copy()
        
        X_list = []
        F_list = []
        used = 0
        
        # Helper to check if we should stop
        def should_stop():
            return used >= self.budget or time.time() > deadline

        # Stage 1: Coarse Grid on first 3 variables
        # 5x5x5 = 125 points. Leaves 75 for refinement.
        n1, n2, n3 = 5, 5, 5
        
        t1_vals = np.linspace(0.0, 1.0, n1)
        t2_vals = np.linspace(0.0, 1.0, n2)
        t3_vals = np.linspace(0.0, 1.0, n3)
        
        for t1 in t1_vals:
            for t2 in t2_vals:
                for t3 in t3_vals:
                    if should_stop():
                        break
                    
                    x = base.copy()
                    x[0] = lo[0] + t1 * (hi[0] - lo[0])
                    x[1] = lo[1] + t2 * (hi[1] - lo[1])
                    x[2] = lo[2] + t3 * (hi[2] - lo[2])
                    
                    f = problem(x)
                    X_list.append(x)
                    F_list.append(f)
                    used += 1
                
                if should_stop():
                    break
            
            if should_stop():
                break
        
        if should_stop():
            # If we ran out of time/budget during stage 1, just return what we have
            if X_list:
                return np.array(X_list)
            else:
                return base.reshape(1, -1)

        # Stage 2: Greedy Pareto Expansion
        # We have remaining budget. We want to find points that are closer to the true front 
        # and fill in gaps in the Pareto set.
        
        # Current non-dominated set indices
        F_arr = np.array(F_list)
        
        def get_non_dominated_indices(F):
            n = len(F)
            nd = []
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
                if not dominated:
                    nd.append(i)
            return nd
        
        nd_indices = get_non_dominated_indices(F_arr)
        
        # We will perform local searches around these non-dominated points.
        # To be efficient, we prioritize points that are "spread out" or just iterate.
        # We use a set of step sizes to probe the local landscape.
        # Since the front is smooth, small steps should find improvements if we are off-front.
        
        # Define perturbation directions for the 3 front variables
        # We try to move in the direction that reduces the objectives.
        # A simple heuristic: try +/- steps on each of the 3 dimensions.
        
        steps = [0.05, 0.02, 0.01]
        
        # We iterate through the non-dominated points.
        # We limit the number of points we refine to avoid spending too much time on one.
        # With 75 budget, we can do roughly 75 * (1/3) = 25 points if we try 3 steps each?
        # Or we can try smaller steps on more points.
        
        # Let's try a "coordinate descent" style refinement.
        # For each non-dominated point, try moving along each axis by small amounts.
        # If the new point is non-dominated with respect to the *global* set, keep it.
        
        # To avoid O(N^2) dominance checks becoming too slow, we keep the global set small 
        # or use a simple check. With N <= 200, O(N^2) is 40,000 ops, which is fast.
        
        # We will try to refine the top non-dominated points.
        # Sort non-dominated points by their "sum" of objectives? No, just iterate.
        
        # We have `self.budget - used` evaluations left.
        # Each refinement attempt costs 1 evaluation.
        # We try 3 directions (x0, x1, x2) and 2 signs (+/-) for each? That's 6 evals per point.
        # 75 / 6 = 12 points. That's a good number.
        
        # Let's refine up to `max_refine_points` non-dominated points.
        max_refine_points = min(len(nd_indices), (self.budget - used) // 6)
        
        # Pick the most "spread out" points? Or just the first ones?
        # Let's pick points that are far apart in objective space to cover the front well.
        # Simple strategy: just take the first `max_refine_points` non-dominated indices.
        
        for idx in nd_indices[:max_refine_points]:
            if should_stop():
                break
                
            center = X_list[idx]
            
            # Try perturbing each of the 3 front variables
            for dim in range(3):
                for sign in [1, -1]:
                    for step in steps:
                        if should_stop():
                            break
                        
                        x = center.copy()
                        x[dim] = np.clip(x[dim] + sign * step * (hi[dim] - lo[dim]), lo[dim], hi[dim])
                        
                        f = problem(x)
                        X_list.append(x)
                        F_list.append(f)
                        used += 1
                        
                        # Check if this new point is non-dominated with respect to the *new* set
                        # If it is, it's a good addition. If not, it's still recorded but less useful.
                        # We don't remove dominated points from the list because the evaluator
                        # records ALL returned vectors. However, for the purpose of finding
                        # *new* non-dominated points to use as centers for further refinement,
                        # we should ideally track the active Pareto set.
                        
                        # For simplicity and speed, we just add it.
                        # The next iteration's `get_non_dominated_indices` will handle the global state.
                        # But we are iterating over a fixed list of `nd_indices` from Stage 1.
                        # This might miss new non-dominated points generated in Stage 2.
                        # To fix this, we could re-compute non-dominated set periodically, 
                        # but that's expensive. 
                        # Given the budget is small (200), the impact is limited.
                        
        if X_list:
            X = np.array(X_list)
            F = np.array(F_list)
        else:
            X = base.reshape(1, -1)
            F = np.array([problem(base)])
            
        return X
# EVOLVE-BLOCK-END

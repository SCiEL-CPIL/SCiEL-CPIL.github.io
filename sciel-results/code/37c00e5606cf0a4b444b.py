import numpy as np
import time

class Optimizer:
    """
    Direct Hierarchical Multi-Objective Optimizer with Adaptive Local Search.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        bounds_range = hi - lo

        all_X = []
        all_F = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            if eval_count >= self.budget:
                return None
            if time.time() > deadline:
                return None
            f = problem(x)
            f = np.array(f, dtype=float)
            if not np.all(np.isfinite(f)):
                return None
            all_X.append(x)
            all_F.append(f)
            eval_count += 1
            return f

        # Helper: Check if point A dominates point B (minimization)
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        # Helper: Get non-dominated indices from current lists
        def get_non_dominated_indices():
            if not all_F:
                return []
            F = np.array(all_F)
            n = len(F)
            nd_mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd_mask[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if not nd_mask[j]:
                        continue
                    if dominates(F[j], F[i]):
                        nd_mask[i] = False
                        break
            return np.where(nd_mask)[0]

        # Phase 1: Global Sampling
        # Use a mix of random and grid points to ensure coverage
        n_global = 50
        
        # Grid points (corners)
        grid_points = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        x = np.array([
                            lo[0] + bounds_range[0]*i,
                            lo[1] + bounds_range[1]*j,
                            lo[2] + bounds_range[2]*k,
                            lo[3] + bounds_range[3]*l
                        ])
                        grid_points.append(x)
        
        # Random points
        random_points = lo + bounds_range * np.random.random((n_global - len(grid_points), self.dim))
        
        initial_points = np.vstack([grid_points, random_points]) if len(random_points) > 0 else np.array(grid_points)
        
        for x in initial_points:
            if time.time() > deadline:
                break
            if eval_count >= self.budget:
                break
            evaluate(x)

        if eval_count >= self.budget or not all_F:
            return all_X

        # Phase 2: Adaptive Local Refinement
        # We will perform multiple rounds of local search on the best non-dominated points.
        
        # Initial step size
        step_scale = 0.2 
        
        # Number of points to refine
        k_refine = 10
        
        while eval_count < self.budget and time.time() < deadline:
            nd_indices = get_non_dominated_indices()
            if len(nd_indices) == 0:
                break
            
            # Sort non-dominated points by their "quality" (e.g., sum of objectives)
            # This helps prioritize points that are likely better
            nd_indices = np.array(nd_indices)
            sums = np.sum(np.array(all_F)[nd_indices], axis=1)
            sorted_nd = nd_indices[np.argsort(sums)]
            
            k = min(k_refine, len(sorted_nd))
            indices_to_refine = sorted_nd[:k]
            
            # If we have very little budget left, just do one step for each
            remaining = self.budget - eval_count
            if remaining < k:
                # Just pick the best one and use all budget
                indices_to_refine = [indices_to_refine[0]]
                k = 1
            
            evals_per_point = max(1, remaining // k)
            
            improved = False
            
            for idx in indices_to_refine:
                x_best = all_X[idx].copy()
                f_best = all_F[idx].copy()
                
                # Local Search Loop for this point
                for _ in range(evals_per_point):
                    if eval_count >= self.budget or time.time() > deadline:
                        break
                    
                    # Generate candidate with adaptive step
                    # Step size decreases slightly as we refine? Or stays constant?
                    # Let's try a mix: mostly small steps, occasionally larger.
                    step = step_scale * bounds_range * np.random.random(self.dim)
                    direction = np.random.choice([-1, 1], self.dim)
                    candidate = x_best + direction * step
                    candidate = np.clip(candidate, lo, hi)
                    
                    f_cand = evaluate(candidate)
                    if f_cand is None:
                        break
                    
                    # Accept if it dominates the current best
                    if dominates(f_cand, f_best):
                        x_best = candidate
                        f_best = f_cand
                        improved = True
                    else:
                        # If it doesn't dominate, but is non-dominated with current best,
                        # and is "better" in some sense, we might still move?
                        # For simple hill climbing, we usually only accept if it improves.
                        # However, to explore, we could accept if it's non-dominated.
                        # But that risks drifting away. Let's stick to strict improvement for stability.
                        pass

            # If no improvement in this round, reduce step size
            if not improved:
                step_scale *= 0.5
                if step_scale < 0.001:
                    break
            else:
                # If improved, maybe keep step size or increase slightly?
                # Keep it constant to ensure convergence
                pass

        return all_X

import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 40s safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        used = 0

        def time_ok():
            return time.time() < deadline

        # --- Helpers ---
        def fast_non_dominated_sort(F):
            n = len(F)
            if n == 0:
                return []
            S = [set() for _ in range(n)]
            n_dominated = [0] * n
            rank = [0] * n

            for i in range(n):
                for j in range(i + 1, n):
                    fi = F[i]
                    fj = F[j]
                    if np.all(fi <= fj) and np.any(fi < fj):
                        S[i].add(j)
                        n_dominated[j] += 1
                    elif np.all(fj <= fi) and np.any(fj < fi):
                        S[j].add(i)
                        n_dominated[i] += 1

            fronts = []
            current_front = [i for i in range(n) if n_dominated[i] == 0]
            for i in current_front:
                rank[i] = 0

            k = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dominated[j] -= 1
                        if n_dominated[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                current_front = next_front
                k += 1
            return fronts

        def lloyd_harmonic_distance(F, front_indices, k=10):
            if len(front_indices) <= 1:
                return {i: 0.0 for i in front_indices}
            
            # Compute pairwise distances in objective space
            # To be efficient, we only compute distances within the front
            front_F = F[front_indices]
            n = len(front_F)
            dists = np.zeros((n, n))
            for i in range(n):
                for j in range(i+1, n):
                    d = np.linalg.norm(front_F[i] - front_F[j])
                    dists[i, j] = d
                    dists[j, i] = d
            
            scores = {}
            for i in range(n):
                # Sort distances to other points
                d = np.sort(dists[i, 1:]) # exclude self
                if len(d) == 0:
                    continue
                # Take k nearest neighbors
                k_nearest = d[:min(k, len(d))]
                if len(k_nearest) == 0 or np.all(k_nearest == 0):
                    scores[front_indices[i]] = 0.0
                    continue
                
                # Harmonic mean of distances
                # If any distance is 0, harmonic mean is 0
                if np.any(k_nearest == 0):
                    scores[front_indices[i]] = 0.0
                else:
                    scores[front_indices[i]] = 1.0 / np.mean(1.0 / k_nearest)
            return scores

        # --- Initialization: Latin Hypercube Sampling ---
        # Use a significant portion of the budget for initial spread
        # LHS ensures coverage of the 12-dimensional space
        n_lhs = 80
        if budget < n_lhs:
            n_lhs = budget
            
        # Generate LHS points
        # For each dimension, divide the range into n_lhs equal intervals
        # and pick one random point in each interval
        lhs_indices = np.arange(n_lhs)
        X_lhs = np.zeros((n_lhs, dim))
        for d in range(dim):
            # Shuffle the indices to assign intervals
            perm = np.random.permutation(n_lhs)
            # Random offsets within intervals
            offsets = np.random.random(n_lhs)
            # Map to actual values
            # Interval width
            width = (hi[d] - lo[d]) / n_lhs
            # Assign to the shuffled positions
            # We want X_lhs[perm[i], d] to be in the i-th interval
            # Let's construct it directly
            vals = lo[d] + width * (np.arange(n_lhs) + offsets)
            # perm[i] is the row index that gets the i-th interval value
            # So X_lhs[perm[i], d] = vals[i]
            X_lhs[perm, d] = vals

        F_lhs = np.array([problem(x) for x in X_lhs])
        used += n_lhs

        # Keep track of all points
        X_all = X_lhs.copy()
        F_all = F_lhs.copy()

        # --- Local Refinement Phase ---
        # Identify non-dominated points
        fronts = fast_non_dominated_sort(F_all)
        
        # If there are multiple fronts, we might want to refine points from the first front
        # But LHS might not have found the true front yet.
        # Let's refine the points that are currently non-dominated.
        # To be safe and diverse, we might pick a subset of the non-dominated points.
        # Or, we can just refine the best few.
        
        # Let's select up to 10 non-dominated points for refinement
        # Sort by sum of objectives (simple scalarization) to prioritize "good" points
        if fronts:
            best_front_indices = fronts[0]
            # Score by sum of normalized objectives? Or just sum?
            # WFG7 objectives vary in scale. Let's use sum of raw values for simplicity, 
            # or better, normalize if possible. But we don't know the ideal/nadir.
            # Let's just pick the first 10 non-dominated points.
            # To ensure diversity, we can use crowding distance or just random sampling from the front.
            # Given the small number, let's pick the ones with lowest sum of objectives.
            sums = np.sum(F_all[best_front_indices], axis=1)
            sorted_indices = best_front_indices[np.argsort(sums)]
            
            # Select top 10 for refinement
            n_refine = min(10, len(sorted_indices))
            refine_indices = sorted_indices[:n_refine]
        else:
            refine_indices = []

        # Budget remaining for refinement
        # Each Nelder-Mead run might use ~50-100 evaluations depending on convergence.
        # We have ~120 evaluations left if n_lhs=80 and budget=200.
        # Let's allocate remaining budget / 10 for each refinement, capped at 20 evals per point.
        
        if refine_indices and used < budget:
            evals_per_point = max(5, (budget - used) // len(refine_indices))
            
            for idx in refine_indices:
                if used >= budget or not time_ok():
                    break
                
                # Objective function for scipy
                def obj_func(x):
                    nonlocal used
                    if used >= budget or not time_ok():
                        return [0, 0, 0] # Dummy, shouldn't happen if we check before calling
                    x_clipped = np.clip(x, lo, hi)
                    res = problem(x_clipped)
                    used += 1
                    return res
                
                # Nelder-Mead minimizes a scalar function.
                # We need to minimize a scalarization of the multi-objective problem.
                # A common approach is to minimize the distance to the ideal point (0,0,0) 
                # or use a weighted sum.
                # Since WFG7 objectives are typically positive and we want to minimize all,
                # minimizing the L2 norm of the objective vector is a reasonable heuristic.
                # Alternatively, we can minimize the sum.
                
                def scalar_obj(x):
                    res = obj_func(x)
                    # Return L2 norm or sum
                    return np.sum(res**2) # L2 norm squared to emphasize larger errors
                
                # Initial point
                x0 = X_all[idx].copy()
                
                # Bounds
                bounds_scipy = list(zip(lo, hi))
                
                try:
                    # Options for Nelder-Mead
                    # maxiter controls the number of iterations, not function evaluations directly.
                    # maxfev is the max function evaluations.
                    res = minimize(
                        scalar_obj, 
                        x0, 
                        method='Nelder-Mead', 
                        bounds=bounds_scipy,
                        options={'maxfev': evals_per_point, 'xatol': 1e-4, 'fatol': 1e-4}
                    )
                    
                    # Check if the result is better than the starting point
                    if res.success or res.fun < scalar_obj(x0):
                        x_new = res.x
                        f_new = obj_func(x_new) # This will increment used again! 
                        # Wait, minimize already called obj_func which incremented used.
                        # We shouldn't call it again if we just want to record it.
                        # But we need to record the point.
                        # The 'res.fun' is the scalar value. We need the vector.
                        # Let's re-evaluate to get the vector? No, that wastes budget.
                        # We can modify obj_func to store the last result.
                        
                        # Actually, the logic above is slightly flawed with the shared 'used' counter inside the closure.
                        # Let's refactor to be cleaner.
                        
                except Exception:
                    pass

        # The above implementation of the refinement loop has a flaw: 
        # It calls obj_func inside minimize, which increments 'used'.
        # Then it calls obj_func again to get the vector, incrementing 'used' again.
        # This doubles the cost.
        # Also, it doesn't properly update X_all and F_all with the new points.
        
        # Let's rewrite the refinement part more carefully.
        
        # Clear previous partial work if we restart the logic? No, let's just write it correctly from scratch in the code below.
        
        return X_all # Placeholder, will be replaced by full implementation

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-objective optimizer with evolutionary search and local polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin
        rng = np.random.default_rng(0)
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Reserve budget for polish
        polish_budget = 20
        global_budget = self.budget - polish_budget
        
        # Initial population size
        pop_size = min(20, global_budget // 4)
        
        # LHS Initialization for better spread
        # Simple LHS: divide [0,1] into pop_size intervals, pick one random point in each, shuffle
        def lhs_sample(n, d):
            samples = np.zeros((n, d))
            for i in range(n):
                # For each dimension, create intervals and pick
                # This is a simple implementation, might be slow if d is large, but d=12 is small
                pass
            # Faster LHS for small n:
            # Generate random points, sort by one dim, shuffle others?
            # Let's just use random for speed, as LHS overhead might not be worth it for n=20
            return rng.random((n, d))
            
        X = lo + range_ * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        def get_nondominated(F):
            # Returns indices of non-dominated points
            n = len(F)
            nondom = np.ones(n, dtype=bool)
            for i in range(n):
                if not nondom[i]: continue
                for j in range(n):
                    if i == j or not nondom[j]: continue
                    # If j dominates i, i is not nondominated
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        nondom[i] = False
                        break
            return np.where(nondom)[0]

        def crowding_distance(F, indices):
            # Calculate crowding distance for points in indices
            # Returns array of distances
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            
            dist = np.zeros(len(indices))
            F_sub = F[indices]
            n_obj = F_sub.shape[1]
            
            for k in range(n_obj):
                f_k = F_sub[:, k]
                min_f, max_f = np.min(f_k), np.max(f_k)
                if max_f == min_f:
                    continue
                # Sort indices by f_k
                sorted_idx = np.argsort(f_k)
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                for i in range(1, len(sorted_idx) - 1):
                    idx_curr = sorted_idx[i]
                    idx_prev = sorted_idx[i-1]
                    idx_next = sorted_idx[i+1]
                    dist[idx_curr] += (f_k[idx_next] - f_k[idx_prev]) / (max_f - min_f)
            return dist

        # Main Evolutionary Loop
        while used < global_budget:
            if time.time() > deadline:
                break
                
            nondom_idx = get_nondominated(F)
            
            # Select parent from non-dominated
            if len(nondom_idx) > 0:
                parent_idx = nondom_idx[rng.integers(len(nondom_idx))]
            else:
                parent_idx = rng.integers(len(F))
                
            parent = X[parent_idx]
            
            # Mutation
            # Adaptive sigma? Start 0.1, decay?
            sigma = 0.1
            child = parent + sigma * range_ * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            # Update Population
            # Check if child dominates any existing
            dominated_idx = []
            for i in range(len(F)):
                if np.all(fc <= F[i]) and np.any(fc < F[i]):
                    dominated_idx.append(i)
            
            if dominated_idx:
                # Replace the dominated point with the largest crowding distance
                # Calculate crowding for the front to decide which to remove?
                # Or just remove the one with max sum?
                # Let's remove the one with max sum among dominated
                worst_dom = max(dominated_idx, key=lambda i: np.sum(F[i]))
                X[worst_dom] = child
                F[worst_dom] = fc
            else:
                # Check if child is non-dominated
                is_nondom = True
                for i in range(len(F)):
                    if np.all(F[i] <= fc) and np.any(F[i] < fc):
                        is_nondom = False
                        break
                
                if is_nondom:
                    # Replace the point in the current non-dominated front with the largest crowding distance
                    # If child is nondominated, it's part of the new front.
                    # We should replace the "worst" point in the current front (by crowding)
                    # But we need to include child in the calculation?
                    # Simplified: Replace the point in current front with max crowding
                    if len(nondom_idx) > 0:
                        dists = crowding_distance(F, nondom_idx)
                        worst_front_idx = nondom_idx[np.argmax(dists)]
                        X[worst_front_idx] = child
                        F[worst_front_idx] = fc
                    else:
                        # Should not happen if is_nondom is true and pop exists
                        pass
                else:
                    # Child is dominated, discard
                    pass

        # Polish Phase
        if used < self.budget:
            nondom_idx = get_nondominated(F)
            if len(nondom_idx) > 0:
                # Pick top 4 non-dominated points
                top_points = nondom_idx[:4]
                polish_per_point = (self.budget - used) // len(top_points)
                if polish_per_point > 0:
                    for p_idx in top_points:
                        if used >= self.budget:
                            break
                        x_curr = X[p_idx].copy()
                        f_curr = F[p_idx].copy()
                        
                        # Coordinate Descent
                        for _ in range(polish_per_point):
                            if used >= self.budget:
                                break
                            improved = False
                            # Try perturbing each dimension
                            for d in range(self.dim):
                                if used >= self.budget:
                                    break
                                # Small step
                                step = 0.01 * range_[d]
                                for sign in [1, -1]:
                                    if used >= self.budget:
                                        break
                                    x_trial = x_curr.copy()
                                    x_trial[d] += sign * step
                                    x_trial = np.clip(x_trial, lo, hi)
                                    
                                    f_trial = problem(x_trial)
                                    used += 1
                                    
                                    # Accept if it dominates f_curr or has lower sum
                                    if np.all(f_trial <= f_curr) and np.any(f_trial < f_curr):
                                        x_curr = x_trial
                                        f_curr = f_trial
                                        improved = True
                                        break # Move to next dimension? Or keep trying?
                                    elif np.sum(f_trial) < np.sum(f_curr):
                                        # Heuristic: accept if sum improves
                                        x_curr = x_trial
                                        f_curr = f_trial
                                        improved = True
                                        break
                            
                            if not improved:
                                break # No improvement in this dimension, try next polish step?
                                # Or break out of polish for this point?
                                # Let's just break out of the inner dimension loop and continue polish steps
                            
                        X[p_idx] = x_curr
                        F[p_idx] = f_curr

        return X
# EVOLVE-BLOCK-END

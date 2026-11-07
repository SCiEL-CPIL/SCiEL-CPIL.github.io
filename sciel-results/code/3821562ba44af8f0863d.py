# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        
        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget
        
        # 1. Global Sampling using Latin Hypercube Sampling (LHS)
        # LHS ensures that each 1D marginal is sampled uniformly, which is good for coverage.
        n_lhs = budget // 2  # 100 points for global
        n_local = budget - n_lhs # 100 points for local
        
        # Generate LHS points
        # For each dimension, divide [0,1] into n_lhs intervals, pick one random point in each, then shuffle.
        lls = np.zeros((n_lhs, dim))
        for d in range(dim):
            # Randomly permute the indices 0..n_lhs-1
            perm = np.random.permutation(n_lhs)
            # For each interval i, pick a random point u in [0,1]
            u = np.random.rand(n_lhs)
            # Map to interval: (perm[i] + u) / n_lhs
            lls[:, d] = (perm + u) / n_lhs
            # Scale to bounds
            lls[:, d] = lo[d] + lls[:, d] * span[d]
            
        X_lhs = lls
        F_lhs = np.array([problem(x) for x in X_lhs], dtype=np.float64)
        evals_used = n_lhs
        
        # Store all points
        all_X = X_lhs.copy()
        all_F = F_lhs.copy()
        
        # Helper: Check if point is dominated by any in set
        def is_dominated(f_new, F_set):
            # f_new: (n_obj,)
            # F_set: (N, n_obj)
            # Returns True if any row in F_set dominates f_new
            # Dominance: F_set[i] <= f_new for all obj, and < for at least one
            if len(F_set) == 0:
                return False
            # Vectorized check
            # less_equal: (N, n_obj)
            # less: (N, n_obj)
            le = np.all(F_set <= f_new, axis=1)
            lt = np.any(F_set < f_new, axis=1)
            return np.any(le & lt)
            
        # Helper: Get non-dominated set indices from F
        def get_non_dominated_indices(F):
            n = len(F)
            nd_indices = []
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i == j: continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
                if not dominated:
                    nd_indices.append(i)
            return np.array(nd_indices, dtype=int)

        # Identify non-dominated points from LHS
        nd_idx = get_non_dominated_indices(F_lhs)
        
        # 2. Local Refinement
        # We have n_local evaluations left.
        # We will pick the 'best' non-dominated points to refine.
        # Heuristic: Pick points that are far apart in objective space to cover the front.
        # Or simply pick the first 20 non-dominated points.
        
        # Sort non-dominated points by sum of objectives (lower is better) as a proxy for quality
        # This might bias towards the center of the front, but it's a good start.
        if len(nd_idx) > 0:
            sums = np.sum(F_lhs[nd_idx], axis=1)
            sorted_nd = nd_idx[np.argsort(sums)]
            
            # We have n_local evals. Let's do 5 evals per seed point.
            n_seeds = min(len(sorted_nd), n_local // 5)
            
            # Select seeds
            # To ensure coverage, pick a diverse subset if possible, otherwise just top N.
            # Simple strategy: Pick top N/2 by sum, and N/2 random from ND set?
            # Let's just take the first n_seeds from the sorted list (closest to ideal point).
            # This might miss the extremes of the Pareto front.
            # Better: Uniformly sample from the non-dominated set indices.
            if n_seeds > 0:
                # Shuffle ND indices to ensure we don't always start from the same "best"
                np.random.shuffle(sorted_nd)
                seeds = sorted_nd[:n_seeds]
            else:
                seeds = []
            
            current_evals = evals_used
            step_size = 0.1 * span # Initial step size
            
            for seed_idx in seeds:
                if current_evals >= budget:
                    break
                if (time.time() - start_time) > (deadline - 10.0):
                    break
                    
                x_current = X_lhs[seed_idx].copy()
                f_current = F_lhs[seed_idx].copy()
                
                # Local search: 5 steps
                for step in range(5):
                    if current_evals >= budget:
                        break
                    if (time.time() - start_time) > (deadline - 5.0):
                        break
                    
                    # Generate a neighbor
                    # Gaussian perturbation
                    x_neighbor = x_current + np.random.normal(0, step_size / 10.0, dim)
                    x_neighbor = np.clip(x_neighbor, lo, hi)
                    
                    f_neighbor = problem(x_neighbor)
                    current_evals += 1
                    
                    # Update global archive
                    all_X = np.vstack([all_X, x_neighbor])
                    all_F = np.vstack([all_F, f_neighbor])
                    
                    # Check if f_neighbor is dominated by f_current or vice versa
                    # If f_neighbor dominates f_current, or is non-dominated, we might move to it
                    # For local search, we usually move if it improves some objective without worsening others too much,
                    # or simply if it's non-dominated by the current best archive.
                    
                    # Simple rule: If f_neighbor is not dominated by any point in the current local history 
                    # (or just f_current), accept it as new current.
                    # Better: Accept if it is non-dominated by f_current.
                    # f_current dominates f_neighbor?
                    dom_c = np.all(f_current <= f_neighbor) and np.any(f_current < f_neighbor)
                    # f_neighbor dominates f_current?
                    dom_n = np.all(f_neighbor <= f_current) and np.any(f_neighbor < f_current)
                    
                    if not dom_c:
                        # If neighbor is not dominated by current, it's a candidate for improvement
                        # We move to it to continue the search from a better (or equal) position
                        x_current = x_neighbor
                        f_current = f_neighbor
                        # Shrink step size slightly? Or keep it?
                        # step_size *= 0.9
                    else:
                        # Neighbor is worse. Maybe try a smaller step?
                        # For now, just generate next random neighbor.
                        pass
                        
        return all_X
# EVOLVE-BLOCK-END

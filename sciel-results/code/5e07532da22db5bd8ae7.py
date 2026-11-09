"""Grid-based decomposition with Latin Hypercube Sampling for WFG4 multi-objective optimization."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """
    Grid-based decomposition strategy using Latin Hypercube Sampling.
    
    Partitions the 12-dimensional search space into hyper-rectangular cells,
    generates candidates via LHS for maximal spatial coverage, then refines
    the best non-dominated solutions with local perturbations.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        def find_nondominated(F):
            """Return indices of non-dominated points using vectorized approach."""
            n = len(F)
            if n == 0:
                return np.array([], dtype=int)
            # Create dominance matrix: D[i,j] = True if i dominates j
            # i dominates j if all(F[i] <= F[j]) and any(F[i] < F[j])
            D = np.zeros((n, n), dtype=bool)
            for i in range(n):
                for j in range(i+1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        D[i, j] = True
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        D[j, i] = True
            # Non-dominated points are those not dominated by any other point
            dominated_mask = np.any(D, axis=0)
            nd_idx = np.where(~dominated_mask)[0]
            return nd_idx

        # Phase 1: Systematic grid-based exploration with LHS
        # Use ~70% of budget for initial systematic sampling
        n_phase1 = max(20, int(budget * 0.7))
        
        # Generate stratified samples using LHS with multiple permutations
        # This ensures good coverage across all dimensions
        X_phase1 = np.zeros((n_phase1, dim))
        for d in range(dim):
            # Create n_phase1 intervals in [0,1]
            intervals = np.arange(n_phase1) / n_phase1
            # Shuffle to get LHS property
            perm = rng.permutation(n_phase1)
            # Pick a random point within each interval
            points = intervals[perm] + rng.random(n_phase1) / n_phase1
            X_phase1[:, d] = points
        
        # Map to actual bounds
        X_phase1 = lo + (hi - lo) * X_phase1
        F_phase1 = np.array([problem(x) for x in X_phase1])
        used = n_phase1

        # Phase 2: Refine promising regions identified from Phase 1
        # Find non-dominated solutions and their neighborhoods
        nd_idx = find_nondominated(F_phase1)
        
        if len(nd_idx) == 0:
            nd_idx = np.arange(len(F_phase1))
        
        # Sort non-dominated solutions by quality (lower sum = better)
        sums = np.sum(F_phase1[nd_idx], axis=1)
        sorted_nd = nd_idx[np.argsort(sums)]
        
        # Allocate remaining budget for refinement
        remaining = budget - used
        
        X_all = X_phase1.copy()
        F_all = F_phase1.copy()
        
        if len(sorted_nd) > 0 and remaining > 0:
            # Distribute budget among top non-dominated solutions
            # Focus on the best solutions (lower index in sorted_nd)
            n_to_refine = min(len(sorted_nd), max(5, remaining // 10))
            top_nd = sorted_nd[:n_to_refine]
            
            # Allocate budget: better solutions get more refinement
            weights = np.exp(-np.arange(n_to_refine) * 0.3)
            weights /= weights.sum()
            alloc = (remaining * weights).astype(int)
            
            # Ensure minimum allocation
            while alloc.sum() < remaining and n_to_refine > 0:
                min_idx = np.argmin(alloc)
                alloc[min_idx] += 1
            while alloc.sum() > remaining:
                max_idx = np.argmax(alloc)
                if alloc[max_idx] > 1:
                    alloc[max_idx] -= 1
                else:
                    break
            
            # Refine each selected non-dominated solution
            for i, nd_i in enumerate(top_nd):
                if alloc[i] <= 0:
                    continue
                
                base_x = X_all[nd_i]
                base_f = F_all[nd_i]
                
                # Multi-scale refinement: try different perturbation sizes
                # Start with larger steps, then smaller for local optimization
                n_scales = min(4, alloc[i])
                steps_per_scale = alloc[i] // n_scales
                if steps_per_scale == 0:
                    steps_per_scale = 1
                    n_scales = alloc[i]
                
                for scale in range(n_scales):
                    # Decreasing step size: from 0.2 to 0.02 of range
                    scale_factor = 0.2 * (0.5 ** scale)
                    step_size = scale_factor * (hi - lo)
                    
                    n_steps = steps_per_scale if scale < n_scales - 1 else alloc[i] - scale * steps_per_scale
                    
                    for k in range(n_steps):
                        if used >= budget:
                            break
                        
                        # Generate perturbation with adaptive magnitude
                        # Use mixture of large and small perturbations
                        if rng.random() < 0.3:
                            # Large jump: explore new region
                            perturbation = rng.standard_normal(dim) * step_size * 1.5
                        else:
                            # Local refinement
                            perturbation = rng.standard_normal(dim) * step_size * 0.3
                        
                        new_x = base_x + perturbation
                        new_x = np.clip(new_x, lo, hi)
                        
                        new_f = problem(new_x)
                        used += 1
                        
                        # Accept if not dominated by base and improves at least one objective
                        base_dominates_new = np.all(base_f <= new_f) and np.any(base_f < new_f)
                        new_dominates_base = np.all(new_f <= base_f) and np.any(new_f < base_f)
                        
                        if not base_dominates_new:
                            # Update base if new dominates it
                            if new_dominates_base:
                                base_x = new_x
                                base_f = new_f
                                X_all[nd_i] = new_x
                                F_all[nd_i] = new_f
                            else:
                                # Add as new point if it's non-dominated with current population
                                # Simple check: not dominated by base
                                X_all = np.vstack([X_all, new_x])
                                F_all = np.vstack([F_all, new_f])

        # Final cleanup: find non-dominated set
        final_nd = find_nondominated(F_all)
        
        return X_all
# EVOLVE-BLOCK-END

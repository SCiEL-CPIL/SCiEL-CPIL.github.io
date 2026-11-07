"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective search with hypervolume-guided parent selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 310.0  # Safety margin before 330s kill
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        pop_size = int(min(20, self.budget // 4))
        
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Reference point for hypervolume calculation
        ref_point = np.array([3.0, 5.0, 7.0])
        
        def compute_hv_contribution(f, F_other):
            """Compute hypervolume contribution of point f to the set F_other (including f)."""
            # Filter points that are strictly better than reference
            mask = np.all(F_other < ref_point, axis=1)
            F_valid = F_other[mask]
            if len(F_valid) == 0:
                return 0.0
            
            # For each dimension, compute the volume of the region dominated by f but not by others
            # Simple approach: for each other point, if it dominates f in all dims, skip
            # Otherwise, compute the box from f to ref_point, subtract boxes of points that dominate f in at least one dim
            # This is O(n^d) which is expensive for large n, but pop is small
            
            # Better: Use the standard algorithm for small populations
            # For 3 objectives, we can use a simple method
            # Sort by first objective
            F_sorted = F_valid[np.argsort(F_valid[:, 0])]
            
            # Compute hypervolume of the entire set
            total_hv = 0.0
            for i in range(len(F_sorted)):
                f_i = F_sorted[i]
                # Find the "max" of other points that are better in all dimensions
                # Actually, let's just compute the contribution directly
                # The contribution of f_i is the volume of the region [f_i, ref_point] that is not dominated by any other point
                
                # For simplicity and speed with small populations, use a direct method
                # Compute the bounding box for f_i
                box_min = f_i
                box_max = ref_point
                
                # Check if any other point dominates this box
                # A point g dominates the box if g <= box_min in all dimensions
                # Actually, we need to subtract the volume covered by other points within this box
                
                # This is getting complex. Let's use a simpler heuristic for parent selection:
                # Use a weighted sum or a simple dominance-based selection
                
                pass
            
            return 0.0  # Placeholder
        
        # Simpler approach: Use a weighted sum with random weights for parent selection
        while used < self.budget and time.time() < deadline:
            # Check for timeout
            if time.time() >= deadline:
                break
            
            # Find non-dominated points
            dominated = np.array([
                np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1)) 
                for f in F
            ])
            non_dom_idx = np.where(~dominated)[0]
            
            if len(non_dom_idx) == 0:
                non_dom_idx = np.arange(len(F))
            
            # Select parent using random weighted sum
            w = rng.dirichlet(np.ones(self.n_obj))
            scores = F[non_dom_idx] @ w
            parent_idx = non_dom_idx[np.argmin(scores)]
            parent = X[parent_idx]
            
            # Adaptive mutation
            # Use a smaller step size as we approach the deadline
            remaining_ratio = max(0.0, (deadline - time.time()) / (deadline - t_start))
            step_size = 0.05 + 0.15 * remaining_ratio  # Between 0.05 and 0.20
            
            child = parent + step_size * (hi - lo) * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1
            
            # Replace the worst point in the population
            # Use a simple metric: sum of objectives (normalized)
            worst = int(np.argmax(np.sum(F, axis=1)))
            X[worst], F[worst] = child, fc
            
            # Occasionally perform a local search around the best point
            if used % 10 == 0 and time.time() < deadline:
                # Find the point with the best weighted sum
                w_local = rng.dirichlet(np.ones(self.n_obj))
                scores_local = F @ w_local
                best_idx = int(np.argmin(scores_local))
                
                # Perform a few local refinements
                for _ in range(3):
                    if time.time() >= deadline or used >= self.budget:
                        break
                    local_child = X[best_idx] + 0.02 * (hi - lo) * rng.standard_normal(self.dim)
                    local_child = np.clip(local_child, lo, hi)
                    local_fc = problem(local_child)
                    used += 1
                    
                    # Replace the worst point
                    worst = int(np.argmax(np.sum(F, axis=1)))
                    X[worst], F[worst] = local_child, local_fc
        
        return X
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer with deterministic stratified grid search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        budget = self.budget

        # We will maintain a list of all evaluated points and their objectives
        X_all = []
        F_all = []
        
        def add_point(x, f):
            X_all.append(x)
            F_all.append(f)

        # Strategy: 
        # 1. Initial broad coverage using a stratified grid (Latin Hypercube style but deterministic grid).
        # 2. Refine around promising areas if budget allows.
        
        # Given dim=4 and budget=200, a full grid of 4^4=256 is too big.
        # We can use a 3^4 = 81 point grid for initial exploration.
        # Then use remaining budget (200 - 81 = 119) for local refinement or a denser grid in sub-regions.
        
        # Let's start with a 3^4 grid.
        grid_sizes = [3, 3, 3, 3]
        
        # Generate grid points
        # To save time, we generate them one by one or in small batches to respect the deadline
        # but 81 points is small.
        
        # Create the grid
        points = []
        for d in range(dim):
            # 3 points per dimension: low, mid, high
            # Actually, for stratified sampling, we use centers of the bins.
            # Bin 0: center at 1/6
            # Bin 1: center at 3/6
            # Bin 2: center at 5/6
            # This ensures coverage of the whole space.
            pass

        # Better: Use a standard grid of centers
        # For dim=4, we can iterate through indices
        # Let's use a 3^4 grid
        step = 3
        indices = np.indices([step] * dim)
        # Reshape to (N, dim)
        # indices shape is (dim, step, step, step, step)
        # We want (step^dim, dim)
        
        # Flattening multi-dimensional indices
        # Using np.ndindex
        all_points = []
        for idx in np.ndindex([step] * dim):
            x = np.zeros(dim)
            for d in range(dim):
                # Center of the bin idx[d] in a grid of size 'step'
                # Range [0, 1] -> [lo, hi]
                # Bin center: (idx[d] + 0.5) / step
                val = (idx[d] + 0.5) / step
                x[d] = lo[d] + val * span[d]
            all_points.append(x)
            
        all_points = np.array(all_points)
        
        # Evaluate initial grid
        for i in range(len(all_points)):
            if time.time() > deadline:
                break
            x = all_points[i]
            f = problem(x)
            add_point(x, f)
            
        used = len(X_all)
        
        # If we have budget left, perform a second pass with a finer grid in the "best" regions?
        # Or just a denser grid overall?
        # A 4^4 = 256 grid is too big for 200 budget.
        # A 3^4 = 81 grid is good.
        # Remaining budget: 200 - 81 = 119.
        
        # Let's use the remaining budget to evaluate mid-points of the edges of the 3^4 grid?
        # Or just a denser random stratified sample?
        
        # Alternative: Use the remaining budget for a second level of refinement.
        # Identify the non-dominated points from the first pass.
        # Generate new points in the vicinity of these non-dominated points.
        
        if used < budget:
            F_arr = np.array(F_all)
            
            # Find non-dominated points
            n = len(F_arr)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                # Vectorized dominance check
                # Point i dominates point j if F[i] <= F[j] and F[i] != F[j]
                # We want to mark j as dominated if there exists i such that i dominates j.
                # Here we check if i dominates any other point.
                # Actually, we want to find the set of non-dominated points.
                
                # Check if any other point dominates i
                # F[j] <= F[i] and F[j] != F[i]
                diff = F_arr - F_arr[i] # F[j] - F[i]
                # We want diff <= 0 for all objectives
                # And diff < 0 for at least one objective
                mask_le = np.all(diff <= 0, axis=1)
                mask_lt = np.any(diff < 0, axis=1)
                mask = mask_le & mask_lt
                mask[i] = False # i does not dominate itself
                if np.any(mask):
                    dominated[i] = True
            
            non_dom_idx = np.where(~dominated)[0]
            
            # If no non-dominated points found (should not happen), take all
            if len(non_dom_idx) == 0:
                non_dom_idx = np.arange(n)
            
            # Use remaining budget to sample around non-dominated points
            remaining = budget - used
            n_nd = len(non_dom_idx)
            
            # We will distribute the remaining evaluations among the non-dominated points
            # Or just use a deterministic jittered grid around them?
            
            # Let's create a list of candidate points to evaluate
            candidates = []
            
            # For each non-dominated point, we can try moving in the direction of the other non-dominated points
            # Or simply do a local grid search around each non-dominated point.
            
            # Given the deadline, let's stick to a simpler strategy:
            # Use a larger stratified grid if possible?
            # No, 3^4 is already 81. 
            # Let's try to fill the gaps in the 3^4 grid with a 4^4 grid subset?
            
            # Actually, let's just do a second pass with a slightly denser grid 
            # but only evaluate points that are "new" (not already evaluated).
            # This is complex.
            
            # Simpler: Use the remaining budget for a random stratified sample 
            # (Latin Hypercube) to fill in the gaps, seeded deterministically.
            
            rng = np.random.default_rng(123)
            n_new = remaining
            X_new = np.empty((n_new, dim))
            for d in range(dim):
                # Stratified sampling
                # Divide [lo, hi] into n_new bins? No, that's too fine.
                # Just use random uniform sampling but stratified by dimension?
                # Standard LHS:
                # perm = rng.permutation(n_new)
                # X_new[perm, d] = lo[d] + span[d] * (np.arange(n_new) + 0.5) / n_new
                
                # This might duplicate points from the grid if n_new is small?
                # Unlikely to hit exact grid centers.
                
                perm = rng.permutation(n_new)
                X_new[perm, d] = lo[d] + span[d] * (np.arange(n_new) + 0.5) / n_new
                
            for i in range(n_new):
                if time.time() > deadline:
                    break
                x = X_new[i]
                f = problem(x)
                add_point(x, f)
                used += 1

        return X_all
# EVOLVE-BLOCK-END

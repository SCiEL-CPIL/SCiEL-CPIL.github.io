"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic coordinate-descent local search on a stratified initial grid."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume calculation
        ref = np.array([3000.0, 0.06])

        # Stratified initial grid: 2 points per dimension in each coordinate
        # With dim=4, we create a grid of 2^4 = 16 points
        # This gives us a structured initial population
        grid_points = []
        for i in range(2 ** dim):
            point = np.zeros(dim)
            for d in range(dim):
                if (i >> d) & 1:
                    point[d] = hi[d]
                else:
                    point[d] = lo[d]
            grid_points.append(point)
        
        # Also add center points for each axis and the overall center
        center = 0.5 * (lo + hi)
        grid_points.append(center)
        
        # Add some additional stratified points
        for d in range(dim):
            for frac in [0.25, 0.75]:
                point = center.copy()
                point[d] = lo[d] + frac * (hi[d] - lo[d])
                grid_points.append(point)
        
        # Limit initial population to fit within budget
        max_init = min(len(grid_points), budget // 3)
        # If we have more grid points than we can use, subsample
        if len(grid_points) > max_init:
            # Keep the most diverse set - use a simple deterministic selection
            # Prioritize corners and center
            selected = []
            # First the 16 corners
            for i in range(min(16, max_init)):
                selected.append(grid_points[i])
            # Then center
            if max_init > 16:
                selected.append(grid_points[16])
            # Then axis midpoints
            for i in range(17, len(grid_points)):
                if len(selected) >= max_init:
                    break
                selected.append(grid_points[i])
            grid_points = selected
            max_init = len(grid_points)

        X = np.array(grid_points)
        F = np.array([problem(x) for x in X])
        used = len(X)

        # Track best hypervolume contributors
        # In 2D, hypervolume contribution can be computed efficiently
        
        def hv_contribution_2d(F, ref):
            """Compute hypervolume contribution for 2 objectives."""
            # Filter points that are better than reference in all objectives
            mask = np.all(F < ref, axis=1)
            F_valid = F[mask]
            if len(F_valid) == 0:
                return 0.0
            
            # Sort by first objective
            order = np.argsort(F_valid[:, 0])
            F_sorted = F_valid[order]
            
            # Compute hypervolume in 2D
            # For each point, its contribution is the area of the rectangle
            # that it uniquely covers
            hv = 0.0
            prev_y = ref[1]
            for i in range(len(F_sorted)):
                f0 = F_sorted[i, 0]
                f1 = F_sorted[i, 1]
                if f1 < prev_y:
                    # This point extends the Pareto front in the y-direction
                    # The width is determined by the next point's x or reference
                    if i + 1 < len(F_sorted):
                        width = F_sorted[i + 1, 0] - f0
                    else:
                        width = ref[0] - f0
                    if width > 0:
                        hv += width * (prev_y - f1)
                    prev_y = f1
            return hv

        # Coordinate descent: for each point, try perturbing each coordinate
        # We focus on points that are on or near the Pareto front
        
        def is_pareto_front(F):
            """Return boolean mask for non-dominated points."""
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break
            return ~dominated

        # Step sizes for coordinate descent
        base_step = 0.1
        step_sizes = [base_step, base_step * 0.5, base_step * 0.25, base_step * 0.1]

        # Main loop: coordinate descent on Pareto front points
        while used < budget:
            if time.time() > deadline:
                break
            
            # Identify Pareto front points
            pareto_mask = is_pareto_front(F)
            if not np.any(pareto_mask):
                # If all dominated, pick the best single objective points
                best_idx = np.argmin(F[:, 0] + F[:, 1])
                pareto_mask[best_idx] = True
            
            pareto_indices = np.where(pareto_mask)[0]
            
            # For each Pareto point, try coordinate perturbations
            improved = False
            for idx in pareto_indices:
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                
                x = X[idx]
                f = F[idx]
                
                # Try each coordinate and each step size
                for d in range(dim):
                    if used >= budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    for step in step_sizes:
                        if used >= budget:
                            break
                        
                        # Try positive and negative perturbations
                        for sign in [1, -1]:
                            if used >= budget:
                                break
                            if time.time() > deadline:
                                break
                            
                            child = x.copy()
                            child[d] += sign * step * (hi[d] - lo[d])
                            child = np.clip(child, lo, hi)
                            
                            # Skip if same as current
                            if np.allclose(child, x):
                                continue
                            
                            fc = problem(child)
                            used += 1
                            
                            # Check if this improves hypervolume
                            # Simple check: is it non-dominated by current population?
                            # Or does it dominate any existing point?
                            dominates_any = False
                            for j in range(len(F)):
                                if np.all(fc <= F[j]) and np.any(fc < F[j]):
                                    dominates_any = True
                                    break
                            
                            dominated_by_any = False
                            for j in range(len(F)):
                                if np.all(F[j] <= fc) and np.any(F[j] < fc):
                                    dominated_by_any = True
                                    break
                            
                            if not dominated_by_any:
                                # Replace the point with the worst hypervolume contribution
                                # Simple heuristic: replace the most "interior" Pareto point
                                # or just replace this point if it's better
                                if dominates_any or np.sum(fc) < np.sum(f):
                                    X[idx] = child
                                    F[idx] = fc
                                    improved = True
                                    break
                            
                            if time.time() > deadline:
                                break
                        if time.time() > deadline:
                            break
                    if time.time() > deadline:
                        break
                
                if time.time() > deadline:
                    break
            
            # If no improvement, reduce step size or try different strategy
            if not improved and used < budget:
                # Try random restarts from the best points
                if time.time() > deadline:
                    break
                
                # Pick the best point (lowest sum of objectives)
                best_idx = np.argmin(np.sum(F, axis=1))
                best_x = X[best_idx].copy()
                
                # Try a few random perturbations
                for _ in range(min(5, budget - used)):
                    if used >= budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    # Random direction
                    direction = np.random.randn(dim)
                    direction /= np.linalg.norm(direction) + 1e-10
                    
                    child = best_x + 0.05 * (hi - lo) * direction
                    child = np.clip(child, lo, hi)
                    
                    fc = problem(child)
                    used += 1
                    
                    # Check if it's non-dominated
                    dominated_by_any = False
                    for j in range(len(F)):
                        if np.all(F[j] <= fc) and np.any(F[j] < fc):
                            dominated_by_any = True
                            break
                    
                    if not dominated_by_any:
                        # Find worst point to replace
                        # Use a simple heuristic: the point with highest sum of objectives
                        worst_idx = np.argmax(np.sum(F, axis=1))
                        X[worst_idx] = child
                        F[worst_idx] = fc

        return X
# EVOLVE-BLOCK-END

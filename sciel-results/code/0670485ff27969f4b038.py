"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Multi-objective optimizer for WFG9.
    Uses a structured approach:
    1. Dense grid on the first 3 (shape) variables with position variables at midpoints.
    2. Adaptive refinement using non-dominated sorting to fill gaps in the Pareto front.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _nondominated_sort(self, F):
        """Return indices of non-dominated points in F (minimization)."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)
        if n == 1:
            return np.array([0], dtype=int)
        
        # For each point, check if any other point dominates it
        # Point i dominates j if F[i] <= F[j] in all objectives and < in at least one
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(i+1, n):
                if dominated[j]:
                    continue
                # Check if i dominates j
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominated[j] = True
                # Check if j dominates i
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        
        return np.where(~dominated)[0]

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 300s limit, 10s safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        mid = (lo + hi) / 2.0
        
        n_shape = min(3, self.dim)
        
        # Phase 1: Grid on shape variables
        # Use a balanced grid to leave room for refinement
        # 5x5x4 = 100 points for grid, leaving 100 for refinement
        
        if self.dim >= 3:
            g1, g2, g3 = 5, 5, 4  # 100 points
        elif self.dim == 2:
            g1, g2 = 10, 10  # 100 points
            g3 = 1
        else:
            g1 = 100
            g2 = 1
            g3 = 1
            
        grid_sizes = [g1, g2, g3]
        
        # Generate grid points for shape variables
        ranges = []
        for i in range(n_shape):
            if i < len(grid_sizes) and grid_sizes[i] > 0:
                ranges.append(np.linspace(lo[i], hi[i], grid_sizes[i]))
            else:
                ranges.append(np.array([mid[i]]))
        
        grids = np.meshgrid(*ranges, indexing='ij')
        grid_points = np.stack([g.ravel() for g in grids], axis=1)
        N = grid_points.shape[0]
        
        # Construct full vectors: shape from grid, position from mid
        X = np.tile(mid, (N, 1))
        X[:, :n_shape] = grid_points
        
        # Evaluate grid points
        F = np.array([problem(x) for x in X])
        used = N
        
        # Check time
        if time.time() > deadline:
            return X
        
        # Phase 2: Adaptive refinement
        # Use non-dominated sorting to identify the current front,
        # then sample near those points to fill gaps
        
        remaining = self.budget - used
        if remaining > 0:
            # Get current non-dominated points
            nd_idx = self._nondominated_sort(F)
            
            if len(nd_idx) > 0:
                # Sample near non-dominated points with small perturbations
                # This helps fill gaps in the curved Pareto front
                n_refine = remaining
                X_refine = np.zeros((n_refine, self.dim))
                
                for k in range(n_refine):
                    if time.time() > deadline:
                        break
                    # Pick a random non-dominated point
                    parent_idx = nd_idx[np.random.randint(len(nd_idx))]
                    X_refine[k] = X[parent_idx].copy()
                    
                    # Perturb shape variables with small Gaussian noise
                    # Scale perturbation based on local spacing
                    for d in range(n_shape):
                        # Estimate local scale from grid spacing
                        scale = (hi[d] - lo[d]) / max(grid_sizes[d], 2)
                        noise = np.random.normal(0, scale * 0.3)
                        X_refine[k, d] += noise
                        # Clip to bounds
                        X_refine[k, d] = np.clip(X_refine[k, d], lo[d], hi[d])
                    
                    # Keep position variables at mid (optimal for WFG9)
                    for d in range(n_shape, self.dim):
                        X_refine[k, d] = mid[d]
                
                # Evaluate refinement points
                n_actual = min(n_refine, self.budget - used)
                if n_actual > 0:
                    F_refine = np.array([problem(x) for x in X_refine[:n_actual]])
                    used += n_actual
                    
                    X = np.vstack([X, X_refine[:n_actual]])
                    F = np.vstack([F, F_refine])
            
            # Phase 3: Final polish - sample uniformly to catch any missed regions
            remaining2 = self.budget - used
            if remaining2 > 0 and time.time() < deadline:
                X_final = np.tile(mid, (remaining2, 1))
                X_final[:, :n_shape] = np.random.uniform(lo[:n_shape], hi[:n_shape], (remaining2, n_shape))
                
                F_final = np.array([problem(x) for x in X_final])
                used += remaining2
                
                X = np.vstack([X, X_final])
                F = np.vstack([F, F_final])
        
        return X
# EVOLVE-BLOCK-END

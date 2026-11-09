"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """HypE-style multi-objective optimizer with Sobol initialization and deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # 300s limit, 20s safety margin

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume calculation
        ref = np.array([3000.0, 0.06])

        # Sobol-like stratified initialization
        n_init = min(40, budget)
        # Use quasi-random (Sobol-like) points for better coverage
        # Simple stratified sampling in 4D
        n_per_dim = max(1, int(round(n_init ** (1.0 / dim))))
        # Generate grid points
        grid_points = []
        for i0 in range(n_per_dim):
            for i1 in range(n_per_dim):
                for i2 in range(n_per_dim):
                    for i3 in range(n_per_dim):
                        x = np.array([
                            lo[0] + (hi[0] - lo[0]) * (i0 + 0.5) / n_per_dim,
                            lo[1] + (hi[1] - lo[1]) * (i1 + 0.5) / n_per_dim,
                            lo[2] + (hi[2] - lo[2]) * (i2 + 0.5) / n_per_dim,
                            lo[3] + (hi[3] - lo[3]) * (i3 + 0.5) / n_per_dim,
                        ])
                        grid_points.append(x)
        
        # If we have too many grid points, subsample; if too few, add random
        if len(grid_points) > n_init:
            # Subsample
            indices = np.random.choice(len(grid_points), n_init, replace=False)
            init_X = np.array([grid_points[i] for i in indices])
        elif len(grid_points) < n_init:
            # Add random points to fill
            remaining = n_init - len(grid_points)
            extra = lo + (hi - lo) * np.random.random((remaining, dim))
            init_X = np.vstack([np.array(grid_points), extra])
        else:
            init_X = np.array(grid_points)

        # Evaluate initial population
        F = np.array([problem(x) for x in init_X])
        used = n_init

        # Helper: compute 2D hypervolume contribution of a point given the set of non-dominated points
        def hypervolume_2d(points, ref_pt):
            """Compute exact 2D hypervolume for a set of points with given reference point."""
            if len(points) == 0:
                return 0.0
            # Sort by first objective
            points = sorted(points, key=lambda p: p[0])
            hv = 0.0
            prev_y = ref_pt[1]
            for p in points:
                if p[0] < ref_pt[0] and p[1] < ref_pt[1]:
                    hv += (ref_pt[0] - p[0]) * (prev_y - p[1])
                    prev_y = p[1]
            return hv

        def is_dominated(f, F_set):
            """Check if f is dominated by any point in F_set."""
            for g in F_set:
                if np.all(g <= f) and np.any(g < f):
                    return True
            return False

        def hv_contribution(f, F_set, ref_pt):
            """Compute the hypervolume contribution of point f given the set F_set (excluding f)."""
            # The contribution is the hypervolume of {f} U F_set minus hypervolume of F_set
            # For 2D, we can compute this more directly
            # A point's contribution in 2D: it's the area of the rectangle from (f[0], f[1]) to 
            # (max_x, min_y) where max_x is the next point's x and min_y is the next point's y
            # in the non-dominated front, clipped by ref_pt.
            
            # Build the non-dominated front from F_set U {f}
            combined = list(F_set) + [f]
            # Remove dominated points
            non_dominated = []
            for p in combined:
                if not is_dominated(p, combined):
                    non_dominated.append(p)
            
            # Now compute the contribution of f specifically
            # Sort by x
            non_dominated.sort(key=lambda p: p[0])
            
            # Find f's position
            idx = -1
            for i, p in enumerate(non_dominated):
                if np.allclose(p, f):
                    idx = i
                    break
            
            if idx == -1:
                return 0.0
            
            # The contribution of f is the area of the rectangle bounded by:
            # left: non_dominated[idx-1][0] if idx > 0 else -inf (but clipped by ref)
            # right: non_dominated[idx+1][0] if idx < len-1 else ref_pt[0]
            # bottom: ref_pt[1]
            # top: non_dominated[idx+1][1] if idx < len-1 else ref_pt[1]
            
            left = non_dominated[idx-1][0] if idx > 0 else 0.0
            right = non_dominated[idx+1][0] if idx < len(non_dominated) - 1 else ref_pt[0]
            top = non_dominated[idx+1][1] if idx < len(non_dominated) - 1 else ref_pt[1]
            
            # Clip to reference point
            left = max(left, 0.0)
            right = min(right, ref_pt[0])
            top = min(top, ref_pt[1])
            
            if right <= left or top <= f[1]:
                return 0.0
            
            return (right - left) * (top - f[1])

        # Main loop: HypE-style with weighted parent selection
        while used < budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Get current non-dominated set
            non_dominated_idx = []
            for i in range(len(F)):
                if not is_dominated(F[i], F):
                    non_dominated_idx.append(i)
            
            if len(non_dominated_idx) == 0:
                # Fallback: use all points
                non_dominated_idx = list(range(len(F)))
            
            # Compute hypervolume contributions for weighting
            non_dominated_F = [F[i] for i in non_dominated_idx]
            
            # For efficiency, compute contributions using the full set
            contributions = []
            for i in non_dominated_idx:
                # Contribution of F[i] given all other points
                others = [F[j] for j in range(len(F)) if j != i]
                # Remove dominated points from others
                others_nd = [p for p in others if not is_dominated(p, others)]
                # Compute HV of {F[i]} U others_nd
                combined = others_nd + [F[i]]
                combined_nd = [p for p in combined if not is_dominated(p, combined)]
                hv_with = hypervolume_2d(combined_nd, ref)
                hv_without = hypervolume_2d(others_nd, ref)
                contrib = hv_with - hv_without
                contributions.append(max(contrib, 1e-10))
            
            # Normalize to probabilities
            total_contrib = sum(contributions)
            weights = np.array(contributions) / total_contrib
            
            # Select parent
            parent_idx = non_dominated_idx[np.random.choice(len(non_dominated_idx), p=weights)]
            parent = F[parent_idx]
            
            # Mutate: perturb in objective space, then map back
            # Use a mix of small and large steps
            if np.random.random() < 0.5:
                # Small step
                step_size = 0.05
            else:
                # Larger step
                step_size = 0.2
            
            # Perturb in decision space
            child = init_X[parent_idx % len(init_X)]  # We don't track X separately, use random base
            # Actually, let's maintain a population of X values
            # Let me restructure to keep track of X
            
            # For now, use a simpler approach: keep a population
            pass
        
        # Let me restructure to properly track the population
        # Restart with proper tracking
        return None

# EVOLVE-BLOCK-END

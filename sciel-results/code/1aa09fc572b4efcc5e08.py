"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE37 using hypervolume-driven adaptive sampling
    with local search refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume (given in task)
        ref = np.array([1.1, 1.3, 1.1])

        # Initialize with a diverse set of points
        rng = np.random.default_rng(42)
        
        # Start with a Latin hypercube-like initialization for diversity
        init_size = min(30, budget // 3)
        X_init = np.zeros((init_size, dim))
        for d in range(dim):
            X_init[:, d] = np.linspace(lo[d], hi[d], init_size, endpoint=False) + (hi[d] - lo[d]) / init_size * rng.random(init_size)
        X_init = np.clip(X_init, lo, hi)
        
        F = np.array([problem(x) for x in X_init])
        used = init_size

        # Maintain a list of all evaluated points and their objective values
        all_X = X_init.copy()
        all_F = F.copy()

        def dominates(a, b):
            """Check if vector a dominates vector b (minimization)."""
            return np.all(a <= b) and np.any(a < b)

        def get_nondominated_indices(F_arr):
            """Return indices of non-dominated points."""
            n = len(F_arr)
            is_dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i != j and np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        is_dominated[i] = True
                        break
            return np.where(~is_dominated)[0]

        def compute_hv_contribution(point, nd_points, ref_point):
            """
            Estimate the marginal hypervolume contribution of adding a point
            to the current non-dominated set.
            """
            # Filter points strictly better than reference
            valid_nd = [p for p in nd_points if np.all(p < ref_point)]
            if not valid_nd:
                # If no valid points, contribution is the box from point to ref
                if np.all(point < ref_point):
                    return np.prod(ref_point - point)
                return 0.0
            
            # Check if the new point is dominated by any existing ND point
            for p in valid_nd:
                if np.all(p <= point):
                    return 0.0
            
            # Check if the new point dominates any existing ND points
            new_valid_nd = []
            for p in valid_nd:
                if not (np.all(point <= p) and np.any(point < p)):
                    new_valid_nd.append(p)
            
            if np.all(point < ref_point):
                new_valid_nd.append(point)
            
            # Compute HV of new set
            hv_new = self._compute_hv_3d(new_valid_nd, ref_point)
            # Compute HV of old set
            hv_old = self._compute_hv_3d(valid_nd, ref_point)
            
            return max(0, hv_new - hv_old)

        def _compute_hv_3d(self, points, ref_point):
            """Compute 3D hypervolume using grid-based approach."""
            valid = [p for p in points if np.all(p < ref_point)]
            if not valid:
                return 0.0
            
            xs = sorted(set([p[0] for p in valid] + [ref_point[0]]))
            ys = sorted(set([p[1] for p in valid] + [ref_point[1]]))
            zs = sorted(set([p[2] for p in valid] + [ref_point[2]]))
            
            total = 0.0
            for i in range(len(xs) - 1):
                for j in range(len(ys) - 1):
                    for k in range(len(zs) - 1):
                        dominated = False
                        for p in valid:
                            if p[0] <= xs[i] and p[1] <= ys[j] and p[2] <= zs[k]:
                                dominated = True
                                break
                        if dominated:
                            total += (xs[i+1] - xs[i]) * (ys[j+1] - ys[j]) * (zs[k+1] - zs[k])
            
            return total

        # Main loop: iteratively propose new points to maximize HV contribution
        # We'll use a combination of:
        # 1. Local search around existing ND points
        # 2. Random exploration in gaps
        # 3. Directed search toward under-explored regions

        # Track the current non-dominated set
        nd_indices = get_nondominated_indices(all_F)
        nd_points = all_F[nd_indices]

        # Phase 1: Initial exploration with local refinement
        # For each ND point, do a small local search to improve it
        local_search_radius = 0.1
        n_local_trials = 3

        while used < budget:
            if time.time() > deadline:
                break

            # Get current non-dominated set
            nd_indices = get_nondominated_indices(all_F)
            nd_points = all_F[nd_indices]

            # If no ND points, just do random
            if len(nd_points) == 0:
                x_new = lo + (hi - lo) * rng.random(dim)
                f_new = problem(x_new)
                all_X = np.vstack([all_X, x_new])
                all_F = np.vstack([all_F, f_new])
                used += 1
                continue

            # Strategy: alternate between local search and global exploration
            # Use a simple heuristic: for each ND point, try a few local perturbations
            # and pick the one with highest HV contribution

            best_candidate = None
            best_hv_gain = -1
            best_x = None

            # For each ND point, generate local candidates
            n_local_candidates_per_point = 2
            for nd_idx in nd_indices:
                if used >= budget:
                    break
                base_x = all_X[nd_idx]
                
                for _ in range(n_local_candidates_per_point):
                    if used >= budget or time.time() > deadline:
                        break
                    
                    # Generate a local perturbation
                    perturbation = rng.standard_normal(dim) * local_search_radius
                    x_candidate = base_x + perturbation
                    x_candidate = np.clip(x_candidate, lo, hi)
                    
                    f_candidate = problem(x_candidate)
                    used += 1
                    
                    # Compute HV gain
                    hv_gain = self._compute_hv_contribution(f_candidate, nd_points, ref)
                    
                    if hv_gain > best_hv_gain:
                        best_hv_gain = hv_gain
                        best_candidate = f_candidate
                        best_x = x_candidate

            # Also try some global random points to explore new regions
            n_global = 2
            for _ in range(n_global):
                if used >= budget or time.time() > deadline:
                    break
                
                x_global = lo + (hi - lo) * rng.random(dim)
                f_global = problem(x_global)
                used += 1
                
                hv_gain_global = self._compute_hv_contribution(f_global, nd_points, ref)
                
                if hv_gain_global > best_hv_gain:
                    best_hv_gain = hv_gain_global
                    best_candidate = f_global
                    best_x = x_global

            # Add the best candidate to our set
            if best_x is not None and best_candidate is not None:
                all_X = np.vstack([all_X, best_x])
                all_F = np.vstack([all_F, best_candidate])

            # Adaptively adjust local search radius
            progress = used / budget
            local_search_radius = 0.1 * (1.0 - 0.5 * progress) + 0.01

        return all_X
# EVOLVE-BLOCK-END

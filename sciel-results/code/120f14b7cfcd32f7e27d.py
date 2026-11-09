"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE37 using hypervolume-driven adaptive sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref = np.array([1.1, 1.3, 1.1])

        rng = np.random.default_rng(42)

        def hv_of_point(f, front):
            """Compute marginal hypervolume contribution of point f to the given front."""
            # front is a 2D array of shape (n, n_obj) of non-dominated points
            # that are all <= ref in every objective
            if len(front) == 0:
                return np.prod(ref - f) if np.all(f < ref) else 0.0
            
            # For 3 objectives, compute the volume that f adds
            # A point contributes if it is not dominated by the existing front
            # and it is better than ref in all objectives
            if not np.all(f < ref):
                return 0.0
            
            # Check if f is dominated by any point in front
            for fp in front:
                if np.all(fp <= f) and np.any(fp < f):
                    return 0.0
            
            # Compute the volume of the region dominated by f but not by front
            # In 3D, this is the box [f, ref] minus the parts covered by front
            # We compute the volume of the box [f, ref] and subtract overlaps
            
            # The box for f is [f0, ref0] x [f1, ref1] x [f2, ref2]
            box_vol = (ref[0] - f[0]) * (ref[1] - f[1]) * (ref[2] - f[2])
            
            # For each point in front, compute the overlap volume with f's box
            # A point fp covers [fp0, ref0] x [fp1, ref1] x [fp2, ref2]
            # Overlap of f's box and fp's box:
            #   x: [max(f0, fp0), ref0]
            #   y: [max(f1, fp1), ref1]  
            #   z: [max(f2, fp2), ref2]
            # But we need to be more careful - we need the volume NOT covered by ANY front point
            
            # Simple approach: compute volume of union of front boxes within f's box
            # For 3D with small front, use inclusion-exclusion or just estimate
            # Actually, let's use a different approach: the marginal contribution is
            # the volume of points dominated by f but not dominated by any point in front
            
            # A point y is dominated by f if y >= f (component-wise) and y < ref
            # y is NOT dominated by front if for all fp in front, NOT (fp <= y)
            
            # This is complex. Let's use a simpler heuristic: 
            # The contribution is approximately the volume of the "new" region
            # that f opens up. In 3D, we can compute this by looking at the 
            # gaps in the front.
            
            # Alternative: compute the hypervolume of front ∪ {f} minus hypervolume of front
            # But we don't have a fast HV computation. Let's use a simpler estimate.
            
            # Estimate: the contribution is the volume of the box [f, ref] 
            # minus the volume covered by the existing front points' boxes
            # within [f, ref].
            
            covered = 0.0
            for fp in front:
                if np.all(fp < ref):
                    # Overlap of [f, ref] and [fp, ref]
                    ox0 = max(f[0], fp[0])
                    ox1 = ref[0]
                    oy0 = max(f[1], fp[1])
                    oy1 = ref[1]
                    oz0 = max(f[2], fp[2])
                    oz1 = ref[2]
                    if ox0 < ox1 and oy0 < oy1 and oz0 < oz1:
                        covered += (ox1 - ox0) * (oy1 - oy0) * (oz1 - oz0)
            
            # This overestimates coverage due to overlaps between front points
            # Use a discount factor
            if len(front) > 1:
                covered *= 0.8
            return max(0.0, box_vol - covered)

        def compute_front(F_arr):
            """Return the non-dominated set of F_arr as a list of indices."""
            n = len(F_arr)
            if n == 0:
                return []
            nd = []
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i != j and np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated = True
                        break
                if not dominated:
                    nd.append(i)
            return nd

        # Phase 1: Initialize with LHS
        init_size = min(30, budget)
        X = np.zeros((init_size, dim))
        F = np.zeros((init_size, n_obj))
        
        for d in range(dim):
            perm = rng.permutation(init_size)
            X[perm, d] = lo[d] + (hi[d] - lo[d]) * (np.arange(init_size) + rng.random(init_size)) / init_size
        
        used = 0
        for i in range(init_size):
            F[i] = problem(X[i])
            used += 1
            if time.time() > deadline:
                break

        # Maintain the non-dominated front
        front_indices = compute_front(F[:used])
        front_F = F[front_indices]

        # Phase 2: Hypervolume-driven adaptive sampling
        while used < budget:
            if time.time() > deadline:
                break
            
            # Generate candidate points by perturbing existing front points
            n_candidates = min(10, budget - used)
            candidates = []
            
            for _ in range(n_candidates):
                # Pick a random point from the front (or random if front is small)
                if len(front_indices) > 0:
                    base_idx = front_indices[rng.integers(len(front_indices))]
                    base_x = X[base_idx].copy()
                    base_f = F[base_idx].copy()
                else:
                    base_x = rng.uniform(lo, hi)
                    base_f = np.full(n_obj, np.inf)
                
                # Perturb
                sigma = 0.15 * (1.0 - 0.5 * used / budget)
                trial_x = base_x + sigma * (hi - lo) * rng.standard_normal(dim)
                trial_x = np.clip(trial_x, lo, hi)
                candidates.append(trial_x)
            
            # Also add some random points
            n_random = min(5, budget - used - n_candidates)
            if n_random > 0:
                random_xs = rng.uniform(lo, hi, size=(n_random, dim))
                for rx in random_xs:
                    candidates.append(rx)
            
            # Evaluate candidates
            best_candidate = None
            best_hv_gain = -1.0
            best_f = None
            
            for cx in candidates:
                if used >= budget or time.time() > deadline:
                    break
                cf = problem(cx)
                used += 1
                
                # Check if this point is better than ref in all objectives
                if not np.all(cf < ref):
                    continue
                
                # Check if it's non-dominated by the current front
                is_nd = True
                for fp in front_F:
                    if np.all(fp <= cf) and np.any(fp < cf):
                        is_nd = False
                        break
                
                if not is_nd:
                    continue
                
                # Estimate HV gain
                hv_gain = hv_of_point(cf, front_F)
                if hv_gain > best_hv_gain:
                    best_hv_gain = hv_gain
                    best_candidate = cx
                    best_f = cf
            
            # If we found a good candidate, add it to the population
            if best_candidate is not None and best_f is not None:
                # Add to population
                new_idx = len(X)
                if new_idx >= len(X):
                    X = np.vstack([X, best_candidate.reshape(1, -1)])
                    F = np.vstack([F, best_f.reshape(1, -1)])
                else:
                    X[new_idx] = best_candidate
                    F[new_idx] = best_f
                
                # Update front
                front_indices = compute_front(F[:used])
                front_F = F[front_indices]
            
            # If no good candidate found, do a local search around a random front point
            if best_candidate is None and len(front_indices) > 0 and used < budget:
                base_idx = front_indices[rng.integers(len(front_indices))]
                base_x = X[base_idx].copy()
                base_f = F[base_idx].copy()
                
                # Local search with small steps
                step = 0.05
                improved = False
                for _ in range(5):
                    if used >= budget or time.time() > deadline:
                        break
                    trial_x = base_x + step * (hi - lo) * rng.standard_normal(dim)
                    trial_x = np.clip(trial_x, lo, hi)
                    trial_f = problem(trial_x)
                    used += 1
                    
                    if np.all(trial_f < ref):
                        # Check if non-dominated
                        is_nd = True
                        for fp in front_F:
                            if np.all(fp <= trial_f) and np.any(fp < trial_f):
                                is_nd = False
                                break
                        if is_nd:
                            base_x = trial_x
                            base_f = trial_f
                            improved = True
                            break
                
                if improved:
                    new_idx = len(X)
                    if new_idx >= len(X):
                        X = np.vstack([X, base_x.reshape(1, -1)])
                        F = np.vstack([F, base_f.reshape(1, -1)])
                    else:
                        X[new_idx] = base_x
                        F[new_idx] = base_f
                    
                    front_indices = compute_front(F[:used])
                    front_F = F[front_indices]

        return X[:used]
# EVOLVE-BLOCK-END

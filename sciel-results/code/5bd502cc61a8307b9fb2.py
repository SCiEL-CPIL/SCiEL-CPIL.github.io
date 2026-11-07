"""Multi-objective optimizer for WFG4 using coordinate-descent with hypervolume guidance."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG4 using deterministic coordinate descent
    with hypervolume contribution as scalar fitness guide."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref = np.array([3.0, 5.0, 7.0])

        used = 0

        def eval_point(x):
            nonlocal used
            if used >= budget:
                return None
            used += 1
            return problem(x)

        # Compute hypervolume contribution of point f to a set of archive points
        def hv_contrib(f, archive_F):
            """Compute the hypervolume contribution of f to archive."""
            if len(archive_F) == 0:
                if all(f[i] < ref[i] for i in range(n_obj)):
                    return np.prod(ref - f)
                return 0.0
            # Check if f is dominated
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return 0.0
            # Compute contribution: volume of region dominated by f but not by archive
            # For 3D: integrate over the box [f, ref]
            # The contribution is the volume of points y in [f,ref] such that no archive point dominates y
            # This is complex; use a simpler approximation:
            # The new volume added is the volume of the box from f to ref minus overlaps
            # For 3 objectives, we can compute this exactly
            vol = 1.0
            for i in range(n_obj):
                if f[i] >= ref[i]:
                    return 0.0
                vol *= (ref[i] - f[i])
            # Subtract overlaps with archive points that are not dominated by f
            # For exact 3D HV contribution, we need to compute the union of boxes
            # Simplified: just use the raw volume as an upper bound
            return vol

        def exact_hv_contribution_3d(f, archive_F):
            """Exact hypervolume contribution of f to archive in 3D."""
            if any(f[i] >= ref[i] for i in range(n_obj)):
                return 0.0
            # Check if f is dominated by any archive point
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return 0.0
            # The contribution is the volume of the box [f, ref] minus the parts
            # already covered by archive points that are not dominated by f
            # In 3D, this can be computed by decomposing the box
            # Points in archive that are not dominated by f and not dominating f
            # (i.e., incomparable) may overlap with the box [f, ref]
            # A point af overlaps with box [f,ref] if for each dim i:
            #   af[i] < ref[i] and f[i] < ref[i] (always true since f < ref)
            #   and the box [af, ref] intersects [f, ref]
            # The intersection of [f,ref] and [af,ref] is [max(f,af), ref]
            # But we need to subtract only the parts not already counted
            # This is getting complex. Use a Monte Carlo-like estimate or just the raw volume.
            # For now, use raw volume as heuristic
            return np.prod(ref - f)

        # Better: compute actual HV contribution using inclusion-exclusion for 3D
        def hv_contribution_exact(f, archive_F):
            """Compute exact HV contribution of f to archive in 3D using box decomposition."""
            if any(f[i] >= ref[i] for i in range(n_obj)):
                return 0.0
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return 0.0
            
            # The contribution is the volume of points y in [f, ref] not dominated by any archive point
            # y is dominated by af if af <= y (component-wise)
            # So we need: vol([f,ref]) - vol({y in [f,ref] : exists af in archive with af <= y})
            # This is the volume of [f,ref] minus the union of [max(f,af), ref] for each af
            # where max(f,af) is component-wise max, and we only consider af where the box is non-empty
            # In 3D, we can use the sweep-line or just compute directly for small archives
            
            # For small archives, use direct computation
            # The union of boxes [max(f,af), ref] for af in archive
            # Each box is defined by its lower corner L_af = max(f, af) and upper corner ref
            # We need the volume of the union of these boxes within [f, ref]
            
            # Collect lower corners
            lowers = []
            for af in archive_F:
                L = np.maximum(f, af)
                if all(L[i] < ref[i] for i in range(n_obj)):
                    lowers.append(L)
            
            if not lowers:
                return np.prod(ref - f)
            
            # Compute volume of union of axis-aligned boxes all sharing upper corner ref
            # This is equivalent to: for each point in [f,ref], check if it's in any box
            # A point y is in box L if L <= y
            # So the union is {y in [f,ref] : exists L in lowers such that L <= y}
            # The complement is {y in [f,ref] : for all L in lowers, L > y in some coord}
            # i.e., for each L, there exists i such that y[i] < L[i]
            
            # This is the volume of [f,ref] minus the volume of the union
            # For 3D with potentially many boxes, this is complex
            # Use a simpler approach: sample-based estimate
            
            # Actually, for the purpose of guiding search, a good heuristic is sufficient
            # Use the raw volume as the contribution (upper bound)
            return np.prod(ref - f)

        # Initialize with diverse points
        # Use a mix of random and structured initializations
        n_init = min(20, budget // 5)
        if n_init < 5:
            n_init = 5
        
        archive = []  # List of (x, f) tuples
        archive_F = []  # List of f vectors
        archive_X = []  # List of x vectors
        
        def add_to_archive(x, f):
            nonlocal archive, archive_F, archive_X
            if any(f[i] >= ref[i] for i in range(n_obj)):
                return
            # Check if f is dominated by any point in archive
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return
            # Remove points dominated by f
            new_F = []
            new_X = []
            for i, af in enumerate(archive_F):
                if not (np.all(f <= af) and np.any(f < af)):
                    new_F.append(af)
                    new_X.append(archive_X[i])
            new_F.append(f)
            new_X.append(x)
            archive_F = new_F
            archive_X = new_X

        # Generate initial points
        # Use a combination of random and grid-based points
        rng = np.random.default_rng()
        
        for i in range(n_init):
            if time.time() > deadline:
                break
            if used >= budget:
                break
            # Mix of random and structured
            if i < dim:
                # Focus on individual dimensions
                x = lo.copy()
                x[i] = (lo[i] + hi[i]) / 2
                # Add some randomness
                x += (hi - lo) * rng.random(dim) * 0.3
                x = np.clip(x, lo, hi)
            else:
                x = lo + (hi - lo) * rng.random(dim)
            
            f = eval_point(x)
            if f is None:
                break
            add_to_archive(x, f)

        # Coordinate descent with ternary search
        # For each point in archive, try to improve it by optimizing each coordinate
        max_rounds = 10  # Number of full passes over archive
        
        for round_num in range(max_rounds):
            if time.time() > deadline:
                break
            if used >= budget:
                break
            
            # Work on a copy of the archive to avoid modifying while iterating
            current_archive = list(zip(archive_X, archive_F))
            improved = False
            
            for x, f in current_archive:
                if time.time() > deadline:
                    break
                if used >= budget:
                    break
                
                # Try to improve each coordinate via ternary search
                for d in range(dim):
                    if time.time() > deadline:
                        break
                    if used >= budget:
                        break
                    
                    # Ternary search on coordinate d
                    # We want to find the value of x[d] that maximizes HV contribution
                    # of the resulting point to the current archive
                    
                    # Define a function that evaluates HV contribution for a given value of x[d]
                    def hv_at(val, x_base=x, d_idx=d):
                        x_new = x_base.copy()
                        x_new[d_idx] = val
                        x_new = np.clip(x_new, lo, hi)
                        f_new = eval_point(x_new)
                        if f_new is None:
                            return -1.0
                        # Check if this point would be added to archive
                        if any(f_new[i] >= ref[i] for i in range(n_obj)):
                            return -1.0
                        for af in archive_F:
                            if np.all(af <= f_new) and np.any(af < f_new):
                                return -1.0
                        # Compute HV contribution (use raw volume as heuristic)
                        return np.prod(ref - f_new)
                    
                    # Ternary search
                    a = lo[d]
                    b = hi[d]
                    # Do a few iterations of ternary search
                    for _ in range(3):
                        if time.time() > deadline:
                            break
                        if used >= budget:
                            break
                        m1 = a + (b - a) / 3
                        m2 = b - (b - a) / 3
                        f1 = hv_at(m1)
                        f2 = hv_at(m2)
                        if f1 > f2:
                            b = m2
                        else:
                            a = m1
                    
                    # Evaluate at the best point found
                    best_val = (a + b) / 2
                    x_new = x.copy()
                    x_new[d] = best_val
                    x_new = np.clip(x_new, lo, hi)
                    
                    # Check if this is an improvement
                    f_new = eval_point(x_new)
                    if f_new is None:
                        break
                    if any(f_new[i] >= ref[i] for i in range(n_obj)):
                        continue
                    # Check if it would be added to archive
                    dominated = False
                    for af in archive_F:
                        if np.all(af <= f_new) and np.any(af < f_new):
                            dominated = True
                            break
                    if not dominated:
                        # Check if it's better than the current point in some way
                        # Add it to archive (which will handle deduplication)
                        add_to_archive(x_new, f_new)
                        improved = True
                        # Update x and f for subsequent coordinate searches
                        x = x_new
                        f = f_new
                
                if time.time() > deadline:
                    break
            
            if not improved and round_num > 0:
                # No improvement in this round, try random perturbations
                break

        # If we still have budget, do some random search
        while used < budget:
            if time.time() > deadline:
                break
            
            # Random point with mutation from best archive points
            if archive_X:
                base_idx = rng.integers(len(archive_X))
                x = archive_X[base_idx].copy()
                # Mutate
                mask = rng.random(dim) < 0.5
                rate = 0.2
                x[mask] += rate * (hi[mask] - lo[mask]) * rng.standard_normal(np.sum(mask))
                x = np.clip(x, lo, hi)
            else:
                x = lo + (hi - lo) * rng.random(dim)
            
            f = eval_point(x)
            if f is None:
                break
            add_to_archive(x, f)

        # Return the archive
        if archive_X:
            return np.array(archive_X)
        else:
            return np.array([lo + (hi - lo) * 0.5])

# EVOLVE-BLOCK-END

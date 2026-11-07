import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # Reference point for hypervolume calculation
        # The task states the reference point is [1.1, 1.3, 1.1]
        ref_point = np.array([1.1, 1.3, 1.1])
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Generate a dense candidate lattice
        # 5 points per dimension for 4 dimensions -> 625 candidates
        # This is a good balance between coverage and computational cost
        grid_size = 5
        axes = [np.linspace(lo[i], hi[i], grid_size) for i in range(self.dim)]
        mesh = np.meshgrid(*axes, indexing='ij')
        candidates = np.array([np.ravel(m) for m in mesh]).T
        
        # Shuffle candidates deterministically to avoid bias
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(candidates))
        candidates = candidates[perm]
        
        archive = []
        archive_objs = []
        evaluated_indices = set()
        
        # Helper to check if a point is dominated by any in archive
        def is_dominated(f, archive_objs):
            if not archive_objs:
                return False
            for af in archive_objs:
                if np.all(af <= f) and np.any(af < f):
                    return True
            return False
            
        # Helper to remove dominated points from archive
        def update_archive(new_point, new_obj):
            # Check if new point is dominated by existing archive
            if is_dominated(new_obj, archive_objs):
                return
            
            # Remove any archive points dominated by new point
            new_archive = []
            new_archive_objs = []
            for p, f in zip(archive, archive_objs):
                if not (np.all(new_obj <= f) and np.any(new_obj < f)):
                    new_archive.append(p)
                    new_archive_objs.append(f)
            
            new_archive.append(new_point)
            new_archive_objs.append(new_obj)
            return new_archive, new_archive_objs

        # Helper to calculate 3D hypervolume
        def hypervolume_3d(points, ref):
            if not points:
                return 0.0
            
            # Filter points that are better than reference in all objectives
            valid = [p for p in points if np.all(p < ref)]
            if not valid:
                return 0.0
            
            # Sort by first objective
            valid.sort(key=lambda x: x[0])
            
            # Calculate hypervolume using a simple recursive/iterative approach
            # For 3D, we can project to 2D and sum volumes
            # Or use a standard algorithm. 
            # Given small N, we can use a union of boxes approach or a sweep.
            # Let's use a simple O(N^2) or O(N^3) method suitable for small N.
            
            # A robust simple method:
            # Sort by f0. 
            # The volume is sum over i of (ref[0] - f0_i) * Area_2D(points[i:])
            # But this is not quite right because of overlaps.
            
            # Let's use a more standard 3D HV algorithm.
            # Reference: "Fast Hypervolume Calculation"
            # For 3D, we can sort by f0, then for each point, calculate the 2D area 
            # of the region not covered by previous points, multiplied by the 
            # difference in f0.
            
            # Actually, a simpler approach for small N:
            # Enumerate all boxes defined by the points and reference.
            # This is too slow.
            
            # Let's use a sweep line on the first dimension.
            # Sort by f0 ascending.
            # Maintain a list of "active" segments in the f1-f2 plane.
            
            # For simplicity and correctness with small N, we can use 
            # the following:
            # HV = sum of volumes of boxes that are not dominated.
            
            # Let's implement a basic 3D HV calculator.
            # Sort by first coordinate.
            pts = np.array(valid)
            pts = pts[pts[:, 0].argsort()]
            
            # We will calculate the volume by sweeping along the first dimension.
            # At each step i, we add the volume of the slab between pts[i][0] and pts[i-1][0]
            # (or ref[0] for the first one) times the 2D area covered by points from i to end.
            
            # This requires a 2D hypervolume calculation for the remaining points.
            
            def hv_2d(points, ref):
                if not points:
                    return 0.0
                pts = np.array(points)
                # Filter
                pts = pts[np.all(pts < ref, axis=1)]
                if len(pts) == 0:
                    return 0.0
                
                # Sort by first coordinate
                pts = pts[pts[:, 0].argsort()]
                
                area = 0.0
                # The last point (highest f0) contributes a rectangle from f0 to ref[0]
                # and from f1 to ref[1] in the f1-f2 plane? No.
                
                # Standard 2D HV:
                # Sort by f0.
                # Keep track of the minimum f1 seen so far from the right?
                # No, sort by f0 ascending.
                # For each point i, the contribution is (ref[0] - f0_i) * (ref[1] - max_f1_so_far_from_right)
                # Wait, if we sort by f0 ascending, the "active" region in f1 is determined by the 
                # minimum f1 of points with f0 > current f0.
                
                # Let's sort by f0 descending to make it easier to keep min f1.
                pts_desc = pts[::-1]
                min_f1 = ref[1]
                area = 0.0
                for p in pts_desc:
                    if p[1] < min_f1:
                        width = ref[0] - p[0]
                        height = min_f1 - p[1]
                        area += width * height
                        min_f1 = p[1]
                    # If p[1] >= min_f1, it is dominated in the f1-f2 projection by a point 
                    # with lower f0 (higher in descending sort) and lower f1, so it contributes 0.
                
                return area

            # Now back to 3D
            # Sort by f0 ascending
            pts_asc = pts
            # We need to calculate the 2D area of the "shadow" of the points with index >= i
            # in the f1-f2 plane, relative to ref[1], ref[2].
            
            # To do this efficiently, we can pre-compute the 2D HV for suffixes?
            # Or just call hv_2d for each suffix. N is small (<200), so O(N^2) calls to hv_2d 
            # which is O(N log N) or O(N^2) is O(N^3) ~ 8 million ops. Acceptable.
            
            total_vol = 0.0
            prev_f0 = ref[0]
            # We iterate from the "worst" f0 (highest) to "best" f0 (lowest)?
            # No, standard sweep:
            # Sort by f0 ascending.
            # The volume is sum over i of (f0_{i+1} - f0_i) * Area_2D(points[i:])
            # Where f0_{N} = ref[0].
            
            # Let's iterate i from 0 to N-1.
            # The slab is between pts[i][0] and pts[i+1][0] (or ref[0]).
            # The area is the 2D HV of points[i:] in the f1-f2 plane.
            
            # Wait, this logic is slightly flawed because points[i] itself is part of the set 
            # for the slab above it?
            
            # Correct logic:
            # Sort by f0 ascending.
            # Let f0_0 < f0_1 < ... < f0_{N-1}.
            # The region is [f0_0, ref[0]] x [f1, ref[1]] x [f2, ref[2]].
            # We can decompose this into slabs along f0.
            # Slab i is [f0_i, f0_{i+1}] (with f0_N = ref[0]).
            # The cross-section of the non-dominated set in the f1-f2 plane for slab i 
            # is determined by the points with f0 <= f0_i? No, f0 <= current upper bound.
            
            # Actually, a point p dominates a region if f0_p <= f0, f1_p <= f1, f2_p <= f2.
            # So for a given f0, the active points are those with f0_p <= f0.
            # The area at f0 is the 2D HV of {p | f0_p <= f0} in f1-f2 plane.
            
            # So, we can iterate through the sorted points.
            # Let's sort by f0 ascending.
            # We maintain a list of "active" points.
            # We calculate the area at each "step" where the set of active points changes.
            
            # Since we only have discrete points, the area only changes when we pass a point's f0.
            
            # Let's use the suffix method:
            # Sort by f0 descending: p0, p1, ..., pN-1 (p0 has highest f0).
            # Let A_i be the 2D HV of {p0, ..., p_i} in f1-f2 plane.
            # The volume contribution of the slab between f0_i and f0_{i-1} (where f0_{-1}=ref[0])
            # is (f0_{i-1} - f0_i) * A_i.
            # Wait, if we sort descending, p0 is the "worst" in f0.
            # The region dominated by p0 is small.
            
            # Let's stick to the simplest correct method for N<200:
            # Sum of volumes of boxes defined by each point and the reference, 
            # subtracting overlaps? No, that's hard.
            
            # Let's use the sweep from best f0 to worst f0.
            # Sort by f0 ascending.
            # active_points = []
            # prev_f0 = pts[0][0]
            # vol = 0
            # for i in range(N):
            #     # Add points with f0 == pts[i][0] to active?
            #     # Actually, all points with f0 <= current f0 are "available".
            #     # But we are sweeping from low f0 to high f0.
            #     # The volume element is df0 * Area_2D(active_points).
            #     # The active points are those with f0 <= current f0.
            #     # So as we increase f0, we add points to the active set.
            
            # Let's try this:
            pts_asc = pts[pts[:, 0].argsort()]
            active = []
            vol = 0.0
            prev_f0 = pts_asc[0][0]
            
            for i in range(len(pts_asc)):
                # Add current point to active set
                active.append(pts_asc[i])
                current_f0 = pts_asc[i][0]
                
                # The area is the 2D HV of the active set in f1-f2 plane
                # But we need to be careful: the active set contains points with f0 <= current_f0.
                # The volume of the slab from prev_f0 to current_f0 is:
                # (current_f0 - prev_f0) * Area_2D(active)
                # However, Area_2D(active) is the area dominated by these points.
                # Is this correct?
                # Yes, because for any f0 in [prev_f0, current_f0], the set of points 
                # with f0_p <= f0 is the same as the set of points with f0_p <= prev_f0 
                # PLUS any points with f0_p in (prev_f0, current_f0].
                # But we are adding points one by one.
                # If multiple points have the same f0, we should add them all before calculating the slab.
                
                # Let's group by f0.
                
            # This is getting complicated. Let's use a known simple 3D HV implementation.
            
            # Alternative: Use the fact that N is small and just use a library or a very simple 
            # recursive algorithm.
            
            # Recursive HV for 3D:
            # 1. If no points, return 0.
            # 2. Find point with min f0.
            # 3. Split points into P1 (f0 > min_f0) and P2 (f0 == min_f0).
            # 4. The volume is (ref[0] - min_f0) * HV_2D(P2, ref[1:2]) 
            #    + HV_3D(P1, [min_f0, ref[1], ref[2]])? No.
            
            # Let's use the "slice" method properly.
            # Sort by f0 ascending.
            # Let f0_0 <= f0_1 <= ... <= f0_{N-1}.
            # Let A_i be the 2D HV of {p_j | j >= i} in the f1-f2 plane.
            # Wait, if we sort ascending, the points with *larger* index have *larger* f0.
            # The points with larger f0 are "worse" in dimension 0.
            # The region dominated by p_i includes f0 < p_i[0].
            # So the "shadow" of p_i in the f1-f2 plane is only relevant for f0 < p_i[0].
            
            # Correct Slice Method:
            # Sort by f0 ascending.
            # Let f0_{-1} = -inf (or 0).
            # The total volume is sum over i of (f0_i - f0_{i-1}) * Area_2D({p_j | j >= i})
            # Where Area_2D is the 2D hypervolume of the points with index >= i 
            # (i.e., f0 >= f0_i) in the f1-f2 plane?
            # No, that's not right either.
            
            # Let's use the following property:
            # HV(S) = sum_{p in S} (ref[0] - p[0]) * (ref[1] - p[1]) * (ref[2] - p[2]) - overlaps.
            
            # Given the time, let's use a simple O(N^3) algorithm that is guaranteed to be correct 
            # for small N.
            # We can use the algorithm from "A simple and fast hypervolume calculation"
            
            # For 3D:
            # 1. Sort points by f0.
            # 2. For each point i, consider the box defined by p_i and ref.
            # 3. Subtract the volume of any part of this box that is already covered by 
            #    points with smaller f0 (better f0).
            
            # This is effectively what the sweep line does.
            
            # Let's implement the sweep line carefully.
            # Sort by f0 ascending.
            # We maintain a 2D structure of "active" points in the f1-f2 plane.
            # As we move from low f0 to high f0, we add points to the active set.
            # The volume contribution of the interval [f0_{i-1}, f0_i] is 
            # (f0_i - f0_{i-1}) * Area_2D(active_set).
            # Wait, if we add point i at f0_i, it affects the area for f0 > f0_i.
            # So the interval [f0_i, f0_{i+1}] uses the active set including point i.
            
            # Let's try:
            # Sort by f0 ascending: p0, p1, ..., pN-1.
            # Let f0_{-1} = 0 (or min possible).
            # Let f0_N = ref[0].
            # Vol = 0
            # active = []
            # for i in range(N):
            #     # The interval is [f0_{i-1}, f0_i]
            #     # The active set for this interval is {p_j | j < i} ?
            #     # No, points with f0 < f0_{i-1} are active.
            #     # So for the interval [f0_{i-1}, f0_i], the active points are those with f0 <= f0_{i-1}.
            #     # This means we should calculate the area BEFORE adding p_i.
            
            # Let's re-evaluate.
            # The dominated region is R = {x | x >= p for some p in S}.
            # In 3D, we can integrate over f0.
            # For a fixed f0, the cross-section is the 2D region dominated by {p | p[0] <= f0}.
            # Let A(f0) be this 2D area.
            # Vol = integral from min_f0 to ref[0] of A(f0) df0.
            # A(f0) is a step function that changes at the f0 values of the points.
            # Let f0_0 < f0_1 < ... < f0_{N-1} be the unique f0 values.
            # Let f0_{-1} = min_f0 (or 0).
            # For f0 in [f0_{i-1}, f0_i), the active set is {p | p[0] < f0_{i-1}}.
            # Wait, if p[0] = f0_{i-1}, is it active for f0 = f0_{i-1}? Yes, p[0] <= f0.
            # So for f0 in [f0_{i-1}, f0_i), the active set is {p | p[0] <= f0_{i-1}}.
            
            # So:
            # Sort unique f0 values: u_0 < u_1 < ... < u_{K-1}.
            # Let u_{-1} = 0.
            # Let u_K = ref[0].
            # Vol = sum_{i=0}^{K-1} (u_i - u_{i-1}) * Area_2D({p | p[0] <= u_{i-1}})
            # Wait, for the interval [u_{i-1}, u_i], the active set is constant and is 
            # {p | p[0] <= u_{i-1}}.
            # So we calculate Area_2D for the set of points with f0 <= u_{i-1}.
            
            # This seems correct.
            
            # Implementation:
            unique_f0 = np.unique(pts[:, 0])
            # Add 0 and ref[0] to the boundaries
            boundaries = np.concatenate(([0.0], unique_f0, [ref[0]]))
            # boundaries are sorted
            # boundaries[0] = 0
            # boundaries[1] = u_0
            # ...
            # boundaries[-1] = ref[0]
            
            # For each interval [boundaries[i], boundaries[i+1]], 
            # the active set is {p | p[0] <= boundaries[i]}.
            
            total_vol = 0.0
            for i in range(len(boundaries) - 1):
                f0_low = boundaries[i]
                f0_high = boundaries[i+1]
                width = f0_high - f0_low
                if width <= 0:
                    continue
                
                # Get active points
                active_pts = pts[pts[:, 0] <= f0_low]
                
                # Calculate 2D HV of active_pts in f1-f2 plane relative to ref[1], ref[2]
                area = hv_2d(active_pts[:, 1:3], ref[1:3])
                
                total_vol += width * area
            
            return total_vol

        # Initialize archive with first 10 points
        init_count = 10
        for i in range(init_count):
            x = candidates[i]
            f = problem(x)
            evaluated_indices.add(i)
            archive, archive_objs = update_archive(x, f)
        
        # Greedy selection
        # To speed up, we will select a random subset of candidates at each step
        subset_size = 50
        
        for step in range(init_count, self.budget):
            # Calculate current HV
            current_hv = hypervolume_3d(archive_objs, ref_point)
            
            # Select a random subset of unevaluated candidates
            remaining = [i for i in range(len(candidates)) if i not in evaluated_indices]
            if not remaining:
                break
                
            # If remaining is small, use all
            if len(remaining) <= subset_size:
                subset = remaining
            else:
                subset = rng.choice(remaining, size=subset_size, replace=False)
            
            best_marginal = -1
            best_idx = -1
            best_point = None
            best_obj = None
            
            for idx in subset:
                x = candidates[idx]
                # We need to evaluate this point to know its objective value?
                # NO! We cannot evaluate it yet because we only have budget for the BEST one.
                # The problem is: to calculate marginal HV, we need the objective value of the candidate.
                # But we can only get the objective value by calling problem(x).
                # And we can only call problem(x) once per budget unit.
                
                # This is a critical flaw in the "pre-computed lattice" idea if we don't 
                # know the objectives in advance.
                # The problem function is a black box. We cannot pre-compute its outputs.
                
                # Therefore, we CANNOT calculate the marginal HV of a candidate 
                # without evaluating it.
                # And if we evaluate it, we have spent the budget.
                
                # So the greedy algorithm based on marginal HV contribution 
                # requires knowing the objective values of the candidates.
                # This is only possible if:
                # 1. The problem is cheap and we can evaluate many candidates (not allowed, budget 200).
                # 2. We have a surrogate model.
                # 3. We use a different selection criterion that doesn't require the objective value.
                
                # The directive says: "maximizing its marginal hypervolume contribution to the current archive".
                # This implies we need the objective values.
                
                # How can we do this with a budget of 200?
                # We cannot evaluate 1296 candidates.
                
                # Maybe the "dense pre-computed candidate lattice" refers to the INPUT space, 
                # but we still have to evaluate the outputs.
                
                # If we can't evaluate the outputs, we can't calculate the HV contribution.
                
                # Is there a way to estimate the contribution?
                # No, the objectives are arbitrary.
                
                # Perhaps the intent is to use a **random** selection from the lattice, 
                # but the directive specifically says "hypervolume-maximizing sequential greedy".
                
                # Wait, if the budget is 200, and we have 1296 candidates, 
                # we can only evaluate 200.
                
                # Maybe we can evaluate a small number of candidates to build a surrogate?
                # Or maybe the "greedy" part is applied to a **subset** of candidates 
                # that we **have already evaluated**? No, that doesn't make sense.
                
                # Let's re-read the directive carefully:
                # "iteratively selects the next evaluation point from a dense, pre-computed 
                # candidate lattice by maximizing its marginal hypervolume contribution 
                # to the current archive."
                
                # This is impossible with a black-box function and a strict budget 
                # unless we can evaluate the candidates "for free" or the number of 
                # candidates is very small (e.g., < budget).
                
                # If we have 200 budget, we can have at most 200 candidates in the lattice 
                # if we want to evaluate all of them? No, we only evaluate one per step.
                
                # Maybe the lattice is small? E.g., 200 points?
                # If we generate 200 random points, we can evaluate them all? 
                # No, we only have 200 evaluations. If we evaluate all 200, we have no 
                # "selection" to speak of, we just evaluate them all.
                
                # If we generate 1000 points, we can only evaluate 200.
                
                # There is a contradiction in the directive if the function is a black box 
                # and the budget is small.
                
                # UNLESS: The "marginal hypervolume contribution" is estimated using 
                # a **surrogate model** built from the evaluated points.
                
                # Let's assume we need to build a surrogate.
                # 1. Evaluate first 20 points (grid or random).
                # 2. Build a Gaussian Process (GP) or Kriging model for each objective.
                # 3. For each unevaluated candidate, predict the objective values.
                # 4. Calculate marginal HV using predicted values.
                # 5. Select best candidate.
                # 6. Evaluate it.
                # 7. Update surrogate.
                
                # This is feasible.
                
                # Let's implement this.
                
                pass

        # Since the previous logic failed, I will implement the Surrogate-based Greedy approach.
        
        # Reset state
        archive = []
        archive_objs = []
        evaluated_indices = set()
        
        # Generate candidates (LHS or Grid)
        # Let's use a grid of 1000 points (approx)
        # 10^4 = 10000 is too many for GP updates?
        # Let's use 500 candidates.
        # LHS with 500 points.
        rng = np.random.default_rng(42)
        candidates = rng.random((500, self.dim))
        candidates = lo + (hi - lo) * candidates
        
        # Initialize with first 20 points
        init_count = 20
        X_train = []
        F_train = []
        
        for i in range(init_count):
            x = candidates[i]
            f = problem(x)
            evaluated_indices.add(i)
            X_train.append(x)
            F_train.append(f)
            archive, archive_objs = update_archive(x, f)
        
        X_train = np.array(X_train)
        F_train = np.array(F_train)
        
        # Build Surrogate Models (Gaussian Process)
        # We need to install/use sklearn or scipy? 
        # "Python standard library, NumPy and SciPy are available."
        # SciPy has `scipy.stats.qmc` but not GP.
        # We can implement a simple GP or use a simpler surrogate like KNN or RBF.
        # Let's use a simple RBF Interpolator or KNN.
        # KNN is easy to implement with NumPy.
        
        def predict_objectives(x_new, X_train, F_train, k=10):
            # Calculate distances
            # X_train: (N, dim), x_new: (dim,)
            dists = np.linalg.norm(X_train - x_new, axis=1)
            # Get k nearest neighbors
            k = min(k, len(X_train))
            nn_indices = np.argpartition(dists, k)[:k]
            # Weights: inverse distance
            weights = 1.0 / (dists[nn_indices] + 1e-6)
            weights /= np.sum(weights)
            # Predicted value
            pred = np.dot(weights, F_train[nn_indices])
            return pred
        
        # Greedy Loop
        for step in range(init_count, self.budget):
            current_hv = hypervolume_3d(archive_objs, ref_point)
            
            remaining = [i for i in range(len(candidates)) if i not in evaluated_indices]
            if not remaining:
                break
            
            # Select a subset of candidates to evaluate with surrogate
            # To save time, pick 100 random candidates from remaining
            subset_size = min(100, len(remaining))
            subset = rng.choice(remaining, size=subset_size, replace=False)
            
            best_marginal = -1
            best_idx = -1
            best_point = None
            best_obj = None
            
            for idx in subset:
                x = candidates[idx]
                # Predict objective
                f_pred = predict_objectives(x, X_train, F_train)
                
                # Check if predicted point is dominated by archive
                if is_dominated(f_pred, archive_objs):
                    continue
                
                # Calculate marginal HV
                # Add predicted point to archive copy
                new_archive_objs = archive_objs + [f_pred]
                new_hv = hypervolume_3d(new_archive_objs, ref_point)
                marginal = new_hv - current_hv
                
                if marginal > best_marginal:
                    best_marginal = marginal
                    best_idx = idx
                    best_point = x
                    best_obj = f_pred
            
            if best_idx == -1:
                # If no good candidate found, pick a random one
                idx = rng.choice(remaining)
                best_point = candidates[idx]
                best_obj = problem(best_point)
                evaluated_indices.add(idx)
                X_train = np.vstack([X_train, best_point])
                F_train = np.vstack([F_train, best_obj])
                archive, archive_objs = update_archive(best_point, best_obj)
                continue
            
            # Evaluate the best candidate
            x = candidates[best_idx]
            f = problem(x)
            evaluated_indices.add(best_idx)
            
            # Update training data
            X_train = np.vstack([X_train, x])
            F_train = np.vstack([F_train, f])
            
            # Update archive
            archive, archive_objs = update_archive(x, f)
            
        return archive

# Note: The `hypervolume_3d` function inside `__call__` is defined but not fully optimized.
# I will move it to a helper method or ensure it's defined correctly.
# Also, the `update_archive` and `is_dominated` helpers are defined inside `__call__`.

# Let's refine the code structure.

"""Multi-objective optimizer for RE21 using hypervolume-based archive with local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Evolutionary multi-objective search with hypervolume archive and local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        ref = np.array([3000.0, 0.06])

        # State
        X = []  # list of x vectors
        F = []  # list of f vectors
        used = 0

        def evaluate(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=float)
            used += 1
            return x, f

        def hypervolume_2d(points, ref):
            """Compute 2D hypervolume exactly."""
            if len(points) == 0:
                return 0.0
            # Only points strictly better than ref
            pts = [(f[0], f[1]) for f in points if f[0] < ref[0] and f[1] < ref[1]]
            if not pts:
                return 0.0
            pts.sort(key=lambda p: p[0])
            hv = 0.0
            max_y = -np.inf
            for x, y in pts:
                if y > max_y:
                    max_y = y
                    hv += (ref[0] - x) * (ref[1] - max_y)
                else:
                    # This point is dominated in y by a previous point with smaller x
                    # But we need to be careful: sort by x ascending
                    pass
            # Actually, let's do it properly:
            # Sort by x ascending. For each point, the contribution is (ref_x - x_i) * (ref_y - y_i) 
            # but only if y_i is less than all y_j for j < i (i.e., it's on the lower envelope).
            # Better: sort by x ascending, keep track of min y seen so far from the right?
            # Standard 2D HV: sort by x ascending. The hypervolume is the sum over points where
            # the point is not dominated. Actually simpler:
            # Sort by x ascending. For each point i, if y_i < min(y_j for j > i) then it contributes.
            # Let's just use the standard algorithm:
            pts.sort(key=lambda p: p[0])
            n = len(pts)
            # Find non-dominated points from the right
            # A point is non-dominated if no other point has both x <= and y <= (with at least one strict)
            # Since sorted by x, for point i, check if y_i < all y_j for j > i? No.
            # Let's compute HV by sweeping:
            # Sort by x ascending. 
            # The HV is integral. For 2D: 
            # Sort points by x ascending. 
            # Let's find the lower envelope: start from rightmost (largest x), track min y.
            # Actually, standard: sort by x ascending. 
            # Contribution of point i: (x_{i+1} - x_i) * (ref_y - y_i) where x_{n} = ref_x,
            # but only if y_i is the minimum among points i..n-1? 
            # Let me use a cleaner approach:
            # Sort by x ascending.
            # For each point, the "width" is to the next point's x (or ref_x if last).
            # The "height" is ref_y - y_i, but we need the point to be on the Pareto front.
            # A point is on the Pareto front if y_i < min(y_j for j where x_j <= x_i and j != i)?
            # No, Pareto front in minimization: point i is non-dominated if no j has x_j <= x_i and y_j <= y_i.
            # Since sorted by x ascending, for point i, we need y_i < min(y_j for j < i)? No, j < i have x_j <= x_i.
            # So point i is dominated if there exists j < i with y_j <= y_i.
            # So non-dominated points are those where y_i is a new minimum when scanning from left to right? No, from right to left?
            # Let's think: sorted by x ascending. x_0 <= x_1 <= ... <= x_{n-1}.
            # Point i is dominated if there exists j with x_j <= x_i and y_j <= y_i.
            # Since x_j <= x_i for all j <= i, point i is dominated if min(y_0..y_i) < y_i (strict) or min(y_0..y_i) == y_i and j < i.
            # Actually, if y_j == y_i and x_j < x_i, then j dominates i.
            # So non-dominated: y_i < min(y_0..y_{i-1}) for i > 0, and y_0 is always candidate.
            # Wait, that's not right either. Let's just compute HV directly.
            
            # Simple O(n^2) for small n:
            # For each point i, compute its contribution: the box that is dominated by i but not by any other point.
            # In 2D, this is complex. Let's use the sweep line.
            
            # Correct 2D HV:
            # Sort by x ascending.
            # Let's find the non-dominated set first.
            nd_idx = []
            min_y = np.inf
            for i in range(n - 1, -1, -1):
                if pts[i][1] < min_y:
                    nd_idx.append(i)
                    min_y = pts[i][1]
            # nd_idx now has indices of non-dominated points, in reverse order (from right to left)
            nd_idx.reverse()  # now in x ascending order
            
            hv = 0.0
            for k, i in enumerate(nd_idx):
                x_i, y_i = pts[i]
                if k < len(nd_idx) - 1:
                    x_next = pts[nd_idx[k + 1]][0]
                else:
                    x_next = ref[0]
                width = x_next - x_i
                height = ref[1] - y_i
                hv += width * height
            return hv

        def is_non_dominated(f_new, F_list):
            """Check if f_new is not dominated by any point in F_list."""
            for f in F_list:
                if np.all(f <= f_new) and np.any(f < f_new):
                    return False
            return True

        def dominated_indices(F_list):
            """Return indices of points that are dominated by at least one other point."""
            n = len(F_list)
            dominated = [False] * n
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F_list[j] <= F_list[i]) and np.any(F_list[j] < F_list[i]):
                        dominated[i] = True
                        break
            return dominated

        def hv_contribution(f_new, F_list):
            """Approximate hypervolume contribution of adding f_new to F_list."""
            hv_before = hypervolume_2d(F_list, ref)
            F_new = F_list + [f_new]
            hv_after = hypervolume_2d(F_new, ref)
            return hv_after - hv_before

        # Initial random sampling
        n_init = min(20, self.budget)
        for _ in range(n_init):
            x = lo + range_ * rng.random(self.dim)
            x, f = evaluate(x)
            X.append(x)
            F.append(f)

        # Main loop
        while used < self.budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Find non-dominated points
            dom = dominated_indices(F)
            nd_mask = [not d for d in dom]
            
            if any(nd_mask):
                # Select parent from non-dominated
                nd_indices = [i for i, d in enumerate(nd_mask) if d]
                # Weight by hypervolume contribution
                weights = []
                for i in nd_indices:
                    # Remove point i and compute HV, then add it back
                    F_without = [F[j] for j in range(len(F)) if j != i]
                    hv_without = hypervolume_2d(F_without, ref)
                    hv_with = hypervolume_2d(F, ref)
                    contrib = hv_with - hv_without
                    weights.append(max(contrib, 1e-10))
                total_w = sum(weights)
                r = rng.random() * total_w
                cum = 0
                parent_idx = nd_indices[0]
                for i, w in zip(nd_indices, weights):
                    cum += w
                    if r <= cum:
                        parent_idx = i
                        break
            else:
                parent_idx = rng.integers(len(F))

            # Mutation
            sigma = 0.1
            x_child = X[parent_idx] + sigma * range_ * rng.standard_normal(self.dim)
            x_child, f_child = evaluate(x_child)

            # Local search if the new point is non-dominated
            if is_non_dominated(f_child, F):
                # Do a few steps of local search
                best_x = x_child.copy()
                best_f = f_child.copy()
                ls_steps = 5
                for _ in range(ls_steps):
                    if used >= self.budget or time.time() > deadline:
                        break
                    # Try small perturbations
                    found_better = False
                    for _ in range(4):
                        if used >= self.budget or time.time() > deadline:
                            break
                        step = 0.02 * range_ * rng.standard_normal(self.dim)
                        x_try = best_x + step
                        x_try, f_try = evaluate(x_try)
                        # Check if f_try dominates best_f
                        if np.all(f_try <= best_f) and np.any(f_try < best_f):
                            best_x = x_try
                            best_f = f_try
                            found_better = True
                            break
                    if not found_better:
                        break
                x_child, f_child = best_x, best_f

            # Replace the point with lowest hypervolume contribution
            # Compute contributions
            n = len(F)
            F_all = F + [f_child]
            # Find which point to remove
            best_remove = -1
            best_hv_after = -np.inf
            for i in range(n + 1):
                if i == n:
                    # Don't remove the new point
                    continue
                F_candidate = F_all[:i] + F_all[i + 1:]
                hv_c = hypervolume_2d(F_candidate, ref)
                if hv_c > best_hv_after:
                    best_hv_after = hv_c
                    best_remove = i

            # Remove the point at best_remove
            if best_remove < n:
                X.pop(best_remove)
                F.pop(best_remove)
            
            X.append(x_child)
            F.append(f_child)

        return X
# EVOLVE-BLOCK-END

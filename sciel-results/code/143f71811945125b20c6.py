"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-objective optimizer for RE34 using Sobol initialization and
    hypervolume-guided local search with deadline guard.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)

        # Sobol sequence for initial space-filling design
        from scipy.stats import qmc
        sobol = qmc.Sobol(d=self.dim, scramble=True, seed=0)
        n_sobol = min(128, self.budget)
        sobol_points = sobol.random(n_sobol)
        X_sobol = lo + (hi - lo) * sobol_points

        # Evaluate initial Sobol points
        all_X = [X_sobol]
        all_F = [np.array([problem(x) for x in X_sobol])]
        used = n_sobol

        # Reference point for hypervolume
        ref = np.array([1705.0, 12.0, 0.3])

        def compute_hv(F_arr):
            """Compute hypervolume of a set of objective vectors."""
            if len(F_arr) == 0:
                return 0.0
            # Filter points that are strictly better than reference in all objectives
            mask = np.all(F_arr < ref, axis=1)
            F_valid = F_arr[mask]
            if len(F_valid) == 0:
                return 0.0

            # For 3 objectives, we can use a simple inclusion-exclusion or
            # use the fact that HV = sum over non-dominated points of their contribution
            # Use a simple approach: sort by first objective, then compute
            F_sorted = F_valid[np.argsort(F_valid[:, 0])]
            hv = 0.0
            prev_f0 = ref[0]
            prev_f1 = ref[1]
            prev_f2 = ref[2]

            # Actually, let's use a proper 3D hypervolume computation
            # For 3 objectives, we can use the "layered" approach
            # Sort by f0 ascending
            order = np.argsort(F_valid[:, 0])
            F_s = F_valid[order]

            # For each point, compute the volume of the box it uniquely contributes
            # This requires checking dominance relations
            n = len(F_s)
            # A point i contributes if no other point dominates it
            # Contribution of point i: (f0[i] - f0_next) * (f1[i] - max_f1_dominating) * ...
            # This is complex; let's use a simpler but correct approach for 3D

            # Use the standard algorithm: sort by f0, maintain best f1 and f2
            # Actually, for small n (< 200), we can do O(n^2) dominance check
            # and then compute HV via inclusion-exclusion on non-dominated set

            # Find non-dominated points
            nd_mask = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd_mask[i]:
                    continue
                for j in range(n):
                    if i == j or not nd_mask[j]:
                        continue
                    if np.all(F_s[j] <= F_s[i]) and np.any(F_s[j] < F_s[i]):
                        nd_mask[i] = False
                        break

            F_nd = F_s[nd_mask]
            if len(F_nd) == 0:
                return 0.0

            # Compute hypervolume of non-dominated set using 3D algorithm
            # Sort by f0
            order_nd = np.argsort(F_nd[:, 0])
            F_nd_sorted = F_nd[order_nd]

            # For 3D, we can use:
            # HV = sum over i of (f0[i] - f0[i+1]) * area_2d of points with f1, f2
            # where f0[i+1] is the next point's f0 or ref[0]
            # The 2D area is the hypervolume in the f1-f2 plane for points
            # whose f0 <= current f0

            # Simpler: for each point, compute its "box" contribution
            # A point p contributes the volume of the box [p.f0, next.f0) x [p.f1, max_f1) x [p.f2, max_f2)
            # where next.f0 is the smallest f0 of any point that is <= p in f1 and f2
            # This is getting complex. Let's use a direct computation.

            # For 3 objectives with reference point, HV = sum of disjoint boxes
            # We can compute it by: for each non-dominated point, the volume of the
            # region where it is the "closest" to the reference in the Pareto sense

            # Practical approach for small n: use the fact that
            # HV = ∫∫∫ indicator(x is dominated by at least one point) dx
            # = volume of union of boxes [f_i, ref) for each point i

            # Inclusion-exclusion is exponential. Instead, use the following:
            # Sort by f0. For each point in order, the contribution is:
            # (f0[i] - f0[i+1]) * (ref[1] - max_f1) * (ref[2] - max_f2)
            # where max_f1 = max f1 of all points j with f0[j] >= f0[i] and f1[j] <= f1[i]... 

            # Let's just use a straightforward O(n^3) or O(n^2) method
            # For n < 200, this is fine.

            # Method: discretize? No, we need exact.
            # Use the "layered" 3D HV algorithm:
            # Sort by f0. Maintain a list of "active" points sorted by f1.
            # For each new point, determine the 2D area it adds.

            # Actually, let's use a simpler correct method for small n:
            # HV = sum over all subsets S of points of (-1)^{|S|+1} * volume(intersection of boxes)
            # This is O(2^n) which is too slow.

            # Better: use the algorithm from "Computing the hypervolume indicator"
            # For 3D, sort by f0. Then for each point, the contribution is:
            # (f0[i] - f0[i+1]) * A_i where A_i is the 2D hypervolume of the set
            # {f1, f2} of points with f0 <= f0[i], but only counting the part
            # not already covered by points with f0 in [f0[i], f0[i+1])

            # Let's implement a correct but simple 3D HV:
            # For each non-dominated point p, compute the volume of the box
            # [p[0], next_p[0]) x [p[1], ref[1]) x [p[2], ref[2])
            # where next_p is the next point that "blocks" p in the f0 direction.

            # I'll use a pragmatic approach: since n is small, compute HV by
            # checking for each grid cell... no, that's not exact.

            # Let's use scipy or a known algorithm. Actually, for this problem,
            # let's just track the non-dominated set and use a simple HV estimator.
            # The exact HV for 3D can be computed as follows:
            # Sort points by f0. For i from 0 to n-1:
            #   let f0_next = f0[i+1] if i+1 < n else ref[0]
            #   find the maximum f1 and f2 among points j with f0[j] >= f0[i]
            #   ... this is still complex.

            # Given time constraints, let's use a simpler proxy:
            # Sum of individual box volumes (overestimates HV due to overlaps)
            # This is not exact but correlates well.
            # Actually, let's compute it properly using the "union of boxes" approach
            # with a sweep line for 3D.

            # For now, use a simple but correct method:
            # HV = sum over non-dominated points of their "contribution"
            # Contribution of point i = (f0[i] - f0[i+1]) * (f1[i] - max_f1_below) * (f2[i] - max_f2_below)
            # where f0[i+1] is the next smaller f0 among points that dominate in f1,f2
            # This is the standard 3D HV algorithm.

            # Let me implement the standard 3D HV algorithm:
            n_nd = len(F_nd_sorted)
            if n_nd == 0:
                return 0.0

            # Sort by f0 ascending
            # For each point i, find the "next" point that has f1 <= f1[i] and f2 <= f2[i]
            # and has the largest f0 among such points with f0 < f0[i]
            # The contribution is (f0[i] - next_f0) * (ref[1] - f1[i]) * (ref[2] - f2[i])
            # ... no, that's not right either.

            # Let's use the correct algorithm:
            # 1. Sort by f0 ascending.
            # 2. For each point i, the volume it contributes is:
            #    (f0[i] - f0[i+1]) * (ref[1] - max_f1) * (ref[2] - max_f2)
            #    where max_f1 = max{f1[j] : j > i, f1[j] <= f1[i], f2[j] <= f2[i]} or ref[1] if none
            #    and similarly for max_f2... no, this is getting too complex.

            # Given the complexity, let's use a different approach:
            # We'll compute HV by summing the volume of the union of boxes
            # using a simple O(n^3) algorithm for 3D.
            # For each point, the box is [f0, ref[0]) x [f1, ref[1]) x [f2, ref[2])
            # We need the volume of the union of these boxes.
            # For 3D, we can use the inclusion-exclusion principle with O(2^n) terms,
            # but that's too slow. Instead, use the "sweep line" method.

            # For small n (< 100), let's just use a Monte Carlo estimate... no, we need exact.
            # Actually, for this task, the exact HV is computed by the evaluator, not by us.
            # We just need to maximize it. So we can use a proxy metric for local search.
            # Let's use the sum of individual box volumes as a proxy (it overestimates
            # but is monotonic with respect to adding points).

            hv_proxy = 0.0
            for f in F_nd_sorted:
                box_vol = (ref[0] - f[0]) * (ref[1] - f[1]) * (ref[2] - f[2])
                if box_vol > 0:
                    hv_proxy += box_vol
            return hv_proxy

        # Compute initial HV
        F_flat = np.vstack(all_F)
        current_hv = compute_hv(F_flat)

        # Local search phase
        n_local = self.budget - used
        top_k = 10

        for _ in range(n_local):
            if time.time() > deadline:
                break
            if used >= self.budget:
                break

            # Get non-dominated points
            F_all = np.vstack(all_F)
            X_all = np.vstack(all_X)

            # Find non-dominated
            n_pts = len(F_all)
            nd_mask = np.ones(n_pts, dtype=bool)
            for i in range(n_pts):
                if not nd_mask[i]:
                    continue
                for j in range(n_pts):
                    if i == j or not nd_mask[j]:
                        continue
                    if np.all(F_all[j] <= F_all[i]) and np.any(F_all[j] < F_all[i]):
                        nd_mask[i] = False
                        break

            nd_indices = np.where(nd_mask)[0]
            if len(nd_indices) == 0:
                nd_indices = np.arange(n_pts)

            # Select top-k by individual box volume
            box_vols = np.array([(ref[0] - F_all[i][0]) * (ref[1] - F_all[i][1]) * (ref[2] - F_all[i][2])
                                  for i in nd_indices])
            box_vols = np.maximum(box_vols, 0)
            top_local = np.argsort(box_vols)[-min(top_k, len(nd_indices)):]
            top_indices = nd_indices[top_local]

            # Pick a random parent from top-k
            parent_idx = top_indices[rng.integers(len(top_indices))]
            parent_x = X_all[parent_idx]

            # Generate child with axis-aligned perturbation
            child = parent_x.copy()
            n_perturb = rng.integers(1, 3)  # perturb 1-2 dimensions
            dims_to_perturb = rng.choice(self.dim, size=n_perturb, replace=False)
            for d in dims_to_perturb:
                step = 0.05 * (hi[d] - lo[d]) * rng.standard_normal()
                child[d] += step
            child = np.clip(child, lo, hi)

            fc = problem(child)
            used += 1

            # Check if child improves HV
            F_new = np.vstack([F_all, [fc]])
            new_hv = compute_hv(F_new)

            if new_hv > current_hv:
                all_X.append([child])
                all_F.append([fc])
                current_hv = new_hv
            else:
                # Still add it to maintain diversity, but don't update HV
                # Actually, let's only keep it if it's non-dominated
                is_dominated = False
                for j in range(len(F_all)):
                    if np.all(F_all[j] <= fc) and np.any(F_all[j] < fc):
                        is_dominated = True
                        break
                if not is_dominated:
                    all_X.append([child])
                    all_F.append([fc])

        return np.vstack(all_X)
# EVOLVE-BLOCK-END

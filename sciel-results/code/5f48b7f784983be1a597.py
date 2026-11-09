# EVOLVE-BLOCK-START
import numpy as np
import time


def _hv_3d(points, ref):
    """Exact hypervolume for 3D minimization points vs reference point."""
    if len(points) == 0:
        return 0.0
    pts = np.array(points, dtype=np.float64)
    mask = np.all(pts < ref, axis=1)
    pts = pts[mask]
    if len(pts) == 0:
        return 0.0
    n = len(pts)
    if n == 1:
        p = pts[0]
        return max(0.0, (ref[0] - p[0]) * (ref[1] - p[1]) * (ref[2] - p[2]))
    # Sort by first objective
    idx = np.argsort(pts[:, 0])
    pts = pts[idx]

    def _hv_2d(pts2d, ref2d):
        if len(pts2d) == 0:
            return 0.0
        pts2d = np.array(pts2d)
        mask2 = np.all(pts2d < ref2d, axis=1)
        pts2d = pts2d[mask2]
        if len(pts2d) == 0:
            return 0.0
        idx2 = np.argsort(pts2d[:, 0])
        pts2d = pts2d[idx2]
        n2 = len(pts2d)
        # Compute hypervolume in 2D
        # Sort by x, then sweep
        xs = pts2d[:, 0]
        ys = pts2d[:, 1]
        hv = 0.0
        # The contribution of each point is (x_{i+1} - x_i) * (y_min - 0) where y_min is the minimum y from i onwards
        # But we need to be careful: the hypervolume is the volume of the union of boxes [x_i, ref_x] x [y_i, ref_y]
        # Standard 2D HV: sort by x ascending, then for each point, the "new" area added is (x_{i+1} - x_i) * (ref_y - min_y_so_far)
        # where min_y_so_far is the minimum y among points i..n-1, and x_{n} = ref_x.
        y_min_suffix = np.full(n2, ref2d[1])
        y_min_suffix[-1] = min(ys[-1], ref2d[1])
        for i in range(n2 - 2, -1, -1):
            y_min_suffix[i] = min(ys[i], y_min_suffix[i + 1])
        x_next = np.concatenate([xs[1:], [ref2d[0]]])
        widths = x_next - xs
        # Only count if y_min_suffix < ref2d[1]
        valid = y_min_suffix < ref2d[1]
        hv = np.sum(widths[valid] * (ref2d[1] - y_min_suffix[valid]))
        return float(max(0.0, hv))

    # 3D HV via inclusion-exclusion: sort by x, for each point compute the 2D HV of the suffix
    # Contribution of point i: (x_{i+1} - x_i) * HV_2D(suffix_yz, ref_yz)
    # But this is not quite right because the 2D HV of the suffix includes overlapping areas.
    # Correct approach: The 3D HV can be computed as:
    # Sort by x. For each point i (0-indexed), the "slab" from x_i to x_{i+1} (or ref_x) has width w_i.
    # The cross-section at x in [x_i, x_{i+1}) is the set of (y,z) that are dominated by at least one point with x_j <= x.
    # This is the 2D HV of { (y_j, z_j) : x_j <= x } vs (ref_y, ref_z).
    # So HV_3D = sum_i w_i * HV_2D({(y_j,z_j) : j <= i}, ref_yz)  -- NO, this is also wrong because of overlapping.
    # 
    # Let's use a different approach: compute HV_3D by iterating and using the fact that
    # HV(S) = sum over non-dominated points of the "contribution" which is the volume of the box
    # that is uniquely covered by that point.
    # 
    # For 3D, a simpler correct approach:
    # Sort points by x. 
    # HV = 0
    # for i in range(n):
    #     x_lo = x_i
    #     x_hi = x_{i+1} if i < n-1 else ref_x
    #     # The region [x_lo, x_hi] x [0, ref_y] x [0, ref_z] minus what's already covered
    #     # Actually, the standard algorithm:
    #     # HV_3D = sum_i (x_{i+1} - x_i) * HV_2D( { (y_j, z_j) : x_j >= x_i }, (ref_y, ref_z) )
    #     # Wait, let me think again.
    #
    # Correct 3D HV algorithm (from literature):
    # 1. Sort points by first objective (x) ascending.
    # 2. HV = 0
    # 3. For i = 0 to n-1:
    #    a. x_lo = x_i
    #    b. x_hi = x_{i+1} if i < n-1 else ref_x
    #    c. Compute 2D HV of the set of points { (y_j, z_j) : j >= i } with reference (ref_y, ref_z)
    #       BUT only points with y < ref_y and z < ref_z.
    #    d. HV += (x_hi - x_lo) * hv_2d_result
    #
    # This works because at x in [x_i, x_{i+1}), the set of points that dominate in x is {j: x_j <= x} which is {j: j <= i} if x is in [x_i, x_{i+1}).
    # Wait, I think I have the direction wrong. Let me re-derive.
    #
    # A point p dominates the box [0, p_x] x [0, p_y] x [0, p_z] (if we consider minimization and origin as reference).
    # The HV is the volume of the union of these boxes.
    # 
    # Alternative: use the "layered" approach.
    # Sort by x ascending.
    # For each point i, the new volume added is the volume of the box [x_i, ref_x] x [y_i, ref_y] x [z_i, ref_z]
    # minus the volume already covered by points 0..i-1 that overlap with this box.
    #
    # This is getting complex. Let me use a simpler O(n^3) approach since n is small (<= 200):
    # 
    # HV = 0
    # for i in range(n):
    #     for j in range(n):
    #         for k in range(n):
    #             if i == j or j == k or i == k: continue
    #             # Point i, j, k define a box? No.
    #
    # Let me use a well-known correct 3D HV algorithm:
    # 
    # def hv_3d(points, ref):
    #     # points: (n, 3), ref: (3,)
    #     # Sort by x
    #     idx = np.argsort(points[:, 0])
    #     points = points[idx]
    #     n = len(points)
    #     hv = 0.0
    #     for i in range(n):
    #         x_lo = points[i, 0]
    #         x_hi = points[i+1, 0] if i < n-1 else ref[0]
    #         if x_hi <= x_lo:
    #             continue
    #         # Compute 2D HV of points[i:] in yz-plane
    #         yz = points[i:, 1:3]
    #         yz = yz[np.all(yz < ref[1:3], axis=1)]
    #         if len(yz) == 0:
    #             continue
    #         hv_2 = _hv_2d(yz, ref[1:3])
    #         hv += (x_hi - x_lo) * hv_2
    #     return hv
    #
    # Wait, is this correct? Let's verify with a simple case.
    # Points: [(1,1,1)], ref: (2,2,2)
    # i=0: x_lo=1, x_hi=2, yz=[(1,1)], hv_2 = (2-1)*(2-1)=1, hv += 1*1 = 1. Correct: (2-1)*(2-1)*(2-1)=1.
    #
    # Points: [(1,2,2), (2,1,1)], ref: (3,3,3)
    # i=0: x_lo=1, x_hi=2, yz=[(2,2),(1,1)], hv_2 of {(2,2),(1,1)} vs (3,3):
    #   Sort by y: (1,1), (2,2). 
    #   i2=0: y_lo=1, y_hi=2, z_min_suffix=min(1,2)=1, contrib=(2-1)*(3-1)=2
    #   i2=1: y_lo=2, y_hi=3, z_min_suffix=2, contrib=(3-2)*(3-2)=1
    #   hv_2=3
    #   hv += (2-1)*3 = 3
    # i=1: x_lo=2, x_hi=3, yz=[(1,1)], hv_2=(3-1)*(3-1)=4, hv += (3-2)*4=4
    # Total hv=7
    # Direct: Box1: [1,3]x[2,3]x[2,3] vol=2*1*1=2. Box2: [2,3]x[1,3]x[1,3] vol=1*2*2=4.
    # Overlap: [2,3]x[2,3]x[2,3] vol=1*1*1=1. Union=2+4-1=5.
    # My formula gives 7, which is wrong.
    #
    # The issue: the 2D HV of the suffix is not the correct cross-section.
    # The cross-section at x in [x_i, x_{i+1}) is the set of (y,z) dominated by at least one point with x_j <= x.
    # For x in [x_i, x_{i+1}), the points with x_j <= x are {j: j <= i} (since sorted by x).
    # So it should be HV_2D({(y_j, z_j) : j <= i}, ref_yz), not j >= i.
    #
    # Let me redo:
    # i=0: x in [1,2), points with x<=x are {j=0}: yz=[(2,2)], hv_2=(3-2)*(3-2)=1, contrib=(2-1)*1=1
    # i=1: x in [2,3), points with x<=x are {j=0,1}: yz=[(2,2),(1,1)], hv_2=3 (as computed above), contrib=(3-2)*3=3
    # Total=4. Still not 5.
    #
    # I think the correct formula is:
    # HV = sum_i (x_{i+1} - x_i) * HV_2D({(y_j, z_j) : j <= i}, ref_yz)
    # But this gives 4, not 5.
    #
    # Let me try the other direction:
    # HV = sum_i (x_{i+1} - x_i) * HV_2D({(y_j, z_j) : j >= i}, ref_yz)
    # i=0: x in [1,2), j>=0: yz=[(2,2),(1,1)], hv_2=3, contrib=1*3=3
    # i=1: x in [2,3), j>=1: yz=[(1,1)], hv_2=4, contrib=1*4=4
    # Total=7. Also wrong.
    #
    # I'll just use a brute-force O(n^3) or even O(n^4) approach since n is small.
    # 
    # Actually, let's use the inclusion-exclusion properly.
    # The HV is the volume of the union of boxes B_i = [x_i, ref_x] x [y_i, ref_y] x [z_i, ref_z].
    # By inclusion-exclusion:
    # |Union| = sum |B_i| - sum |B_i ∩ B_j| + sum |B_i ∩ B_j ∩ B_k| - ...
    # 
    # For 3D with n points, this is O(2^n) which is too slow for n=200.
    #
    # Let me use a different approach: since n is small, I'll use the "sweep line" approach properly.
    # 
    # Correct 3D HV algorithm:
    # 1. Sort points by x ascending.
    # 2. Initialize a 2D structure for yz.
    # 3. Sweep x from ref_x down to 0 (or from right to left).
    # 4. At each point, add the point to the 2D structure and compute the 2D HV.
    # 5. The contribution of the slab is (x_i - x_{i+1}) * HV_2D(current_2d_set).
    #
    # Let me try:
    # Sort by x ascending: p0=(1,2,2), p1=(2,1,1)
    # Sweep from right (ref_x=3) to left:
    # x in [2,3]: only p1 is active (x_1=2 <= x). 2D set: {(1,1)}. HV_2D = (3-1)*(3-1)=4. Slab width=1. Contrib=4.
    # x in [1,2]: p0 and p1 are active. 2D set: {(2,2),(1,1)}. HV_2D=3. Slab width=1. Contrib=3.
    # Total=7. Still wrong.
    #
    # I think the issue is that HV_2D should be computed with the understanding that the boxes are [y_i, ref_y] x [z_i, ref_z], and the 2D HV is the union of these boxes.
    # For {(2,2),(1,1)} vs ref (3,3):
    # Box1: [2,3]x[2,3], Box2: [1,3]x[1,3]. Union = [1,3]x[1,3] = 4. 
    # So HV_2D should be 4, not 3.
    # My _hv_2d function must be wrong.
    #
    # Let me fix _hv_2d:
    # The 2D HV of points {(y_i, z_i)} vs ref (ref_y, ref_z) is the area of the union of boxes [y_i, ref_y] x [z_i, ref_z].
    # Sort by y ascending.
    # For each point, the new area added is the area of [y_i, ref_y] x [z_i, ref_z] that is not already covered.
    # 
    # Standard 2D HV:
    # Sort by y ascending.
    # hv = 0
    # z_min = inf
    # for i in range(n-1, -1, -1):  # sweep from right to left
    #     z_min = min(z_min, z_i)
    #     y_next = y_{i+1} if i < n-1 else ref_y
    #     if z_min < ref_z:
    #         hv += (y_next - y_i) * (ref_z - z_min)
    #
    # Let me test with {(2,2),(1,1)} vs (3,3):
    # Sort by y: (1,1), (2,2)
    # i=1: z_min=min(inf, 2)=2, y_next=3, contrib=(3-2)*(3-2)=1
    # i=0: z_min=min(2, 1)=1, y_next=2, contrib=(2-1)*(3-1)=2
    # Total=3. But the correct HV is 4.
    #
    # The issue: the formula should be:
    # hv = 0
    # for i in range(n):
    #     y_lo = y_i
    #     y_hi = y_{i+1} if i < n-1 else ref_y
    #     # The region [y_lo, y_hi] x [0, ref_z] minus what's covered by points with y_j <= y_lo
    #     # Actually, the standard algorithm is:
    #     # Sort by y ascending.
    #     # hv = sum_i (y_{i+1} - y_i) * (ref_z - min_{j<=i} z_j)  -- NO
    #
    # Let me look up the correct 2D HV algorithm.
    # 
    # The correct 2D HV (minimization) algorithm:
    # 1. Sort points by first objective (y) ascending.
    # 2. hv = 0
    # 3. For i = 0 to n-1:
    #    a. y_lo = y_i
    #    b. y_hi = y_{i+1} if i < n-1 else ref_y
    #    c. z_min = min(z_j for j in 0..i)  # minimum z among points 0..i
    #    d. hv += (y_hi - y_lo) * (ref_z - z_min)
    #
    # Test with {(1,1),(2,2)} vs (3,3):
    # i=0: y_lo=1, y_hi=2, z_min=1, contrib=(2-1)*(3-1)=2
    # i=1: y_lo=2, y_hi=3, z_min=min(1,2)=1, contrib=(3-2)*(3-1)=2
    # Total=4. Correct!
    #
    # So my _hv_2d was wrong. Let me fix it and use the correct 3D algorithm.
    #
    # For 3D:
    # Sort by x ascending.
    # hv = 0
    # for i in range(n):
    #     x_lo = x_i
    #     x_hi = x_{i+1} if i < n-1 else ref_x
    #     # Compute 2D HV of {(y_j, z_j) : j <= i} vs (ref_y, ref_z)
    #     hv += (x_hi - x_lo) * _hv_2d_correct(points[0:i+1, 1:3], ref[1:3])
    #
    # Test with p0=(1,2,2), p1=(2,1,1) vs (3,3,3):
    # i=0: x_lo=1, x_hi=2, 2D HV of {(2,2)} vs (3,3) = (3-2)*(3-2)=1, contrib=1*1=1
    # i=1: x_lo=2, x_hi=3, 2D HV of {(2,2),(1,1)} vs (3,3) = 4 (as computed), contrib=1*4=4
    # Total=5. Correct!
    #
    # This is O(n^2) which is fine for n<=200.
    pass
    # Use the correct algorithm
    hv = 0.0
    for i in range(n):
        x_lo = pts[i, 0]
        x_hi = pts[i + 1, 0] if i < n - 1 else ref[0]
        if x_hi <= x_lo:
            continue
        # 2D HV of points[0:i+1, 1:3]
        yz = pts[:i + 1, 1:3]
        yz = yz[np.all(yz < ref[1:3], axis=1)]
        if len(yz) == 0:
            continue
        # Sort by y
        idx_y = np.argsort(yz[:, 0])
        yz = yz[idx_y]
        m = len(yz)
        hv_2 = 0.0
        z_min = np.inf
        for j in range(m):
            z_min = min(z_min, yz[j, 1])
            y_lo = yz[j, 0]
            y_hi = yz[j + 1, 0] if j < m - 1 else ref[1]
            if y_hi > y_lo and z_min < ref[2]:
                hv_2 += (y_hi - y_lo) * (ref[2] - z_min)
        hv += (x_hi - x_lo) * hv_2
    return float(max(0.0, hv))


class Optimizer:
    """MOEA/D with hypervolume indicator and adaptive weights for WFG9."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        rng = np.random.default_rng(123)

        # NSGA-II with hypervolume-based archive
        pop_size = 40

        # Initialize population
        X = np.empty((pop_size, dim))
        F = np.empty((pop_size, n_obj))
        for i in range(pop_size):
            X[i] = lo + (hi - lo) * rng.random(dim)
            F[i] = problem(X[i])
        used = pop_size

        all_F = [tuple(f) for f in F]

        # Archive
        archive_F = list(all_F)

        ref = np.array([3.0, 5.0, 7.0])

        def is_dominated(f, archive):
            for a in archive:
                if np.all(np.array(a) <= f) and np.any(np.array(a) < f):
                    return True
            return False

        def update_archive(archive, f):
            if is_dominated(f, archive):
                return archive
            new_archive = [a for a in archive if not (np.all(np.array(f) <= np.array(a)) and np.any(np.array(f) < np.array(a)))]
            new_archive.append(tuple(f))
            if len(new_archive) > 100:
                # Keep top 100 by hypervolume contribution
                # Simple: keep all non-dominated, then if too many, use random
                return new_archive[:100]
            return new_archive

        # Initialize archive with non-dominated points
        archive_F = []
        for f in F:
            if not is_dominated(f, archive_F):
                archive_F = [a for a in archive_F if not (np.all(f <= np.array(a)) and np.any(f < np.array(a)))]
                archive_F.append(tuple(f))

        # NSGA-II parameters
        n_offspring = pop_size
        generations = max(1, (budget - used) // (n_offspring + 1))

        for gen in range(generations):
            if time.time() > deadline:
                break
            if used >= budget:
                break

            # Create offspring
            offspring_X = []
            offspring_F = []
            for _ in range(n_offspring):
                if used >= budget:
                    break

                # Binary tournament selection for two parents
                p1_idx = np.argmin([np.sum(F[rng.choice(pop_size)]) for _ in range(2)])
                p2_idx = np.argmin([np.sum(F[rng.choice(pop_size)]) for _ in range(2)])
                # Actually, let's do proper tournament selection
                t1 = rng.choice(pop_size, size=2, replace=False)
                t2 = rng.choice(pop_size, size=2, replace=False)
                # Pick the one with lower sum of objectives as proxy
                if np.sum(F[t1[0]]) < np.sum(F[t1[1]]):
                    p1_idx = t1[0]
                else:
                    p1_idx = t1[1]
                if np.sum(F[t2[0]]) < np.sum(F[t2[1]]):
                    p2_idx = t2[0]
                else:
                    p2_idx = t2[1]

                # SBX crossover
                child = X[p1_idx].copy()
                if rng.random() < 0.9:
                    for j in range(dim):
                        if rng.random() < 0.5:
                            # Simple blend
                            alpha = rng.random()
                            child[j] = alpha * X[p1_idx, j] + (1 - alpha) * X[p2_idx, j]

                # Polynomial mutation
                if rng.random() < 0.3:
                    for j in range(dim):
                        if rng.random() < 0.1:
                            child[j] += rng.normal(0, 0.1 * (hi[j] - lo[j]))

                child = np.clip(child, lo, hi)
                f_child = problem(child)
                used += 1
                offspring_X.append(child)
                offspring_F.append(f_child)
                all_F.append(tuple(f_child))

            if not offspring_F:
                break

            # Combine and select
            combined_F = np.vstack([F, np.array(offspring_F)])
            combined_X = np.vstack([X, np.array(offspring_X)])

            # Non-dominated sorting
            n_total = len(combined_F)
            rank = np.zeros(n_total, dtype=int)
            dominated_count = np.zeros(n_total, dtype=int)
            dominates = [[] for _ in range(n_total)]

            for i in range(n_total):
                for j in range(i + 1, n_total):
                    fi = combined_F[i]
                    fj = combined_F[j]
                    if np.all(fi <= fj) and np.any(fi < fj):
                        dominated_count[j] += 1
                        dominates[i].append(j)
                    elif np.all(fj <= fi) and np.any(fj < fi):
                        dominated_count[i] += 1
                        dominates[j].append(i)

            # Front 0
            front0 = [i for i in range(n_total) if dominated_count[i] == 0]
            current_front = front0
            front_id = 0
            while current_front:
                for i in current_front:
                    rank[i] = front_id
                next_front = []
                for i in current_front:
                    for j in dominates[i]:
                        dominated_count[j] -= 1
                        if dominated_count[j] == 0:
                            next_front.append(j)
                current_front = next_front
                front_id += 1

            # Select top pop_size by rank
            selected_indices = np.argsort(rank, kind='stable')[:pop_size]

            X = combined_X[selected_indices]
            F = combined_F[selected_indices]

            # Update archive
            for f in F:
                if not is_dominated(f, archive_F):
                    archive_F = [a for a in archive_F if not (np.all(f <= np.array(a)) and np.any(f < np.array(a)))]
                    archive_F.append(tuple(f))

        return all_F

# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    @staticmethod
    def _dominates(a, b):
        """Check if vector a dominates vector b (minimization)."""
        return np.all(a <= b) and np.any(a < b)

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 300.0 - 10.0  # safety margin
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref_point = np.array([1.1, 1.3, 1.1])

        rng = np.random.default_rng(42)

        # Archive: list of (x, f) tuples for non-dominated points
        archive_X = []
        archive_F = []
        eval_count = 0

        def update_archive(x, f):
            nonlocal archive_X, archive_F
            f = np.asarray(f, dtype=float)
            # Check if new point is dominated by any archive point
            for i in range(len(archive_F)):
                if self._dominates(archive_F[i], f):
                    return
            # Remove archive points dominated by new point
            keep = []
            for i in range(len(archive_F)):
                if self._dominates(f, archive_F[i]):
                    continue
                keep.append(i)
            archive_F = [archive_F[i] for i in keep]
            archive_X = [archive_X[i] for i in keep]
            archive_F.append(f)
            archive_X.append(x)

        def calc_hv(points, ref):
            """
            Calculate the hypervolume of a set of points in 3D.
            Points are assumed to be non-dominated and better than ref.
            Uses a sweep-line approach: sort by f0, then compute 2D union area of remaining boxes.
            """
            if len(points) == 0:
                return 0.0
            
            # Filter points that are strictly better than ref
            pts = [p for p in points if np.all(p < ref)]
            if len(pts) == 0:
                return 0.0
            
            # Sort by first objective
            pts.sort(key=lambda x: x[0])
            
            hv = 0.0
            # The sweep line maintains the "active" boxes that extend beyond the current x-slice
            # For each point i, it contributes a box of width (pts[i+1][0] - pts[i][0]) or (ref[0] - pts[i][0])
            # The height/depth is the union of the 2D projections of the active boxes.
            
            # Since N is small, we can use a simpler O(N^2) or O(N^3) approach for the 2D part.
            # Let's use the standard 3D HV algorithm:
            # Sort by x.
            # For i from 0 to N-1:
            #   width = (pts[i+1][0] if i < N-1 else ref[0]) - pts[i][0]
            #   active = [j for j in range(i, N) if pts[j][1] < ref[1] and pts[j][2] < ref[2]]
            #   Actually, the "active" set for the slice starting at pts[i][0] includes all points j >= i 
            #   that have pts[j][1] < ref[1] and pts[j][2] < ref[2].
            #   Wait, the union of boxes in the yz-plane for points j >= i.
            #   The area of the union of rectangles [pts[j][1], ref[1]] x [pts[j][2], ref[2]].
            
            def union_area_2d(boxes, ref_y, ref_z):
                # boxes: list of (y, z)
                # Union of rectangles [y, ref_y] x [z, ref_z]
                if not boxes:
                    return 0.0
                
                # Sort by y
                boxes.sort(key=lambda x: x[0])
                area = 0.0
                # Sweep in y
                # Keep track of active z-intervals?
                # Since N is small, O(N^2) is fine.
                # For each box i, its unique contribution in y is determined by the next box's y or ref_y.
                # But we must account for overlap in z.
                
                # Simpler: Inclusion-Exclusion is too slow.
                # Let's use the property that all rectangles share the top-right corner (ref_y, ref_z).
                # The union area is the area of the bounding box minus the holes?
                # No, it's the area covered by at least one rectangle.
                # Since they all extend to ref, the union is determined by the "lower envelope" of the y-z coordinates?
                # Actually, for rectangles anchored at top-right, the union area can be computed by sorting by y.
                # Let's just do a grid-based approximation? No, we need exact or close to exact for ranking.
                # Let's use O(N^2) exact calculation for small N.
                # For each point i, consider the vertical strip from y_i to y_{i+1} (or ref_y).
                # In this strip, the "active" rectangles are those with y_j <= y_i (i.e., j <= i in sorted order? No, y_j <= y_current).
                # If sorted by y ascending, for the strip starting at y_i, the active rectangles are those with index <= i?
                # No, a rectangle [y_j, ref_y] is active in [y_i, y_{i+1}] if y_j <= y_i.
                # So active set is {j | y_j <= y_i}.
                # The height of the union in this strip is the union of z-intervals [z_j, ref_z] for j in active set.
                
                # Let's implement this.
                n = len(boxes)
                # Sort by y
                sorted_boxes = sorted(boxes, key=lambda x: x[0])
                total_area = 0.0
                
                for i in range(n):
                    y_start = sorted_boxes[i][0]
                    y_end = sorted_boxes[i+1][0] if i < n-1 else ref_y
                    if y_end <= y_start:
                        continue
                    
                    # Active boxes are those with y_j <= y_start (which is y_i)
                    # Since sorted, these are indices 0..i
                    active_z = [sorted_boxes[j][1] for j in range(i+1)]
                    
                    # Calculate union of intervals [z_j, ref_z]
                    if not active_z:
                        continue
                    
                    # Sort z intervals
                    active_z.sort()
                    current_max_z = -np.inf
                    strip_area = 0.0
                    # The union of [z_j, ref_z] is [min_z, ref_z] minus gaps?
                    # No, since they all go to ref_z, the union is simply [min(active_z), ref_z].
                    # Wait, is that true?
                    # Rectangles: [y_j, ref_y] x [z_j, ref_z].
                    # In the y-strip [y_i, y_{i+1}], the covered z-range is the union of [z_j, ref_z] for all j with y_j <= y_i.
                    # Since all intervals end at ref_z, the union is [min(z_j), ref_z].
                    # Yes! Because if you have intervals [a, C] and [b, C], their union is [min(a,b), C].
                    min_z = min(active_z)
                    height = ref_z - min_z
                    if height > 0:
                        strip_area += (y_end - y_start) * height
                    
                    total_area += strip_area
                
                return total_area

            for i in range(len(pts)):
                x_start = pts[i][0]
                x_end = pts[i+1][0] if i < len(pts)-1 else ref[0]
                if x_end <= x_start:
                    continue
                
                # Active boxes for x-slice [x_start, x_end] are those with x_j <= x_start.
                # Since sorted by x, these are indices 0..i.
                active_yz = [(pts[j][1], pts[j][2]) for j in range(i+1) if pts[j][1] < ref[1] and pts[j][2] < ref[2]]
                
                if not active_yz:
                    continue
                
                area = union_area_2d(active_yz, ref[1], ref[2])
                hv += (x_end - x_start) * area
            
            return hv

        def hv_contribution(index, F_nd, ref):
            """Calculate HV contribution of point at index in F_nd."""
            N = len(F_nd)
            if N == 0:
                return 0.0
            
            # Full HV
            full_hv = calc_hv(F_nd, ref)
            
            # HV without point i
            subset = [F_nd[j] for j in range(N) if j != index]
            subset_hv = calc_hv(subset, ref)
            
            return full_hv - subset_hv

        # Phase 1: Latin Hypercube Sampling for initial coverage
        n_init = max(30, min(100, budget // 3))
        
        # Generate LHS points with stratified sampling
        lxs = np.zeros((n_init, dim))
        for d in range(dim):
            intervals = np.linspace(lo[d], hi[d], n_init + 1)
            perm = rng.permutation(n_init)
            for i in range(n_init):
                row = perm[i]
                lxs[row, d] = intervals[i] + rng.random() * (intervals[i + 1] - intervals[i])
        
        # Evaluate initial population
        for i in range(n_init):
            if time.time() >= deadline:
                break
            x = lxs[i]
            f = problem(x)
            eval_count += 1
            update_archive(x, f)

        # Phase 2: Hypervolume-Indicative Evolutionary Algorithm (HVEA)
        remaining = budget - n_init
        if remaining <= 0:
            return np.array(archive_X) if archive_X else np.array([lo + (hi - lo) * 0.5])

        pop_size = max(15, min(40, remaining // 3))
        
        # Initialize DE population from archive + random
        de_X = []
        de_F = []
        
        if archive_F:
            # Select points with high HV contribution
            if len(archive_F) > 1:
                contribs = [hv_contribution(i, archive_F, ref_point) for i in range(len(archive_F))]
                archive_idx = np.argsort(contribs)[::-1]
            else:
                archive_idx = [0]
            
            for i in archive_idx[:min(pop_size, len(archive_idx))]:
                de_X.append(archive_X[i].copy())
                de_F.append(archive_F[i].copy())
        
        # Fill rest with random points
        while len(de_X) < pop_size:
            x = lo + (hi - lo) * rng.random(dim)
            f = problem(x)
            eval_count += 1
            update_archive(x, f)
            de_X.append(x.copy())
            de_F.append(f.copy())
            
        # DE parameters
        F_factor = 0.9
        CR = 0.95
        
        while eval_count < budget:
            if time.time() >= deadline:
                break
            
            # Select parent based on HV contribution
            if len(archive_F) > 1:
                contribs = [hv_contribution(i, archive_F, ref_point) for i in range(len(archive_F))]
                # Normalize contributions for probability
                contribs = np.array(contribs)
                if contribs.sum() > 0:
                    probs = contribs / contribs.sum()
                    parent_idx = rng.choice(len(archive_F), p=probs)
                else:
                    parent_idx = rng.integers(len(archive_F))
            else:
                parent_idx = 0
            
            base = archive_X[parent_idx].copy()
            base_f = archive_F[parent_idx].copy()
            
            # Generate two random distinct indices from archive (or population if archive is small)
            if len(archive_X) >= 3:
                r1_idx, r2_idx = rng.choice(len(archive_X), size=2, replace=False)
                r1 = archive_X[r1_idx]
                r2 = archive_X[r2_idx]
            elif len(de_X) >= 3:
                r1_idx, r2_idx = rng.choice(len(de_X), size=2, replace=False)
                r1 = de_X[r1_idx]
                r2 = de_X[r2_idx]
            else:
                r1 = base + rng.random(dim) * 0.1
                r2 = base + rng.random(dim) * 0.1
            
            # Mutation
            mutant = base + F_factor * (r1 - r2)
            mutant = np.clip(mutant, lo, hi)
            
            # Crossover
            trial = np.zeros(dim)
            jrand = rng.integers(dim)
            for j in range(dim):
                if rng.random() < CR or j == jrand:
                    trial[j] = mutant[j]
                else:
                    trial[j] = base[j]
            
            # Evaluate
            f_trial = problem(trial)
            eval_count += 1
            update_archive(trial, f_trial)
            
            # Update DE population: replace worst point (by sum) or random if needed
            if len(de_X) > 0:
                sums = np.array([np.sum(f) for f in de_F])
                worst_idx = np.argmax(sums)
                if np.sum(f_trial) < sums[worst_idx]:
                    de_X[worst_idx] = trial.copy()
                    de_F[worst_idx] = f_trial.copy()
                # Also occasionally replace a random point to maintain diversity
                elif rng.random() < 0.1:
                    rand_idx = rng.integers(len(de_X))
                    de_X[rand_idx] = trial.copy()
                    de_F[rand_idx] = f_trial.copy()

        # Phase 3: Local refinement with multiple weight vectors
        if archive_F:
            # Select top points from archive based on HV contribution
            if len(archive_F) > 1:
                contribs = [hv_contribution(i, archive_F, ref_point) for i in range(len(archive_F))]
                archive_idx = np.argsort(contribs)[::-1]
            else:
                archive_idx = [0]
            
            n_refine = min(10, len(archive_idx), max(3, (budget - eval_count) // 4))
            
            # Structured weight grid
            weights_list = [
                np.ones(n_obj) / n_obj,
                np.array([0.6, 0.2, 0.2]),
                np.array([0.2, 0.6, 0.2]),
                np.array([0.2, 0.2, 0.6]),
                np.array([0.4, 0.35, 0.25]),
                np.array([0.25, 0.35, 0.4]),
                np.array([0.35, 0.4, 0.25]),
                np.array([0.45, 0.3, 0.25]),
                np.array([0.25, 0.45, 0.3]),
                np.array([0.3, 0.25, 0.45]),
            ]
            
            for i in range(n_refine):
                if eval_count >= budget:
                    break
                if time.time() >= deadline:
                    break
                
                x0 = archive_X[archive_idx[i]].copy()
                
                for w in weights_list:
                    if eval_count >= budget:
                        break
                    if time.time() >= deadline:
                        break
                    
                    def objective(x, w=w):
                        nonlocal eval_count
                        f = problem(x)
                        eval_count += 1
                        return np.dot(w, f)
                    
                    try:
                        result = minimize(objective, x0, method='Nelder-Mead',
                                        bounds=list(zip(lo, hi)),
                                        options={'maxiter': 50, 'xatol': 1e-5, 'fatol': 1e-5})
                        if result.fun < np.inf:
                            x_best = result.x
                            f_best = problem(x_best)
                            eval_count += 1
                            update_archive(x_best, f_best)
                    except:
                        pass

        # Return the archive
        if archive_X:
            return np.array(archive_X)
        return np.array([lo + (hi - lo) * 0.5])
# EVOLVE-BLOCK-END

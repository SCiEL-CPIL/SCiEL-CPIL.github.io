"""Multi-objective optimizer for WFG7 using hypervolume-driven search."""
import time
import numpy as np


class Optimizer:
    """Evolutionary multi-objective optimizer with exact hypervolume-based selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _calc_hv_3d(self, points, ref):
        """
        Calculate the exact hypervolume of a set of points in 3D.
        points: Nx3 array
        ref: 3-array
        """
        if len(points) == 0:
            return 0.0
        
        # Filter points that are better than ref in all objectives
        mask = np.all(points < ref, axis=1)
        pts = points[mask]
        
        if len(pts) == 0:
            return 0.0
            
        # Sort by first objective (f0)
        idx = np.argsort(pts[:, 0])
        pts = pts[idx]
        
        def area_2d(points_2d, ref_2d):
            """
            Calculate area of union of rectangles in 2D.
            points_2d: Mx2 array
            ref_2d: 2-array
            """
            if len(points_2d) == 0:
                return 0.0
            
            # Sort by first coordinate
            idx2 = np.argsort(points_2d[:, 0])
            p2 = points_2d[idx2]
            
            total_area = 0.0
            active_intervals = []
            
            for i in range(len(p2)):
                # Add current point's interval [p2[i][1], ref_2d[1]]
                active_intervals.append((p2[i][1], ref_2d[1]))
                
                # Determine width of this slice
                if i == len(p2) - 1:
                    width = ref_2d[0] - p2[i][0]
                else:
                    width = p2[i+1][0] - p2[i][0]
                
                # Calculate union length of active intervals
                if not active_intervals:
                    h = 0.0
                else:
                    # Sort by start
                    active_intervals.sort(key=lambda x: x[0])
                    merged = []
                    for interval in active_intervals:
                        if not merged:
                            merged.append([interval[0], interval[1]])
                        else:
                            last = merged[-1]
                            if interval[0] <= last[1]:
                                last[1] = max(last[1], interval[1])
                            else:
                                merged.append([interval[0], interval[1]])
                    h = sum(end - start for start, end in merged)
                
                total_area += width * h
                
            return total_area

        total_hv = 0.0
        ref_2d = ref[1:]
        
        for i in range(len(pts)):
            # Get projections of pts[0..i] onto (f1, f2) plane
            proj = pts[0:i+1, 1:]
            a = area_2d(proj, ref_2d)
            
            if i == len(pts) - 1:
                width = ref[0] - pts[i][0]
            else:
                width = pts[i+1][0] - pts[i][0]
            
            total_hv += width * a
            
        return total_hv

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        width = hi - lo

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume (scoring reference)
        ref = np.array([3.0, 5.0, 7.0])

        # --- Initial population ---
        pop_size = 24
        X = lo + width * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size

        # --- Main search loop ---
        while used < budget:
            if time.time() > deadline - 5.0:
                break

            # Differential Evolution style mutation
            # Pick 3 distinct parents
            i1, i2, i3 = np.random.choice(pop_size, 3, replace=False)
            
            # DE/rand/1 mutation: v = x_i1 + F * (x_i2 - x_i3)
            F_factor = 0.8
            v = X[i1] + F_factor * (X[i2] - X[i3])
            
            # Add some Gaussian noise for exploration
            v += 0.1 * width * np.random.standard_normal(dim)
            v = np.clip(v, lo, hi)
            
            fc = problem(v)
            used += 1
            
            # Add to population
            X = np.vstack([X, v])
            F = np.vstack([F, fc])
            
            # If population exceeds size, remove worst point based on HV contribution
            if len(X) > pop_size:
                # Calculate total HV
                hv_total = self._calc_hv_3d(F, ref)
                
                min_contrib = float('inf')
                min_idx = -1
                
                for i in range(len(X)):
                    # Create F without i
                    mask = np.ones(len(X), dtype=bool)
                    mask[i] = False
                    F_without_i = F[mask]
                    hv_without_i = self._calc_hv_3d(F_without_i, ref)
                    contrib = hv_total - hv_without_i
                    
                    if contrib < min_contrib:
                        min_contrib = contrib
                        min_idx = i
                
                # Remove the point with minimum contribution
                X = np.delete(X, min_idx, axis=0)
                F = np.delete(F, min_idx, axis=0)
            
            # Adaptive mutation scale
            progress = used / budget
            if progress < 0.5:
                F_factor = 0.8
            else:
                F_factor = 0.5

        return X

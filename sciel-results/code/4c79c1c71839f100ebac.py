"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """NSGA-II with Hypervolume-based selection for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.ref_point = np.array([3.0, 5.0, 7.0])

    def _non_dominated_sort(self, F):
        n = len(F)
        S = [[] for _ in range(n)]
        domination_count = np.zeros(n, dtype=int)
        fronts = []

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    domination_count[i] += 1

        current_front = [i for i in range(n) if domination_count[i] == 0]
        if not current_front:
            return [list(range(n))]

        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in S[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front = next_front

        assigned = set()
        for f in fronts:
            for i in f:
                assigned.add(i)
        if len(assigned) < n:
            remaining = [i for i in range(n) if i not in assigned]
            fronts.append(remaining)

        return fronts

    def _hv_2d_union_area(self, rects):
        """Calculate area of union of rectangles in 2D. rects: list of (x1, y1, x2, y2)"""
        if not rects:
            return 0.0
        
        # Collect unique x coordinates
        xs = set()
        for r in rects:
            xs.add(r[0])
            xs.add(r[2])
        xs = sorted(list(xs))
        
        if len(xs) < 2:
            return 0.0
            
        total_area = 0.0
        for i in range(len(xs) - 1):
            x1 = xs[i]
            x2 = xs[i+1]
            width = x2 - x1
            if width <= 0:
                continue
            
            # Find y-intervals that cover this x-slice
            y_intervals = []
            for r in rects:
                if r[0] <= x1 and r[2] >= x2: # Rectangle covers this x-slice
                    y_intervals.append((r[1], r[3]))
            
            if not y_intervals:
                continue
                
            # Merge y-intervals
            y_intervals.sort(key=lambda k: k[0])
            merged = []
            for y1, y2 in y_intervals:
                if not merged:
                    merged.append([y1, y2])
                else:
                    if y1 <= merged[-1][1]:
                        merged[-1][1] = max(merged[-1][1], y2)
                    else:
                        merged.append([y1, y2])
                        
            slice_area = sum(y2 - y1 for y1, y2 in merged)
            total_area += width * slice_area
            
        return total_area

    def _hypervolume_3d(self, points, ref):
        """Calculate hypervolume of points in 3D with reference point ref."""
        # Filter points strictly better than ref
        mask = np.all(points < ref, axis=1)
        points = points[mask]
        if len(points) == 0:
            return 0.0
            
        # Sort by first objective (f1)
        points = points[np.argsort(points[:, 0])]
        
        total_hv = 0.0
        n = len(points)
        
        for i in range(n):
            p = points[i]
            # Box volume for point i
            box_vol = (ref[0] - p[0]) * (ref[1] - p[1]) * (ref[2] - p[2])
            
            # Calculate volume of intersection with previous points
            # We need the area of the union of rectangles in f2-f3 plane
            # defined by points j < i
            
            rects = []
            for j in range(i):
                pj = points[j]
                # Intersection in f2: [max(pj[1], p[1]), ref[1]]
                # Intersection in f3: [max(pj[2], p[2]), ref[2]]
                x1 = max(pj[1], p[1])
                x2 = ref[1]
                y1 = max(pj[2], p[2])
                y2 = ref[2]
                
                if x1 < x2 and y1 < y2:
                    rects.append((x1, y1, x2, y2))
                    
            if not rects:
                total_hv += box_vol
            else:
                # Calculate area of union of these rects
                # Note: The "width" in f1 is (ref[0] - p[0])
                # The volume to subtract is Area_Union * (ref[0] - p[0])
                area_union = self._hv_2d_union_area(rects)
                subtract_vol = area_union * (ref[0] - p[0])
                total_hv += max(0, box_vol - subtract_vol)
                
        return total_hv

    def _hv_contribution(self, points, ref, idx):
        """Calculate HV contribution of points[idx] to the set points."""
        # Full set
        full_hv = self._hypervolume_3d(points, ref)
        # Set without idx
        mask = np.ones(len(points), dtype=bool)
        mask[idx] = False
        reduced_hv = self._hypervolume_3d(points[mask], ref)
        return full_hv - reduced_hv

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        rng = np.random.default_rng(42)

        # Population size 100 to maximize spread with 200 budget
        pop_size = 100
        
        # Initial population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X], dtype=float)
        used = pop_size

        # If budget allows, do one generation of offspring
        if used < self.budget:
            n_offspring = min(pop_size, self.budget - used)
            
            # Create offspring
            offspring_X = []
            for _ in range(n_offspring):
                # Tournament selection
                i = rng.integers(pop_size)
                j = rng.integers(pop_size)
                # Simple domination check for tournament
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    p1 = X[i]
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    p1 = X[j]
                else:
                    p1 = X[i] if rng.random() < 0.5 else X[j]
                    
                k = rng.integers(pop_size)
                l = rng.integers(pop_size)
                if np.all(F[k] <= F[l]) and np.any(F[k] < F[l]):
                    p2 = X[k]
                elif np.all(F[l] <= F[k]) and np.any(F[l] < F[k]):
                    p2 = X[l]
                else:
                    p2 = X[k] if rng.random() < 0.5 else X[l]

                # SBX Crossover
                child = p1.copy()
                if rng.random() < 0.9: # Crossover probability
                    eta_c = 20.0
                    for d in range(self.dim):
                        if rng.random() < 0.5:
                            if abs(p1[d] - p2[d]) > 1e-14:
                                if p1[d] < p2[d]:
                                    x1, x2 = p1[d], p2[d]
                                else:
                                    x1, x2 = p2[d], p1[d]
                                    
                                u = rng.random()
                                if u <= 1.0 / (eta_c + 1.0):
                                    beta = (2.0 * u) ** (1.0 / (eta_c + 1.0))
                                elif u >= 1.0 - 1.0 / (eta_c + 1.0):
                                    beta = (2.0 * (1.0 - u)) ** (1.0 / (eta_c + 1.0))
                                else:
                                    beta = 1.0
                                    
                                c1 = 0.5 * ((1.0 + beta) * x1 + (1.0 - beta) * x2)
                                c2 = 0.5 * ((1.0 - beta) * x1 + (1.0 + beta) * x2)
                                
                                if rng.random() < 0.5:
                                    child[d] = c1
                                else:
                                    child[d] = c2
                            
                # Polynomial Mutation
                eta_m = 20.0
                for d in range(self.dim):
                    if rng.random() < 1.0 / self.dim:
                        u = rng.random()
                        if u < 0.5:
                            xy = (child[d] - lo[d]) / (hi[d] - lo[d])
                            beta = (1.0 + (self.dim + 1.0) * u) ** (1.0 / (eta_m + 1.0))
                            child[d] = child[d] - (hi[d] - lo[d]) * ((1.0 - xy) ** beta)
                        else:
                            xy = (hi[d] - child[d]) / (hi[d] - lo[d])
                            beta = (1.0 + (self.dim + 1.0) * (1.0 - u)) ** (1.0 / (eta_m + 1.0))
                            child[d] = child[d] + (hi[d] - lo[d]) * ((1.0 - xy) ** beta)
                            
                child = np.clip(child, lo, hi)
                offspring_X.append(child)

            offspring_X = np.array(offspring_X)
            offspring_F = np.array([problem(x) for x in offspring_X], dtype=float)
            used += n_offspring

            # Combine
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            
            # Non-dominated sort
            fronts = self._non_dominated_sort(combined_F)
            
            # Select next population using Hypervolume-based selection
            next_X = []
            next_F = []
            remaining_to_select = pop_size
            
            for front in fronts:
                if len(front) <= remaining_to_select:
                    for idx in front:
                        next_X.append(combined_X[idx])
                        next_F.append(combined_F[idx])
                    remaining_to_select -= len(front)
                else:
                    # Need to select remaining_to_select from this front
                    # Use Greedy HV Selection
                    candidates = list(front)
                    selected = []
                    
                    # Pre-calculate HV of current selected set (which is empty for this front initially, 
                    # but we have the fixed points from previous fronts)
                    # Actually, we should consider the HV contribution relative to the ALREADY SELECTED points.
                    
                    fixed_X = np.array(next_X) if next_X else np.empty((0, self.dim))
                    fixed_F = np.array(next_F) if next_F else np.empty((0, self.n_obj))
                    
                    # We want to maximize HV(fixed_F union selected_from_front)
                    # Greedy:
                    for _ in range(remaining_to_select):
                        best_idx = -1
                        best_contrib = -1.0
                        
                        # Current set to test against
                        current_selected_F = np.array([combined_F[i] for i in selected]) if selected else np.empty((0, self.n_obj))
                        
                        # We need to calculate HV(fixed_F union current_selected_F union {candidate})
                        # vs HV(fixed_F union current_selected_F)
                        
                        base_F = np.vstack([fixed_F, current_selected_F]) if (fixed_F.size > 0 or current_selected_F.size > 0) else np.empty((0, self.n_obj))
                        base_hv = self._hypervolume_3d(base_F, self.ref_point)
                        
                        for idx in candidates:
                            if idx in selected:
                                continue
                            test_F = np.vstack([base_F, combined_F[idx]])
                            test_hv = self._hypervolume_3d(test_F, self.ref_point)
                            contrib = test_hv - base_hv
                            if contrib > best_contrib:
                                best_contrib = contrib
                                best_idx = idx
                                
                        if best_idx != -1:
                            selected.append(best_idx)
                            next_X.append(combined_X[best_idx])
                            next_F.append(combined_F[best_idx])
                        else:
                            break
                    
                    remaining_to_select = 0
                    break

            X = np.array(next_X)
            F = np.array(next_F)

        # Return non-dominated points
        fronts = self._non_dominated_sort(F)
        best_front = fronts[0] if fronts else [0]
        return X[best_front]
# EVOLVE-BLOCK-END

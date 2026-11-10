"""Multi-objective optimizer on WFG4 with hypervolume-estimation-based island model."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Island model with hypervolume-estimation-based sub-regions for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def _dominates(self, a, b):
        """Return True if a dominates b (a <= b in all, a < b in at least one)."""
        return np.all(a <= b) and np.any(a < b)

    def _non_dominated_sort(self, F):
        """Return list of fronts (each a list of indices) via fast non-dominated sort."""
        n = len(F)
        S = [[] for _ in range(n)]
        nd = [0] * n
        rank = [0] * n
        fronts = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(F[i], F[j]):
                    S[i].append(j)
                elif self._dominates(F[j], F[i]):
                    nd[i] += 1
            if nd[i] == 0:
                rank[i] = 0
                fronts.append([i])
        front_idx = 0
        while front_idx < len(fronts):
            next_front = []
            for i in fronts[front_idx]:
                for j in S[i]:
                    nd[j] -= 1
                    if nd[j] == 0:
                        rank[j] = front_idx + 1
                        next_front.append(j)
            front_idx += 1
            if next_front:
                fronts.append(next_front)
        return fronts

    def _hv_3d(self, pts, ref):
        """Compute 3D hypervolume for points dominated by ref using exact algorithm."""
        n = len(pts)
        if n == 0:
            return 0.0
        # Filter points strictly better than ref
        mask = np.all(pts < ref, axis=1)
        pts = pts[mask]
        n = len(pts)
        if n == 0:
            return 0.0
        
        # Sort by first objective
        order = np.argsort(pts[:, 0])
        pts = pts[order]
        
        hv = 0.0
        # For each point, compute the 2D area it contributes in the yz-plane
        # that is not covered by points with smaller x1
        active_yz = []  # list of (y, z) pairs for points with smaller x1
        
        for i in range(n):
            x1, y, z = pts[i]
            # Check if this point is dominated in yz by any active point
            dominated_yz = False
            for ay, az in active_yz:
                if ay <= y and az <= z and (ay < y or az < z):
                    dominated_yz = True
                    break
            
            if not dominated_yz:
                # Compute the 2D area of [y, ref[1]] x [z, ref[2]] not covered by active points
                area = self._area_2d_exact(y, z, ref[1], ref[2], active_yz)
                hv += (ref[0] - x1) * area
            
            active_yz.append((y, z))
        
        return hv

    def _area_2d_exact(self, y, z, ref_y, ref_z, active):
        """Compute exact 2D area of [y, ref_y] x [z, ref_z] not covered by active rectangles."""
        full_area = (ref_y - y) * (ref_z - z)
        if full_area <= 0:
            return 0.0
        
        if len(active) == 0:
            return full_area
        
        # For exact computation, we need to find the union of intersections
        # The uncovered area = full_area - union of covered areas
        # For small active sets, use a sweep line or grid approach
        
        # Collect all y and z coordinates
        ys = [y, ref_y]
        zs = [z, ref_z]
        for ay, az in active:
            if ay >= y and ay <= ref_y:
                ys.append(ay)
            if az >= z and az <= ref_z:
                zs.append(az)
        
        ys = sorted(set(ys))
        zs = sorted(set(zs))
        
        # Create a grid and check which cells are covered
        covered = 0.0
        for i in range(len(ys) - 1):
            for j in range(len(zs) - 1):
                cell_y1, cell_y2 = ys[i], ys[i+1]
                cell_z1, cell_z2 = zs[j], zs[j+1]
                if cell_y2 <= cell_y1 or cell_z2 <= cell_z1:
                    continue
                
                # Check if this cell is covered by any active rectangle
                cell_cy = (cell_y1 + cell_y2) / 2
                cell_cz = (cell_z1 + cell_z2) / 2
                
                is_covered = False
                for ay, az in active:
                    if ay <= cell_cy <= ref_y and az <= cell_cz <= ref_z:
                        # Check if the cell is within [ay, ref_y] x [az, ref_z]
                        if ay <= cell_y1 and az <= cell_z1 and ref_y >= cell_y2 and ref_z >= cell_z2:
                            is_covered = True
                            break
                
                if not is_covered:
                    covered += (cell_y2 - cell_y1) * (cell_z2 - cell_z1)
        
        return covered

    def __call__(self, problem, bounds):
        deadline = time.time() + 280.0
        dim, n_obj = self.dim, self.n_obj
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        rng = np.random
        ref = np.array([3.0, 5.0, 7.0])
        
        # Define weight regions for island model
        # Each island focuses on a different weight vector
        num_islands = 4
        weights = np.array([
            [0.5, 0.25, 0.25],
            [0.25, 0.5, 0.25],
            [0.25, 0.25, 0.5],
            [1/3, 1/3, 1/3]
        ])
        
        pop_per_island = 8
        evals = 0
        
        # Initialize islands
        islands = []
        for k in range(num_islands):
            X_island = np.zeros((pop_per_island, dim))
            F_island = np.zeros((pop_per_island, n_obj))
            
            # LHS initialization for this island
            lhs = np.zeros((pop_per_island, dim))
            for d in range(dim):
                bins = np.linspace(0, 1, pop_per_island + 1)
                pts = rng.uniform(bins[:-1], bins[1:])
                rng.shuffle(pts)
                lhs[:, d] = pts
            X_island = lo + lhs * span
            
            for j in range(pop_per_island):
                F_island[j] = problem(X_island[j])
                evals += 1
            
            islands.append({'X': X_island, 'F': F_island})
        
        # Evolution loop
        migration_interval = 10
        gen = 0
        
        while evals < self.budget:
            if time.time() > deadline:
                break
            
            gen += 1
            progress = evals / self.budget
            
            for k in range(num_islands):
                if evals >= self.budget:
                    break
                
                X = islands[k]['X']
                F = islands[k]['F']
                weight = weights[k]
                
                # Non-dominated sort within this island
                fronts = self._non_dominated_sort(F)
                
                # Select parent from first front
                if len(fronts[0]) >= 2:
                    parent_idx = rng.randint(len(fronts[0]))
                    parent_idx = fronts[0][parent_idx]
                else:
                    parent_idx = rng.randint(pop_per_island)
                
                parent = X[parent_idx].copy()
                
                # Weight-specific mutation intensity
                # Islands with more balanced weights use smaller steps
                weight_spread = np.max(weight) - np.min(weight)
                base_sigma = 0.25 * (1 - progress) + 0.05
                sigma = span * base_sigma * (1.0 + weight_spread)
                
                child = parent + sigma * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)
                
                fc = problem(child)
                evals += 1
                
                # Replacement: accept if child is non-dominated in this island
                dominated = False
                for j in range(pop_per_island):
                    if self._dominates(F[j], fc):
                        dominated = True
                        break
                
                if not dominated:
                    # Replace worst point in this island
                    if len(fronts) > 1:
                        last_front = fronts[-1]
                        replace_idx = last_front[rng.randint(len(last_front))]
                    else:
                        replace_idx = rng.randint(pop_per_island)
                    
                    X[replace_idx] = child
                    F[replace_idx] = fc
            
            # Migration: every migration_interval generations, share best points
            if gen % migration_interval == 0:
                for k in range(num_islands):
                    if evals >= self.budget:
                        break
                    
                    # Find the best point in this island based on weighted sum
                    F = islands[k]['F']
                    weighted_scores = F @ weights[k]
                    best_local_idx = np.argmin(weighted_scores)
                    
                    # Find worst point in this island
                    fronts = self._non_dominated_sort(F)
                    if len(fronts) > 1:
                        last_front = fronts[-1]
                        worst_idx = last_front[0]
                    else:
                        worst_idx = np.argmax(weighted_scores)
                    
                    # Check if best local point is better than worst
                    if not self._dominates(F[worst_idx], F[best_local_idx]):
                        # Migrate: replace worst with best from another island
                        # Pick a random other island
                        other_islands = [m for m in range(num_islands) if m != k]
                        other_k = other_islands[rng.randint(len(other_islands))]
                        
                        F_other = islands[other_k]['F']
                        X_other = islands[other_k]['X']
                        
                        # Find best in other island
                        weighted_other = F_other @ weights[other_k]
                        best_other_idx = np.argmin(weighted_other)
                        
                        # Only migrate if it's actually better
                        if not self._dominates(F[worst_idx], F_other[best_other_idx]):
                            islands[k]['X'][worst_idx] = X_other[best_other_idx].copy()
                            islands[k]['F'][worst_idx] = F_other[best_other_idx].copy()
        
        # Combine all islands and return the best non-dominated set
        all_X = np.vstack([island['X'] for island in islands])
        all_F = np.vstack([island['F'] for island in islands])
        
        # Return the full population (evaluator will compute HV from all recorded points)
        return all_X
# EVOLVE-BLOCK-END

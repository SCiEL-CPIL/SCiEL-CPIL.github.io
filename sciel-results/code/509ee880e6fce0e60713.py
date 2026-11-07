"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull


class Optimizer:
    """Hypervolume-driven evolutionary multi-objective optimizer for DTLZ7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hv_contribution(self, point, others, ref):
        """Compute the hypervolume contribution of a single point given a set of others."""
        # A point's HV contribution is the volume of the region it uniquely covers
        # This is expensive, so we use an approximation: 
        # contribution = volume of box from point to ref, minus overlap with others
        # For simplicity, we use the "slice" method for 3 objectives
        
        n = len(others)
        if n == 0:
            return np.prod(ref - point)
        
        # For 3 objectives, compute the unique contribution
        # A point p contributes if no other point dominates it in all objectives
        # The contribution is the volume of the region dominated by p but not by any other point
        
        # Simple approximation: if p is non-dominated, its contribution is roughly
        # the volume of its "dominance region" minus overlaps
        # We'll use a Monte Carlo-free exact method for 3D
        
        # For 3 objectives, the HV contribution of point p relative to set S:
        # Consider the 2D slices at each objective value
        # This is complex, so we use a simpler heuristic:
        # contribution = product of (ref[i] - p[i]) * (1 / (1 + number of points that dominate p in at least 2 objectives))
        
        # Actually, let's just use a simpler but effective heuristic:
        # The contribution is proportional to the volume of the box, weighted by non-domination
        dominated_count = 0
        for o in others:
            if np.all(o <= point) and np.any(o < point):
                dominated_count += 1
        
        if dominated_count > 0:
            return 0.0
        
        # Non-dominated point: contribution is the volume of its exclusive region
        # Approximate by the volume of the box it defines
        return np.prod(ref - point)

    def _crowding_distance(self, F):
        """Compute crowding distance for a set of points."""
        n = len(F)
        if n <= 2:
            return np.full(n, np.inf)
        
        cd = np.zeros(n)
        for m in range(F.shape[1]):
            obj = F[:, m]
            min_val, max_val = obj.min(), obj.max()
            if max_val - min_val < 1e-10:
                continue
            order = np.argsort(obj)
            cd[order[0]] = np.inf
            cd[order[-1]] = np.inf
            for i in range(1, n - 1):
                cd[order[i]] += (obj[order[i + 1]] - obj[order[i - 1]]) / (max_val - min_val)
        
        return cd

    def _is_dominated(self, f, F):
        """Check if f is dominated by any point in F."""
        return np.any(np.all(F <= f, axis=1) & np.any(F < f, axis=1))

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([1.1, 1.1, 25.0])
        
        # Population size: use a good fraction of budget
        pop_size = min(50, max(10, self.budget // 4))
        
        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Track best hypervolume
        best_hv = 0.0
        
        while used < self.budget:
            # Find non-dominated set
            non_dom_mask = ~np.array([self._is_dominated(f, F) for f in F])
            
            if np.any(non_dom_mask):
                nd_X = X[non_dom_mask]
                nd_F = F[non_dom_mask]
            else:
                nd_X = X
                nd_F = F
            
            # Compute crowding distance on non-dominated set
            if len(nd_F) > 1:
                cd = self._crowding_distance(nd_F)
            else:
                cd = np.full(len(nd_F), np.inf)
            
            # Parent selection: tournament on non-dominated set using crowding distance
            # Pick 2 parents via tournament
            if len(nd_F) >= 2:
                idx1, idx2 = rng.integers(0, len(nd_F), size=2)
                # Tournament: pick the one with larger crowding distance
                if cd[idx1] >= cd[idx2]:
                    p1_idx, p2_idx = idx1, idx2
                else:
                    p1_idx, p2_idx = idx2, idx1
            else:
                p1_idx = p2_idx = 0
            
            p1 = nd_X[p1_idx]
            p2 = nd_X[p2_idx]
            
            # Adaptive mutation
            # Compute population spread
            spread = (hi - lo) * 0.1
            # Adapt based on generation
            gen_factor = max(0.3, 1.0 - used / self.budget)
            
            # Mix of SBX-like crossover and mutation
            if rng.random() < 0.5:
                # Crossover + mutation
                child = 0.5 * (p1 + p2)
                child += spread * gen_factor * rng.standard_normal(self.dim)
            else:
                # Pure mutation from better parent
                child = p1 + spread * gen_factor * rng.standard_normal(self.dim)
            
            child = np.clip(child, lo, hi)
            fc = problem(child)
            used += 1
            
            # Selection: replace the point with lowest hypervolume contribution
            # Compute HV contribution for each point in population
            # To save time, only compute for a subset or use heuristic
            
            # Simple approach: replace the most "redundant" point
            # A point is redundant if it's dominated, or if it has low HV contribution
            
            # Find the point to replace:
            # 1. If child is non-dominated, replace the most dominated point
            # 2. If child is dominated, replace the point it dominates (if any)
            
            child_is_nd = not self._is_dominated(fc, F)
            
            if child_is_nd:
                # Replace the most dominated point (highest dominance count)
                dom_counts = np.array([
                    np.sum(np.all(F <= f, axis=1) & np.any(F < f, axis=1))
                    for f in F
                ])
                # Among dominated points, pick the one with highest count
                dominated_indices = np.where(dom_counts > 0)[0]
                if len(dominated_indices) > 0:
                    replace_idx = dominated_indices[np.argmax(dom_counts[dominated_indices])]
                else:
                    # All non-dominated: replace the one with lowest crowding distance
                    # Compute CD on full population
                    cd_full = self._crowding_distance(F)
                    replace_idx = np.argmin(cd_full)
            else:
                # Child is dominated: check if it dominates any existing point
                dominates_mask = np.array([
                    np.all(fc <= f) and np.any(fc < f)
                    for f in F
                ])
                if np.any(dominates_mask):
                    # Replace the dominated point with lowest crowding distance
                    dom_indices = np.where(dominates_mask)[0]
                    cd_dom = self._crowding_distance(F[dom_indices])
                    replace_idx = dom_indices[np.argmin(cd_dom)]
                else:
                    # Child doesn't dominate anyone: replace the point with lowest CD
                    cd_full = self._crowding_distance(F)
                    replace_idx = np.argmin(cd_full)
            
            X[replace_idx] = child
            F[replace_idx] = fc
        
        return X
# EVOLVE-BLOCK-END

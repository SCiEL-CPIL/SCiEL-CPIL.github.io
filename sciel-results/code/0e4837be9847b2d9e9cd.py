"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import qmc


class Optimizer:
    """
    Multi-objective optimizer for RE34 (5 vars, 3 objectives).
    Phase 1: Sobol quasi-random sampling for uniform coverage (first 150 evals).
    Phase 2: Hypervolume-contribution-based local search (remaining 50 evals).
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hypervolume_3d(self, points, ref):
        """
        Compute exact 3D hypervolume for a set of points (all better than ref).
        points: (N, 3) array
        ref: (3,) array
        Returns: hypervolume scalar.
        """
        if len(points) == 0:
            return 0.0
        # Sort by first objective (ascending, i.e., better first)
        idx = np.argsort(points[:, 0])
        p = points[idx]
        r = ref
        # Hypervolume contribution of each point in 3D:
        # For point i (sorted by f0), contribution = (f0_{i+1} - f0_i) * area_2d of points j>=i in (f1,f2)
        # But we need to be careful: standard O(n^2) or O(n log n) for 3D.
        # Use a simple O(n^2) approach: for each point, compute the volume it uniquely contributes.
        # Actually, let's use the standard recursive/iterative approach.
        # For 3D, we can do:
        # Sort by f0. For each i from 0 to n-1:
        #   width_i = (p[i+1,0] if i < n-1 else r[0]) - p[i,0]
        #   But we need the 2D hypervolume of points i..n-1 projected on (f1,f2) with ref (r[1],r[2])
        # This is O(n^2) which is fine for n<=200.
        
        n = len(p)
        hv = 0.0
        for i in range(n):
            # Width in dimension 0
            f0_next = p[i + 1, 0] if i < n - 1 else r[0]
            width = f0_next - p[i, 0]
            if width <= 0:
                continue
            # 2D hypervolume of points i..n-1 in (f1, f2) space, ref (r[1], r[2])
            # Only consider points that are better than ref in f1 and f2
            sub = p[i:, [1, 2]]
            sub = sub[(sub[:, 0] < r[1]) & (sub[:, 1] < r[2])]
            if len(sub) == 0:
                continue
            # 2D hypervolume: sort by f1
            sub_idx = np.argsort(sub[:, 0])
            sub_sorted = sub[sub_idx]
            hv_2d = 0.0
            for j in range(len(sub_sorted)):
                f1_next = sub_sorted[j + 1, 0] if j < len(sub_sorted) - 1 else r[1]
                w1 = f1_next - sub_sorted[j, 0]
                if w1 <= 0:
                    continue
                # Max extent in f2 for points j..end
                max_f2 = np.max(sub_sorted[j:, 1])
                h2 = r[2] - max_f2
                if h2 <= 0:
                    continue
                hv_2d += w1 * h2
            hv += width * hv_2d
        return hv

    def _marginal_hv_contributions(self, X, F, ref):
        """
        Compute marginal hypervolume contribution of each point in the set.
        Returns array of contributions (same length as X).
        """
        n = len(X)
        if n == 0:
            return np.array([])
        # Total HV
        total_hv = self._hypervolume_3d(F, ref)
        # For each point, HV without it
        contribs = np.zeros(n)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            hv_without = self._hypervolume_3d(F[mask], ref)
            contribs[i] = total_hv - hv_without
        return contribs

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Reference point for RE34
        ref = np.array([1705.0, 12.0, 0.3])
        
        # Phase 1: Sobol quasi-random sampling
        n_sobol = min(150, budget)
        n_local = budget - n_sobol
        
        X = []
        F = []
        
        # Generate Sobol sequence
        sampler = qmc.Sobol(d=dim, scramble=True)
        # We need to generate n_sobol points
        sobol_points = sampler.random(n_sobol)
        # Map to [lo, hi]
        sobol_points = lo + (hi - lo) * sobol_points
        
        for i in range(n_sobol):
            x = sobol_points[i]
            f = np.array(problem(x))
            X.append(x)
            F.append(f)
        
        X = np.array(X)
        F = np.array(F)
        
        # Phase 2: Hypervolume-contribution-based local search
        if n_local > 0 and len(X) > 0:
            for _ in range(n_local):
                # Compute marginal contributions
                contribs = self._marginal_hv_contributions(X, F, ref)
                # Find the point with the lowest contribution
                worst_idx = int(np.argmin(contribs))
                
                # Generate a Gaussian perturbation of the worst point
                # Adaptive step size based on bounds
                scale = 0.1 * (hi - lo)
                perturbation = scale * rng.standard_normal(dim)
                new_x = X[worst_idx] + perturbation
                new_x = np.clip(new_x, lo, hi)
                
                new_f = np.array(problem(new_x))
                
                # Replace the worst point
                X[worst_idx] = new_x
                F[worst_idx] = new_f
        
        return X

    def _hypervolume_3d(self, points, ref):
        """
        Compute exact 3D hypervolume for a set of points (all better than ref).
        points: (N, 3) array
        ref: (3,) array
        Returns: hypervolume scalar.
        """
        if len(points) == 0:
            return 0.0
        
        # Filter points strictly better than reference in all objectives
        mask = np.all(points < ref, axis=1)
        p = points[mask]
        if len(p) == 0:
            return 0.0
        
        # Sort by first objective (ascending)
        idx = np.argsort(p[:, 0])
        p = p[idx]
        r = ref
        n = len(p)
        
        hv = 0.0
        for i in range(n):
            f0_next = p[i + 1, 0] if i < n - 1 else r[0]
            width = f0_next - p[i, 0]
            if width <= 0:
                continue
            
            # 2D hypervolume of points i..n-1 in (f1, f2) space
            sub = p[i:, [1, 2]]
            sub = sub[(sub[:, 0] < r[1]) & (sub[:, 1] < r[2])]
            if len(sub) == 0:
                continue
            
            sub_idx = np.argsort(sub[:, 0])
            sub_sorted = sub[sub_idx]
            m = len(sub_sorted)
            
            hv_2d = 0.0
            for j in range(m):
                f1_next = sub_sorted[j + 1, 0] if j < m - 1 else r[1]
                w1 = f1_next - sub_sorted[j, 0]
                if w1 <= 0:
                    continue
                max_f2 = np.max(sub_sorted[j:, 1])
                h2 = r[2] - max_f2
                if h2 <= 0:
                    continue
                hv_2d += w1 * h2
            
            hv += width * hv_2d
        return hv
# EVOLVE-BLOCK-END

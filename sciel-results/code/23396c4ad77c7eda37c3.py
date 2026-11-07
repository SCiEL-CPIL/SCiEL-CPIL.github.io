"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer: Halton init, grid decomposition + local simplex polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _halton(self, n, dim, base=None):
        """Generate Halton sequence points in [0,1]^dim."""
        if base is None:
            base = list(range(2, 2 + dim))
        points = np.zeros((n, dim))
        for d in range(dim):
            b = base[d]
            for i in range(1, n + 1):
                f = 1.0
                r = 0.0
                idx = i
                while idx > 0:
                    idx, rem = divmod(idx, b)
                    f /= b
                    r += f * rem
                points[i - 1, d] = r
        return points

    def _nondominated_mask(self, F):
        """Return boolean mask of non-dominated points in F (n x n_obj)."""
        n = len(F)
        mask = np.ones(n, dtype=bool)
        for i in range(n):
            if not mask[i]:
                continue
            for j in range(n):
                if i == j or not mask[j]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    mask[i] = False
                    break
        return mask

    def _hypervolume_contribution(self, F, ref):
        """Compute HV contribution of each point (2D case)."""
        n = len(F)
        contrib = np.zeros(n)
        # Filter to points better than ref
        valid = np.all(F < ref, axis=1)
        Fv = F[valid]
        if len(Fv) < 2:
            return contrib
        # Sort by first objective
        order = np.argsort(Fv[:, 0])
        Fv = Fv[order]
        # For 2D: contribution of point i = (Fv[i,1] - Fv[i+1,1]) * (ref[0] - Fv[i,0])
        # Actually: contribution = area of rectangle that point i uniquely covers
        # Sort by obj1 ascending, then contribution = (ref[1] - Fv[i,1]) - (ref[1] - Fv[i+1,1]) if Fv[i,1] < Fv[i+1,1]
        # Simpler: for each point, its contribution is the area it adds when inserted
        # For 2D with ref: sort by x ascending, keep track of min y seen
        # Contribution of point i = (ref[1] - y_i) * (x_i - x_{i+1}) where y_i < all subsequent y
        # Let me just compute it properly for 2D
        # After sorting by x ascending:
        # The non-dominated subset (in sorted order) has decreasing y
        # Contribution of point i = (x_i - x_{i-1}) * (ref[1] - y_i) where x_{-1}=0 (or ref[0])
        # Actually standard 2D HV: sort by x ascending, for each point that is non-dominated
        # (i.e., y is less than all previous y's in the sorted order... wait, we minimize)
        # Let me just do: for each valid point, compute its unique contribution
        # In 2D: point (x,y) contributes (ref[0]-x)*(ref[1]-y) minus overlap
        # Easiest: compute total HV, then for each point compute HV without it
        # But that's O(n^2). For n up to 200, that's fine.
        total_hv = self._hv_2d(Fv, ref)
        for i in range(n):
            if not valid[i]:
                contrib[i] = 0.0
                continue
            # HV without point i
            F_without = np.delete(F, i, axis=0)
            F_without = F_without[np.all(F_without < ref, axis=1)]
            if len(F_without) < 2:
                hv_without = 0.0
            else:
                hv_without = self._hv_2d(F_without, ref)
            contrib[i] = total_hv - hv_without
        return contrib

    def _hv_2d(self, F, ref):
        """Compute 2D hypervolume for points better than ref."""
        if len(F) < 2:
            return 0.0
        order = np.argsort(F[:, 0])
        F = F[order]
        # Keep only non-dominated (in 2D: y must be strictly decreasing as x increases)
        # After sorting by x, keep points where y is less than all previous kept y's
        kept = []
        min_y = np.inf
        for i in range(len(F)):
            if F[i, 1] < min_y:
                kept.append(i)
                min_y = F[i, 1]
        if len(kept) == 0:
            return 0.0
        # Compute HV: for each kept point, contribution = (x_{i+1} - x_i) * (ref[1] - y_i)
        # where x_{n} = ref[0]
        hv = 0.0
        for k, i in enumerate(kept):
            x_next = F[kept[k + 1], 0] if k + 1 < len(kept) else ref[0]
            width = x_next - F[i, 0]
            height = ref[1] - F[i, 1]
            if width > 0 and height > 0:
                hv += width * height
        return hv

    def _simplex_minimize(self, problem, x0, bounds, budget, n_obj, ref, deadline):
        """Local Nelder-Mead style simplex search on a scalarized objective.
        Returns (best_x, best_f, n_evals_used)."""
        t0 = time.time()
        dim = len(x0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Scalarize: use weighted sum with weights based on proximity to ref
        # f_scalar = sum(w_i * (f_i - ref_i)) where w_i encourages balance
        # Actually, use the "augmented Tchebycheff" or just weighted sum
        # For local search, use a random weighted sum
        w = np.random.dirichlet(np.ones(n_obj))
        
        def scalar_f(x):
            f = problem(x)
            # Penalize if outside bounds (shouldn't happen after clip)
            return np.dot(w, f)
        
        # Initialize simplex
        simplex = [x0.copy()]
        for i in range(dim):
            perturbation = x0.copy()
            step = 0.05 * (hi[i] - lo[i]) if hi[i] > lo[i] else 0.05
            perturbation[i] += step
            perturbation = np.clip(perturbation, lo, hi)
            simplex.append(perturbation)
        
        F_simplex = [scalar_f(x) for x in simplex]
        n_evals = len(simplex)
        
        # Nelder-Mead iterations
        for _ in range(budget - n_evals):
            if time.time() > deadline:
                break
            # Sort by objective value
            order = np.argsort(F_simplex)
            simplex = [simplex[i] for i in order]
            F_simplex = [F_simplex[i] for i in order]
            
            # Reflection
            centroid = np.mean(simplex[:-1], axis=0)
            x_reflect = centroid + 1.0 * (centroid - simplex[-1])
            x_reflect = np.clip(x_reflect, lo, hi)
            f_reflect = scalar_f(x_reflect)
            n_evals += 1
            
            if f_reflect < F_simplex[0]:
                # Expansion
                x_expand = centroid + 2.0 * (x_reflect - centroid)
                x_expand = np.clip(x_expand, lo, hi)
                f_expand = scalar_f(x_expand)
                n_evals += 1
                if f_expand < f_reflect:
                    simplex[-1] = x_expand
                    F_simplex[-1] = f_expand
                else:
                    simplex[-1] = x_reflect
                    F_simplex[-1] = f_reflect
            elif f_reflect < F_simplex[-2]:
                simplex[-1] = x_reflect
                F_simplex[-1] = f_reflect
            else:
                # Contraction
                x_contract = centroid + 0.5 * (simplex[-1] - centroid)
                x_contract = np.clip(x_contract, lo, hi)
                f_contract = scalar_f(x_contract)
                n_evals += 1
                if f_contract < F_simplex[-1]:
                    simplex[-1] = x_contract
                    F_simplex[-1] = f_contract
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        F_simplex[i] = scalar_f(simplex[i])
                        n_evals += 1
                        if n_evals > budget:
                            break
                    if n_evals > budget:
                        break
        
        # Find best point
        best_idx = np.argmin(F_simplex)
        best_x = simplex[best_idx].copy()
        best_f = problem(best_x)
        return best_x, best_f, n_evals

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin (330s limit, leave 50s)
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        ref = np.array([3000.0, 0.06])
        
        # Phase 1: Halton sequence initialization for uniform coverage
        n_init = min(40, budget // 5)
        halton_pts = self._halton(n_init, dim)
        X = lo + (hi - lo) * halton_pts
        F = np.array([problem(x) for x in X])
        used = n_init
        
        # Phase 2: Grid-based decomposition with local simplex polish
        # Partition the space into a grid of hypercubes
        # With 4D and ~160 remaining evaluations, use a 2x2x2x2 = 16 grid
        grid_size = 2
        n_cells = grid_size ** dim  # 16 cells
        
        # For each cell, do a local simplex search
        cell_budget = max(5, (budget - used - 20) // n_cells)  # Reserve 20 for final polish
        rng = np.random.default_rng(42)
        
        for cell_idx in range(n_cells):
            if used >= budget or time.time() > deadline:
                break
            
            # Compute cell bounds
            cell_lo = np.zeros(dim)
            cell_hi = np.zeros(dim)
            for d in range(dim):
                # Extract cell index for dimension d
                idx_d = (cell_idx // (grid_size ** d)) % grid_size
                cell_lo[d] = lo[d] + idx_d * (hi[d] - lo[d]) / grid_size
                cell_hi[d] = cell_lo[d] + (hi[d] - lo[d]) / grid_size
            
            # Start point: center of cell + small random perturbation
            center = (cell_lo + cell_hi) / 2.0
            perturbation = 0.1 * (cell_hi - cell_lo) * rng.standard_normal(dim)
            x_start = np.clip(center + perturbation, cell_lo, cell_hi)
            
            # Local simplex search
            best_x, best_f, n_used = self._simplex_minimize(
                problem, x_start, bounds, cell_budget, n_obj, ref, deadline
            )
            used += n_used
            X = np.vstack([X, best_x])
            F = np.vstack([F, best_f])
        
        # Phase 3: Final polish on best points using HV contribution
        remaining = budget - used
        if remaining > 0:
            # Select top points by HV contribution for final polish
            n_polish = min(remaining, 10)
            if n_polish > 0:
                # Compute HV contributions
                contrib = self._hypervolume_contribution(F, ref)
                # Select top contributors
                top_indices = np.argsort(contrib)[-n_polish:]
                for idx in top_indices:
                    if used >= budget or time.time() > deadline:
                        break
                    # Small local refinement
                    x_refine = X[idx] + 0.02 * (hi - lo) * rng.standard_normal(dim)
                    x_refine = np.clip(x_refine, lo, hi)
                    f_refine = problem(x_refine)
                    used += 1
                    X = np.vstack([X, x_refine])
                    F = np.vstack([F, f_refine])
        
        return X
# EVOLVE-BLOCK-END

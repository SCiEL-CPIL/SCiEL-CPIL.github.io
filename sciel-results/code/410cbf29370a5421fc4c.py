"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead optimizer minimizing hypervolume loss for RE21."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hv_2d(self, front, ref):
        """Compute exact 2D hypervolume for a list of points (Pareto front)."""
        if len(front) == 0:
            return 0.0
        pts = np.array(front, dtype=float)
        mask = np.all(pts < ref, axis=1)
        pts = pts[mask]
        if len(pts) == 0:
            return 0.0
        order = np.argsort(pts[:, 0])
        pts = pts[order]
        front = [pts[0]]
        for p in pts[1:]:
            if p[1] < front[-1][1]:
                front.append(p)
        if len(front) == 0:
            return 0.0
        hv = 0.0
        for i in range(len(front)):
            x_next = front[i + 1][0] if i + 1 < len(front) else ref[0]
            y_curr = front[i][1]
            if y_curr >= ref[1]:
                continue
            width = x_next - front[i][0]
            height = ref[1] - y_curr
            hv += width * height
        return hv

    def _non_dominated_front(self, F):
        """Return indices of non-dominated points in F (2D minimization)."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=int)
        F = np.array(F, dtype=float)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        return np.where(~dominated)[0]

    def _hv_contribution_2d(self, point, front, ref):
        """Compute HV contribution of a single point in a 2D front."""
        if len(front) == 0:
            return 0.0
        pts = np.array(front, dtype=float)
        mask = np.all(pts < ref, axis=1)
        pts = pts[mask]
        if len(pts) == 0:
            return 0.0
        p = np.array(point, dtype=float)
        if not np.all(p < ref):
            return 0.0
        
        order = np.argsort(pts[:, 0])
        pts = pts[order]
        
        nd_front = [pts[0]]
        for pt in pts[1:]:
            if pt[1] < nd_front[-1][1]:
                nd_front.append(pt)
        
        if len(nd_front) == 0:
            return 0.0
        
        all_pts = np.vstack([np.array(nd_front), p.reshape(1, -1)])
        order = np.argsort(all_pts[:, 0])
        all_pts = all_pts[order]
        
        nd_all = [all_pts[0]]
        for pt in all_pts[1:]:
            if pt[1] < nd_all[-1][1]:
                nd_all.append(pt)
        
        hv_with = 0.0
        for i in range(len(nd_all)):
            x_next = nd_all[i + 1][0] if i + 1 < len(nd_all) else ref[0]
            y_curr = nd_all[i][1]
            if y_curr >= ref[1]:
                continue
            width = x_next - nd_all[i][0]
            height = ref[1] - y_curr
            hv_with += width * height
        
        hv_without = 0.0
        for i in range(len(nd_front)):
            x_next = nd_front[i + 1][0] if i + 1 < len(nd_front) else ref[0]
            y_curr = nd_front[i][1]
            if y_curr >= ref[1]:
                continue
            width = x_next - nd_front[i][0]
            height = ref[1] - y_curr
            hv_without += width * height
        
        return hv_with - hv_without

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        ref = np.array([3000.0, 0.06])
        rng = np.random

        all_X = []
        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = problem(x)
            all_X.append(x.copy())
            all_F.append(f)
            used += 1
            return f

        # Phase 1: Stratified random sampling for initial coverage
        n_stratified = 30
        # Use stratified sampling in each dimension
        n_per_dim = int(np.ceil(n_stratified ** (1.0 / self.dim)))
        grid = np.meshgrid(*[np.linspace(0.05, 0.95, n_per_dim) for _ in range(self.dim)])
        grid = np.stack([g.ravel() for g in grid], axis=1)
        # Shuffle and take first n_stratified
        perm = rng.permutation(len(grid))
        grid = grid[perm[:n_stratified]]
        
        for g in grid:
            if time.time() > deadline or used >= self.budget:
                break
            x = lo + (hi - lo) * g
            evaluate(x)

        # Phase 2: Multi-start Nelder-Mead
        # Number of starts based on remaining budget
        remaining = self.budget - used
        if remaining > 0:
            # Each NM run uses about 15-25 evaluations
            n_starts = min(remaining // 15, 12)
            if n_starts < 1:
                n_starts = 1
            
            # Choose starting points: mix of best front points and random
            front_idx = self._non_dominated_front(all_F)
            start_points = []
            
            if len(front_idx) > 0:
                # Add best front points
                front_points = [all_F[i] for i in front_idx]
                contributions = np.zeros(len(front_idx))
                for k, idx in enumerate(front_idx):
                    contributions[k] = self._hv_contribution_2d(
                        all_F[idx], front_points, ref
                    )
                top_k = min(4, len(front_idx))
                top_indices = front_idx[np.argsort(contributions)[-top_k:]]
                for idx in top_indices:
                    start_points.append(all_X[idx].copy())
            
            # Add random points
            while len(start_points) < n_starts:
                start_points.append(lo + (hi - lo) * rng.random(self.dim))
            
            start_points = start_points[:n_starts]
            
            # Run Nelder-Mead from each start
            for sp in start_points:
                if time.time() > deadline or used >= self.budget:
                    break
                
                # Nelder-Mead parameters
                alpha = 1.0  # reflection
                gamma = 2.0  # expansion
                rho = 0.5    # contraction
                sigma = 0.5  # shrink
                max_iter = 20  # max iterations per NM run
                
                # Initialize simplex
                simplex = [sp.copy()]
                for i in range(self.dim):
                    xi = sp.copy()
                    if sp[i] > lo[i]:
                        xi[i] = max(lo[i], sp[i] - 0.1 * (hi[i] - lo[i]))
                    else:
                        xi[i] = min(hi[i], sp[i] + 0.1 * (hi[i] - lo[i]))
                    simplex.append(xi)
                
                # Evaluate initial simplex
                simplex_F = []
                for sx in simplex:
                    if time.time() > deadline or used >= self.budget:
                        break
                    f = evaluate(sx)
                    simplex_F.append(f)
                
                if len(simplex_F) < self.dim + 1:
                    continue
                
                # NM iterations
                for it in range(max_iter):
                    if time.time() > deadline or used >= self.budget:
                        break
                    
                    # Sort by first objective (proxy for quality)
                    order = np.argsort([f[0] for f in simplex_F])
                    simplex = [simplex[i] for i in order]
                    simplex_F = [simplex_F[i] for i in order]
                    
                    # Check convergence
                    spread = np.max([np.max(np.abs(simplex[i] - simplex[0])) for i in range(1, len(simplex))])
                    if spread < 1e-6 * np.max(hi - lo):
                        break
                    
                    # Centroid of all but worst
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    # Reflection
                    x_r = centroid + alpha * (centroid - simplex[-1])
                    x_r = np.clip(x_r, lo, hi)
                    if used >= self.budget:
                        break
                    f_r = evaluate(x_r)
                    
                    if f_r[0] < simplex_F[0][0]:
                        # Expansion
                        x_e = centroid + gamma * (centroid - simplex[-1])
                        x_e = np.clip(x_e, lo, hi)
                        if used >= self.budget:
                            break
                        f_e = evaluate(x_e)
                        if f_e[0] < f_r[0]:
                            simplex[-1] = x_e
                            simplex_F[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            simplex_F[-1] = f_r
                    elif f_r[0] < simplex_F[-1][0]:
                        # Accept reflection
                        simplex[-1] = x_r
                        simplex_F[-1] = f_r
                    else:
                        # Contraction
                        if f_r[0] < simplex_F[-1][0]:
                            # Outside contraction
                            x_c = centroid + rho * (centroid - simplex[-1])
                        else:
                            # Inside contraction
                            x_c = centroid - rho * (centroid - simplex[-1])
                        x_c = np.clip(x_c, lo, hi)
                        if used >= self.budget:
                            break
                        f_c = evaluate(x_c)
                        
                        if f_c[0] < simplex_F[-1][0]:
                            simplex[-1] = x_c
                            simplex_F[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                if used >= self.budget:
                                    break
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], lo, hi)
                                simplex_F[i] = evaluate(simplex[i])

        # Phase 3: Local refinement around best front
        remaining = self.budget - used
        if remaining > 0:
            front_idx = self._non_dominated_front(all_F)
            if len(front_idx) > 0:
                front_points = [all_F[i] for i in front_idx]
                contributions = np.zeros(len(front_idx))
                for k, idx in enumerate(front_idx):
                    contributions[k] = self._hv_contribution_2d(
                        all_F[idx], front_points, ref
                    )
                
                top_k = min(5, len(front_idx))
                top_indices = front_idx[np.argsort(contributions)[-top_k:]]
                
                local_steps = min(remaining, 20)
                for step in range(local_steps):
                    if time.time() > deadline or used >= self.budget:
                        break
                    parent_idx = top_indices[step % len(top_indices)]
                    parent = all_X[parent_idx].copy()
                    
                    small_mut = 0.02 * (hi - lo)
                    child = parent + rng.standard_normal(self.dim) * small_mut
                    evaluate(child)

        return all_X
# EVOLVE-BLOCK-END

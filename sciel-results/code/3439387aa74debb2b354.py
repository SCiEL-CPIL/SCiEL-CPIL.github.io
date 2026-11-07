"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-driven multi-objective optimizer for RE21 using archive + local search."""

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

    def _hv_contribution_2d(self, point, front, ref):
        """Compute HV contribution of a single point to the front."""
        if len(front) == 0:
            p = np.array(point, dtype=float)
            if np.all(p < ref):
                return (ref[0] - p[0]) * (ref[1] - p[1])
            return 0.0
        
        pts = np.array(front, dtype=float)
        mask = np.all(pts < ref, axis=1)
        pts = pts[mask]
        if len(pts) == 0:
            p = np.array(point, dtype=float)
            if np.all(p < ref):
                return (ref[0] - p[0]) * (ref[1] - p[1])
            return 0.0
            
        p = np.array(point, dtype=float)
        if not np.all(p < ref):
            return 0.0
            
        # Check if dominated by existing front
        for fp in pts:
            if np.all(fp <= p) and np.any(fp < p):
                return 0.0
                
        # Build combined front
        combined = np.vstack([pts, [p]])
        hv_with = self._hv_2d(combined, ref)
        hv_without = self._hv_2d(pts, ref)
        return hv_with - hv_without

    def _nelder_mead_minimize(self, func, x0, bounds, max_iter=30, tol=1e-8):
        """Simple Nelder-Mead implementation for bounded minimization."""
        dim = len(x0)
        lo, hi = bounds
        simplex = [np.array(x0, dtype=float)]
        
        # Create initial simplex
        for i in range(dim):
            new_pt = simplex[0].copy()
            step = 0.1 * (hi[i] - lo[i])
            new_pt[i] += step
            new_pt = np.clip(new_pt, lo, hi)
            simplex.append(new_pt)
            
        fvals = [func(pt) for pt in simplex]
        
        for _ in range(max_iter):
            # Sort by function value
            indices = np.argsort(fvals)
            simplex = [simplex[i] for i in indices]
            fvals = [fvals[i] for i in indices]
            
            if abs(fvals[-1] - fvals[0]) < tol:
                break
                
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + (centroid - simplex[-1])
            xr = np.clip(xr, lo, hi)
            fr = func(xr)
            
            if fr < fvals[0]:
                # Expansion
                xe = centroid + 2 * (centroid - simplex[-1])
                xe = np.clip(xe, lo, hi)
                fe = func(xe)
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                xc = centroid + 0.5 * (simplex[-1] - centroid)
                xc = np.clip(xc, lo, hi)
                fc = func(xc)
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], lo, hi)
                        fvals[i] = func(simplex[i])
                        
        return min(simplex, key=lambda x: func(x)), min(fvals)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        ref = np.array([3000.0, 0.06])
        budget = self.budget

        all_X = []
        all_F = []
        used = 0

        def evaluate(x):
            nonlocal used
            if used >= budget:
                return None
            f = problem(x)
            used += 1
            all_X.append(x)
            all_F.append(f)
            return f

        # Phase 1: Initial diverse sampling (20% of budget)
        n_init = max(10, budget // 5)
        X_init = lo + (hi - lo) * np.random.random((n_init, self.dim))
        for x in X_init:
            evaluate(x)
            
        # Also sample corners and midpoints
        special_points = [
            lo,
            hi,
            (lo + hi) / 2,
            lo + (hi - lo) * [0.25, 0.25, 0.25, 0.25],
            lo + (hi - lo) * [0.75, 0.75, 0.75, 0.75],
            lo + (hi - lo) * [0.0, 1.0, 0.0, 1.0],
            lo + (hi - lo) * [1.0, 0.0, 1.0, 0.0],
        ]
        for x in special_points:
            if used < budget:
                evaluate(np.clip(x, lo, hi))

        # Build initial archive
        archive = []
        for i in range(len(all_F)):
            f = all_F[i]
            x = all_X[i]
            if np.all(f < ref):
                # Check if dominated by existing archive
                dominated = False
                for af in archive:
                    if np.all(af <= f) and np.any(af < f):
                        dominated = True
                        break
                if not dominated:
                    # Remove points dominated by new point
                    archive = [af for af in archive if not (np.all(f <= af) and np.any(f < af))]
                    archive.append(f)
                    archive_x = None  # We'll track x separately if needed

        # Phase 2: Local search from promising points
        # Use remaining budget for Nelder-Mead from top contributors
        remaining = budget - used
        if remaining > 5:
            # Select starting points: best HV contributors + random
            start_points = []
            
            # Get non-dominated points
            nd_indices = []
            for i in range(len(all_F)):
                f = all_F[i]
                dominated = False
                for j in range(len(all_F)):
                    if i == j:
                        continue
                    if np.all(all_F[j] <= f) and np.any(all_F[j] < f):
                        dominated = True
                        break
                if not dominated:
                    nd_indices.append(i)
                    
            # Add non-dominated points as starts
            for idx in nd_indices[:min(10, len(nd_indices))]:
                start_points.append(all_X[idx].copy())
                
            # Add some random starts
            n_random_starts = min(5, remaining // 10)
            for _ in range(n_random_starts):
                start_points.append(lo + (hi - lo) * np.random.random(self.dim))
                
            # Define objective: minimize distance to reference point (L1 norm)
            def hypervolume_loss(x):
                f = problem(x)
                if not np.all(f < ref):
                    return 1e6  # Penalty
                # Use negative HV contribution as proxy
                # Simpler: minimize sum of distances to ref
                return np.sum(ref - f) if np.all(f < ref) else 1e6
            
            # But we need to be careful about evaluations
            # Let's do a simpler local search: coordinate descent with line search
            
            for sp in start_points:
                if time.time() > deadline:
                    break
                if used >= budget - 3:
                    break
                    
                best_x = sp.copy()
                best_f = problem(best_x)
                used += 1
                all_X.append(best_x)
                all_F.append(best_f)
                
                # Coordinate descent
                for _ in range(5):  # 5 iterations
                    if used >= budget - 1:
                        break
                    improved = False
                    for d in range(self.dim):
                        if used >= budget - 1:
                            break
                        # Try moving in positive direction
                        step = 0.1 * (hi[d] - lo[d])
                        trial_x = best_x.copy()
                        trial_x[d] += step
                        trial_x = np.clip(trial_x, lo, hi)
                        trial_f = problem(trial_x)
                        used += 1
                        all_X.append(trial_x)
                        all_F.append(trial_f)
                        
                        if np.all(trial_f <= best_f):
                            best_x = trial_x
                            best_f = trial_f
                            improved = True
                        else:
                            # Try negative direction
                            if used >= budget - 1:
                                break
                            trial_x = best_x.copy()
                            trial_x[d] -= step
                            trial_x = np.clip(trial_x, lo, hi)
                            trial_f = problem(trial_x)
                            used += 1
                            all_X.append(trial_x)
                            all_F.append(trial_f)
                            
                            if np.all(trial_f <= best_f):
                                best_x = trial_x
                                best_f = trial_f
                                improved = True
                    
                    if not improved:
                        break

        return all_X
# EVOLVE-BLOCK-END

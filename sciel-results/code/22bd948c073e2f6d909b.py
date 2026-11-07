"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Grid-decomposition multi-objective optimizer with local simplex polish."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 270.0  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume
        ref = np.array([3000.0, 0.06])

        def hypervolume_2d(points, ref_pt):
            """Compute hypervolume of 2D points w.r.t. reference point."""
            if len(points) == 0:
                return 0.0
            # Filter points strictly better than reference
            mask = np.all(points < ref_pt, axis=1)
            pts = points[mask]
            if len(pts) == 0:
                return 0.0
            # Sort by first objective
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            hv = 0.0
            for i in range(len(pts)):
                # Width in obj1
                if i == len(pts) - 1:
                    w1 = ref_pt[0] - pts[i, 0]
                else:
                    w1 = pts[i + 1, 0] - pts[i, 0]
                if w1 <= 0:
                    continue
                # Height in obj2: min of all subsequent points' obj2 and ref
                h2 = min(pts[i:, 1].min(), ref_pt[1] - pts[i, 1])
                if h2 > 0:
                    hv += w1 * h2
            return hv

        def non_dominated(points):
            """Return indices of non-dominated points."""
            n = len(points)
            nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not nd[i]:
                    continue
                for j in range(n):
                    if i == j or not nd[j]:
                        continue
                    if np.all(points[j] <= points[i]) and np.any(points[j] < points[i]):
                        nd[i] = False
                        break
            return np.where(nd)[0]

        # Phase 1: Halton sequence for initial coverage
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def halton_sequence(n, dim):
            seq = np.zeros((n, dim))
            for d in range(dim):
                # Use first few primes
                primes = [2, 3, 5, 7, 11, 13, 17, 19]
                base = primes[d % len(primes)]
                for i in range(n):
                    seq[i, d] = halton(i + 1, base)
            return seq

        # Budget allocation
        n_init = min(40, budget // 5)
        n_grid = min(100, budget // 2)
        n_local = budget - n_init - n_grid

        # Phase 1: Halton initialization
        halton_pts = halton_sequence(n_init, dim)
        X_init = lo + (hi - lo) * halton_pts
        F_init = np.array([problem(x) for x in X_init])
        used = n_init

        # Check deadline
        if time.time() > deadline - 30:
            return X_init

        # Phase 2: Grid-based decomposition
        # Partition space into hypercubes
        n_per_dim = 3  # 3^4 = 81 cells for dim=4
        n_cells = n_per_dim ** dim
        # Allocate evaluations per cell
        evals_per_cell = max(1, n_grid // n_cells)

        # Generate grid centers
        grid_centers = []
        for d in range(dim):
            centers_d = np.linspace(lo[d], hi[d], n_per_dim + 1)
            # Cell centers
            cell_centers_d = (centers_d[:-1] + centers_d[1:]) / 2.0
            grid_centers.append(cell_centers_d)

        # Create all cell centers
        from itertools import product
        cell_indices = list(product(*[range(n_per_dim)] * dim))
        X_grid = np.array([
            np.array([grid_centers[d][idx] for d, idx in enumerate(cell_idx)])
            for cell_idx in cell_indices
        ])

        # Evaluate grid points
        F_grid = np.zeros((len(X_grid), n_obj))
        for i in range(len(X_grid)):
            if time.time() > deadline - 30:
                break
            F_grid[i] = problem(X_grid[i])
            used += 1

        # Phase 3: Local simplex search on best cells
        # Select top cells based on hypervolume contribution
        all_X = np.vstack([X_init, X_grid])
        all_F = np.vstack([F_init, F_grid])

        # Find non-dominated points
        nd_idx = non_dominated(all_F)
        if len(nd_idx) > 0:
            nd_F = all_F[nd_idx]
            # Compute HV contribution of each non-dominated point
            hv_full = hypervolume_2d(nd_F, ref)
            
            # For local search, pick points that are well-spread
            # Use crowding distance
            def crowding_distance(points):
                n = len(points)
                if n <= 2:
                    return np.ones(n)
                cd = np.zeros(n)
                for m in range(n_obj):
                    order = np.argsort(points[:, m])
                    cd[order[0]] = np.inf
                    cd[order[-1]] = np.inf
                    for i in range(1, n - 1):
                        cd[order[i]] += (points[order[i+1], m] - points[order[i-1], m]) / (
                            points[:, m].max() - points[:, m].min() + 1e-10
                        )
                return cd

            cd = crowding_distance(nd_F)
            # Select top points by crowding distance for local search
            n_local_pts = min(10, len(nd_idx))
            local_idx = nd_idx[np.argsort(cd)[-n_local_pts:]]
        else:
            local_idx = np.array([], dtype=int)

        # Local simplex search
        if len(local_idx) > 0 and used < budget:
            simplex_step = 0.05 * (hi - lo)
            for idx in local_idx:
                if used >= budget or time.time() > deadline - 20:
                    break
                x0 = all_X[idx].copy()
                f0 = all_F[idx].copy()
                # Simplex: start with x0, then add perturbed points
                simplex = [x0.copy()]
                for d in range(dim):
                    xp = x0.copy()
                    xp[d] += simplex_step[d]
                    xp = np.clip(xp, lo, hi)
                    simplex.append(xp)
                
                F_simplex = np.array([problem(x) for x in simplex])
                used += dim
                
                # Nelder-Mead iterations
                n_iter = max(5, (budget - used) // (len(local_idx) * (dim + 1)))
                for _ in range(n_iter):
                    if used >= budget or time.time() > deadline - 20:
                        break
                    # Sort by sum of objectives
                    order = np.argsort(np.sum(F_simplex, axis=1))
                    simplex = [simplex[i] for i in order]
                    F_simplex = F_simplex[order]
                    
                    # Centroid of all but worst
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    # Reflection
                    x_ref = centroid + (centroid - simplex[-1])
                    x_ref = np.clip(x_ref, lo, hi)
                    f_ref = problem(x_ref)
                    used += 1
                    
                    if np.sum(f_ref) < np.sum(F_simplex[0]):
                        # Expansion
                        x_exp = centroid + 2 * (x_ref - centroid)
                        x_exp = np.clip(x_exp, lo, hi)
                        f_exp = problem(x_exp)
                        used += 1
                        if np.sum(f_exp) < np.sum(f_ref):
                            simplex[-1] = x_exp
                            F_simplex[-1] = f_exp
                        else:
                            simplex[-1] = x_ref
                            F_simplex[-1] = f_ref
                    elif np.sum(f_ref) < np.sum(F_simplex[-1]):
                        simplex[-1] = x_ref
                        F_simplex[-1] = f_ref
                    else:
                        # Contraction
                        x_con = centroid + 0.5 * (simplex[-1] - centroid)
                        x_con = np.clip(x_con, lo, hi)
                        f_con = problem(x_con)
                        used += 1
                        if np.sum(f_con) < np.sum(F_simplex[-1]):
                            simplex[-1] = x_con
                            F_simplex[-1] = f_con
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                F_simplex[i] = problem(simplex[i])
                                used += 1
                            if used >= budget or time.time() > deadline - 20:
                                break
                    
                    if time.time() > deadline - 20:
                        break
                
                # Add best simplex point to results
                best_in_simplex = np.argmin(np.sum(F_simplex, axis=1))
                all_X = np.vstack([all_X, simplex[best_in_simplex]])
                all_F = np.vstack([all_F, F_simplex[best_in_simplex]])

        # Phase 4: Final polish - refine non-dominated set
        if used < budget and time.time() < deadline - 10:
            nd_idx = non_dominated(all_F)
            if len(nd_idx) > 0:
                # Fine-tune around non-dominated points
                n_polish = min(20, budget - used)
                step = 0.02 * (hi - lo)
                for _ in range(n_polish):
                    if used >= budget or time.time() > deadline - 10:
                        break
                    idx = nd_idx[np.random.randint(len(nd_idx))]
                    x_new = all_X[idx] + step * np.random.randn(dim)
                    x_new = np.clip(x_new, lo, hi)
                    f_new = problem(x_new)
                    used += 1
                    all_X = np.vstack([all_X, x_new])
                    all_F = np.vstack([all_F, f_new])

        return all_X
# EVOLVE-BLOCK-END

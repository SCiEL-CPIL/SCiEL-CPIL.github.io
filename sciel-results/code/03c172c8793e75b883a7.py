"""Multi-objective optimizer for ZDT4 using Structure-Exploiting Global Search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Optimizer that exploits ZDT4's known structure:
    - f1 = 1 + 10*(n-1) + sum(xi^2 - 10*cos(4*pi*xi)) for i=2..n
    - f2 = g(x1) * h(f1, g(x1))
    - g(x1) = 1 + 100*(x1^2 - 5*sin(2*pi*x1))^2
    - h(u, g) = 1 - sqrt(u/g)
    
    On the Pareto front: xi = 0.5 for i>=2 (so f1=1), and x1 varies.
    We use a two-phase approach:
    1. Global random search with structure-aware perturbation
    2. Local refinement around promising points using Nelder-Mead
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo

        def is_deadline_exceeded():
            return (time.time() - start_time) > deadline

        def random_point():
            return lo + span * np.random.random(dim)

        # Known structure: for i >= 2, xi = 0.5 gives f1 = 1 (minimum possible)
        # x1 can vary to sweep the front
        structure_point = lo + span * np.random.random(dim)
        if dim >= 2:
            # Set x1 to a random value, and set x2..xn to 0.5 (midpoint of [-5, 5] typically)
            # But we don't know the exact bounds, so use midpoint
            structure_point[1:] = (lo[1:] + hi[1:]) / 2.0

        # Phase 1: Global search - mix of random and structure-aware points
        n_global = min(budget, 60)
        if n_global < 5:
            n_global = max(5, budget // 3)

        X_list = []
        F_list = []
        used = 0

        # Add structure points
        for i in range(min(10, n_global)):
            pt = structure_point.copy()
            # Perturb x1 randomly
            pt[0] = lo[0] + span[0] * np.random.random()
            # Small perturbation to other coords
            if i % 3 == 0 and dim > 2:
                jitter = np.random.normal(0, 0.1 * span, dim)
                pt += jitter
                pt = np.clip(pt, lo, hi)
            X_list.append(pt)
            F_list.append(problem(pt))
            used += 1

        # Add pure random points
        while used < n_global:
            pt = random_point()
            X_list.append(pt)
            F_list.append(problem(pt))
            used += 1

        X = np.array(X_list)
        F = np.array(F_list)

        # Phase 2: Local refinement using Nelder-Mead on scalarized objectives
        # Use weighted sum with different weights to cover the front
        n_local = min(budget - used, 8)
        if n_local < 3:
            n_local = max(3, (budget - used) // 2)

        # Find initial non-dominated points for local search
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        # Get non-dominated indices
        nd_mask = np.ones(len(F), dtype=bool)
        for i in range(len(F)):
            for j in range(len(F)):
                if i != j and dominates(F[j], F[i]):
                    nd_mask[i] = False
                    break

        nd_indices = np.where(nd_mask)[0]
        if len(nd_indices) == 0:
            nd_indices = np.array([0])

        # Select diverse points for local search
        if len(nd_indices) > n_local:
            # Select evenly spaced by f1 value
            f1_vals = F[nd_indices, 0]
            order = np.argsort(f1_vals)
            selected = nd_indices[order]
            step = max(1, len(selected) // n_local)
            local_seeds = selected[::step][:n_local]
        else:
            local_seeds = nd_indices

        # For each seed, do local search minimizing weighted sum
        for seed_idx in local_seeds:
            if used >= budget or is_deadline_exceeded():
                break

            # Weight for this local search - based on position in front
            f1_min = F[:, 0].min()
            f1_max = F[:, 0].max()
            f1_range = f1_max - f1_min if f1_max > f1_min else 1.0
            w = (F[seed_idx, 0] - f1_min) / f1_range
            w = np.clip(w, 0.05, 0.95)

            # Nelder-Mead on weighted sum
            x0 = X[seed_idx].copy()

            def objective(x):
                fx = problem(x)
                return w * fx[0] + (1.0 - w) * fx[1]

            # Simple Nelder-Mead implementation
            n_vars = dim
            # Initial simplex
            simplex = [x0.copy() for _ in range(n_vars + 1)]
            fvals = [objective(simplex[0])]
            for k in range(n_vars):
                dx = x0.copy()
                step_size = 0.25 * (hi[k] - lo[k])
                dx[k] += step_size
                dx = np.clip(dx, lo, hi)
                simplex[k + 1] = dx
                fvals.append(objective(dx))
                used += 1

            max_iter = 30
            for iteration in range(max_iter):
                if used >= budget or is_deadline_exceeded():
                    break

                # Sort by objective value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                centroid = np.clip(centroid, lo, hi)

                # Reflection
                xr = centroid + 1.0 * (centroid - simplex[-1])
                xr = np.clip(xr, lo, hi)
                fr = objective(xr)
                used += 1

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + 2.0 * (centroid - simplex[-1])
                    xe = np.clip(xe, lo, hi)
                    fe = objective(xe)
                    used += 1
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
                    fc = objective(xc)
                    used += 1
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for k in range(1, n_vars + 1):
                            simplex[k] = (simplex[0] + simplex[k]) / 2.0
                            simplex[k] = np.clip(simplex[k], lo, hi)
                            fvals[k] = objective(simplex[k])
                            used += 1
                            if used >= budget or is_deadline_exceeded():
                                break
                        if used >= budget or is_deadline_exceeded():
                            break

                # Keep best point
                best_idx = np.argmin(fvals)
                X_list.append(simplex[best_idx])
                F_list.append(problem(simplex[best_idx]))
                used += 1

        # Phase 3: Final random fill to use remaining budget
        while used < budget and not is_deadline_exceeded():
            pt = random_point()
            X_list.append(pt)
            F_list.append(problem(pt))
            used += 1

        # Combine all evaluated points
        X_all = np.array(X_list[:used])
        F_all = np.array(F_list[:used])

        return X_all
# EVOLVE-BLOCK-END

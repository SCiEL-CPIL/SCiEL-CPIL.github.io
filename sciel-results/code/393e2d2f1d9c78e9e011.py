"""Multi-objective optimizer for WFG7 using LHS initialization + Nelder-Mead local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """LHS-based diverse initialization followed by Nelder-Mead local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _lhs(self, n, dim, rng):
        """Latin Hypercube Sampling: n points in [0,1]^dim."""
        points = np.zeros((n, dim))
        for j in range(dim):
            perm = rng.permutation(n)
            points[:, j] = (perm + rng.random(n)) / n
        return points

    def _dominance_rank(self, F):
        """Non-dominated sorting: return rank array (0 = non-dominated)."""
        n = len(F)
        rank = np.zeros(n, dtype=int)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j or dominated[j]:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominated[j] = True
                    rank[j] = max(rank[j], 1)
        # Assign ranks based on dominance count
        for i in range(n):
            if not dominated[i]:
                rank[i] = 0
        return rank

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo

        # Phase 1: LHS initialization with diverse coverage
        n_lhs = min(40, self.budget // 5)
        lhs_unit = self._lhs(n_lhs, self.dim, rng)
        X_lhs = lo + lhs_unit * span
        F_lhs = np.array([problem(x) for x in X_lhs], dtype=float)
        used = n_lhs

        # Keep track of all evaluated points for hypervolume
        all_X = list(X_lhs)
        all_F = list(F_lhs)

        # Phase 2: Nelder-Mead local refinement on non-dominated points
        # Sort by dominance rank, refine the best (non-dominated) points
        rank = self._dominance_rank(F_lhs)
        # Get indices of non-dominated points, sort by sum of objectives
        nd_idx = np.where(rank == 0)[0]
        if len(nd_idx) == 0:
            nd_idx = np.argsort(np.sum(F_lhs, axis=1))[:5]
        else:
            # Sort non-dominated by sum of objectives
            nd_sorted = nd_idx[np.argsort(np.sum(F_lhs[nd_idx], axis=1))]
            nd_idx = nd_sorted

        # Allocate remaining budget for local search
        remaining_budget = self.budget - used
        n_refine = min(len(nd_idx), max(1, remaining_budget // 15))

        # For each non-dominated point, run Nelder-Mead on a scalarized objective
        for i in range(n_refine):
            if used >= self.budget:
                break
            if time.time() > deadline:
                break

            base_x = X_lhs[nd_idx[i]]
            base_f = F_lhs[nd_idx[i]].copy()

            # Create scalarized objective: weighted sum + penalty for being outside
            # Use different weights for each refinement to explore different parts
            weights = rng.dirichlet(np.ones(self.n_obj))

            def scalar_obj(x, _w=weights):
                f = problem(x)
                return float(np.dot(f, _w))

            # Nelder-Mead with bounded box constraint handled by clipping
            def bounded_obj(x):
                x_clip = np.clip(x, lo, hi)
                return scalar_obj(x_clip)

            # Initial simplex: perturb from base point
            x0 = base_x.copy()
            options = {
                'maxiter': max(10, (self.budget - used) // (n_refine - i)),
                'xatol': 1e-4,
                'fatol': 1e-4,
            }

            # Use a smaller initial step size
            init_step = 0.05 * span

            # Create initial simplex
            simplex = [x0.copy()]
            for k in range(self.dim):
                pt = x0.copy()
                pt[k] += init_step[k]
                pt = np.clip(pt, lo, hi)
                simplex.append(pt)

            result = minimize(
                bounded_obj,
                x0,
                method='Nelder-Mead',
                options=options
            )

            if result.success or result.fun < np.inf:
                x_best = np.clip(result.x, lo, hi)
                f_best = problem(x_best)
                all_X.append(x_best)
                all_F.append(f_best)
                used += 1

                # Also add intermediate points from the optimization if available
                # (Nelder-Mead doesn't easily expose all evaluations, so we just add the final)

        # Phase 3: If we have budget left, do more random + local search
        if used < self.budget and time.time() < deadline:
            # Add some more random points and refine
            n_extra = min(self.budget - used, 20)
            extra_unit = self._lhs(n_extra, self.dim, rng)
            X_extra = lo + extra_unit * span
            for x in X_extra:
                if used >= self.budget or time.time() > deadline:
                    break
                f = problem(x)
                all_X.append(x)
                all_F.append(f)
                used += 1

        # Phase 4: Final polish - refine the best points found so far
        if used < self.budget and time.time() < deadline:
            all_F_arr = np.array(all_F)
            rank_final = self._dominance_rank(all_F_arr)
            best_idx = np.where(rank_final == 0)[0]
            if len(best_idx) == 0:
                best_idx = [np.argmin(np.sum(all_F_arr, axis=1))]

            # Sort by sum
            best_sorted = best_idx[np.argsort(np.sum(all_F_arr[best_idx], axis=1))]
            n_polish = min(len(best_sorted), max(1, (self.budget - used) // 10))

            for i in range(n_polish):
                if used >= self.budget or time.time() > deadline:
                    break

                base_x = all_X[best_sorted[i]]
                weights = rng.dirichlet(np.ones(self.n_obj))

                def scalar_obj(x, _w=weights):
                    f = problem(x)
                    return float(np.dot(f, _w))

                def bounded_obj(x):
                    x_clip = np.clip(x, lo, hi)
                    return scalar_obj(x_clip)

                options = {
                    'maxiter': max(5, (self.budget - used) // max(1, n_polish - i)),
                    'xatol': 1e-5,
                    'fatol': 1e-5,
                }

                result = minimize(
                    bounded_obj,
                    base_x,
                    method='Nelder-Mead',
                    options=options
                )

                if result.fun < np.inf:
                    x_best = np.clip(result.x, lo, hi)
                    f_best = problem(x_best)
                    all_X.append(x_best)
                    all_F.append(f_best)
                    used += 1

        return all_X
# EVOLVE-BLOCK-END

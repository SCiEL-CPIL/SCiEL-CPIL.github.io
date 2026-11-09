"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic LHS + Nelder-Mead local refinement for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        budget = self.budget
        dim = self.dim
        n_obj = self.n_obj

        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        def non_dominated_set(F):
            idx = []
            for i, f in enumerate(F):
                dominated = False
                for j, g in enumerate(F):
                    if i != j and dominates(g, f):
                        dominated = True
                        break
                if not dominated:
                    idx.append(i)
            return np.array(idx, dtype=int)

        # Phase 1: Latin Hypercube sampling for guaranteed coverage
        n_lhs = min(budget, 80)
        rng = np.random.default_rng(42)
        # LHS: for each dimension, divide [0,1] into n_lhs strata, pick one uniform point per stratum
        lhs_points = np.zeros((n_lhs, dim))
        for d in range(dim):
            strata = np.arange(n_lhs) + rng.random(n_lhs)
            lhs_points[:, d] = strata / n_lhs
        X_lhs = lo + (hi - lo) * lhs_points
        F_lhs = np.array([problem(x) for x in X_lhs])
        used = n_lhs

        # Phase 2: Nelder-Mead local refinement on non-dominated points
        # We'll use scipy's minimize with Nelder-Mead for each ND point
        from scipy.optimize import minimize

        # Get non-dominated set
        nd_idx = non_dominated_set(F_lhs)
        if len(nd_idx) == 0:
            nd_idx = np.array([0])

        # Sort ND points by sum of objectives (proxy for quality)
        nd_F = F_lhs[nd_idx]
        nd_X = X_lhs[nd_idx]
        order = np.argsort(np.sum(nd_F, axis=1))
        nd_idx_sorted = nd_idx[order]
        nd_F_sorted = nd_F[order]
        nd_X_sorted = nd_X[order]

        # Allocate remaining budget to Nelder-Mead runs
        remaining = budget - used
        # Each Nelder-Mead run uses at most ~dim*4 evals (heuristic)
        max_nmead_per_point = max(5, min(20, remaining // max(1, len(nd_idx_sorted))))
        n_refine_points = min(len(nd_idx_sorted), max(1, remaining // max(5, max_nmead_per_point)))

        # Store all evaluated points
        all_X = [X_lhs.copy()]
        all_F = [F_lhs.copy()]

        for i in range(n_refine_points):
            if used >= budget:
                break
            if time.time() > deadline:
                break

            base_x = nd_X_sorted[i].copy()
            base_f = nd_F_sorted[i].copy()

            # Define scalar objective: sum of objectives (for Nelder-Mead)
            # But we need to handle the multi-objective nature.
            # Strategy: minimize the sum, but check if it improves the ND front.
            def scalar_obj(x):
                x = np.clip(x, lo, hi)
                return np.sum(problem(x))

            # Initial simplex
            x0 = base_x.copy()
            f0 = problem(x0)
            used += 1

            # Create initial simplex
            simplex = [x0.copy()]
            for d in range(dim):
                x_simplex = x0.copy()
                if x0[d] > 0.5 * (hi[d] - lo[d]):
                    x_simplex[d] -= 0.05 * (hi[d] - lo[d])
                else:
                    x_simplex[d] += 0.05 * (hi[d] - lo[d])
                x_simplex = np.clip(x_simplex, lo, hi)
                simplex.append(x_simplex)

            # Evaluate simplex
            simplex_F = [f0]
            for d in range(1, dim + 1):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                fs = problem(simplex[d])
                simplex_F.append(fs)
                used += 1

            # Nelder-Mead iterations
            max_iter_nmead = min(30, budget - used)
            for it in range(max_iter_nmead):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break

                # Sort simplex by scalar objective
                order_simplex = np.argsort([np.sum(f) for f in simplex_F])
                simplex = [simplex[j] for j in order_simplex]
                simplex_F = [simplex_F[j] for j in order_simplex]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                worst_f = simplex_F[-1]

                # Reflection
                x_ref = centroid + (centroid - worst)
                x_ref = np.clip(x_ref, lo, hi)
                f_ref = problem(x_ref)
                used += 1
                scalar_ref = np.sum(f_ref)
                scalar_best = np.sum(simplex_F[0])
                scalar_worst = np.sum(worst_f)

                if scalar_ref < scalar_best:
                    # Expansion
                    x_exp = centroid + 1.5 * (centroid - worst)
                    x_exp = np.clip(x_exp, lo, hi)
                    f_exp = problem(x_exp)
                    used += 1
                    scalar_exp = np.sum(f_exp)
                    if scalar_exp < scalar_ref:
                        simplex[-1] = x_exp
                        simplex_F[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        simplex_F[-1] = f_ref
                elif scalar_ref < scalar_worst:
                    simplex[-1] = x_ref
                    simplex_F[-1] = f_ref
                else:
                    # Contraction
                    x_con = centroid + 0.5 * (worst - centroid)
                    x_con = np.clip(x_con, lo, hi)
                    f_con = problem(x_con)
                    used += 1
                    scalar_con = np.sum(f_con)
                    if scalar_con < scalar_worst:
                        simplex[-1] = x_con
                        simplex_F[-1] = f_con
                    else:
                        # Shrink
                        for d in range(1, dim + 1):
                            simplex[d] = simplex[0] + 0.5 * (simplex[d] - simplex[0])
                            simplex[d] = np.clip(simplex[d], lo, hi)
                            f_shrink = problem(simplex[d])
                            used += 1
                            simplex_F[d] = f_shrink
                        if used >= budget:
                            break

            # Add best point from Nelder-Mead to all_X/all_F
            best_idx = np.argmin([np.sum(f) for f in simplex_F])
            all_X.append(simplex[best_idx].reshape(1, -1))
            all_F.append(simplex_F[best_idx].reshape(1, -1))

        # Combine all points
        all_X = np.vstack(all_X)
        all_F = np.vstack(all_F)

        return all_X

# EVOLVE-BLOCK-END

"""Multi-objective optimizer for RE34 using MOEA/D decomposition with L-BFGS-B refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """MOEA/D-based multi-objective optimizer with Tchebycheff decomposition
    and L-BFGS-B local search for systematic Pareto front coverage."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random.default_rng(42)

        # Storage for all evaluated points
        all_X = []
        all_F = []
        eval_count = [0]

        def count_eval(x):
            if eval_count[0] >= self.budget:
                return None
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=np.float64)
            all_X.append(x.copy())
            all_F.append(f.copy())
            eval_count[0] += 1
            return f

        def get_non_dominated(F_arr):
            """Return indices of non-dominated points."""
            n = len(F_arr)
            if n == 0:
                return []
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    if np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated[i] = True
                        break
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        dominated[j] = True
            return [i for i in range(n) if not dominated[i]]

        # ================================================================
        # Phase 1: Initial sampling - Latin hypercube + random
        # ================================================================
        n_init = min(20, self.budget // 5)

        for i in range(self.dim):
            perm = rng.permutation(n_init)
            for j in range(n_init):
                x = lo + (hi - lo) * (perm[j] + rng.random()) / n_init
                count_eval(x)

        # ================================================================
        # Phase 2: MOEA/D with Tchebycheff scalarization + L-BFGS-B
        # ================================================================
        # Generate weight vectors on the simplex (200 points for 3 objectives)
        n_weights = 12
        weights = []

        # Extreme points
        for i in range(self.n_obj):
            w = np.zeros(self.n_obj)
            w[i] = 1.0
            weights.append(w)

        # Uniform weights
        n_inner = n_weights - self.n_obj
        if n_inner > 0:
            for _ in range(n_inner):
                w = rng.random(self.n_obj)
                w /= w.sum()
                weights.append(w)

        weights = np.array(weights)

        # Tchebycheff scalarization with adaptive reference point
        def tchebycheff_scalarization(x, w, z):
            """Tchebycheff: max_i w_i * |f_i(x) - z_i|"""
            if eval_count[0] >= self.budget:
                return np.inf
            fx = count_eval(x)
            if fx is None:
                return np.inf
            # Use current ideal point z
            return np.max(w * np.abs(fx - z))

        for w in weights:
            if time.time() > deadline or eval_count[0] >= self.budget:
                break

            # Current ideal point (component-wise minimum of all evaluated)
            F_arr = np.array(all_F)
            z = np.min(F_arr, axis=0)

            # Choose best starting point based on Tchebycheff value
            if len(all_F) > 0:
                F_arr = np.array(all_F)
                tcheb_vals = np.array([np.max(w * np.abs(f - z)) for f in F_arr])
                best_idx = np.argmin(tcheb_vals)
                x0 = all_X[best_idx].copy()
            else:
                x0 = (lo + hi) / 2.0

            def obj(x, w=w, z=z):
                if eval_count[0] >= self.budget:
                    return np.inf
                fx = count_eval(x)
                if fx is None:
                    return np.inf
                return np.max(w * np.abs(fx - z))

            try:
                result = minimize(
                    obj, x0, method='L-BFGS-B',
                    bounds=list(zip(lo, hi)),
                    options={'maxiter': 30, 'maxls': 10}
                )
                if result.fun < np.inf:
                    x_res = np.clip(result.x, lo, hi)
                    f_res = count_eval(x_res)
            except Exception:
                pass

        # ================================================================
        # Phase 3: Gap-filling search using reference point based
        # adaptive scalarization
        # ================================================================
        if eval_count[0] < self.budget and time.time() < deadline:
            F_arr = np.array(all_F)
            X_arr = np.array(all_X)

            # Reference point for gap-filling
            z_ref = np.array([1705.0, 12.0, 0.3])

            def hypervolume_gap(x, w, z):
                """Augmented epsilon-constraint style: maximize minimum
                improvement in hypervolume contribution."""
                if eval_count[0] >= self.budget:
                    return np.inf
                fx = count_eval(x)
                if fx is None:
                    return np.inf
                # Weighted sum with emphasis on under-explored regions
                # Use normalized objectives
                F_norm = np.array(all_F)
                f_min = np.min(F_norm, axis=0)
                f_max = np.max(F_norm, axis=0)
                f_range = np.where(f_max - f_min > 1e-12, f_max - f_min, 1.0)
                fx_norm = (fx - f_min) / f_range
                return np.sum(w * fx_norm)

            # Generate more weight vectors focused on gaps
            n_gap_weights = 8
            for _ in range(n_gap_weights):
                if time.time() > deadline or eval_count[0] >= self.budget:
                    break

                w = rng.random(self.n_obj)
                w /= w.sum()

                # Start from random point in the current Pareto front region
                nd_idx = get_non_dominated(F_arr)
                if len(nd_idx) > 0:
                    # Pick a random non-dominated point as starting point
                    start_idx = nd_idx[rng.integers(len(nd_idx))]
                    x0 = X_arr[start_idx].copy()
                    # Add some noise
                    x0 += 0.1 * (hi - lo) * rng.standard_normal(self.dim)
                    x0 = np.clip(x0, lo, hi)
                else:
                    x0 = lo + (hi - lo) * rng.random(self.dim)

                z = np.min(F_arr, axis=0) if len(F_arr) > 0 else np.zeros(self.n_obj)

                def obj_gap(x, w=w, z=z):
                    if eval_count[0] >= self.budget:
                        return np.inf
                    fx = count_eval(x)
                    if fx is None:
                        return np.inf
                    # Tchebycheff with adaptation
                    return np.max(w * np.abs(fx - z))

                try:
                    result = minimize(
                        obj_gap, x0, method='L-BFGS-B',
                        bounds=list(zip(lo, hi)),
                        options={'maxiter': 25, 'maxls': 10}
                    )
                    if result.fun < np.inf:
                        x_res = np.clip(result.x, lo, hi)
                        f_res = count_eval(x_res)
                except Exception:
                    pass

        # ================================================================
        # Phase 4: Multi-start L-BFGS-B from boundary of Pareto front
        # ================================================================
        if eval_count[0] < self.budget and time.time() < deadline:
            F_arr = np.array(all_F)
            X_arr = np.array(all_X)
            nd_idx = get_non_dominated(F_arr)

            if len(nd_idx) >= 2:
                # Use pairs of non-dominated points to generate intermediate starts
                n_pairs = min(6, (self.budget - eval_count[0]) // 8)
                for _ in range(n_pairs):
                    if time.time() > deadline or eval_count[0] >= self.budget:
                        break

                    i1 = nd_idx[rng.integers(len(nd_idx))]
                    i2 = nd_idx[rng.integers(len(nd_idx))]
                    if i1 == i2:
                        continue

                    # Interpolate between two Pareto points
                    alpha = rng.random()
                    x0 = alpha * X_arr[i1] + (1 - alpha) * X_arr[i2]
                    x0 = np.clip(x0, lo, hi)

                    # Weight vector orthogonal to the segment in objective space
                    f1 = F_arr[i1]
                    f2 = F_arr[i2]
                    w = np.abs(f2 - f1)
                    if np.sum(w) > 1e-12:
                        w /= np.sum(w)
                    else:
                        w = rng.random(self.n_obj)
                        w /= w.sum()

                    z = np.min(F_arr, axis=0)

                    def obj_interp(x, w=w, z=z):
                        if eval_count[0] >= self.budget:
                            return np.inf
                        fx = count_eval(x)
                        if fx is None:
                            return np.inf
                        return np.max(w * np.abs(fx - z))

                    try:
                        result = minimize(
                            obj_interp, x0, method='L-BFGS-B',
                            bounds=list(zip(lo, hi)),
                            options={'maxiter': 20, 'maxls': 8}
                        )
                        if result.fun < np.inf:
                            x_res = np.clip(result.x, lo, hi)
                            f_res = count_eval(x_res)
                    except Exception:
                        pass

        # ================================================================
        # Phase 5: Final local refinement around best non-dominated points
        # ================================================================
        if eval_count[0] < self.budget and time.time() < deadline:
            F_arr = np.array(all_F)
            X_arr = np.array(all_X)

            nd_idx = get_non_dominated(F_arr)

            n_refine = min(5, len(nd_idx), max(1, (self.budget - eval_count[0]) // 10))
            for idx in nd_idx[:n_refine]:
                if time.time() > deadline or eval_count[0] >= self.budget:
                    break

                x_nd = X_arr[idx]
                step_size = 0.02 * (hi - lo)

                for _ in range(3):
                    if eval_count[0] >= self.budget or time.time() > deadline:
                        break
                    x_new = x_nd + step_size * rng.standard_normal(self.dim)
                    x_new = np.clip(x_new, lo, hi)
                    f_new = count_eval(x_new)
                    if f_new is None:
                        break

        return all_X
# EVOLVE-BLOCK-END

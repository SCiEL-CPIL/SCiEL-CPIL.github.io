"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer combining weighted-sum L-BFGS-B,
    NSGA-II-style evolutionary search, and adaptive local refinement."""

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

        # ================================================================
        # Phase 1: Initial sampling - random + Latin hypercube for coverage
        # ================================================================
        n_init = min(20, self.budget // 5)

        # Latin hypercube for uniform coverage
        for i in range(self.dim):
            perm = rng.permutation(n_init)
            for j in range(n_init):
                x = lo + (hi - lo) * (perm[j] + rng.random()) / n_init
                count_eval(x)

        # ================================================================
        # Phase 2: Weighted-sum L-BFGS-B decomposition
        # ================================================================
        # Generate diverse weight vectors on the simplex
        n_weights = 12
        weights = []

        # Extreme points (each objective individually)
        for i in range(self.n_obj):
            w = np.zeros(self.n_obj)
            w[i] = 1.0
            weights.append(w)

        # Uniform random weights on simplex
        for _ in range(n_weights - self.n_obj):
            w = rng.random(self.n_obj)
            w /= w.sum()
            weights.append(w)

        weights = np.array(weights)

        for w in weights:
            if time.time() > deadline or eval_count[0] >= self.budget:
                break

            # Choose best starting point based on current weighted sum
            if len(all_F) > 0:
                F_arr = np.array(all_F)
                weighted_sums = F_arr @ w
                best_idx = np.argmin(weighted_sums)
                x0 = all_X[best_idx].copy()
            else:
                x0 = (lo + hi) / 2.0

            def obj(x, w=w):
                if eval_count[0] >= self.budget:
                    return np.inf
                fx = count_eval(x)
                if fx is None:
                    return np.inf
                return np.sum(fx * w)

            try:
                result = minimize(
                    obj, x0, method='L-BFGS-B',
                    bounds=list(zip(lo, hi)),
                    options={'maxiter': 40, 'maxls': 15}
                )
                if result.fun < np.inf:
                    x_res = np.clip(result.x, lo, hi)
                    f_res = count_eval(x_res)
            except Exception:
                pass

        # ================================================================
        # Phase 3: NSGA-II style evolutionary search
        # ================================================================
        if eval_count[0] < self.budget and time.time() < deadline:
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

            def crowding_distance(F_arr, nd_idx):
                """Compute crowding distance for non-dominated points."""
                n = len(nd_idx)
                if n <= 2:
                    return np.full(n, np.inf)

                distances = np.zeros(n)
                F_nd = F_arr[nd_idx]

                for m in range(self.n_obj):
                    sorted_idx = np.argsort(F_nd[:, m])
                    distances[sorted_idx[0]] = np.inf
                    distances[sorted_idx[-1]] = np.inf

                    f_min = F_nd[sorted_idx[0], m]
                    f_max = F_nd[sorted_idx[-1], m]
                    span = f_max - f_min
                    if span < 1e-12:
                        continue

                    for k in range(1, n - 1):
                        distances[sorted_idx[k]] += (
                            F_nd[sorted_idx[k + 1], m] - F_nd[sorted_idx[k - 1], m]
                        ) / span

                return distances

            # Build initial population from non-dominated points
            F_arr = np.array(all_F)
            nd_idx = get_non_dominated(F_arr)

            # Population size for evolutionary phase
            pop_size = max(8, min(25, (self.budget - eval_count[0]) // 8))

            # If we don't have enough non-dominated points, add random ones
            while len(nd_idx) < pop_size and eval_count[0] < self.budget:
                x = lo + (hi - lo) * rng.random(self.dim)
                f = count_eval(x)
                if f is not None:
                    nd_idx.append(len(all_F) - 1)

            # Take first pop_size
            pop_idx = nd_idx[:pop_size]

            # Adaptive mutation scale
            total_remaining = self.budget - eval_count[0]
            mutation_scale = 0.15 * (hi - lo)

            while eval_count[0] < self.budget and time.time() < deadline:
                F_arr = np.array(all_F)
                nd_idx = get_non_dominated(F_arr)

                if len(nd_idx) < 2:
                    nd_idx = list(range(len(F_arr)))

                # Compute crowding distances for selection
                cd = crowding_distance(F_arr, nd_idx)

                # Tournament selection with crowding distance
                def tournament_select():
                    t1 = rng.integers(len(nd_idx))
                    t2 = rng.integers(len(nd_idx))
                    if cd[t1] > cd[t2]:
                        return nd_idx[t1]
                    elif cd[t2] > cd[t1]:
                        return nd_idx[t2]
                    else:
                        return nd_idx[rng.integers(2)]

                p1_idx = tournament_select()
                p2_idx = tournament_select()

                # Simulated binary crossover
                x1 = all_X[p1_idx]
                x2 = all_X[p2_idx]
                beta = 0.5
                alpha = np.where(rng.random(self.dim) < beta,
                                 rng.power(2 / (beta + 1), self.dim),
                                 1.0 - rng.power(2 / (beta + 1), self.dim))
                child = alpha * x1 + (1 - alpha) * x2
                child = np.clip(child, lo, hi)

                # Adaptive Gaussian mutation (decays over time)
                progress = eval_count[0] / self.budget
                current_scale = mutation_scale * (1.0 - 0.5 * progress)
                if rng.random() < 0.7:
                    child += current_scale * rng.standard_normal(self.dim)
                    child = np.clip(child, lo, hi)

                fc = count_eval(child)
                if fc is None:
                    break

        # ================================================================
        # Phase 4: Final local refinement around best non-dominated points
        # ================================================================
        if eval_count[0] < self.budget and time.time() < deadline:
            F_arr = np.array(all_F)
            X_arr = np.array(all_X)

            def get_non_dominated(F_arr):
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

            nd_idx = get_non_dominated(F_arr)

            # Refine top non-dominated points with small perturbations
            n_refine = min(6, len(nd_idx), max(1, (self.budget - eval_count[0]) // 10))
            for idx in nd_idx[:n_refine]:
                if time.time() > deadline or eval_count[0] >= self.budget:
                    break

                x_nd = X_arr[idx]
                step_size = 0.03 * (hi - lo)

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

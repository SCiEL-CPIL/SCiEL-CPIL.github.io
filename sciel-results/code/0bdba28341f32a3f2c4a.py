from scipy.optimize import minimize
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE34 using structured exploration and local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _non_dominated_mask(self, F):
        """Return a boolean mask of non-dominated points in F (n x n_obj array)."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=bool)

        F_i = F[:, np.newaxis, :]  # (n, 1, n_obj)
        F_j = F[np.newaxis, :, :]  # (1, n, n_obj)

        leq = np.all(F_i <= F_j, axis=2)  # (n, n)
        lt = np.any(F_i < F_j, axis=2)    # (n, n)

        dominates = leq & lt
        dominated = np.any(dominates, axis=0)

        return ~dominated

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        rng = np.random.default_rng(123)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget

        ref_point = np.array([1705.0, 12.0, 0.3])

        def time_left():
            return deadline - time.time()

        def is_deadline():
            return time.time() > deadline - 10

        # Phase 1: Structured initialization with LHS for better coverage
        n_initial = min(60, budget // 3)
        X = np.random.uniform(lo, hi, size=(n_initial, dim))
        
        # Add some structured points at corners and midpoints
        structured = []
        for i in range(2**dim):
            x = np.array([(hi[j] if (i >> j) & 1 else lo[j]) for j in range(dim)])
            structured.append(x)
        # Add midpoint
        structured.append((lo + hi) / 2)
        structured = np.array(structured)
        
        # Combine and sample
        all_init = np.vstack([X, structured])
        if len(all_init) > n_initial:
            idx = rng.choice(len(all_init), size=n_initial, replace=False)
            X = all_init[idx]
        else:
            X = all_init
        
        F = np.array([problem(x) for x in X])
        used = len(X)
        all_F = F.copy()

        # Phase 2: Surrogate-assisted search (GP + EI on weighted scalarization)

        def matern52_kernel(X1, X2, length_scale, variance):
            X1 = X1[:, :, np.newaxis]
            X2 = X2[:, np.newaxis, :]
            sq_dist = np.sum((X1 - X2) ** 2, axis=2)
            z = np.sqrt(3.0 * sq_dist) / length_scale
            K = variance * (1.0 + z + z ** 2 / 3.0) * np.exp(-z)
            return K

        def predict_gp(X_train, y_train, X_pred, length_scale, variance, noise):
            n_train = len(y_train)
            K = matern52_kernel(X_train, X_train, length_scale, variance)
            K[np.diag_indices_from(K)] += noise
            K_pred = matern52_kernel(X_train, X_pred, length_scale, variance)
            try:
                K_inv_y = np.linalg.solve(K, y_train)
            except np.linalg.LinAlgError:
                K_inv_y = np.linalg.lstsq(K, y_train, rcond=None)[0]
            mean = K_pred.T @ K_inv_y
            K_pred_pred = matern52_kernel(X_pred, X_pred, length_scale, variance)
            try:
                K_inv = np.linalg.inv(K)
            except:
                K_inv = np.linalg.pinv(K)
            var = np.diag(K_pred_pred) - np.sum(K_pred * (K_pred @ K_inv), axis=1)
            var = np.maximum(var, 1e-10)
            std = np.sqrt(var)
            return mean, std

        def estimate_length_scale(X, y):
            if len(X) < 2:
                return 0.5
            from scipy.spatial.distance import pdist
            dists = pdist(X, 'euclidean')
            if len(dists) == 0:
                return 0.5
            return max(0.1, np.median(dists) * 0.5)

        surrogate_budget = int(budget * 0.55)
        n_weights = 12
        W = rng.random((n_weights, n_obj))
        W /= W.sum(axis=1, keepdims=True)
        best_vals = np.full(n_weights, np.inf)

        while used < surrogate_budget:
            if is_deadline():
                break

            for w_idx in range(n_weights):
                w = W[w_idx]
                y_w = F @ w
                best_vals[w_idx] = np.min(y_w)

            n_candidates = 30
            candidates = np.random.uniform(lo, hi, size=(n_candidates, dim))
            nd_mask = self._non_dominated_mask(F)
            nd_idx = np.where(nd_mask)[0]
            if len(nd_idx) > 0:
                n_pert = min(20, n_candidates // 2)
                for _ in range(n_pert):
                    pi = nd_idx[rng.integers(len(nd_idx))]
                    pert = X[pi] + 0.1 * (hi - lo) * rng.standard_normal(dim)
                    pert = np.clip(pert, lo, hi)
                    candidates = np.vstack([candidates, pert])

            candidates = np.clip(candidates, lo, hi)

            best_candidate = None
            best_ei = -np.inf

            for w_idx in range(n_weights):
                w = W[w_idx]
                y_w = F @ w
                ls = estimate_length_scale(X, y_w)
                var_y = max(np.var(y_w), 1e-6)
                noise = 1e-4 * var_y

                try:
                    mean_pred, std_pred = predict_gp(X, y_w, candidates, ls, var_y, noise)
                except:
                    continue

                improvement = best_vals[w_idx] - mean_pred
                z = improvement / (std_pred + 1e-10)
                ei = improvement * norm.cdf(z) + std_pred * norm.pdf(z)
                ei = np.nan_to_num(ei, nan=0.0, posinf=0.0, neginf=0.0)

                idx_best = np.argmax(ei)
                if ei[idx_best] > best_ei:
                    best_ei = ei[idx_best]
                    best_candidate = candidates[idx_best].copy()

            if best_candidate is None:
                best_candidate = np.random.uniform(lo, hi, size=(dim,))

            fc = problem(best_candidate)
            used += 1
            all_F = np.vstack([all_F, fc])

            X = np.vstack([X, best_candidate])
            F = np.vstack([F, fc])

            if len(F) > 60:
                nd_mask = self._non_dominated_mask(F)
                nd_idx = np.where(nd_mask)[0]
                dom_idx = np.where(~nd_mask)[0]
                if len(nd_idx) > 40:
                    keep_idx = nd_idx[:40]
                else:
                    keep_idx = nd_idx
                    if len(dom_idx) > 0:
                        n_add = min(20, len(dom_idx))
                        add_idx = dom_idx[rng.choice(len(dom_idx), size=n_add, replace=False)]
                        keep_idx = np.concatenate([keep_idx, add_idx])
                keep_idx = np.unique(keep_idx)
                X = X[keep_idx]
                F = F[keep_idx]

        # Phase 3: Local refinement with weighted scalarization
        if used < budget and time_left() > 40:
            nd_mask = self._non_dominated_mask(F)
            nd_idx = np.where(nd_mask)[0]

            if len(nd_idx) > 0:
                F_nd = F[nd_idx]
                F_min = F_nd.min(axis=0)
                F_max = F_nd.max(axis=0)
                F_range = F_max - F_min
                F_range[F_range == 0] = 1.0
                F_norm = (F_nd - F_min) / F_range
                dists = np.sqrt(np.sum(F_norm ** 2, axis=1))
                sorted_nd = nd_idx[np.argsort(dists)]

                points_to_refine = sorted_nd[:min(10, len(sorted_nd))]
                n_weights_refine = min(8, max(3, (budget - used) // 10))
                W_refine = rng.random((n_weights_refine, n_obj))
                W_refine /= W_refine.sum(axis=1, keepdims=True)

                for pt_idx in points_to_refine:
                    if used >= budget or time_left() < 40:
                        break

                    x0 = X[pt_idx].copy()

                    for w in W_refine:
                        if used >= budget or time_left() < 40:
                            break

                        eval_count = [0]

                        def obj(x, w=w):
                            eval_count[0] += 1
                            return np.dot(w, problem(x))

                        try:
                            res = minimize(obj, x0, method='L-BFGS-B',
                                          bounds=list(zip(lo, hi)),
                                          options={'maxiter': 30, 'ftol': 1e-14, 'gtol': 1e-12})
                            if res.success and eval_count[0] > 0:
                                fx = problem(res.x)
                                used += eval_count[0] + 1
                                all_F = np.vstack([all_F, fx])

                                if np.any(fx < F[pt_idx]):
                                    X[pt_idx] = res.x
                                    F[pt_idx] = fx
                        except:
                            used += eval_count[0]

        return X

    def _find_worst_index(self, F, ref_point):
        """Find the index of the worst point in the population using hypervolume contribution."""
        if len(F) == 0:
            return 0

        nd_mask = self._non_dominated_mask(F)
        n = len(F)

        # Compute approximate hypervolume contribution for each point
        # Points that dominate others or are far from reference are "worse"
        F_min = F.min(axis=0)
        F_max = F.max(axis=0)
        F_range = F_max - F_min
        F_range[F_range == 0] = 1.0
        F_norm = (F - F_min) / F_range

        dist = np.sqrt(np.sum(F_norm ** 2, axis=1))
        penalty = np.where(~nd_mask, 100.0, 0.0)

        # Also penalize points that are close to the reference point (less useful)
        ref_dist = np.sqrt(np.sum((F - ref_point) ** 2, axis=1))
        ref_penalty = -0.1 * ref_dist  # Points closer to ref get higher penalty (worse)

        quality = dist + penalty + ref_penalty
        return np.argmax(quality)
# EVOLVE-BLOCK-END

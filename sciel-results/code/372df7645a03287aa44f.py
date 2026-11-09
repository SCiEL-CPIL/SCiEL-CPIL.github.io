"""Multi-objective optimizer using Bayesian Optimization with MOEI acquisition."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import qmc
from scipy.optimize import minimize


class Optimizer:
    """Bayesian Optimization with multi-objective expected improvement for RE37."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Phase 1: Initial sampling with Sobol sequence
        n_init = min(64, budget)
        sampler = qmc.Sobol(d=dim, scramble=True)
        init_points = sampler.random(n_init)
        X_init = lo + (hi - lo) * init_points
        F_init = np.array([problem(x) for x in X_init])
        used = n_init

        # Track all evaluated points
        X_all = X_init.copy()
        F_all = F_init.copy()

        def get_non_dominated_indices(F_arr):
            """Return indices of non-dominated points in F_arr (minimization)."""
            n = len(F_arr)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(i + 1, n):
                    if dominated[j]:
                        continue
                    if np.all(F_arr[i] <= F_arr[j]) and np.any(F_arr[i] < F_arr[j]):
                        dominated[j] = True
                    elif np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        dominated[i] = True
                        break
            return np.where(~dominated)[0]

        # Phase 2: Bayesian Optimization with MOEI
        remaining = budget - used
        if remaining > 0:
            nd_idx = get_non_dominated_indices(F_all)
            if len(nd_idx) == 0:
                nd_idx = np.arange(len(F_all))

            # Reference point for MOEI (slightly worse than current best)
            ref_point = np.max(F_all, axis=0) + 0.1

            # GP hyperparameters (simple RBF kernel)
            # We'll use a simplified GP with fixed hyperparameters for speed
            def gp_predict(X_train, Y_train, X_test):
                """Simple GP prediction with RBF kernel."""
                n_train = len(X_train)
                # Standardize
                X_mean = np.mean(X_train, axis=0)
                X_std = np.std(X_train, axis=0) + 1e-8
                X_train_std = (X_train - X_mean) / X_std
                X_test_std = (X_test - X_mean) / X_std

                # RBF kernel
                def rbf_kernel(X1, X2, length_scale=0.5):
                    # Compute pairwise distances
                    diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
                    dist_sq = np.sum(diff ** 2, axis=2)
                    return np.exp(-0.5 * dist_sq / (length_scale ** 2))

                K = rbf_kernel(X_train_std, X_train_std) + 1e-6 * np.eye(n_train)
                Ks = rbf_kernel(X_train_std, X_test_std)

                # Solve for weights
                try:
                    L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
                except:
                    alpha = np.linalg.solve(K + 1e-4 * np.eye(n_train), Y_train)

                # Prediction
                mu = Ks.T @ alpha
                v = np.linalg.solve(L, Ks)
                var = 1.0 - np.sum(v ** 2, axis=0)
                var = np.maximum(var, 1e-8)
                std = np.sqrt(var)

                return mu, std

            # MOEI acquisition function
            def moei_acquisition(x, X_train, F_train, nd_points):
                """Multi-objective expected improvement."""
                x = np.array(x).reshape(1, -1)
                try:
                    mu, std = gp_predict(X_train, F_train, x)
                except:
                    return -1.0

                # For each non-dominated point, compute EI
                ei_sum = 0.0
                for nd_p in nd_points:
                    f_nd = F_train[nd_p]
                    # Expected improvement for each objective
                    for k in range(n_obj):
                        mu_k = mu[0, k]
                        std_k = std[0, k]
                        if std_k < 1e-8:
                            continue
                        z = (f_nd - mu_k) / std_k
                        # Standard normal CDF and PDF
                        from scipy.stats import norm
                        ei = std_k * (z * norm.cdf(z) + norm.pdf(z))
                        ei_sum += max(ei, 0.0)
                return ei_sum

            # Iterative BO loop
            rng = np.random.default_rng(42)
            n_bo_iter = min(remaining, 136)  # Leave some budget for final refinement

            for bo_iter in range(n_bo_iter):
                if used >= budget or time.time() > deadline:
                    break

                # Get current non-dominated points
                nd_idx = get_non_dominated_indices(F_all)
                if len(nd_idx) == 0:
                    nd_idx = np.arange(len(F_all))
                nd_points = nd_idx[:min(20, len(nd_idx))]  # Limit to top 20 for speed

                # Candidate points: Sobol-based candidates + local perturbations
                n_candidates = 32
                cand_sampler = qmc.Sobol(d=dim, scramble=True)
                cand_points = cand_sampler.random(n_candidates)
                cand_X = lo + (hi - lo) * cand_points

                # Also add local perturbations around non-dominated points
                local_cands = []
                for idx in nd_points[:5]:
                    for _ in range(4):
                        perturbation = 0.1 * (hi - lo) * rng.standard_normal(dim)
                        local_cand = X_all[idx] + perturbation
                        local_cand = np.clip(local_cand, lo, hi)
                        local_cands.append(local_cand)
                local_cands = np.array(local_cands) if local_cands else np.empty((0, dim))

                all_candidates = np.vstack([cand_X, local_cands]) if len(local_cands) > 0 else cand_X

                # Evaluate acquisition function
                acq_values = []
                for cand in all_candidates:
                    acq_val = moei_acquisition(cand, X_all, F_all, nd_points)
                    acq_values.append(acq_val)
                acq_values = np.array(acq_values)

                # Select best candidate
                best_cand_idx = np.argmax(acq_values)
                x_new = all_candidates[best_cand_idx]

                # Evaluate
                f_new = problem(x_new)
                used += 1
                X_all = np.vstack([X_all, x_new.reshape(1, -1)])
                F_all = np.vstack([F_all, f_new.reshape(1, -1)])

                # Periodically check time
                if bo_iter % 10 == 0 and time.time() > deadline:
                    break

            # Phase 3: Final local refinement around best non-dominated points
            remaining = budget - used
            if remaining > 0 and time.time() < deadline:
                nd_idx_final = get_non_dominated_indices(F_all)
                if len(nd_idx_final) > 0:
                    sums = np.sum(F_all[nd_idx_final], axis=1)
                    top_idx = nd_idx_final[np.argsort(sums)[:min(5, len(nd_idx_final))]]
                    for idx in top_idx:
                        if used >= budget or time.time() > deadline:
                            break
                        for _ in range(min(remaining, 10)):
                            if used >= budget or time.time() > deadline:
                                break
                            child = X_all[idx] + 0.05 * (hi - lo) * rng.standard_normal(dim)
                            child = np.clip(child, lo, hi)
                            fc = problem(child)
                            used += 1
                            X_all = np.vstack([X_all, child.reshape(1, -1)])
                            F_all = np.vstack([F_all, fc.reshape(1, -1)])

        # Return the last population (return value is ignored by evaluator)
        return X_all
# EVOLVE-BLOCK-END

"""Bayesian Optimization with GP surrogate + Nelder-Mead local polish for f24 (Lunacek bi-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """GP-based Bayesian Optimization for global search + Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 290.0

        best_x = np.zeros(dim)
        best_f = float('inf')
        calls = 0

        def check_deadline():
            return time.time() >= deadline or calls >= budget

        def eval_func(x):
            nonlocal calls, best_f, best_x
            if check_deadline():
                return float('inf')
            x = np.asarray(x, dtype=np.float64)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(777)

        # Phase 1: Initial exploration - sample to find both Lunacek basins
        # Lunacek bi-Rastrigin has two global minima; we need broad coverage
        n_init = min(40, budget // 10)
        X_samples = []
        y_samples = []

        for i in range(n_init):
            if check_deadline():
                break
            if i < dim + 1:
                # Latin hypercube-like stratified samples
                x = rng.uniform(-4.5, 4.5, dim)
            elif i % 4 == 0:
                # Focus on potential basin centers
                center = rng.choice([-3.0, 3.0, 0.0, -1.5, 1.5])
                x = np.full(dim, center) + rng.normal(0, 1.5, dim)
            else:
                x = rng.uniform(-4.5, 4.5, dim)
            x = np.clip(x, -5, 5)
            fx = eval_func(x)
            if np.isfinite(fx):
                X_samples.append(x.copy())
                y_samples.append(fx)

        if not X_samples:
            return best_x, best_f

        X_arr = np.array(X_samples)
        y_arr = np.array(y_samples)

        # Phase 2: Bayesian Optimization loop with GP surrogate
        # Use a simple GP with RBF kernel and EI acquisition
        n_bo = min(200, budget - calls - 50)  # reserve 50 for local polish
        kappa = 1.5  # exploration-exploitation balance

        for bo_iter in range(n_bo):
            if check_deadline():
                break

            # Fit GP: y = f(X) + noise
            # Use normalized inputs for better conditioning
            X_min = X_arr.min(axis=0)
            X_max = X_arr.max(axis=0)
            X_range = np.where(X_max - X_min < 1e-10, 1.0, X_max - X_min)
            X_norm = (X_arr - X_min) / X_range

            y_mean = y_arr.mean()
            y_std = max(y_arr.std(), 1e-10)
            y_norm = (y_arr - y_mean) / y_std

            # GP with RBF kernel
            # K(x, x') = sigma_f^2 * exp(-0.5 * (x-x')^T * L^-1 * (x-x'))
            # For simplicity, use isotropic RBF with length-scale adaptation
            # Compute pairwise distances
            Xn = X_norm
            n_pts = len(Xn)

            # Pairwise squared distances
            diff = Xn[:, np.newaxis, :] - Xn[np.newaxis, :, :]
            sq_dist = np.sum(diff ** 2, axis=2)

            # Adaptive length scale based on data spread
            # Use median distance heuristic
            if n_pts > 1:
                triu_idx = np.triu_indices(n_pts, k=1)
                med_dist = np.median(np.sqrt(sq_dist[triu_idx]))
                length_scale = max(med_dist * 0.5, 0.1)
            else:
                length_scale = 1.0

            # RBF kernel matrix
            sigma_f_sq = 1.0
            noise = 1e-6
            K = sigma_f_sq * np.exp(-0.5 * sq_dist / (length_scale ** 2))
            K[np.diag_indices_from(K)] += noise

            # Cholesky decomposition for stable inversion
            try:
                L_chol = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K[np.diag_indices_from(K)] += 1e-4
                try:
                    L_chol = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    break

            # Posterior mean and variance at candidate points
            # We'll evaluate EI on a grid of candidate points

            # Generate candidate points: mix of random and local perturbations
            n_candidates = 50
            candidates = []

            # Random candidates
            for _ in range(20):
                candidates.append(rng.uniform(-4.8, 4.8, dim))

            # Local perturbations around best points
            top_k = min(5, len(X_arr))
            y_sorted_idx = np.argsort(y_arr)[:top_k]
            for idx in y_sorted_idx:
                x_base = X_arr[idx]
                for scale in [0.3, 0.8, 1.5]:
                    candidates.append(x_base + rng.normal(0, scale, dim))

            # Additional structured candidates
            for _ in range(10):
                x = rng.uniform(-4.8, 4.8, dim)
                # Snap one coordinate to a basin center
                d = rng.integers(0, dim)
                x[d] = rng.choice([-3.0, 3.0, 0.0])
                candidates.append(x)

            candidates = np.array(candidates)
            candidates = np.clip(candidates, -5, 5)

            # Normalize candidates
            C_norm = (candidates - X_min) / X_range

            # Compute posterior mean and variance
            # K(X_new, X_train)
            diff_c = C_norm[:, np.newaxis, :] - Xn[np.newaxis, :, :]
            sq_dist_c = np.sum(diff_c ** 2, axis=2)
            K_new = sigma_f_sq * np.exp(-0.5 * sq_dist_c / (length_scale ** 2))

            # Solve for alpha = K^-1 * y_norm
            try:
                alpha = np.linalg.solve(L_chol.T, np.linalg.solve(L_chol, y_norm))
            except np.linalg.LinAlgError:
                break

            # Posterior mean
            mu_new = K_new @ alpha
            # Posterior variance
            v = np.linalg.solve(L_chol, K_new.T)
            var_new = sigma_f_sq - np.sum(v ** 2, axis=0)
            var_new = np.maximum(var_new, 1e-10)
            sigma_new = np.sqrt(var_new)

            # Expected Improvement acquisition
            f_best = y_arr.min()
            f_best_norm = (f_best - y_mean) / y_std

            Z = (mu_new - f_best_norm) / sigma_new
            EI = sigma_new * (norm.pdf(Z) + Z * norm.cdf(Z))

            # Select best candidate by EI
            best_idx = np.argmax(EI)
            x_next = candidates[best_idx].copy()

            # Evaluate
            fx = eval_func(x_next)
            if np.isfinite(fx):
                X_arr = np.vstack([X_arr, x_next])
                y_arr = np.append(y_arr, fx)

        # Phase 3: Local polish with Nelder-Mead
        remaining = budget - calls
        if remaining > 20 and not check_deadline():
            try:
                result = minimize(
                    lambda x: eval_func(x),
                    best_x.copy(),
                    method='Nelder-Mead',
                    options={'maxiter': min(remaining // 2, 400),
                             'fatol': 1e-12,
                             'xatol': 1e-10,
                             'adaptive': True}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        # Phase 4: (1+1)-ES fine-tuning
        if not check_deadline():
            sigma_fine = 0.1
            x = best_x.copy()
            fx = best_f
            no_improve = 0
            max_no_improve = 40

            while not check_deadline():
                direction = rng.standard_normal(dim)
                candidate = x + sigma_fine * direction
                candidate = np.clip(candidate, -5, 5)
                fc = eval_func(candidate)
                if np.isfinite(fc) and fc < fx:
                    x = candidate
                    fx = fc
                    no_improve = 0
                    sigma_fine = min(sigma_fine * 1.01, 0.5)
                else:
                    no_improve += 1
                    sigma_fine = max(sigma_fine * 0.95, 1e-6)
                if no_improve >= max_no_improve:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END

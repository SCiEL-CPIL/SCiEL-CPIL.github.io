"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


class Optimizer:
    """Bayesian Optimization with Gaussian Process surrogate for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        d = self.dim
        budget = self.budget

        # Initial points: include origin, random, and some structured points
        # Büche-Rastrigin has a global minimum at origin (before transformation)
        # and local minima at other locations
        initial_points = []
        # Origin
        initial_points.append(np.zeros(d))
        # Random points
        for _ in range(9):
            initial_points.append(rng.uniform(-5, 5, d))
        # Some structured points near known Rastrigin minima patterns
        # Rastrigin minima in 5D are at integer points, but global is at 0
        # Add some points near boundaries and midpoints
        for _ in range(4):
            x = np.zeros(d)
            for i in range(d):
                if rng.random() < 0.3:
                    x[i] = rng.choice([-4, -3, -2, -1, 1, 2, 3, 4])
                else:
                    x[i] = rng.uniform(-5, 5)
            initial_points.append(x)

        X = np.array(initial_points)
        X = np.clip(X, -5, 5)
        n_init = len(X)
        
        # Evaluate initial points
        Y = np.array([func(x) for x in X])

        best_x = X[np.argmin(Y)]
        best_y = Y.min()

        # GP hyperparameters
        # Length scales: start with moderate values
        lengthscales = np.ones(d) * 2.0
        signal_var = 1.0
        noise_var = 1e-4

        # Adaptive hyperparameter tuning will be done every N iterations
        n_calls = n_init
        remaining = budget - n_init

        # Acquisition function: Expected Improvement
        def compute_ei(x_new, X_train, Y_train, ls, sv, nv):
            """Compute Expected Improvement for a single point."""
            x_new = x_new.reshape(1, -1)
            # RBF kernel
            def rbf(X1, X2, ls):
                # Compute pairwise distances
                X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)
                X2_sq = np.sum(X2**2, axis=1).reshape(1, -1)
                dist_sq = X1_sq + X2_sq - 2 * X1 @ X2.T
                dist_sq = np.maximum(dist_sq, 0)
                # Divide by length scales squared
                dist_sq /= (ls**2).reshape(1, -1)
                return np.exp(-0.5 * dist_sq)

            K = rbf(X_train, X_train, ls) + nv * np.eye(len(X_train))
            Ks = rbf(X_train, x_new, ls)

            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
                v = np.linalg.solve(L, Ks)
                Kss = rbf(x_new, x_new, ls)[0, 0]
                var = sv + nv - Kss + np.sum(v**2, axis=0)
                var = max(var, 1e-10)
                std = np.sqrt(var)
                # Best value so far
                f_best = Y_train.min()
                # EI
                z = (f_best - Y_train.mean()) / std if std > 1e-10 else 0
                # Use proper GP mean prediction
                mu = Ks.T @ alpha
                ei = (f_best - mu) * norm.cdf(z) + std * norm.pdf(z)
                return ei[0] if ei.size > 0 else 0.0
            except np.linalg.LinAlgError:
                return 0.0

        # Better EI computation
        def compute_ei_batch(x_candidates, X_train, Y_train, ls, sv, nv):
            """Compute EI for multiple candidate points."""
            n_cand = len(x_candidates)
            ei_values = np.zeros(n_cand)
            for i, x_new in enumerate(x_candidates):
                ei_values[i] = compute_ei(x_new, X_train, Y_train, ls, sv, nv)
            return ei_values

        # Main optimization loop
        # Use a mix of local search around best point and global exploration
        n_evals_per_iter = 5  # Evaluate 5 candidates per iteration
        iterations = remaining // n_evals_per_iter

        for it in range(iterations):
            if n_calls >= budget:
                break

            # Generate candidate points
            candidates = []
            
            # 1. Random points for exploration
            n_random = max(1, n_evals_per_iter // 3)
            for _ in range(n_random):
                candidates.append(rng.uniform(-5, 5, d))
            
            # 2. Points near current best (exploitation)
            n_local = max(1, n_evals_per_iter // 3)
            for _ in range(n_local):
                step = 0.5 * (1 - n_calls / budget)  # Decrease step over time
                candidates.append(best_x + step * rng.standard_normal(d))
            
            # 3. Points from GP-guided acquisition
            n_gp = n_evals_per_iter - n_random - n_local
            if n_gp > 0:
                # Generate a pool of candidates and select top by EI
                pool_size = 50
                pool = []
                for _ in range(pool_size):
                    # Mix of random and perturbed best
                    if rng.random() < 0.5:
                        pool.append(rng.uniform(-5, 5, d))
                    else:
                        step = 1.0 * (1 - n_calls / budget)
                        pool.append(best_x + step * rng.standard_normal(d))
                pool = np.array(pool)
                pool = np.clip(pool, -5, 5)
                
                ei_vals = compute_ei_batch(pool, X, Y, lengthscales, signal_var, noise_var)
                top_idx = np.argsort(ei_vals)[-n_gp:]
                for idx in top_idx:
                    candidates.append(pool[idx])

            # Clip all candidates
            candidates = np.array(candidates)
            candidates = np.clip(candidates, -5, 5)

            # Evaluate candidates
            for x_new in candidates:
                if n_calls >= budget:
                    break
                y_new = func(x_new)
                n_calls += 1
                X = np.vstack([X, x_new.reshape(1, -1)])
                Y = np.append(Y, y_new)
                if y_new < best_y:
                    best_y = y_new
                    best_x = x_new.copy()

            # Occasionally re-tune hyperparameters (simple: adjust based on improvement)
            if (it + 1) % 10 == 0:
                # Simple heuristic: if we're making progress, keep going; else explore more
                # Adjust noise variance based on function values spread
                y_range = Y.max() - Y.min()
                if y_range > 0:
                    noise_var = max(1e-6, y_range * 1e-4)
                # Adjust signal variance
                signal_var = max(0.1, y_range * 0.1)

        return best_x, best_y
# EVOLVE-BLOCK-END

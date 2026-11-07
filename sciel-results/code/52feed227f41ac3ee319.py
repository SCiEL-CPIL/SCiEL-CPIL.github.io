"""Surrogate-assisted Bayesian optimization for BBOB f9."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


class Optimizer:
    """Bayesian optimization with GP surrogate and Expected Improvement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _gp_predict(self, X_train, y_train, X_test, length_scale=1.0, noise=1e-6):
        """
        Simple GP with RBF kernel. Returns mean and variance at X_test.
        Uses subsampling for large training sets to keep computation tractable.
        """
        n = X_train.shape[0]
        
        # If training set is too large, subsample for GP fit
        max_train = 150
        if n > max_train:
            # Keep best points and random sample
            y_sorted_idx = np.argsort(y_train)
            n_keep_best = min(50, n)
            best_indices = y_sorted_idx[:n_keep_best]
            remaining_indices = np.setdiff1d(np.arange(n), best_indices)
            n_random = max_train - n_keep_best
            if n_random > 0 and len(remaining_indices) > 0:
                rand_indices = remaining_indices[np.random.choice(len(remaining_indices), n_random, replace=False)]
            else:
                rand_indices = np.array([], dtype=int)
            selected = np.concatenate([best_indices, rand_indices])
            X_train_sub = X_train[selected]
            y_train_sub = y_train[selected]
        else:
            X_train_sub = X_train
            y_train_sub = y_train

        n_sub = X_train_sub.shape[0]

        # Compute kernel matrix K
        diff = X_train_sub[:, np.newaxis, :] - X_train_sub[np.newaxis, :, :]
        sq_dists = np.sum(diff ** 2, axis=-1)
        K = np.exp(-0.5 * sq_dists / (length_scale ** 2))
        K += noise * np.eye(n_sub)

        # Normalize y for numerical stability
        y_mean = np.mean(y_train_sub)
        y_std = np.std(y_train_sub) + 1e-10
        y_norm = (y_train_sub - y_mean) / y_std

        # Cholesky decomposition
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            K += 1e-4 * np.eye(n_sub)
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                return np.full(X_test.shape[0], y_mean), np.full(X_test.shape[0], y_std ** 2)

        # Solve for alpha
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))

        # Predict at X_test
        diff_test = X_test[:, np.newaxis, :] - X_train_sub[np.newaxis, :, :]
        sq_dists_test = np.sum(diff_test ** 2, axis=-1)
        K_star = np.exp(-0.5 * sq_dists_test / (length_scale ** 2))

        # Mean prediction
        mu = K_star @ alpha * y_std + y_mean

        # Variance prediction
        v = np.linalg.solve(L, K_star.T)
        var = 1.0 - np.sum(v ** 2, axis=0)
        var = np.maximum(var, 1e-10)
        var *= y_std ** 2

        return mu, var

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initial sampling: use LHS for good coverage
        n_init = min(30, budget)
        X = np.zeros((n_init, dim))
        for d in range(dim):
            perm = rng.permutation(n_init)
            X[:, d] = (perm + 0.5) / n_init * 10.0 - 5.0

        y = np.zeros(n_init)
        for i in range(n_init):
            y[i] = func(X[i])

        # Adaptive length scale estimation
        best_idx = np.argmin(y)
        length_scale = 1.0

        calls_made = n_init
        best_x = X[best_idx].copy()
        best_y = y[best_idx]

        # Track progress for adaptive behavior
        last_improvement_call = 0
        stagnation_count = 0

        # Bayesian optimization loop
        while calls_made < budget:
            # Generate candidate points for acquisition
            # More candidates as we go deeper
            n_candidates = min(500, 200 + (calls_made - n_init) // 5)
            
            # Mix of exploration and exploitation candidates
            X_cand_list = []
            
            # 1. Random exploration
            n_explore = int(n_candidates * 0.3)
            X_rand = rng.uniform(-5, 5, (n_explore, dim))
            X_cand_list.append(X_rand)
            
            # 2. Local exploitation around best
            n_local = int(n_candidates * 0.4)
            # Adaptive local search radius
            local_radius = max(0.1, length_scale * 0.3)
            X_local = best_x + rng.normal(0, local_radius, (n_local, dim))
            X_local = np.clip(X_local, -5, 5)
            X_cand_list.append(X_local)
            
            # 3. Intermediate points between best and other good points
            n_intermediate = n_candidates - n_explore - n_local
            if n_intermediate > 0 and len(y) > 5:
                # Get top 3 points
                top3_idx = np.argsort(y)[:3]
                X_top3 = X[top3_idx]
                X_intermediate = []
                for i in range(n_intermediate):
                    base = X_top3[i % 3]
                    # Perturb with decreasing radius
                    radius = local_radius * (1 + i / n_intermediate)
                    pt = base + rng.normal(0, radius * 0.5, dim)
                    pt = np.clip(pt, -5, 5)
                    X_intermediate.append(pt)
                X_intermediate = np.array(X_intermediate)
                X_cand_list.append(X_intermediate)
            
            X_cand = np.vstack(X_cand_list)
            
            # Predict with GP
            mu, var = self._gp_predict(X, y, X_cand, length_scale)
            std = np.sqrt(var)
            
            # Expected Improvement with adaptive exploration bonus
            eps = 1e-8
            y_best = np.min(y)
            improvement = y_best - mu
            z = improvement / (std + eps)
            EI = improvement * norm.cdf(z) + std * norm.pdf(z)
            EI = np.where(std > eps, EI, 0.0)
            
            # Add small exploration bonus for unexplored regions
            # Compute distance to nearest observed point
            if len(X) < 200:
                # Use a simple distance-based bonus
                X_cand_expanded = X_cand[:, np.newaxis, :]
                X_obs_expanded = X[np.newaxis, :, :]
                dists = np.sqrt(np.sum((X_cand_expanded - X_obs_expanded) ** 2, axis=2))
                min_dists = np.min(dists, axis=1)
                # Bonus for far points
                exploration_bonus = 0.05 * min_dists / (length_scale + 1e-8)
                EI += exploration_bonus
            
            # Select best candidate
            idx = np.argmax(EI)
            x_new = X_cand[idx]
            
            # Evaluate
            f_new = func(x_new)
            calls_made += 1
            
            # Update dataset
            X = np.vstack([X, x_new])
            y = np.append(y, f_new)
            
            # Update best
            if f_new < best_y:
                best_y = f_new
                best_x = x_new.copy()
                last_improvement_call = calls_made
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            # Adapt length scale based on progress
            if calls_made % 20 == 0 and len(y) > 10:
                # Re-estimate length scale from recent improvements
                sorted_idx = np.argsort(y)
                top_k = min(5, len(y))
                best_points = X[sorted_idx[:top_k]]
                if top_k > 1:
                    dists = []
                    for i in range(top_k):
                        for j in range(i + 1, top_k):
                            dists.append(np.linalg.norm(best_points[i] - best_points[j]))
                    if dists:
                        new_ls = max(0.1, np.mean(dists) * 0.4)
                        # Smooth update
                        length_scale = 0.7 * length_scale + 0.3 * new_ls
                
                # If stagnating, increase exploration
                if stagnation_count > 30:
                    length_scale = max(0.5, length_scale * 1.2)

        return best_x, best_y
# EVOLVE-BLOCK-END

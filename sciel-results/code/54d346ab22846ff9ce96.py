"""Bayesian Optimization with adaptive Gaussian Process surrogate for BBOB f14."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


class Optimizer:
    """Bayesian Optimization using an adaptive Gaussian Process surrogate."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Latin Hypercube sampling for initial points
        n_initial = min(25, budget // 4)
        if n_initial < 10:
            n_initial = 10

        # Generate LHS points
        x_init = np.zeros((n_initial, dim))
        for d in range(dim):
            idx = rng.permutation(n_initial)
            x_init[idx, d] = (idx + 0.5) / n_initial
        x_init = x_init * 10.0 - 5.0

        # Evaluate initial points
        X = x_init.copy()
        Y = np.array([func(x) for x in X])
        n_calls = n_initial

        # Adaptive GP parameters
        # Standardization parameters
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0) + 1e-8

        # Hyperparameters - will be adapted
        length_scale = 1.5
        variance = 1.0
        noise_var = 1e-6

        best_x = X[np.argmin(Y)]
        best_y = Y.min()
        best_y_history = [best_y]

        def standardize(x):
            return (x - X_mean) / X_std

        def matern_kernel(X1, X2, length_scale, variance):
            """Compute Matern 5/2 kernel between two sets of points."""
            X1_std = standardize(X1)
            X2_std = standardize(X2)
            diff = X1_std[:, np.newaxis, :] - X2_std[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=-1))
            sqrt5_d = np.sqrt(5) * dist / length_scale
            K = variance * (1 + sqrt5_d + sqrt5_d ** 2 / 3.0) * np.exp(-sqrt5_d)
            return K

        def expected_improvement(y_mean, y_std, f_best, xi=0.01):
            """Compute EI for a set of candidate points."""
            y_best = f_best - xi
            z = (y_best - y_mean) / (y_std + 1e-12)
            ei = (y_best - y_mean) * norm.cdf(z) + (y_std + 1e-12) * norm.pdf(z)
            ei = np.maximum(ei, 0.0)
            return ei

        def gp_predict(X_test, length_scale, variance, noise_var):
            """Predict mean and variance at X_test given training data X, Y."""
            n_train = len(X)
            K_train = matern_kernel(X, X, length_scale, variance) + noise_var * np.eye(n_train)
            K_test = matern_kernel(X, X_test, length_scale, variance)

            # Use Cholesky for numerical stability
            try:
                L = np.linalg.cholesky(K_train)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y))
            except np.linalg.LinAlgError:
                try:
                    alpha = np.linalg.solve(K_train, Y)
                except np.linalg.LinAlgError:
                    alpha = np.linalg.lstsq(K_train, Y, rcond=None)[0]

            mu = K_test.T @ alpha

            # Compute variance
            v = np.linalg.solve(L, K_test.T) if 'L' in dir() else K_test.T @ np.linalg.inv(K_train)
            K_test_test = matern_kernel(X_test, X_test, length_scale, variance)
            var = np.diag(K_test_test) - np.sum(v ** 2, axis=0)
            var = np.maximum(var, 1e-12)
            return mu, np.sqrt(var)

        def adapt_hyperparameters():
            """Adapt GP hyperparameters based on progress."""
            nonlocal length_scale, variance, noise_var, X_mean, X_std

            # Update standardization
            X_mean = X.mean(axis=0)
            X_std = X.std(axis=0) + 1e-8

            # Adapt length_scale based on improvement rate
            if len(best_y_history) > 5:
                recent_improvement = best_y_history[-5] - best_y
                total_range = Y.max() - Y.min() + 1e-8
                improvement_ratio = max(0, recent_improvement / total_range)

                # If making good progress, shrink length_scale (exploit more)
                # If stuck, expand length_scale (explore more)
                if improvement_ratio > 0.01:
                    length_scale = max(0.3, length_scale * 0.9)
                elif improvement_ratio < 0.001:
                    length_scale = min(3.0, length_scale * 1.2)

            # Adapt variance based on data spread
            y_range = Y.max() - Y.min()
            variance = max(0.1, y_range ** 2 / 10.0 + 0.1)

            # Adapt noise
            noise_var = max(1e-8, 1e-6 * (1 + len(X) / 100.0))

        def generate_candidates(n_random, n_local, best_x):
            """Generate a mix of random and local candidates."""
            candidates = []

            # Random candidates
            X_rand = rng.uniform(-5, 5, (n_random, dim))
            candidates.append(X_rand)

            # Local candidates around best point
            if n_local > 0:
                # Adaptive local radius based on progress
                progress = n_calls / budget
                local_radius = 2.0 * (1.0 - 0.7 * progress)  # Shrinks over time

                X_local = best_x + rng.normal(0, local_radius, (n_local, dim))
                # Clip to bounds
                X_local = np.clip(X_local, -5, 5)
                candidates.append(X_local)

            # Some candidates at the boundaries (helps for box-constrained problems)
            if n_calls < budget * 0.5:
                n_boundary = max(5, n_random // 4)
                X_boundary = rng.choice([-5, 5], size=(n_boundary, dim))
                # Mix with some random interior
                X_boundary[:n_boundary//2] = rng.uniform(-5, 5, (n_boundary//2, dim))
                candidates.append(X_boundary)

            return np.vstack(candidates)

        # Main optimization loop
        while n_calls < budget:
            # Determine candidate allocation based on progress
            progress = n_calls / budget
            if progress < 0.3:
                # Early: more exploration
                n_random = 150
                n_local = 50
            elif progress < 0.6:
                # Mid: balanced
                n_random = 100
                n_local = 100
            else:
                # Late: more exploitation
                n_random = 50
                n_local = 150

            # Generate candidates
            X_cand = generate_candidates(n_random, n_local, best_x)
            n_candidates = len(X_cand)

            # Predict mean and std at candidates
            mu, std = gp_predict(X_cand, length_scale, variance, noise_var)

            # Compute EI
            ei = expected_improvement(mu, std, best_y)

            # Select best candidate
            idx = np.argmax(ei)
            x_new = X_cand[idx]
            y_new = func(x_new)
            n_calls += 1

            # Update training set
            X = np.vstack([X, x_new])
            Y = np.append(Y, y_new)

            # Update best
            if y_new < best_y:
                best_y = y_new
                best_x = x_new

            best_y_history.append(best_y)

            # Adapt hyperparameters every 25 evaluations
            if n_calls % 25 == 0:
                adapt_hyperparameters()

        return best_x, best_y

# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm as scipy_norm


class Optimizer:
    """Bayesian Optimization with GP surrogate and hypervolume-based EI acquisition."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        ref_point = np.array([3.0, 5.0, 7.0])

        # Initialize with random samples
        n_init = 10
        X_train = np.zeros((n_init, dim))
        for i in range(n_init):
            X_train[i] = lo + (hi - lo) * np.random.random(dim)

        F_train = np.array([problem(x) for x in X_train])
        used = n_init

        # Track non-dominated solutions
        archive_X = [X_train[0]]
        archive_F = [F_train[0]]
        for i in range(1, n_init):
            if not self._is_dominated(F_train[i], archive_F):
                archive_X.append(X_train[i])
                archive_F.append(F_train[i])

        # Bayesian optimization loop
        while used < self.budget:
            if time.time() > deadline:
                break

            # Fit GP surrogate for each objective
            gps = []
            for k in range(n_obj):
                gp = self._fit_gp(X_train, F_train[:, k])
                gps.append(gp)

            # Generate candidate points
            n_candidates = 200
            candidates = np.zeros((n_candidates, dim))
            for i in range(n_candidates):
                candidates[i] = lo + (hi - lo) * np.random.random(dim)

            # Compute acquisition function (hypervolume-based EI)
            acq_values = np.zeros(n_candidates)
            for i in range(n_candidates):
                x_cand = candidates[i]
                # Predict mean and std for each objective
                means = np.array([gp.predict_mean(x_cand) for gp in gps])
                stds = np.array([gp.predict_std(x_cand) for gp in gps])

                # For minimization, EI is high when prediction is low and uncertain
                # Simple EI: for each objective, EI_k = max(0, best_k - mean_k + 0.5*std_k)
                best_k = np.array([np.min(archive_F[:, k]) for k in range(n_obj)])
                ei = np.maximum(0, best_k - means + 0.5 * stds)
                acq_values[i] = np.sum(ei)

            # Select best candidate
            best_idx = np.argmax(acq_values)
            x_new = candidates[best_idx]
            f_new = problem(x_new)
            used += 1

            # Update training data
            X_train = np.vstack([X_train, x_new])
            F_train = np.vstack([F_train, f_new])

            # Update archive
            if not self._is_dominated(f_new, archive_F):
                # Remove dominated solutions from archive
                new_archive_X = [x_new]
                new_archive_F = [f_new]
                for i in range(len(archive_X)):
                    if not self._is_dominated(archive_F[i], f_new):
                        new_archive_X.append(archive_X[i])
                        new_archive_F.append(archive_F[i])
                archive_X = new_archive_X
                archive_F = new_archive_F

        return archive_X

    def _is_dominated(self, f, archive_F):
        """Check if f is dominated by any solution in archive_F."""
        for g in archive_F:
            if np.all(g <= f) and np.any(g < f):
                return True
        return False

    def _fit_gp(self, X, y):
        """Fit a simple GP using RBF kernel with normalization."""
        X_norm = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-9)
        y_mean = np.mean(y)
        y_std = np.std(y) + 1e-9
        y_norm = (y - y_mean) / y_std

        # Simple GP with RBF kernel
        n = len(X_norm)
        if n == 1:
            return _SimpleGP(X_norm, y_norm, y_mean, y_std, np.array([1.0]), np.array([0.01]))

        # Compute kernel matrix
        K = self._rbf_kernel(X_norm, X_norm, length_scale=0.5)
        K += 1e-6 * np.eye(n)

        # Add noise
        K += 0.01 * np.eye(n)

        # Cholesky decomposition
        try:
            L = np.linalg.cholesky(K)
        except:
            L = np.linalg.cholesky(K + 1e-4 * np.eye(n))

        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))

        return _SimpleGP(X_norm, y_norm, y_mean, y_std, alpha, np.array([0.5]))

    def _rbf_kernel(self, X1, X2, length_scale):
        """Compute RBF kernel between X1 and X2."""
        sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
        sq_dists = np.maximum(sq_dists, 0)
        return np.exp(-0.5 * sq_dists / length_scale**2)


class _SimpleGP:
    """Simple GP wrapper for prediction."""

    def __init__(self, X_train, y_train, y_mean, y_std, alpha, length_scale):
        self.X_train = X_train
        self.y_train = y_train
        self.y_mean = y_mean
        self.y_std = y_std
        self.alpha = alpha
        self.length_scale = length_scale

    def predict_mean(self, x):
        x_norm = (x - np.mean(self.X_train, axis=0)) / (np.std(self.X_train, axis=0) + 1e-9)
        k = self._rbf_kernel_single(x_norm, self.X_train, self.length_scale)
        mean = np.dot(k, self.alpha)
        return mean * self.y_std + self.y_mean

    def predict_std(self, x):
        x_norm = (x - np.mean(self.X_train, axis=0)) / (np.std(self.X_train, axis=0) + 1e-9)
        k = self._rbf_kernel_single(x_norm, self.X_train, self.length_scale)
        k_xx = 1.0
        var = k_xx - np.dot(k, self.alpha)
        std = np.sqrt(max(var, 0.01))
        return std * self.y_std

    def _rbf_kernel_single(self, x1, X2, length_scale):
        sq_dists = np.sum(x1**2) + np.sum(X2**2, axis=1) - 2 * x1 @ X2.T
        sq_dists = np.maximum(sq_dists, 0)
        return np.exp(-0.5 * sq_dists / length_scale**2)
# EVOLVE-BLOCK-END

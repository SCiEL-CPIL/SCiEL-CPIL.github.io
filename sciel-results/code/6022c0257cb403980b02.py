"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """Bayesian Optimization with GP surrogate and EI acquisition,
    followed by local refinement with Nelder-Mead."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim

        # Phase 1: Latin Hypercube initialization for good coverage
        n_init = 20
        X = np.zeros((n_init, dim))
        for d in range(dim):
            perm = rng.permutation(n_init)
            X[:, d] = (perm + rng.random(n_init)) / n_init
        X = X * 10.0 - 5.0  # map to [-5, 5]

        Y = np.array([func(x) for x in X])

        # Phase 2: Bayesian Optimization loop
        n_bo = self.budget - n_init - 10  # reserve 10 for final refinement
        n_bo = max(n_bo, 50)

        best_x = X[np.argmin(Y)]
        best_f = Y.min()

        for i in range(n_bo):
            if time.time() > deadline:
                break

            # Fit GP surrogate
            X_train = X
            Y_train = Y

            # Normalize Y for better GP behavior
            y_mean = Y_train.mean()
            y_std = Y_train.std() + 1e-10
            Y_norm = (Y_train - y_mean) / y_std

            # RBF kernel with ARD
            lengthscales = np.ones(dim) * 2.0
            sigma_f = 1.0
            noise = 1e-6

            # Compute kernel matrix
            K = self._rbf_kernel(X_train, X_train, lengthscales) + noise * np.eye(len(X_train))
            K += 1e-8 * np.eye(len(X_train))  # jitter for numerical stability

            # Predict at candidate points
            n_candidates = 200
            X_cand = rng.uniform(-5, 5, (n_candidates, dim))

            K_s = self._rbf_kernel(X_cand, X_train, lengthscales)
            K_ss = self._rbf_kernel(X_cand, X_cand, lengthscales)

            # Solve for posterior
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_norm))
            except np.linalg.LinAlgError:
                # Fallback to random point if GP fails
                x_next = rng.uniform(-5, 5, dim)
                f_next = func(x_next)
                X = np.vstack([X, x_next])
                Y = np.append(Y, f_next)
                if f_next < best_f:
                    best_f = f_next
                    best_x = x_next
                continue

            mu = K_s @ alpha
            v = np.linalg.solve(L, K_s.T)
            var = sigma_f ** 2 - np.sum(v ** 2, axis=1)
            var = np.maximum(var, 1e-10)
            std = np.sqrt(var)

            # Expected Improvement acquisition
            f_best_norm = (best_f - y_mean) / y_std
            Z = (f_best_norm - mu) / std
            ei = std * (norm.pdf(Z) + Z * norm.cdf(Z))
            ei = np.where(std > 1e-8, ei, 0.0)

            # Add small exploration bonus for early iterations
            if i < 10:
                # Distance-based exploration
                for j in range(len(X_train)):
                    dist = np.sqrt(np.sum((X_cand - X_train[j]) ** 2, axis=1))
                    ei += 0.1 * np.exp(-dist / 2.0)

            idx = np.argmax(ei)
            x_next = X_cand[idx]
            f_next = func(x_next)

            X = np.vstack([X, x_next])
            Y = np.append(Y, f_next)

            if f_next < best_f:
                best_f = f_next
                best_x = x_next

        # Phase 3: Local refinement with Nelder-Mead
        if time.time() < deadline - 10:
            try:
                result = minimize(
                    func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 100,
                        'xatol': 1e-6,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x
            except Exception:
                pass

        # Phase 4: Try a few more random restarts with local optimization if time permits
        remaining = self.budget - len(Y)
        if remaining > 5 and time.time() < deadline - 20:
            for _ in range(min(remaining, 5)):
                if time.time() > deadline:
                    break
                x_try = rng.uniform(-5, 5, dim)
                try:
                    result = minimize(
                        func,
                        x_try,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 50,
                            'xatol': 1e-6,
                            'fatol': 1e-10,
                            'adaptive': True
                        }
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = result.x
                except Exception:
                    pass

        return best_x, best_f

    def _rbf_kernel(self, X1, X2, lengthscales):
        """Compute RBF kernel matrix between X1 and X2."""
        # X1: (n1, dim), X2: (n2, dim)
        # lengthscales: (dim,)
        sq_dists = (
            np.sum(X1 ** 2, axis=1)[:, None] +
            np.sum(X2 ** 2, axis=1)[None, :] -
            2 * X1 @ X2.T
        )
        sq_dists = np.maximum(sq_dists, 0)
        # Scale by lengthscales
        scaled = sq_dists / (lengthscales ** 2)
        return np.exp(-0.5 * scaled)
# EVOLVE-BLOCK-END

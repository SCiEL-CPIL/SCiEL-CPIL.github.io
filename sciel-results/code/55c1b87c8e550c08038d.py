"""Surrogate-assisted derivative-free optimizer for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    """
    Gaussian Process surrogate-assisted search.
    - Initial Latin Hypercube Sampling
    - GP with Matern 5/2 kernel
    - Randomized Expected Improvement acquisition
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0

    def _count_call(self, func, x):
        """Evaluate func(x) and track call count."""
        self.calls += 1
        return func(x)

    def _matern52(self, X1, X2, length_scale=1.0, variance=1.0):
        """Compute Matern 5/2 kernel matrix between X1 and X2."""
        # X1: (N1, D), X2: (N2, D)
        # dist^2
        X1_sq = np.sum(X1**2, axis=1, keepdims=True)
        X2_sq = np.sum(X2**2, axis=1, keepdims=True).T
        dist_sq = X1_sq + X2_sq - 2 * X1 @ X2.T
        dist_sq = np.maximum(dist_sq, 0)
        d = np.sqrt(dist_sq)
        sqrt_3_d = np.sqrt(3) * d / length_scale
        K = variance * (1 + sqrt_3_d + (sqrt_3_d**2) / 3) * np.exp(-sqrt_3_d)
        return K

    def _gp_update(self, X, Y, noise=1e-6):
        """
        Update GP internal state (K, L, alpha).
        X: (N, D), Y: (N,)
        """
        self.X = X
        self.Y = Y
        self.noise = noise
        
        # Add small noise to diagonal for numerical stability
        K = self._matern52(X, X)
        K[np.diag_indices_from(K)] += noise
        
        # Cholesky decomposition
        try:
            self.L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            # Fallback if singular
            K[np.diag_indices_from(K)] += 1e-4
            self.L = np.linalg.cholesky(K)
        
        # Solve L alpha = Y
        self.alpha = np.linalg.solve(self.L, Y)
        self.alpha = np.linalg.solve(self.L.T, self.alpha)
        
        # Normalization (optional, but helps if Y has large mean)
        # We will predict raw values, so keep Y as is.

    def _gp_predict(self, X_new):
        """
        Predict mean and variance for X_new.
        X_new: (M, D)
        Returns: mean (M,), var (M,)
        """
        K_star = self._matern52(X_new, self.X)
        K_star_star = self._matern52(X_new, X_new) # Not needed for variance if we use K_s - K_sx K_x^-1 K_xs^T
        
        # Mean: K_star @ alpha
        mean = K_star @ self.alpha
        
        # Variance: K(x, x) - K_star @ L^-1 L^-T K_star^T
        # Let V = solve(L, K_star.T)
        V = np.linalg.solve(self.L, K_star.T)
        var = 1.0 - np.sum(V**2, axis=1) # Assuming kernel variance is 1.0
        
        # Clip negative variance
        var = np.maximum(var, 1e-10)
        
        return mean, var

    def _latin_hypercube(self, n, dim, bounds=(-5, 5), rng=None):
        """Generate Latin Hypercube samples."""
        if rng is None:
            rng = np.random.default_rng(42)
        
        X = np.zeros((n, dim))
        for i in range(dim):
            perm = rng.permutation(n)
            # Uniform in [0, 1)
            u = (perm + rng.random(n)) / n
            # Scale to bounds
            X[:, i] = bounds[0] + u * (bounds[1] - bounds[0])
        return X

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        self.calls = 0

        # Phase 1: Initial Latin Hypercube Sampling
        n_init = min(30, budget)
        rng_init = np.random.default_rng(42)
        X = self._latin_hypercube(n_init, dim, rng=rng_init)
        F = np.array([self._count_call(func, x) for x in X])

        best_idx = np.argmin(F)
        x_best = X[best_idx].copy()
        f_best = F[best_idx]

        # Initialize GP
        self._gp_update(X, F)

        # Phase 2: Iterative GP-guided search
        remaining = budget - self.calls
        rng = np.random.default_rng(123)
        
        # Acquisition parameters
        kappa = 1.0 # Exploration-exploitation balance
        
        while remaining > 0:
            # Generate random candidates for acquisition
            n_cand = 200
            X_cand = rng.uniform(-5, 5, size=(n_cand, dim))
            
            # Predict
            mean, var = self._gp_predict(X_cand)
            std = np.sqrt(var)
            
            # Expected Improvement
            # EI = (f_best - mean - kappa * std) * Phi((f_best - mean - kappa * std) / std) + std * phi(...)
            # Simplified: Just use mean - kappa * std (Lower Confidence Bound)
            # This is faster and often sufficient.
            scores = mean - kappa * std
            
            # Pick best candidate (lowest score)
            best_cand_idx = np.argmin(scores)
            x_next = X_cand[best_cand_idx]
            
            # Evaluate
            f_next = self._count_call(func, x_next)
            
            # Update
            X = np.vstack([X, x_next.reshape(1, -1)])
            F = np.append(F, f_next)
            
            if f_next < f_best:
                f_best = f_next
                x_best = x_next.copy()
            
            # Update GP
            self._gp_update(X, F)
            
            remaining = budget - self.calls
            
            # Adaptive kappa? Start high, decrease?
            # Or keep constant. Constant is safer.
            
        return x_best, f_best


# EVOLVE-BLOCK-END

"""Sparse GP contextual bandit for non-stationary linear rewards.

Models the global reward surface over the augmented feature vector
(context concatenated with one-hot arm encoding) using a sparse
inducing-point variational GP with proper FITC approximation.
"""
# EVOLVE-BLOCK-START
import numpy as np


class Policy:
    """Sparse (variational) GP bandit over augmented context+arm features."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.d_aug = dim + n_arms

        # Inducing-point count
        self.M = int(min(64, max(16, 2 * self.d_aug)))

        # Kernel hyperparameters
        self.length_scale = 1.0
        self.sigma_f = 1.0
        self.noise = 0.1

        # Random inducing points
        rng = np.random.default_rng(0)
        self.Z = rng.normal(0.0, self.length_scale, size=(self.M, self.d_aug))

        # Precompute kernel matrices
        self.Kuu = self._compute_kernel(self.Z, self.Z)
        self.Kuu += self.noise * np.eye(self.M)

        # Variational posterior q(u) = N(m, S)
        self.m = np.zeros(self.M)
        self.S = self.sigma_f ** 2 * np.eye(self.M)

        # Cholesky factor of S
        self.S_L = np.linalg.cholesky(self.S + 1e-8 * np.eye(self.M))

        # Exploration coefficient
        self.beta = 1.5
        self.beta_decay = 0.999
        self.beta_min = 0.3
        self.t = 0

        # Forgetting for non-stationarity
        self.forget = 0.99
        self.forget_min = 0.95

        # Reward statistics for adaptive forgetting
        self.reward_mean = 0.0
        self.reward_var = 1.0

    def _compute_kernel(self, X, Y):
        """Compute RBF kernel between X and Y."""
        X_sq = np.sum(X ** 2, axis=1)[:, None]
        Y_sq = np.sum(Y ** 2, axis=1)[None, :]
        D2 = X_sq + Y_sq - 2.0 * (X @ Y.T)
        np.maximum(D2, 0.0, out=D2)
        return self.sigma_f ** 2 * np.exp(-D2 / (2.0 * self.length_scale ** 2))

    def _kz(self, x):
        """Kernel between single point x and all inducing points."""
        x = x.reshape(1, -1)
        return self._compute_kernel(x, self.Z).reshape(-1)

    def _predict(self, x):
        """Predictive mean and variance under FITC approximation."""
        kz = self._kz(x)
        kxx = self.sigma_f ** 2

        # Solve S^{-1} kz using Cholesky factor
        S_inv_kz = np.linalg.solve(self.S_L.T, np.linalg.solve(self.S_L, kz))

        # Predictive mean
        mean = kz @ self.m + kz @ S_inv_kz @ self.m

        # Predictive variance (proper FITC formula)
        # var = kxx - kz S^{-1} kz + kz S^{-1} (Kuu - S) S^{-1} kz
        S_inv_kz_kz = kz @ S_inv_kz
        Kuu_minus_S = self.Kuu - self.S
        correction = S_inv_kz @ Kuu_minus_S @ S_inv_kz

        var = kxx - S_inv_kz_kz + correction
        var = max(var, 1e-6)

        return mean, var

    def _recompute_S_factor(self):
        """Recompute Cholesky factor of S, ensuring positive-definiteness."""
        # Symmetrize and add small ridge for numerical stability
        self.S = 0.5 * (self.S + self.S.T)
        # Check eigenvalues and add ridge if needed
        eigvals = np.linalg.eigvalsh(self.S)
        if eigvals.min() < 1e-8:
            self.S += (1e-8 - eigvals.min()) * np.eye(self.M)
        self.S_L = np.linalg.cholesky(self.S)

    def select(self, context):
        """Select arm using GP mean + exploration bonus."""
        scores = np.zeros(self.n)
        ctx = np.asarray(context, dtype=float).reshape(-1)
        for a in range(self.n):
            x = np.zeros(self.d_aug)
            x[:self.d] = ctx
            x[self.d + a] = 1.0
            mean, var = self._predict(x)
            scores[a] = mean + self.beta * np.sqrt(var)
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        """Update variational posterior after observing reward."""
        ctx = np.asarray(context, dtype=float).reshape(-1)
        x = np.zeros(self.d_aug)
        x[:self.d] = ctx
        x[self.d + arm] = 1.0

        kz = self._kz(x)
        kxx = self.sigma_f ** 2

        # Compute S^{-1} kz
        S_inv_kz = np.linalg.solve(self.S_L.T, np.linalg.solve(self.S_L, kz))

        # Predictive mean and variance
        pred_mean = kz @ self.m + kz @ S_inv_kz @ self.m
        Kuu_minus_S = self.Kuu - self.S
        correction = S_inv_kz @ Kuu_minus_S @ S_inv_kz
        pred_var = max(kxx - kz @ S_inv_kz + correction, 1e-6)

        # Update precision
        lam = 1.0 / (pred_var + self.noise)

        # Update mean: m <- m + S S^{-1} kz * (reward - pred_mean) * lam
        # S S^{-1} kz = kz
        delta_m = kz * (reward - pred_mean) * lam
        self.m = self.forget * self.m + delta_m

        # Update covariance using Sherman-Morrison formula
        # S_new = S - lam * (S @ kz) @ (S @ kz).T
        S_kz = self.S @ kz
        self.S = self.S - lam * np.outer(S_kz, S_kz)

        # Forgetting: shrink S toward prior covariance
        prior_S = self.sigma_f ** 2 * np.eye(self.M)
        self.S = self.forget * self.S + (1.0 - self.forget) * prior_S

        # Ensure S is PD
        self._recompute_S_factor()

        # Adaptive forgetting: increase forgetting rate if reward variance is high
        if self.t > 10:
            self.forget = max(self.forget_min, 0.99 - 0.03 * min(1.0, self.reward_var / 5.0))

        # Adaptive exploration: decay beta toward beta_min
        self.beta = max(self.beta_min, self.beta * self.beta_decay)

        # Track reward statistics
        alpha = 0.02
        self.reward_mean = (1 - alpha) * self.reward_mean + alpha * reward
        self.reward_var = (1 - alpha) * self.reward_var + alpha * (reward - self.reward_mean) ** 2

        self.t += 1
# EVOLVE-BLOCK-END

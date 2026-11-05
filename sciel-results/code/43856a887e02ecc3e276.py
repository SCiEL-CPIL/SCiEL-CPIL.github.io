"""Sparse GP contextual bandit for non-stationary linear rewards.

Models the global reward surface over the augmented feature vector
(context concatenated with one-hot arm encoding) using a sparse
inducing-point variational GP with a sliding-window recency bias.
"""
# EVOLVE-BLOCK-START
import numpy as np


class Policy:
    """Sparse (variational) GP bandit over augmented context+arm features."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.d_aug = dim + n_arms

        # Inducing-point count: keep small for speed and stability.
        self.M = int(min(30, max(12, self.d_aug)))

        # Kernel hyperparameters (RBF + linear mixture).
        self.length_scale = 1.0
        self.sigma_rbf = 1.0
        self.sigma_lin = 0.3
        self.noise = 0.1

        # Random inducing points, centered near origin.
        rng = np.random.default_rng(42)
        self.Z = rng.normal(0.0, self.length_scale, size=(self.M, self.d_aug))

        # Precompute Kuu and factorize once; we will maintain a running
        # variational mean and a fixed covariance structure for speed.
        self._recompute_Kuu()

        # Cholesky of Kuu for stable solves
        self.Kuu_chol = np.linalg.cholesky(self.Kuu + 1e-6 * np.eye(self.M))

        # Variational mean over inducing values; covariance held at prior scale.
        self.m = np.zeros(self.M)

        # Exploration coefficient with slower decay.
        self.beta = 1.5
        self.beta_decay = 0.9995
        self.t = 0

        # Forgetting factor for non-stationarity (exponential decay of mean).
        self.forget = 0.98

        # Recency weighting: store recent (x, reward) pairs for a fast
        # local correction on top of the GP.
        self.recent_window = 30
        self.recent_X = []
        self.recent_Y = []

        # Precompute Kuu inverse for faster solves
        self.Kuu_inv = np.linalg.inv(self.Kuu)

    def _recompute_Kuu(self):
        Z = self.Z
        sq = (Z ** 2).sum(1)
        D2 = sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T)
        np.maximum(D2, 0.0, out=D2)
        K_rbf = self.sigma_rbf ** 2 * np.exp(-D2 / (2.0 * self.length_scale ** 2))
        K_lin = self.sigma_lin ** 2 * (Z @ Z.T)
        self.Kuu = K_rbf + K_lin + self.noise * np.eye(self.M)

    def _kz(self, x):
        x = x.reshape(1, -1)
        diff = x - self.Z
        D2 = (diff ** 2).sum(1)
        k_rbf = self.sigma_rbf ** 2 * np.exp(-D2 / (2.0 * self.length_scale ** 2))
        k_lin = self.sigma_lin ** 2 * (x @ self.Z.T).reshape(-1)
        return k_rbf + k_lin

    def _kxx(self):
        return self.sigma_rbf ** 2 + self.sigma_lin ** 2

    def _predict(self, x):
        """Returns (mean, variance) using variational GP with fixed covariance."""
        kz = self._kz(x)
        # Use precomputed inverse for speed
        Kuu_inv_m = self.Kuu_inv @ self.m
        mean = kz @ Kuu_inv_m

        # Predictive variance: kxx - kz^T Kuu^{-1} kz
        Kuu_inv_kz = self.Kuu_inv @ kz
        var = self._kxx() - kz @ Kuu_inv_kz
        var = max(var, 1e-6)
        return mean, var

    def select(self, context):
        ctx = np.asarray(context, dtype=float).reshape(-1)
        scores = np.zeros(self.n)
        
        # Precompute recent window arrays once if available
        if self.recent_X:
            Xr = np.array(self.recent_X)
            Yr = np.array(self.recent_Y)
            weights = np.exp(-0.15 * np.arange(len(Xr) - 1, -1, -1))
            weights /= weights.sum()
        else:
            Xr = None
            Yr = None
            weights = None

        for a in range(self.n):
            x = np.zeros(self.d_aug)
            x[:self.d] = ctx
            x[self.d + a] = 1.0
            gp_mean, gp_var = self._predict(x)
            
            # Add recency correction
            recency_bonus = 0.0
            if Xr is not None:
                diffs = Xr - x
                D2 = (diffs ** 2).sum(1)
                k_vals = self.sigma_rbf ** 2 * np.exp(-D2 / (2.0 * self.length_scale ** 2)) + \
                         self.sigma_lin ** 2 * (Xr @ x)
                recency_bonus = (weights * k_vals * Yr).sum()
            
            scores[a] = gp_mean + recency_bonus + self.beta * np.sqrt(gp_var)
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        ctx = np.asarray(context, dtype=float).reshape(-1)
        x = np.zeros(self.d_aug)
        x[:self.d] = ctx
        x[self.d + arm] = 1.0

        # Update variational mean with forgetting
        kz = self._kz(x)
        pred_mean, _ = self._predict(x)
        residual = reward - pred_mean
        
        # Update m in direction that increases predictive mean at x
        Kuu_inv_kz = self.Kuu_inv @ kz
        self.m = self.forget * self.m + Kuu_inv_kz * residual * 0.7

        # Store in recent window
        self.recent_X.append(x)
        self.recent_Y.append(reward)
        if len(self.recent_X) > self.recent_window:
            self.recent_X.pop(0)
            self.recent_Y.pop(0)

        # Decay exploration
        self.beta *= self.beta_decay
        self.t += 1
# EVOLVE-BLOCK-END

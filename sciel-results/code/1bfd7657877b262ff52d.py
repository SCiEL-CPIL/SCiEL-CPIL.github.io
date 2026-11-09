"""Adaptive Exponential Forgetting Thompson Sampling with Sherman-Morrison-Woodbury updates."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Non-stationary linear contextual bandit policy using:
    - Discounted Least Squares with adaptive forgetting factor (gamma)
    - Thompson Sampling for exploration
    - Sherman-Morrison-Woodbury rank-1 updates for O(d^2) per step
    - Adaptive gamma based on residual magnitude
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0

        # Regularization
        self.reg = 1.0

        # Thompson Sampling noise variance
        self.sigma_sq = 1.0

        # Per-arm statistics: A (covariance inverse), b (moment)
        self.A = [self.reg * np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.A_inv = [np.eye(dim) / self.reg for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]

        # Adaptive forgetting parameters
        self.base_gamma = 0.97
        self.min_gamma = 0.80
        self.max_gamma = 0.995

        # EWMA of squared residuals to drive adaptation
        self.residual_sq_ewma = [0.0 for _ in range(n_arms)]
        self.ewma_alpha = 0.25

        # Current gamma for each arm
        self.gamma = [self.base_gamma for _ in range(n_arms)]

        # Deadline guard
        self.start_time = time.time()
        self.deadline = 115.0

    def _update_gamma(self, arm, residual_sq):
        """
        Update the forgetting factor based on the magnitude of the residual.
        Large residual -> model is wrong -> forget old data faster (lower gamma).
        Small residual -> model is good -> keep memory (higher gamma).
        """
        scale = 1.0
        factor = 1.0 / (1.0 + self.residual_sq_ewma[arm] * scale)

        new_gamma = self.min_gamma + (self.max_gamma - self.min_gamma) * factor

        # Smooth the gamma update to avoid oscillation
        self.gamma[arm] = 0.85 * self.gamma[arm] + 0.15 * new_gamma

    def _recompute_arm_stats(self, arm):
        """Recompute inverse and theta from current A and b using SMMW."""
        # Use Sherman-Morrison-Woodbury for rank-1 update:
        # A_new = gamma * A_old + x x^T
        # A_new^{-1} = (gamma * A_old + x x^T)^{-1}
        # Using SMMW: (A + u v^T)^{-1} = A^{-1} - A^{-1} u v^T A^{-1} / (1 + v^T A^{-1} u)
        # Here A = gamma * A_old, u = x, v = x
        # A_new^{-1} = (gamma * A_old)^{-1} - (gamma * A_old)^{-1} x x^T (gamma * A_old)^{-1} / (1 + x^T (gamma * A_old)^{-1} x)
        #             = (1/gamma) * A_old^{-1} - (1/gamma)^2 * A_old^{-1} x x^T A_old^{-1} / (1 + (1/gamma) * x^T A_old^{-1} x)
        
        gamma = self.gamma[arm]
        A_inv_old = self.A_inv[arm]
        x = None  # We'll handle this in update instead
        
        # Actually, let's just do the full update in the update method
        # and recompute here only if needed
        try:
            self.A_inv[arm] = np.linalg.inv(self.A[arm])
            self.theta[arm] = self.A_inv[arm] @ self.b[arm]
        except np.linalg.LinAlgError:
            self.A_inv[arm] = np.eye(self.d) / self.reg
            self.theta[arm] = np.zeros(self.d)

    def select(self, context):
        # Check deadline
        if time.time() - self.start_time > self.deadline:
            context = np.asarray(context, dtype=float)
            means = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(means))
        
        context = np.asarray(context, dtype=float)
        scores = []
        
        for a in range(self.n):
            theta_mean = self.theta[a]
            
            # Sample theta from posterior N(theta_mean, sigma^2 * A_inv[a])
            try:
                L = np.linalg.cholesky(self.A[a])
                z = np.random.randn(self.d)
                w = np.linalg.solve(L.T, z)
                theta_sample = theta_mean + self.sigma_sq * w
            except np.linalg.LinAlgError:
                theta_sample = theta_mean
            
            score = theta_sample @ context
            scores.append(score)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=float)
        reward = float(reward)
        
        # 1. Compute residual
        pred = self.theta[arm] @ context
        residual = reward - pred
        residual_sq = residual ** 2
        
        # 2. Update EWMA of squared residuals
        self.residual_sq_ewma[arm] = (1 - self.ewma_alpha) * self.residual_sq_ewma[arm] + self.ewma_alpha * residual_sq
        
        # 3. Update adaptive gamma
        self._update_gamma(arm, residual_sq)
        
        # 4. Update statistics with forgetting using SMMW
        gamma = self.gamma[arm]
        
        # A_new = gamma * A_old + x x^T
        # b_new = gamma * b_old + reward * x
        
        # For A_inv, use Sherman-Morrison-Woodbury:
        # Let M = gamma * A_old, so M^{-1} = (1/gamma) * A_old^{-1}
        # A_new = M + x x^T
        # A_new^{-1} = M^{-1} - M^{-1} x x^T M^{-1} / (1 + x^T M^{-1} x)
        
        M_inv = (1.0 / gamma) * self.A_inv[arm]
        M_inv_x = M_inv @ context
        denom = 1.0 + context @ M_inv_x
        
        if abs(denom) > 1e-10:
            # A_new^{-1} = M_inv - M_inv @ x @ x^T @ M_inv / denom
            # = M_inv - (M_inv @ x) @ (x^T @ M_inv) / denom
            # = M_inv - np.outer(M_inv_x, M_inv_x) / denom
            self.A_inv[arm] = M_inv - np.outer(M_inv_x, M_inv_x) / denom
        else:
            # Fallback to full inversion
            self.A[arm] = gamma * self.A[arm] + np.outer(context, context)
            try:
                self.A_inv[arm] = np.linalg.inv(self.A[arm])
            except np.linalg.LinAlgError:
                self.A_inv[arm] = np.eye(self.d) / self.reg
        
        # Update A and b for future reference (but we primarily use A_inv)
        self.A[arm] = gamma * self.A[arm] + np.outer(context, context)
        self.b[arm] = gamma * self.b[arm] + reward * context
        
        # 5. Recompute theta
        self.theta[arm] = self.A_inv[arm] @ self.b[arm]
        
        self.t += 1
# EVOLVE-BLOCK-END

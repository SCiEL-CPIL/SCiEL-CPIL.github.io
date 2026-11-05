"""Non-stationary linear contextual bandit with model-and-optimize approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit using:
    - Per-arm discounted least squares with adaptive forgetting
    - Thompson Sampling for exploration (sampling from posterior)
    - Prediction-error-driven adaptation of forgetting rate
    - Augmented context with intercept term
    - Deadline guard for safety
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim  # original dimension
        self.d_aug = dim + 1  # augmented dimension (with intercept)
        self.start_time = time.time()
        self.deadline = 140.0  # safety margin before 150s kill

        # Inverse covariance matrices V = A^{-1} for each arm (augmented)
        self.V = [np.eye(self.d_aug) for _ in range(n_arms)]
        self.b = [np.zeros(self.d_aug) for _ in range(n_arms)]
        self.counts = [0] * n_arms

        # Adaptive forgetting parameters
        self.base_gamma = 0.93
        self.gamma_min = 0.80
        self.gamma_max = 0.99
        self.prediction_errors = [0.0] * n_arms

        # Thompson Sampling noise
        self.noise_std = 1.0

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _augment(self, context):
        """Augment context with intercept term."""
        ctx = np.asarray(context, dtype=np.float64)
        return np.concatenate([ctx, [1.0]])

    def _adaptive_gamma(self, arm, prediction_error):
        """Adapt forgetting factor based on prediction error."""
        c = self.counts[arm]
        if c < 5:
            return self.base_gamma

        # Track running average of prediction errors
        self.prediction_errors[arm] = 0.9 * self.prediction_errors[arm] + 0.1 * prediction_error

        pe = self.prediction_errors[arm]
        # Higher prediction error -> faster forgetting (lower gamma)
        # Scale prediction error to adjust gamma
        pe_scale = min(pe, 5.0)
        gamma = self.base_gamma - 0.08 * pe_scale
        gamma = max(self.gamma_min, min(self.gamma_max, gamma))
        return gamma

    def select(self, context):
        ctx_aug = self._augment(context)

        # Deadline guard: if close to deadline, use greedy
        if not self._check_deadline():
            # Greedy fallback: pick arm with highest mean estimate
            scores = []
            for a in range(self.n):
                theta = self.V[a] @ self.b[a]
                scores.append(theta @ ctx_aug)
            return int(np.argmax(scores))

        # Thompson Sampling: sample from posterior for each arm
        scores = []
        for a in range(self.n):
            # Posterior: theta | D ~ N(theta_hat, V)
            theta_hat = self.V[a] @ self.b[a]
            # Sample from N(theta_hat, V)
            try:
                # Use Cholesky decomposition for sampling
                L = np.linalg.cholesky(self.V[a])
                z = np.random.randn(self.d_aug)
                theta_sample = theta_hat + L @ z
            except np.linalg.LinAlgError:
                # Fallback if covariance is not positive definite
                theta_sample = theta_hat

            scores.append(theta_sample @ ctx_aug)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        ctx_aug = self._augment(context)
        reward = float(reward)

        # Compute prediction error before update
        theta_hat = self.V[arm] @ self.b[arm]
        predicted = theta_hat @ ctx_aug
        prediction_error = abs(reward - predicted)

        # Track statistics
        self.counts[arm] += 1

        # Adaptive forgetting
        gamma = self._adaptive_gamma(arm, prediction_error)

        # Update inverse covariance using Sherman-Morrison with forgetting
        # V_new = (gamma * V_old^{-1} + x x^T)^{-1}
        # By Sherman-Morrison: (A + u v^T)^{-1} = A^{-1} - A^{-1} u v^T A^{-1} / (1 + v^T A^{-1} u)
        # Here A = gamma * V_old^{-1}, so A^{-1} = (1/gamma) * V_old
        # u = v = x
        # V_new = (1/gamma) * V_old - (1/gamma)^2 * V_old x x^T V_old / (1 + (1/gamma) * x^T V_old x)
        #       = (1/gamma) * V_old - (1/gamma) * V_old x x^T V_old / (gamma + x^T V_old x)

        V_old = self.V[arm]
        Vx = V_old @ ctx_aug
        xVx = ctx_aug @ Vx  # scalar
        denom = gamma + xVx

        if denom > 1e-12:
            self.V[arm] = (1.0 / gamma) * V_old - (1.0 / gamma) * np.outer(Vx, Vx) / denom
        else:
            # Fallback: reset to identity scaled
            self.V[arm] = np.eye(self.d_aug)

        # Update b with forgetting
        self.b[arm] = gamma * self.b[arm] + reward * ctx_aug
# EVOLVE-BLOCK-END

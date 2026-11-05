"""
Non-stationary linear contextual bandit with Bayesian Online Changepoint Detection (BOCPD).
Per-arm linear regression with adaptive forgetting and changepoint-triggered resets.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.dim = dim

        # Deadline guard
        self.start_time = time.time()
        self.deadline = 145.0
        self.round = 0

        # Per-arm linear regression state
        # V = (lambda*I + sum x x^T)^-1, b = sum r*x, theta = V @ b
        self.lam = 2.0
        self.V = [np.eye(dim) / self.lam for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]

        # Per-arm observation count since last reset
        self.counts = [0] * n_arms

        # BOCPD state per arm: posterior over age (0 = just reset)
        # We track a small set of possible ages and their probabilities
        self.max_age = 200
        self.age_posteriors = [np.zeros(self.max_age + 1) for _ in range(n_arms)]
        for a in range(n_arms):
            self.age_posteriors[a][0] = 1.0  # Start with age 0

        # Residual tracking for changepoint detection
        self.residual_history = [[] for _ in range(n_arms)]
        self.residual_ema = np.zeros(n_arms)
        self.residual_var_ema = np.zeros(n_arms)
        self.residual_alpha = 0.15

        # Adaptive forgetting base rate
        self.base_gamma = 0.95
        self.min_gamma = 0.80
        self.max_gamma = 0.995

        # Exploration
        self.alpha = 1.5
        self.alpha_min = 0.3
        self.alpha_max = 3.0

        # Fallback tracking
        self.best_arm = 0
        self.best_reward = -np.inf
        self.arm_mean_reward = [0.0] * n_arms
        self.arm_count = [0] * n_arms

        # For applying gamma decay to V and b
        self.gamma = [self.base_gamma] * n_arms

    def _check_deadline(self):
        if time.time() - self.start_time >= self.deadline:
            return True
        return False

    def _reset_arm(self, arm):
        """Reset the linear model for an arm (changepoint detected)."""
        self.V[arm] = np.eye(self.dim) / self.lam
        self.b[arm] = np.zeros(self.dim)
        self.theta[arm] = np.zeros(self.dim)
        self.counts[arm] = 0
        self.age_posteriors[arm] = np.zeros(self.max_age + 1)
        self.age_posteriors[arm][0] = 1.0
        self.residual_history[arm] = []
        self.residual_ema[arm] = 0.0
        self.residual_var_ema[arm] = 0.0

    def _apply_forgetting(self, arm, gamma):
        """Apply exponential forgetting to V and b."""
        self.V[arm] = self.V[arm] / gamma
        self.b[arm] = self.b[arm] * gamma
        self.theta[arm] = self.V[arm] @ self.b[arm]

    def _update_bochpd(self, arm, residual):
        """
        Bayesian Online Changepoint Detection update.
        Returns True if a changepoint is detected (posterior mass on age 0 is high).
        """
        # Update residual statistics
        r = residual
        self.residual_ema[arm] = (1 - self.residual_alpha) * self.residual_ema[arm] + self.residual_alpha * r
        r2 = r * r
        self.residual_var_ema[arm] = (1 - self.residual_alpha) * self.residual_var_ema[arm] + self.residual_alpha * r2

        # Store residual for prediction
        self.residual_history[arm].append(r)
        if len(self.residual_history[arm]) > 50:
            self.residual_history[arm].pop(0)

        # Compute prediction error under each age hypothesis
        # Under age t, we expect residuals to have mean ~0 and variance based on recent history
        # Use a simplified predictive distribution

        # Growth vector: P(no change) for each age
        no_change_prob = 0.995  # Base probability of no change
        growth = np.full(self.max_age + 1, no_change_prob)
        growth[0] = 0.0  # Age 0 cannot persist (it means just changed)

        # Predictive probability of residual under each age
        # Under age t, the model has been stable for t steps
        # Use the residual EMA and variance as a proxy
        pred_mean = self.residual_ema[arm]
        pred_var = max(self.residual_var_ema[arm], 0.1)

        # For young ages (recent reset), we have less confidence
        # Scale the predictive likelihood
        predictive = np.zeros(self.max_age + 1)
        for t in range(self.max_age + 1):
            # Effective variance increases with age (more uncertainty about true mean)
            eff_var = pred_var * (1 + t * 0.01)
            # Gaussian likelihood
            diff = r - pred_mean
            predictive[t] = np.exp(-0.5 * diff * diff / max(eff_var, 1e-6))
            predictive[t] /= np.sqrt(2 * np.pi * max(eff_var, 1e-6))

        # Normalize predictive
        pred_sum = predictive.sum()
        if pred_sum > 0:
            predictive /= pred_sum
        else:
            predictive = np.ones(self.max_age + 1) / (self.max_age + 1)

        # Update posterior
        # P(age t+1 | r) ∝ P(r | age t) * P(no change | age t) * P(age t)
        # P(age 0 | r) ∝ P(r | just changed) * (1 - sum P(no change | age t) * P(age t))

        posterior = self.age_posteriors[arm]
        new_posterior = np.zeros(self.max_age + 1)

        # Predictive probability under "just changed" (age 0 after update)
        # Use a broad prior
        just_changed_likelihood = 1.0 / np.sqrt(2 * np.pi * max(pred_var * 2, 0.5))
        diff = r - pred_mean
        just_changed_likelihood = np.exp(-0.5 * diff * diff / max(pred_var * 2, 0.5)) / np.sqrt(2 * np.pi * max(pred_var * 2, 0.5))

        # Probability of no changepoint
        p_no_change = np.sum(posterior * growth)
        p_change = 1.0 - p_no_change

        # New posterior
        # Age 0: just had a changepoint
        new_posterior[0] = just_changed_likelihood * p_change
        # Ages 1..max_age: no changepoint, age incremented
        for t in range(1, self.max_age + 1):
            if t - 1 <= self.max_age:
                new_posterior[t] = predictive[t - 1] * growth[t - 1] * posterior[t - 1]

        # Normalize
        total = new_posterior.sum()
        if total > 1e-10:
            new_posterior /= total
        else:
            new_posterior = np.zeros(self.max_age + 1)
            new_posterior[0] = 1.0

        self.age_posteriors[arm] = new_posterior

        # Check if changepoint detected: high mass on young ages
        p_young = new_posterior[:5].sum()
        if p_young > 0.6:
            return True
        return False

    def _update_adaptive_gamma(self, arm, residual):
        """Adapt the forgetting rate based on residual magnitude."""
        r = abs(residual)
        # Higher residual -> faster forgetting (lower gamma)
        if r > 2.0:
            self.gamma[arm] = max(self.min_gamma, self.gamma[arm] * 0.95)
        elif r < 0.3:
            self.gamma[arm] = min(self.max_gamma, self.gamma[arm] * 1.02)
        else:
            target = 0.95 - 0.1 * min(r, 3.0) / 3.0
            self.gamma[arm] = self.gamma[arm] * 0.9 + target * 0.1

    def _update_adaptive_alpha(self):
        """Adapt exploration coefficient."""
        if self.round < 30:
            self.alpha = self.alpha_max
        elif self.round < 150:
            self.alpha = self.alpha_max - (self.alpha_max - 1.0) * (self.round - 30) / 120.0
        else:
            self.alpha = max(self.alpha_min, 1.0 - 0.3 * (self.round - 150) / 2000.0)

    def select(self, context):
        self.round += 1

        if self._check_deadline():
            return self.best_arm

        context = np.asarray(context, dtype=np.float64).flatten()

        self._update_adaptive_alpha()

        scores = np.zeros(self.n)
        for a in range(self.n):
            mean = self.theta[a] @ context
            # Uncertainty: context^T V_a context
            var = context @ self.V[a] @ context
            if var < 0:
                var = 0.0
            # Scale exploration by how uncertain we are about the regime
            # If BOCPD says we're in a new regime, explore more
            regime_uncertainty = self.age_posteriors[a][:10].sum()
            alpha_eff = self.alpha * (1.0 + regime_uncertainty)
            scores[a] = mean + alpha_eff * np.sqrt(var)

        # Small random tiebreaker
        scores += np.random.normal(0, 0.01, self.n)

        arm = int(np.argmax(scores))
        return arm

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64).flatten()
        reward = float(reward)

        # Track for fallback
        self.arm_count[arm] += 1
        if self.arm_count[arm] == 1:
            self.arm_mean_reward[arm] = reward
        else:
            self.arm_mean_reward[arm] = (self.arm_mean_reward[arm] * (self.arm_count[arm] - 1) + reward) / self.arm_count[arm]

        if self.arm_count[arm] >= 5:
            if self.arm_mean_reward[arm] > self.best_reward:
                self.best_reward = self.arm_mean_reward[arm]
                self.best_arm = arm

        # Prediction before update
        pred = self.theta[arm] @ context
        residual = reward - pred

        # Adapt gamma
        self._update_adaptive_gamma(arm, residual)

        # Apply forgetting
        self._apply_forgetting(arm, self.gamma[arm])

        # BOCPD update - check for changepoint
        changepoint = self._update_bochpd(arm, residual)

        if changepoint:
            self._reset_arm(arm)
            # After reset, add the current observation to the fresh model
            V = self.V[arm]
            Vx = V @ context
            denom = 1.0 + context @ Vx
            if denom < 1e-10:
                denom = 1e-10
            self.V[arm] = V - np.outer(Vx, Vx) / denom
            self.b[arm] = reward * context
            self.theta[arm] = self.V[arm] @ self.b[arm]
            self.counts[arm] = 1
            self.age_posteriors[arm] = np.zeros(self.max_age + 1)
            self.age_posteriors[arm][1] = 1.0  # Age 1 after first observation
        else:
            # Add new observation using Sherman-Morrison
            V = self.V[arm]
            Vx = V @ context
            denom = 1.0 + context @ Vx
            if denom < 1e-10:
                denom = 1e-10
            self.V[arm] = V - np.outer(Vx, Vx) / denom
            self.b[arm] = self.b[arm] + reward * context
            self.theta[arm] = self.V[arm] @ self.b[arm]
            self.counts[arm] += 1
# EVOLVE-BLOCK-END

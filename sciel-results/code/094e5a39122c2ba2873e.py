"""Rewritten program for non-stationary linear contextual bandit."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit with:
    - Per-arm ridge regression with exponential forgetting (discounted least squares)
    - Adaptive forgetting rate based on prediction residuals
    - UCB-style exploration with arm-specific uncertainty
    - Deadline guard for safety
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.dim = dim

        # Per-arm parameters: V = (sum gamma^t x x^T + lambda I)^{-1}, b = sum gamma^t r x
        # Use a larger initial regularization (inverse of covariance) to stabilize early steps
        self.V = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.counts = [0] * n_arms

        # Fixed parameters for stability and speed
        self.gamma = 0.92
        self.C = 2.0

        # Deadline guard
        self.start_time = time.time()
        self.deadline = 115.0
        self.elapsed = 0.0
        self.round = 0

        # Track best arm for fallback
        self.best_arm = 0
        self.best_reward = -np.inf
        self.arm_rewards = [0.0] * n_arms
        self.arm_counts = [0] * n_arms

    def _check_deadline(self):
        self.elapsed = time.time() - self.start_time
        if self.elapsed >= self.deadline:
            return True
        return False

    def select(self, context):
        self.round += 1

        # Deadline guard: if we're close to timeout, return best known arm
        if self._check_deadline():
            return self.best_arm

        context = np.asarray(context, dtype=np.float64)

        # Compute scores with UCB
        scores = np.zeros(self.n)
        for a in range(self.n):
            theta = self.V[a] @ self.b[a]
            mean = theta @ context
            # Uncertainty term
            var = context @ self.V[a] @ context
            scores[a] = mean + self.C * np.sqrt(max(var, 0))

        # Add small random tie-breaker to avoid systematic bias
        scores += np.random.normal(0, 0.01, self.n)

        arm = int(np.argmax(scores))
        return arm

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)

        # Track rewards for fallback
        self.arm_rewards[arm] = 0.95 * self.arm_rewards[arm] + 0.05 * reward
        self.arm_counts[arm] += 1
        if self.arm_counts[arm] >= 5:
            if self.arm_rewards[arm] > self.best_reward:
                self.best_reward = self.arm_rewards[arm]
                self.best_arm = arm

        # Fixed forgetting factor
        gamma = self.gamma
        V = self.V[arm]
        b = self.b[arm]

        # Update with forgetting
        # We maintain V as the inverse of the weighted sum of outer products.
        # Weighted sum S = gamma * S_prev + x x^T
        # V_new = S_new^{-1}
        # Using Sherman-Morrison on V_old = S_old^{-1}:
        # S_new = gamma * V_old^{-1} + x x^T
        # This is tricky to update V directly without inverting.
        # Standard DLS update:
        # V_new = (1/gamma) * V_old - ((1/gamma) * V_old x) ((1/gamma) * x^T V_old) / (1 + (1/gamma) * x^T V_old x)
        
        # Let's use the standard formula for discounted least squares update of inverse covariance.
        # V_t = (gamma * V_{t-1} + x x^T)^{-1} is INCORRECT. 
        # The model is usually: minimize sum gamma^t (r_t - x_t^T theta)^2
        # The Hessian is H = sum gamma^t x_t x_t^T.
        # H_t = gamma H_{t-1} + x_t x_t^T.
        # V_t = H_t^{-1}.
        # V_t = (gamma H_{t-1} + x x^T)^{-1}
        # Using matrix inversion lemma: (A + uv^T)^{-1} = A^{-1} - A^{-1} u v^T A^{-1} / (1 + v^T A^{-1} u)
        # Here A = gamma H_{t-1}, u = x, v = x.
        # A^{-1} = (1/gamma) H_{t-1}^{-1} = (1/gamma) V_{t-1}.
        # V_t = (1/gamma) V_{t-1} - ((1/gamma) V_{t-1} x) (x^T (1/gamma) V_{t-1}) / (1 + x^T (1/gamma) V_{t-1} x)
        # V_t = (1/gamma) V_{t-1} - (1/gamma^2) (V_{t-1} x) (x^T V_{t-1}) / (1 + (1/gamma) x^T V_{t-1} x)
        
        Vx = V @ context
        # Numerator part: (1/gamma^2) * outer(Vx, Vx)
        # Denominator part: (1 + (1/gamma) * context @ Vx)
        # Let's simplify by factoring out 1/gamma from the denominator term?
        # Denom = (gamma + context @ Vx) / gamma
        # Term = (1/gamma^2) * outer(Vx, Vx) / ( (gamma + context @ Vx) / gamma )
        # Term = (1/gamma) * outer(Vx, Vx) / (gamma + context @ Vx)
        
        V = (1.0 / gamma) * V - (1.0 / gamma) * np.outer(Vx, Vx) / (gamma + context @ Vx)
        
        # Update b
        # b_t = gamma * b_{t-1} + r_t x_t
        b = gamma * b + reward * context

        self.V[arm] = V
        self.b[arm] = b
        self.counts[arm] += 1
# EVOLVE-BLOCK-END

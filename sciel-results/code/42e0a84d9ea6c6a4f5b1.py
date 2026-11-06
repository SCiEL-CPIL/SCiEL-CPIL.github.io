"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Policy:
    """Global linear model with arm-specific biases and forgetting factor."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        # Global model
        self.A = np.eye(dim)
        self.b = np.zeros(dim)
        # Arm-specific biases
        self.arm_bias = np.zeros(n_arms)
        self.arm_count = np.zeros(n_arms)
        # Forgetting factor
        self.gamma = 0.995
        # Exploration parameter
        self.alpha = 1.0
        # Track arm means for bias estimation
        self.arm_reward_sum = np.zeros(n_arms)

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)
        inv = np.linalg.inv(self.A)
        theta = inv @ self.b
        scores = []
        for a in range(self.n):
            # Global linear prediction + arm bias
            pred = theta @ context + self.arm_bias[a]
            # Uncertainty from global model
            var = context @ inv @ context
            # Add small uncertainty for arm bias (shrinks as arm is sampled more)
            bias_var = 1.0 / (1.0 + self.arm_count[a])
            total_var = var + bias_var
            scores.append(pred + self.alpha * np.sqrt(total_var))
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        # Update global model with forgetting
        self.A = self.gamma * self.A + np.outer(context, context)
        self.b = self.gamma * self.b + reward * context
        # Update arm-specific bias using exponential moving average
        self.arm_count[arm] += 1
        # Update bias: track deviation from what global model would predict
        # Simple approach: maintain running mean of rewards for each arm,
        # and bias = arm_mean - global_mean_estimate
        # But we don't have a global mean estimate easily.
        # Alternative: bias captures the arm's intercept effect.
        # Use exponential moving average of (reward - global_pred)
        inv = np.linalg.inv(self.A)
        theta = inv @ self.b
        global_pred = theta @ context
        residual = reward - global_pred
        # Update arm bias with forgetting
        if self.arm_count[arm] == 1:
            self.arm_bias[arm] = residual
        else:
            self.arm_bias[arm] = self.gamma * self.arm_bias[arm] + (1 - self.gamma) * residual
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Non-stationary Linear Contextual Bandit with per-arm adaptive forgetting."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.inv_A = [np.eye(dim) for _ in range(n_arms)]
        
        # Exploration parameter
        self.alpha = 1.8
        
        # Forgetting factor parameters
        self.base_gamma = 0.98
        self.min_gamma = 0.95
        self.max_gamma = 0.999
        
        # Per-arm adaptive forgetting state
        self.rewards_ema = [0.0 for _ in range(n_arms)]
        self.errors_ema = [0.0 for _ in range(n_arms)]
        self.counts = [0 for _ in range(n_arms)]
        
        # Timing guard
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0

    def _compute_theta(self, arm):
        """Compute theta = inv_A @ b for a specific arm."""
        self.theta[arm] = self.inv_A[arm] @ self.b[arm]

    def select(self, context):
        # Check deadline
        if time.time() > self.deadline:
            # Return a deterministic arm if past deadline (safety)
            return 0
            
        scores = []
        for a in range(self.n):
            # Ensure theta is up to date
            self._compute_theta(a)
            theta = self.theta[a]
            inv_A = self.inv_A[a]
            
            # Predicted reward
            pred = theta @ context
            
            # Confidence interval width
            # Use the upper bound of the confidence interval
            # sqrt(context^T inv_A context)
            v = context @ inv_A @ context
            ci = self.alpha * np.sqrt(max(v, 0.0))
            
            score = pred + ci
            scores.append(score)
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Update counts
        self.counts[arm] += 1
        
        # Update EMA of rewards
        self.rewards_ema[arm] = 0.9 * self.rewards_ema[arm] + 0.1 * reward
        
        # Compute error (squared)
        # We need the predicted reward before update to compute the error
        # But we don't store the exact prediction from select. 
        # We can approximate the error using the current model.
        # However, the error should be measured against the *previous* model's prediction.
        # Let's recompute the prediction with the *current* (pre-update) model state.
        
        # Actually, we can just track the raw reward and use the difference from the EMA
        # to estimate volatility.
        
        error = reward - self.rewards_ema[arm]
        self.errors_ema[arm] = 0.9 * self.errors_ema[arm] + 0.1 * (error ** 2)
        
        # Adaptive forgetting factor
        # If the error is high, we want to forget faster (lower gamma)
        # If the error is low, we want to forget slower (higher gamma)
        # Map the error magnitude to a gamma between min_gamma and max_gamma
        
        # Normalize error
        # We need a scale. Let's use a simple heuristic.
        # High error -> low gamma (fast forgetting)
        # Low error -> high gamma (slow forgetting)
        
        # Use a sigmoid-like mapping or linear clipping
        # Let's try: gamma = max_gamma - (max_gamma - min_gamma) * tanh(error_ema / scale)
        # Scale is an estimate of typical reward variance. 
        # Since we don't know the scale, let's just use the error_ema directly with a fixed scale.
        # Or simpler: just use the error_ema to adjust gamma.
        
        # Let's use a simpler approach: 
        # gamma = base_gamma - adjustment
        # adjustment is proportional to error_ema
        
        # To keep it stable, let's clamp the adjustment.
        err_norm = self.errors_ema[arm]
        # If err_norm is large, we want faster forgetting.
        # Let's assume reward scale is around 1.
        # If err_norm > 1, increase forgetting rate.
        
        # Linear mapping with clipping
        # gamma = max_gamma - (max_gamma - min_gamma) * min(1.0, err_norm / 2.0)
        
        gamma = self.max_gamma - (self.max_gamma - self.min_gamma) * min(1.0, err_norm / 2.0)
        gamma = max(self.min_gamma, min(self.max_gamma, gamma))
        
        # Update A and b with forgetting
        self.A[arm] = gamma * self.A[arm] + np.outer(context, context)
        self.b[arm] = gamma * self.b[arm] + reward * context
        
        # Update inverse A using Sherman-Morrison or just recompute
        # Since dim is small (likely 5-10), recomputing inverse is cheap
        # But we can use the rank-1 update formula for efficiency if needed.
        # For now, recompute to ensure numerical stability.
        try:
            self.inv_A[arm] = np.linalg.inv(self.A[arm])
        except np.linalg.LinAlgError:
            # Fallback if singular
            self.inv_A[arm] = np.eye(self.d)
            
        # Update theta
        self._compute_theta(arm)
# EVOLVE-BLOCK-END

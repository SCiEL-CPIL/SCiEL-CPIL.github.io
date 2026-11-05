"""Non-stationary linear contextual bandit with adaptive Kalman-like filtering."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Non-stationary linear contextual bandit using:
    - Per-arm adaptive Kalman-like filters for parameter estimation
    - Process noise (Q) and measurement noise (R) adaptation based on residuals
    - UCB exploration with covariance-based uncertainty
    - Deadline guard for safety
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = 145.0  # Safety margin before 150s kill

        # Initialization
        # Initial parameters: zero
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        
        # Initial covariance: large diagonal to allow quick initial learning
        # This corresponds to low initial confidence
        self.P = [100.0 * np.eye(dim) for _ in range(n_arms)]
        
        self.counts = [0] * n_arms
        
        # Adaptive parameters
        # Base process noise scalar (multiplied by identity)
        self.base_Q = 0.1
        self.Q_min = 0.01
        self.Q_max = 1.0
        
        # Base measurement noise
        self.R = 0.1
        
        # Exploration constant
        self.C = 1.5
        
        # Residual tracking for adaptation
        self.residual_ema = [0.0] * n_arms  # EMA of squared residuals
        self.residual_ema_alpha = 0.1  # Rate of EMA update

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _adapt_Q(self, arm, pred_error_sq):
        """
        Adapt process noise Q based on current prediction error.
        If current error is significantly larger than the historical average,
        increase Q to allow the state to move faster (track drift).
        If current error is smaller, decrease Q to smooth the state (exploit stability).
        """
        # Update EMA of squared residuals
        # pred_error_sq is (reward - pred)^2
        self.residual_ema[arm] = (1 - self.residual_ema_alpha) * self.residual_ema[arm] + self.residual_ema_alpha * pred_error_sq
        
        # Compare current error to EMA
        # Avoid division by zero
        avg_err = max(self.residual_ema[arm], 1e-6)
        
        # Ratio of current to average error
        ratio = pred_error_sq / avg_err
        
        # Map ratio to Q scale
        # If ratio > 1, error is high -> increase Q
        # If ratio < 1, error is low -> decrease Q
        # Use log scale for smoothness and bounded range
        # log(ratio) is negative if ratio < 1, positive if > 1
        log_ratio = np.log(ratio)
        
        # Scale adjustment: adjust base_Q by factor exp(log_ratio * sensitivity)
        # Sensitivity 1.0 means if error doubles, Q doubles
        sensitivity = 1.0
        adjustment = np.exp(log_ratio * sensitivity)
        
        Q_val = self.base_Q * adjustment
        
        return max(self.Q_min, min(self.Q_max, Q_val))

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)

        # Deadline guard: if close to deadline, use greedy
        if not self._check_deadline():
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        # UCB with adaptive exploration
        scores = []
        for a in range(self.n):
            mu = self.theta[a] @ context
            # Uncertainty: sqrt(x^T P x)
            # P is symmetric positive semi-definite (ideally)
            uncertainty = np.sqrt(max(0.0, context @ self.P[a] @ context))
            scores.append(mu + self.C * uncertainty)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        self.counts[arm] += 1

        # 1. Calculate prediction and error
        pred = self.theta[arm] @ context
        error = reward - pred
        pred_error_sq = error * error

        # 2. Adapt Process Noise Q
        Q_scalar = self._adapt_Q(arm, pred_error_sq)
        Q = Q_scalar * np.eye(self.d)
        
        # Measurement noise R (scalar, identity)
        R = self.R * np.eye(self.d)
        # Since R is scalar * I, we can optimize the Kalman update terms

        # 3. Kalman-like Update
        # Predicted Covariance: P_pred = P + Q
        # Innovation Covariance: S = R + x^T P_pred x
        # Kalman Gain: K = P_pred x / S
        
        P_old = self.P[arm]
        P_pred = P_old + Q
        
        # Compute S = R + x^T P_pred x
        # R is scalar * I, so x^T R x = R_scalar * ||x||^2
        x_norm_sq = context @ context
        S = self.R * x_norm_sq + context @ P_pred @ context
        
        if S < 1e-9:
            S = 1e-9
            
        # Kalman Gain vector K
        K = (P_pred @ context) / S
        
        # Update Parameter Theta
        self.theta[arm] = self.theta[arm] + K * error
        
        # Update Covariance P (Joseph form or standard)
        # Standard: P_new = (I - K x^T) P_pred
        # Optimized for R = r * I:
        # P_new = P_pred - K (x^T P_pred)
        # Note: x^T P_pred = (P_pred x)^T
        Px = P_pred @ context
        self.P[arm] = P_pred - np.outer(K, Px)
        
        # Symmetrize P to prevent numerical drift from breaking PSD
        self.P[arm] = (self.P[arm] + self.P[arm].T) / 2.0
        
# EVOLVE-BLOCK-END

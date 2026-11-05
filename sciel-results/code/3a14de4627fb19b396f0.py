"""Non-stationary linear contextual bandit with dual-timescale Kalman filter."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary Linear Contextual Bandit using a dual-timescale Kalman filter.
    
    Core Idea:
    - Model arm weights as a random walk (process noise Q).
    - Use Kalman Filter updates for state estimation.
    - Dual-timescale: 
        1. Fast scale: Standard KF update for reward prediction.
        2. Slow scale: Adaptive Process Noise (Q) based on prediction error.
           - If prediction error is high, increase Q (trust model less, expect drift).
           - If prediction error is low, decrease Q (trust model more, assume stability).
    - This avoids the instability of fixed/heuristic forgetting factors by tying
      adaptation directly to observed performance.
    
    State:
    - theta[a]: Estimated weight vector for arm a.
    - P[a]: Covariance matrix for theta[a].
    
    Update:
    - Prediction: theta_pred = theta, P_pred = P + Q (Random Walk)
    - Update: Standard KF update with observation y = reward, H = context.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = 140.0  # Safety margin before 150s kill

        # Per-arm state
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Initialize P with large value to allow rapid initial learning
        self.P = [np.eye(dim) * 100.0 for _ in range(n_arms)]
        
        # Process noise base and adaptation params
        # Q is the covariance of the random walk step.
        # We model Q = q_scalar * I for simplicity, but adapt q_scalar.
        self.q_base = 0.05
        self.q_min = 0.01
        self.q_max = 5.0
        
        # Adaptation params for Q
        self.error_threshold = 1.0  # Scale of expected noise
        self.adapt_gain = 0.5       # How aggressively to adapt Q
        
        # Exploration
        # UCB style: mu + alpha * sqrt(context^T P context)
        self.alpha = 1.5
        self.alpha_decay = 0.999
        self.alpha_min = 0.5
        
        # Tracking
        self.round_num = 0
        self.arm_counts = [0] * n_arms
        self.pred_errors = [np.zeros(10) for _ in range(n_arms)] # Circular buffer
        self.pred_error_idx = [0] * n_arms

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _adapt_q(self, arm, pred_error):
        """
        Adapt process noise Q based on recent prediction errors.
        High error -> High Q (expecting change).
        Low error -> Low Q (stable).
        """
        c = self.arm_counts[arm]
        if c < 5:
            return self.q_base

        # Get recent errors
        errs = self.pred_errors[arm]
        mean_err = np.mean(errs)
        
        # Normalize error relative to threshold
        # If mean_err > threshold, we are in a non-stationary regime
        ratio = mean_err / self.error_threshold
        
        # Log-linear scaling of Q
        # q = q_base * (ratio ** adapt_gain)
        # Clamp
        q = self.q_base * (max(ratio, 1e-3) ** self.adapt_gain)
        
        return max(self.q_min, min(self.q_max, q))

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)

        if not self._check_deadline():
            # Greedy fallback
            scores = []
            for a in range(self.n):
                scores.append(self.theta[a] @ context)
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            mu = self.theta[a] @ context
            # Uncertainty: context^T P context
            P_ctx = self.P[a] @ context
            var_term = context @ P_ctx
            if var_term < 0:
                var_term = 0.0
            bonus = self.alpha * np.sqrt(var_term)
            scores.append(mu + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        self.round_num += 1
        
        # Decay alpha slightly to reduce exploration over time
        self.alpha = max(self.alpha_min, self.alpha * self.alpha_decay)
        
        # 1. Prediction Step (Random Walk)
        # Adapt Q based on history
        q = self._adapt_q(arm, 0.0) # Pass 0, we update error buffer below
        
        # theta_pred = theta (Identity transition)
        # P_pred = P + q * I
        P_pred = self.P[arm] + q * np.eye(self.d)

        # 2. Update Step (Kalman Filter)
        # Observation: y = reward
        # H = context (1 x d)
        # R = observation noise variance. Assume 1.0 for Gaussian bandit.
        R = 1.0
        
        # Innovation
        y_hat = self.theta[arm] @ context
        innovation = reward - y_hat
        
        # Innovation Covariance
        S = context @ P_pred @ context + R
        
        # Kalman Gain
        if S > 1e-10:
            K = P_pred @ context / S
        else:
            K = np.zeros(self.d)
        
        # State Update
        self.theta[arm] = self.theta[arm] + K * innovation
        
        # Covariance Update (Joseph form for stability)
        # P_new = (I - K H) P_pred (I - K H)^T + K R K^T
        I_KH = np.eye(self.d) - np.outer(K, context)
        self.P[arm] = I_KH @ P_pred @ I_KH.T + np.outer(K, K) * R
        
        # Symmetrize
        self.P[arm] = 0.5 * (self.P[arm] + self.P[arm].T)

        # 3. Update Tracking Stats
        self.arm_counts[arm] += 1
        
        # Store prediction error (before update) for Q adaptation
        err_idx = self.pred_error_idx[arm]
        self.pred_errors[arm][err_idx] = abs(innovation)
        self.pred_error_idx[arm] = (err_idx + 1) % 10

# EVOLVE-BLOCK-END

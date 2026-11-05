import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        
        # State variables: theta is the weight vector, P is the covariance matrix
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Initialize P with high uncertainty to encourage exploration initially
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.count = [0 for _ in range(n_arms)]
        
        # Adaptive Process Noise (Q) parameters
        self.Q_base = 3e-4
        self.Q_max = 0.04
        self._innov_ema = [0.0 for _ in range(n_arms)]
        self._innov_alpha = 0.12  # Smoothing factor for EMA
        
        # Observation Noise (R)
        self.obs_noise = 0.1
        
        # Exploration parameters
        # Using the refined exploration strategy from inspirations:
        # Base alpha 1.2, min 0.3, decay based on time and count
        self.alpha_base = 1.2
        self.alpha_min = 0.3
        self.alpha_decay_exp = 0.1
        
        # Reset mechanism for catastrophic drift
        self._consec_large = [0 for _ in range(n_arms)]
        self._reset_threshold = 3
        self._reset_cov_scale = 5.0
        
        # Deadline guard
        self._start_time = time.time()
        self._deadline = 112.0  # Safety margin before 150s kill

    def _adaptive_Q(self, arm):
        """
        Compute adaptive process noise based on EMA of normalized squared innovations.
        High innovation suggests the true parameters are moving, so we increase Q
        to allow P to grow and theta to update more aggressively.
        """
        if self.count[arm] < 3:
            return self.Q_base
        
        ema_val = self._innov_ema[arm]
        # Map EMA to [0, 1] range using tanh to ensure bounded Q
        scaled = np.tanh(ema_val / 0.4)
        q = self.Q_base + (self.Q_max - self.Q_base) * scaled
        return q

    def _maybe_reset(self, arm, innov_sq_norm):
        """
        If too many consecutive large normalized innovations, reset covariance.
        This helps escape from a state where the filter has converged to a wrong solution
        due to a sudden regime change.
        """
        # Threshold for 'large' normalized innovation
        if innov_sq_norm > 4.0:
            self._consec_large[arm] += 1
        else:
            self._consec_large[arm] = max(0, self._consec_large[arm] - 1)
        
        if self._consec_large[arm] >= self._reset_threshold:
            # Soft reset: inflate covariance to encourage re-exploration
            self.P[arm] = np.eye(self.d) * self._reset_cov_scale
            self._consec_large[arm] = 0

    def select(self, context):
        # Deadline check: if close to deadline, exploit best estimate to ensure completion
        if time.time() - self._start_time > self._deadline:
            context_vec = np.asarray(context, dtype=np.float64)
            scores = [np.dot(self.theta[a], context_vec) for a in range(self.n)]
            return int(np.argmax(scores))

        context_vec = np.asarray(context, dtype=np.float64)
        scores = np.zeros(self.n)
        
        # Adaptive alpha: start high for exploration, decay to exploitation
        # Incorporates time decay and count-based decay from inspirations
        for a in range(self.n):
            mu = np.dot(self.theta[a], context_vec)
            # Variance estimate: x^T P x
            P_x = np.dot(self.P[a], context_vec)
            var = np.dot(context_vec, P_x)
            if var < 0:
                var = 0.0
            std = np.sqrt(var)
            
            # Combine time decay and count decay
            # alpha_t = max(alpha_min, alpha_base * (1+t)^(-exp) * (1+sqrt(count))^(-1))
            # The count factor ensures that even if time is small, heavily visited arms explore less
            # The time factor ensures global exploration decreases over time
            count_factor = 1.0 / (1.0 + np.sqrt(self.count[a]))
            time_factor = (1.0 + self.t) ** (-self.alpha_decay_exp)
            alpha_t = max(self.alpha_min, self.alpha_base * time_factor * count_factor)
            
            scores[a] = mu + alpha_t * std
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1
        a = arm
        x = np.asarray(context, dtype=np.float64)
        r = reward
        
        # Deadline guard: stop heavy computation if we are running out of time
        if time.time() - self._start_time > self._deadline:
            self.count[a] += 1
            return

        # 1. Predict Step: Account for process noise (non-stationarity)
        Q = self._adaptive_Q(a)
        P_pred = self.P[a] + Q * np.eye(self.d)
        
        # 2. Calculate Innovation
        pred = np.dot(self.theta[a], x)
        y = r - pred
        
        # 3. Update Step (Kalman Filter with Joseph Form for stability)
        # K = P_pred x (x^T P_pred x + sigma^2)^-1
        Px = P_pred @ x
        denom = x @ Px + self.obs_noise
        if denom < 1e-12:
            denom = 1e-12
            
        K = Px / denom
        
        # Update Theta
        self.theta[a] = self.theta[a] + K * y
        
        # Update P using Joseph Form:
        # P_new = (I - K x^T) P_pred (I - K x^T)^T + K sigma^2 K^T
        # This form is numerically stable and guarantees P_new is positive semi-definite.
        I = np.eye(self.d)
        IKxT = I - np.outer(K, x)
        P_new = IKxT @ P_pred @ IKxT.T + np.outer(K, K) * self.obs_noise
        
        # Ensure symmetry
        self.P[a] = (P_new + P_new.T) / 2.0
        
        # 4. Update Adaptive Q Metrics
        # Normalize innovation by the predicted innovation variance (denom)
        innov_sq_norm = (y * y) / denom
        self._innov_ema[a] = (1.0 - self._innov_alpha) * self._innov_ema[a] + self._innov_alpha * innov_sq_norm
        
        # 5. Check for Reset
        self._maybe_reset(a, innov_sq_norm)
        
        self.count[a] += 1

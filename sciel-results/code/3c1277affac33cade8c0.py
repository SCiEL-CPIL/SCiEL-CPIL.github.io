import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        
        # Dual-mode Kalman Filter States
        # Mode 0: Stable (low Q base, slow adaptation)
        # Mode 1: Volatile (high Q base, fast adaptation)
        self.theta = [[np.zeros(dim) for _ in range(2)] for _ in range(n_arms)]
        self.P = [[np.eye(dim) * 10.0 for _ in range(2)] for _ in range(n_arms)]
        self.count = [0 for _ in range(n_arms)]
        
        # Adaptive Q Parameters per Mode
        # Base Q: Minimum process noise
        # Max Q: Maximum process noise
        # Gain: How quickly the EMA drives Q towards Max
        self.Q_base = [3e-4, 1e-3]   # Stable vs Volatile
        self.Q_max = [0.02, 0.1]     # Stable vs Volatile
        self.Q_gain = [0.5, 0.2]     # Denominator for tanh scaling (lower = more sensitive)
        
        # Innovation EMA tracking per mode
        self._innov_ema = [[0.0 for _ in range(2)] for _ in range(n_arms)]
        self._innov_alpha = 0.15     # EMA smoothing factor
        
        # Observation Noise (R)
        # Lower R allows sharper tracking. 0.1 is a good balance for Gaussian rewards.
        self.obs_noise = 0.1
        
        # Exploration parameters
        self.alpha_base = 1.2
        self.alpha_min = 0.3
        self.alpha_decay_exp = 0.1
        
        # Reset Mechanism
        # If innovations are consistently large, we might be in a completely new regime.
        # Soft reset: Inflate covariance to encourage exploration.
        self._consec_large = [[0 for _ in range(2)] for _ in range(n_arms)]
        self._reset_threshold = 4
        self._reset_cov_scale = 5.0
        
        # Deadline guard
        self._start_time = time.time()
        self._deadline = 112.0

    def _adaptive_Q(self, arm, mode):
        """
        Compute adaptive process noise Q for a specific arm and mode.
        Uses tanh saturation of the innovation EMA to bound Q.
        """
        if self.count[arm] < 3:
            return self.Q_base[mode]
        
        ema_val = self._innov_ema[arm][mode]
        # Scale EMA by mode-specific gain. 
        # Volatile mode has lower gain denominator, so it reacts more strongly to innovations?
        # Actually, we want Volatile mode to have higher Q generally.
        # Let's use the gain to control sensitivity.
        scaled = np.tanh(ema_val / self.Q_gain[mode])
        q = self.Q_base[mode] + (self.Q_max[mode] - self.Q_base[mode]) * scaled
        return q

    def _maybe_reset(self, arm, mode, innov_sq):
        """
        Check for catastrophic drift. If standardized innovation is large 
        for consecutive steps, inflate covariance.
        """
        # Threshold for 'large' standardized innovation squared
        # innov_sq is (y^2) / (x^T P x + R). Ideally ~1.
        # If > 4.0, it's a significant outlier.
        if innov_sq > 4.0:
            self._consec_large[arm][mode] += 1
        else:
            self._consec_large[arm][mode] = max(0, self._consec_large[arm][mode] - 1)
        
        if self._consec_large[arm][mode] >= self._reset_threshold:
            # Soft reset: Inflate covariance
            self.P[arm][mode] = np.eye(self.d) * self._reset_cov_scale
            self._consec_large[arm][mode] = 0

    def _get_active_mode(self, arm):
        """
        Determine the active mode for scoring.
        We prefer the mode with the lower recent innovation (better fit).
        If stable mode fits well, use it (lower variance -> better exploration bonus).
        If volatile mode fits better, use it (tracks drift).
        """
        if self.count[arm] < 3:
            return 0
        
        # Compare the EMA of squared innovations.
        # Lower EMA means the filter is tracking the rewards well.
        ema_stable = self._innov_ema[arm][0]
        ema_volatile = self._innov_ema[arm][1]
        
        # If volatile is significantly better, switch to it.
        # Otherwise, stick with stable to avoid unnecessary switching noise.
        if ema_volatile < ema_stable * 0.8: # 20% better
            return 1
        return 0

    def _blend_modes(self, arm, context_vec):
        """
        Blend the predictions and uncertainties from both modes using 
        softmax weights based on the negative innovation EMAs.
        Returns (blended_mu, blended_std).
        """
        if self.count[arm] < 3:
            mode = 0
            mu = np.dot(self.theta[arm][mode], context_vec)
            P_x = self.P[arm][mode] @ context_vec
            var = float(context_vec @ P_x)
            if var < 0:
                var = 0.0
            std = np.sqrt(var)
            return mu, std

        # Compute logits: negative EMA (lower innovation is better)
        # Scale by a temperature to control how sharp the switching is
        # T=1.0 is standard. Lower T makes it sharper (harder switch).
        T = 1.0
        logit_0 = -self._innov_ema[arm][0] / T
        logit_1 = -self._innov_ema[arm][1] / T
        
        # Softmax for stability
        max_logit = max(logit_0, logit_1)
        exp_0 = np.exp(logit_0 - max_logit)
        exp_1 = np.exp(logit_1 - max_logit)
        sum_exp = exp_0 + exp_1
        w_0 = exp_0 / sum_exp
        w_1 = exp_1 / sum_exp
        
        # Blend means
        mu_0 = np.dot(self.theta[arm][0], context_vec)
        mu_1 = np.dot(self.theta[arm][1], context_vec)
        mu = w_0 * mu_0 + w_1 * mu_1
        
        # Blend variances (approximate: weighted sum of variances + variance of the mixture)
        # For exploration, we want a conservative (higher) estimate of uncertainty if the modes disagree.
        # Let's just use weighted average of standard deviations for simplicity, 
        # or weighted variance. 
        # Actually, blending covariances is complex. Let's blend the scalar variances.
        P_x_0 = self.P[arm][0] @ context_vec
        var_0 = float(context_vec @ P_x_0)
        if var_0 < 0: var_0 = 0.0
        
        P_x_1 = self.P[arm][1] @ context_vec
        var_1 = float(context_vec @ P_x_1)
        if var_1 < 0: var_1 = 0.0
        
        # Weighted variance
        var = w_0 * var_0 + w_1 * var_1
        
        # Add extra uncertainty if the modes disagree significantly (disagreement bonus)
        diff = abs(mu_0 - mu_1)
        # If the two modes predict very different values, the true value is uncertain.
        # Add a term proportional to the disagreement.
        disagreement_penalty = 0.5 * diff * diff
        var += disagreement_penalty
        
        std = np.sqrt(var)
        return mu, std

    def select(self, context):
        # Deadline check
        if time.time() - self._start_time > self._deadline:
            context_vec = np.asarray(context, dtype=np.float64)
            scores = np.zeros(self.n)
            for a in range(self.n):
                # Use the active mode for greedy selection in emergency
                mode = self._get_active_mode(a)
                scores[a] = np.dot(self.theta[a][mode], context_vec)
            return int(np.argmax(scores))

        context_vec = np.asarray(context, dtype=np.float64)
        scores = np.zeros(self.n)
        
        for a in range(self.n):
            # Use blended mode selection for smoother transitions
            mu, std = self._blend_modes(a, context_vec)
            
            # Exploration Bonus
            # Decay over time and with visit count
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
        
        # Deadline guard: minimize work if running out of time
        if time.time() - self._start_time > self._deadline:
            self.count[a] += 1
            return

        # Update BOTH modes for the pulled arm.
        # This keeps both filters warm and allows the _get_active_mode to switch instantly.
        for mode in range(2):
            # 1. Predict Step with adaptive process noise
            Q = self._adaptive_Q(a, mode)
            P_pred = self.P[a][mode] + Q * np.eye(self.d)
            
            # 2. Calculate Innovation
            pred = float(self.theta[a][mode] @ x)
            y = r - pred
            
            # 3. Kalman Filter Update (Joseph Form for stability)
            Px = P_pred @ x
            denom = float(x @ Px) + self.obs_noise
            if denom < 1e-12:
                denom = 1e-12
                
            K = Px / denom
            
            self.theta[a][mode] = self.theta[a][mode] + K * y
            
            I = np.eye(self.d)
            IKxT = I - np.outer(K, x)
            P_new = IKxT @ P_pred @ IKxT.T + np.outer(K, K) * self.obs_noise
            self.P[a][mode] = (P_new + P_new.T) / 2.0
            
            # 4. Update innovation EMA
            innov_sq_norm = (y * y) / denom
            self._innov_ema[a][mode] = (1.0 - self._innov_alpha) * self._innov_ema[a][mode] + self._innov_alpha * innov_sq_norm
            
            # 5. Check for reset
            self._maybe_reset(a, mode, innov_sq_norm)
            
        self.count[a] += 1

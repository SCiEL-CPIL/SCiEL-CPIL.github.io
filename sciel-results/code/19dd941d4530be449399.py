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
        # Two regimes: stable (low Q) and volatile (high Q)
        self.Q_stable = 1e-4
        self.Q_volatile = 0.02
        self._innov_ema = [0.0 for _ in range(n_arms)]
        self._innov_alpha = 0.15  # Smoothing factor for EMA
        
        # Regime probabilities: P(regime=volatile | history)
        # Start in stable mode
        self._vol_prob = [0.0 for _ in range(n_arms)]
        
        # Transition probabilities for regime switching
        # P(stable -> volatile), P(volatile -> stable)
        self._p_sv = 0.1  # 10% chance to enter volatile mode each step
        self._p_vs = 0.3  # 30% chance to return to stable mode each step
        
        # Observation Noise (R)
        self.obs_noise = 0.1
        
        # Exploration parameters
        self.alpha_init = 1.5
        self.alpha_min = 0.3
        self.alpha_decay = 100.0
        
        # Deadline guard
        self._start_time = time.time()
        self._deadline = 110.0  # Safety margin before 150s kill

    def _update_regime(self, arm, innov_sq_norm):
        """
        Update regime probability based on innovation magnitude.
        Large innovations increase probability of volatile regime.
        """
        # Likelihood of innovation under each regime
        # Under stable: innovation variance ~ obs_noise + small Q
        # Under volatile: innovation variance ~ obs_noise + large Q
        # Use simple heuristic: if innov_sq_norm > threshold, increase vol probability
        
        # Threshold based on expected chi-squared distribution
        # For 1D, chi-squared(1) has mean 1, so values > 4 are unusual
        threshold = 4.0
        
        if innov_sq_norm > threshold:
            # Large innovation: increase volatile probability
            self._vol_prob[arm] = min(1.0, self._vol_prob[arm] + 0.2)
        else:
            # Small innovation: decrease volatile probability
            self._vol_prob[arm] = max(0.0, self._vol_prob[arm] - 0.1)
        
        # Apply transition probabilities
        p_vol = self._vol_prob[arm]
        p_stable = 1.0 - p_vol
        
        # New probability of being in volatile mode
        # P(volatile_t) = P(volatile_{t-1}) * P(volatile|volatile) + P(stable_{t-1}) * P(volatile|stable)
        new_p_vol = p_vol * (1.0 - self._p_vs) + p_stable * self._p_sv
        self._vol_prob[arm] = new_p_vol

    def _adaptive_Q(self, arm):
        """
        Compute adaptive process noise based on regime probability.
        Blend stable and volatile Q based on probability of volatile regime.
        """
        p_vol = self._vol_prob[arm]
        q = (1.0 - p_vol) * self.Q_stable + p_vol * self.Q_volatile
        return q

    def select(self, context):
        # Deadline check: if close to deadline, exploit best estimate to ensure completion
        if time.time() - self._start_time > self._deadline:
            context_vec = np.asarray(context, dtype=np.float64)
            scores = [np.dot(self.theta[a], context_vec) for a in range(self.n)]
            return int(np.argmax(scores))

        context_vec = np.asarray(context, dtype=np.float64)
        scores = np.zeros(self.n)
        
        # Adaptive alpha: start high for exploration, decay to exploitation
        alpha = self.alpha_min + (self.alpha_init - self.alpha_min) * np.exp(-self.t / self.alpha_decay)
        
        for a in range(self.n):
            mu = np.dot(self.theta[a], context_vec)
            # Variance estimate: x^T P x
            P_x = np.dot(self.P[a], context_vec)
            var = np.dot(context_vec, P_x)
            if var < 0:
                var = 0.0
            std = np.sqrt(var)
            scores[a] = mu + alpha * std
            
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
        Px = P_pred @ x
        denom = x @ Px + self.obs_noise
        if denom < 1e-12:
            denom = 1e-12
            
        K = Px / denom
        
        # Update Theta
        self.theta[a] = self.theta[a] + K * y
        
        # Update P using Joseph Form
        I = np.eye(self.d)
        IKxT = I - np.outer(K, x)
        P_new = IKxT @ P_pred @ IKxT.T + np.outer(K, K) * self.obs_noise
        
        # Ensure symmetry
        self.P[a] = (P_new + P_new.T) / 2.0
        
        # 4. Update Adaptive Q Metrics and Regime Probability
        innov_sq_norm = (y * y) / denom
        self._innov_ema[a] = (1.0 - self._innov_alpha) * self._innov_ema[a] + self._innov_alpha * innov_sq_norm
        self._update_regime(a, innov_sq_norm)
        
        self.count[a] += 1

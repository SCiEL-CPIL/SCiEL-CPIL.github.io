import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        
        # Regime-switching: two Kalman filter states per arm
        # Mode 0 = "stable" (low process noise), Mode 1 = "volatile" (high process noise)
        self.theta = [[np.zeros(dim) for _ in range(2)] for _ in range(n_arms)]
        self.P = [[np.eye(dim) * 10.0 for _ in range(2)] for _ in range(n_arms)]
        self.count = [0 for _ in range(n_arms)]
        
        # Process noise per mode
        self.Q_stable = 1e-4
        self.Q_volatile = 0.02
        
        # HMM transition probabilities (per-step)
        # P(stable->stable), P(stable->volatile), P(volatile->stable), P(volatile->volatile)
        self.P_ss = 0.995
        self.P_sv = 0.005
        self.P_vs = 0.05
        self.P_vv = 0.95
        
        # Innovation-based emission: threshold for switching to volatile
        self._innov_ema = [0.0 for _ in range(n_arms)]
        self._innov_alpha = 0.15
        
        # Observation Noise (R)
        self.obs_noise = 0.1
        
        # Exploration parameters
        self.alpha_base = 1.2
        self.alpha_min = 0.3
        self.alpha_decay_exp = 0.1
        
        # Deadline guard
        self._start_time = time.time()
        self._deadline = 112.0

    def _get_active_mode(self, arm):
        """
        Determine the active mode (0=stable, 1=volatile) for an arm based on
        the EMA of normalized squared innovations.
        """
        if self.count[arm] < 3:
            return 0
        # If EMA of normalized innovation is high, we're in volatile regime
        if self._innov_ema[arm] > 1.5:
            return 1
        return 0

    def select(self, context):
        # Deadline check: if close to deadline, exploit best estimate to ensure completion
        if time.time() - self._start_time > self._deadline:
            context_vec = np.asarray(context, dtype=np.float64)
            scores = np.zeros(self.n)
            for a in range(self.n):
                # Use weighted combination of both modes
                mode = self._get_active_mode(a)
                mu = np.dot(self.theta[a][mode], context_vec)
                scores[a] = mu
            return int(np.argmax(scores))

        context_vec = np.asarray(context, dtype=np.float64)
        scores = np.zeros(self.n)
        
        for a in range(self.n):
            mode = self._get_active_mode(a)
            mu = np.dot(self.theta[a][mode], context_vec)
            P_x = self.P[a][mode] @ context_vec
            var = float(context_vec @ P_x)
            if var < 0:
                var = 0.0
            std = np.sqrt(var)
            
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

        # Determine current mode before update
        mode = self._get_active_mode(a)
        
        # 1. Predict Step with mode-specific process noise
        Q = self.Q_stable if mode == 0 else self.Q_volatile
        P_pred = self.P[a][mode] + Q * np.eye(self.d)
        
        # 2. Calculate Innovation
        pred = float(self.theta[a][mode] @ x)
        y = r - pred
        
        # 3. Kalman Filter Update (Joseph Form)
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
        self._innov_ema[a] = (1.0 - self._innov_alpha) * self._innov_ema[a] + self._innov_alpha * innov_sq_norm
        
        # 5. Regime switching: propagate both modes
        # If we were in stable mode and see large innovation, start feeding volatile mode
        # If we were in volatile mode and see small innovation, start feeding stable mode
        # The inactive mode gets a "lazy" update: it slowly converges toward the active mode
        inactive = 1 - mode
        if mode == 0 and innov_sq_norm > 2.0:
            # Large innovation in stable mode: initialize volatile toward current state
            # with inflated covariance to allow fast adaptation
            self.theta[a][1] = self.theta[a][0].copy()
            self.P[a][1] = self.P[a][0].copy() + 2.0 * np.eye(self.d)
        elif mode == 1 and innov_sq_norm < 0.8:
            # Small innovation in volatile mode: stable mode catches up
            self.theta[a][0] = self.theta[a][0] + 0.3 * (self.theta[a][1] - self.theta[a][0])
            self.P[a][0] = self.P[a][0] + 0.3 * (self.P[a][1] - self.P[a][0])
            self.P[a][0] = (self.P[a][0] + self.P[a][0].T) / 2.0
        else:
            # No regime switch: inactive mode slowly tracks active mode
            self.theta[a][inactive] = self.theta[a][inactive] + 0.05 * (self.theta[a][mode] - self.theta[a][inactive])
            self.P[a][inactive] = self.P[a][inactive] + 0.05 * (self.P[a][mode] - self.P[a][inactive])
            self.P[a][inactive] = (self.P[a][inactive] + self.P[a][inactive].T) / 2.0
        
        self.count[a] += 1

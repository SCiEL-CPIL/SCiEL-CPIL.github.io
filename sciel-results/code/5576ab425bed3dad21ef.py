import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0
        
        # Unified model dimension: d (context) + n (one-hot arm)
        # We model reward as: r = [x, e_a]^T theta
        # where x is context, e_a is one-hot vector for arm a
        self.d_full = self.d + self.n
        
        # Kalman Filter State
        # theta: (d_full,) vector of weights
        # P: (d_full, d_full) covariance matrix
        
        # Initialize with strong prior (zero mean, high variance)
        self.theta = np.zeros(self.d_full)
        self.P = np.eye(self.d_full) * 100.0  # High initial uncertainty
        
        # Process noise (Q) and Measurement noise (R)
        # In non-stationary setting, Q represents how much the true weights drift.
        # We use adaptive Q: if innovation is large, assume faster drift.
        self.R = 1.0  # Assume unit variance for Gaussian rewards
        self.Q_base = 0.01  # Base process noise
        
        # UCB exploration parameter
        self.alpha = 1.5
        
        # Time step
        self.t = 0

    def _build_features(self, context, arm):
        """
        Construct the full feature vector for a given context and arm.
        x_full = [x_context, one_hot_arm]
        """
        x = np.array(context, dtype=np.float64)
        e = np.zeros(self.n)
        e[arm] = 1.0
        return np.concatenate([x, e])

    def select(self, context):
        if time.time() > self.deadline:
            # Fallback: Greedy
            context_vec = np.array(context)
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                x_full = self._build_features(context, a)
                mu = np.dot(self.theta, x_full)
                # Simplified fallback: just use mean
                if mu > best_score:
                    best_score = mu
                    best_arm = a
            return best_arm

        context_vec = np.array(context)
        scores = np.zeros(self.n)
        
        for a in range(self.n):
            # Construct full feature vector for arm a
            x_full = self._build_features(context, a)
            
            # Predicted mean
            mu = np.dot(self.theta, x_full)
            
            # Uncertainty term for UCB
            # Var = x^T P x
            px = np.dot(self.P, x_full)
            var = np.dot(x_full, px)
            
            if var < 0:
                var = 0.0
                
            std = np.sqrt(var)
            
            # UCB score
            scores[a] = mu + self.alpha * std
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if time.time() > self.deadline:
            return

        # Construct full feature vector for the chosen arm
        x_full = self._build_features(context, arm)
        
        # --- Kalman Filter Update ---
        
        # 1. Innovation (Prediction Error)
        pred = np.dot(self.theta, x_full)
        innovation = reward - pred
        
        # 2. Innovation Covariance
        S = np.dot(x_full, np.dot(self.P, x_full)) + self.R
        if S < 1e-12:
            S = 1e-12
            
        # 3. Kalman Gain
        K = np.dot(self.P, x_full) / S
        
        # 4. State Update
        self.theta = self.theta + K * innovation
        
        # 5. Covariance Update (Joseph form for numerical stability)
        # P_new = (I - K x^T) P (I - K x^T)^T + K R K^T
        I = np.eye(self.d_full)
        IK = I - np.outer(K, x_full)
        self.P = np.dot(IK, np.dot(self.P, IK.T)) + np.outer(K, K) * self.R
        
        # Symmetrize P to prevent drift from numerical errors
        self.P = (self.P + self.P.T) / 2.0
        
        # 6. Adaptive Process Noise (Prediction Step)
        # In standard KF, we would predict theta_{t+1} = theta_t and P_{t+1} = P_t + Q.
        # Here, we are in the "update" step which corresponds to time t.
        # The next "select" will act as the prediction step for time t+1.
        # To model non-stationarity, we add process noise Q to P immediately.
        # Adaptive Q: Scale Q by the magnitude of the innovation.
        # If the model is surprised (large innovation), the environment might have changed,
        # so we increase uncertainty to allow faster adaptation.
        
        # Innovation magnitude heuristic
        innov_mag = abs(innovation)
        
        # Base Q is a small matrix. We scale it.
        # A common approach is Q = q * I.
        # We make q adaptive.
        q_val = self.Q_base * (1.0 + innov_mag)
        
        # Add process noise to covariance
        # This effectively says: "The true weights may have drifted by an amount characterized by Q"
        self.P = self.P + q_val * np.eye(self.d_full)
        
        self.t += 1

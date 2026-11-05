import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0
        
        # Kalman Filter Parameters
        # Initial state covariance P (uncertainty)
        # High initial uncertainty encourages exploration
        self.P = [np.eye(self.d) * 10.0 for _ in range(self.n)]
        
        # State estimates theta
        self.theta = [np.zeros(self.d) for _ in range(self.n)]
        
        # Measurement noise R base
        self.R_base = 1.0
        
        # Base process noise Q
        self.base_Q = 0.01
        
        # UCB parameter
        self.alpha = 1.0
        
        # Adaptive noise tracking
        # We keep a short history of squared innovations per arm to scale Q and R
        self.innovation_hist = [np.zeros(50) for _ in range(self.n)]
        self.hist_idx = [0 for _ in range(self.n)]
        
        self.t = 0

    def select(self, context):
        if time.time() > self.deadline:
            # Fallback: Greedy
            context_vec = np.array(context)
            scores = [np.dot(self.theta[a], context_vec) for a in range(self.n)]
            return int(np.argmax(scores))

        context_vec = np.array(context)
        scores = np.zeros(self.n)
        
        for a in range(self.n):
            # Predicted mean
            mu = np.dot(self.theta[a], context_vec)
            
            # Uncertainty term for UCB
            # In Kalman Filter, P is the covariance of the state estimate.
            # The variance of the prediction x^T theta is x^T P x.
            px = np.dot(self.P[a], context_vec)
            var = np.dot(context_vec, px)
            
            if var < 0:
                var = 0.0
                
            std = np.sqrt(var)
            
            # UCB score
            scores[a] = mu + self.alpha * std
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if time.time() > self.deadline:
            return

        context_vec = np.array(context)
        
        # 1. Calculate Innovation (Prediction Error)
        pred = np.dot(self.theta[arm], context_vec)
        innovation = reward - pred
        
        # 2. Update Innovation History for Adaptive R and Q
        self.innovation_hist[arm][self.hist_idx[arm]] = innovation ** 2
        self.hist_idx[arm] = (self.hist_idx[arm] + 1) % 50
        
        # Calculate average recent squared innovation
        recent_innovations = self.innovation_hist[arm]
        avg_innov_sq = np.mean(recent_innovations)
        
        # 3. Adaptive Observation Noise R
        # If the average squared error is high, it might be due to high noise.
        # We estimate R to be at least half of the observed variance, capped by base initially?
        # Actually, let R_eff be the max of base R and a fraction of the observed variance.
        # This prevents the filter from over-trusting noisy data.
        R_eff = max(self.R_base, avg_innov_sq * 0.5)
        
        # 4. Adaptive Process Noise Q
        # The residual variance after accounting for R_eff is attributed to state change.
        residual_var = max(0.0, avg_innov_sq - R_eff)
        
        # Scale Q based on residual variance relative to R_eff
        # If residual is 0, Q is base_Q. If residual is large, Q increases.
        noise_scale = 1.0 + (residual_var / R_eff)
        # Cap the scale to prevent explosion
        noise_scale = min(noise_scale, 10.0)
        
        Q = self.base_Q * noise_scale * np.eye(self.d)
        
        # 5. Kalman Predict Step
        # State prediction: theta_pred = theta (random walk model, no drift)
        # Covariance prediction: P_pred = P + Q
        P_pred = self.P[arm] + Q
        
        # 6. Kalman Update Step
        # Innovation covariance: S = H P_pred H^T + R_eff
        # Here H = x^T (1 x d), so S = x^T P_pred x + R_eff (scalar)
        P_pred_x = np.dot(P_pred, context_vec)
        S = np.dot(context_vec, P_pred_x) + R_eff
        
        # Avoid division by zero
        if S < 1e-12:
            S = 1e-12
            
        # Kalman Gain: K = P_pred H^T / S
        # K = P_pred x / S (d x 1 vector)
        K = P_pred_x / S
        
        # State Update: theta = theta_pred + K * innovation
        self.theta[arm] = self.theta[arm] + K * innovation
        
        # Covariance Update: P = (I - K H) P_pred
        # P = P_pred - K (x^T P_pred)
        self.P[arm] = P_pred - np.outer(K, P_pred_x)
        
        # Numerical stability: Symmetrize P
        self.P[arm] = (self.P[arm] + self.P[arm].T) / 2.0
        
        # Optional: Ensure PSD by clipping negative eigenvalues?
        # For speed, we rely on symmetrization and the structure of the update.
        # If P becomes indefinite, var in select() will be clamped to 0.
        
        self.t += 1

import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        
        # State variables: posterior mean and covariance for each arm
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.P = [np.eye(dim) * 5.0 for _ in range(n_arms)]
        
        # Adaptive noise parameters for non-stationarity
        self.innovation_ema = [0.0 for _ in range(n_arms)]
        self.innovation_decay = 0.8
        self.q_base = 0.05
        self.q_max = 0.5
        self.q_scale = 1.5
        
        # Observation noise
        self.sigma2 = 1.0
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = 145.0

    def _get_adaptive_q(self, arm_idx):
        # Adaptive process noise based on standardized innovation magnitude
        # High innovation -> high Q -> faster adaptation
        innovation = self.innovation_ema[arm_idx]
        q_val = self.q_base + (self.q_max - self.q_base) * np.tanh(self.q_scale * innovation)
        return np.eye(self.d) * q_val

    def select(self, context):
        # Deadline check
        if time.time() - self.start_time > self.deadline:
            # Fallback: just return the arm with highest current theta prediction
            context = np.asarray(context)
            preds = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(preds))
        
        context = np.asarray(context)
        
        # Thompson Sampling: sample from posterior for each arm
        samples = []
        for a in range(self.n):
            # Sample theta from N(theta[a], P[a])
            try:
                # Use Cholesky for numerical stability
                L = np.linalg.cholesky(self.P[a] + 1e-6 * np.eye(self.d))
                z = np.random.randn(self.d)
                theta_sample = self.theta[a] + L @ z
            except np.linalg.LinAlgError:
                # Fallback to standard normal if Cholesky fails
                std = np.sqrt(np.diag(self.P[a]))
                theta_sample = self.theta[a] + std * np.random.randn(self.d)
            
            # Compute reward prediction with sampled theta
            sample_reward = theta_sample @ context
            samples.append(sample_reward)
        
        return int(np.argmax(samples))

    def update(self, context, arm, reward):
        self.t += 1
        context = np.asarray(context)
        
        # 1. Calculate innovation (prediction error)
        pred = self.theta[arm] @ context
        innovation = reward - pred
        
        # 2. Update innovation EMA (standardized innovation magnitude)
        # Normalize by expected innovation variance
        S = context @ self.P[arm] @ context + self.sigma2
        S_std = np.sqrt(max(S, 1e-6))
        standardized_innovation = abs(innovation) / S_std
        self.innovation_ema[arm] = self.innovation_decay * self.innovation_ema[arm] + (1 - self.innovation_decay) * standardized_innovation
        
        # 3. Calculate Adaptive Q based on standardized innovation
        Q = self._get_adaptive_q(arm)
        
        # 4. Prediction Step: P_pred = P + Q
        P_pred = self.P[arm] + Q
        
        # 5. Update Step (Kalman filter)
        # Compute Kalman Gain: K = P_pred x^T (x^T P_pred x + sigma^2)^-1
        Px = P_pred @ context
        denom = context @ Px + self.sigma2
        K = Px / denom
        
        # Update Theta (posterior mean)
        self.theta[arm] = self.theta[arm] + K * innovation
        
        # Update P (posterior covariance) using Joseph form for stability
        I = np.eye(self.d)
        IKxT = I - np.outer(K, context)
        P_new = IKxT @ P_pred @ IKxT.T + np.outer(K, K) * self.sigma2
        
        # Ensure symmetry and positive semi-definiteness
        P_new = (P_new + P_new.T) / 2.0
        # Clip small negative eigenvalues
        eigvals, eigvecs = np.linalg.eigh(P_new)
        eigvals = np.clip(eigvals, 1e-8, None)
        P_new = eigvecs @ np.diag(eigvals) @ eigvecs.T
        P_new = (P_new + P_new.T) / 2.0
        
        self.P[arm] = P_new

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Per-Arm Kalman Filter with Context-Scaled Process Noise."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        # Initialize state covariance matrices (P) and mean vectors (theta)
        self.P = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        
        # Kalman filter parameters
        self.Q_base = 0.01       # Base process noise variance
        self.Q_lambda = 0.005    # Scaling factor for context norm
        self.R = 1.0             # Observation noise variance
        self.alpha = 0.995       # Forgetting factor for stability (optional blend)
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0  # 115s safety margin

    def select(self, context):
        # Check deadline
        if time.time() > self.deadline:
            # Return a random arm or best guess if we're out of time
            return int(np.random.randint(self.n))

        context = np.asarray(context, dtype=np.float64)
        scores = []
        
        for a in range(self.n):
            # Predict step: Inflate covariance to account for drift
            # Process noise Q is scaled by the norm of the context
            norm_sq = np.dot(context, context)
            Q = (self.Q_base + self.Q_lambda * norm_sq) * np.eye(self.d)
            
            # Update covariance: P = P + Q (for the current step)
            # Note: In a standard KF, P_pred = F P F^T + Q. Here F=I.
            P_pred = self.P[a] + Q
            
            # Compute score: Expected reward + Uncertainty bonus
            mean_reward = np.dot(self.theta[a], context)
            
            # Uncertainty: sqrt(context^T P_pred context)
            # We use the predicted covariance for the bonus to encourage exploration
            var = np.dot(context, P_pred @ context)
            if var < 0:
                var = 0.0
            std = np.sqrt(var)
            
            # UCB parameter
            beta = 1.0
            scores.append(mean_reward + beta * std)
            
            # Store P_pred temporarily? No, we update P in update().
            # However, the standard KF update uses the predicted P.
            # We need to store the predicted P or recompute it in update.
            # Since update is called immediately after select with the same context,
            # we can recompute Q and P_pred in update, or store it.
            # Storing is safer to ensure consistency if context changes slightly (it shouldn't).
            # Let's store P_pred for the selected arm only to save memory/time?
            # Actually, the update only happens for the selected arm.
            # But select computes for all arms. We only need P_pred for the selected arm in update.
            # So we don't need to store all of them. We will recompute in update.

        best_arm = int(np.argmax(scores))
        return best_arm

    def update(self, context, arm, reward):
        if time.time() > self.deadline:
            return

        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        # Recompute Process Noise
        norm_sq = np.dot(context, context)
        Q = (self.Q_base + self.Q_lambda * norm_sq) * np.eye(self.d)
        
        # Predicted Covariance
        P_pred = self.P[arm] + Q

        # Measurement Update (Kalman Gain)
        # K = P_pred H^T (H P_pred H^T + R)^-1
        # H = context (1 x d), so H P_pred H^T is scalar
        # Let S = H P_pred H^T + R = context^T P_pred context + R
        S = np.dot(context, P_pred @ context) + self.R
        if S < 1e-12:
            S = 1e-12
            
        # Kalman Gain vector
        K = (P_pred @ context) / S

        # Update Mean
        residual = reward - np.dot(self.theta[arm], context)
        self.theta[arm] = self.theta[arm] + K * residual

        # Update Covariance
        # P = (I - K H) P_pred
        # Since H is context (row vector), K H = K outer context
        I_KH = np.eye(self.d) - np.outer(K, context)
        self.P[arm] = I_KH @ P_pred
        
        # Optional: Symmetrize to prevent numerical drift
        self.P[arm] = (self.P[arm] + self.P[arm].T) / 2.0

        # Optional: Forgetting/Decay to handle non-stationarity more aggressively if needed
        # The Q term handles drift. If Q is too small, we might be stuck.
        # If Q is too large, we forget too fast.
        # The current design adapts Q to context norm.
# EVOLVE-BLOCK-END

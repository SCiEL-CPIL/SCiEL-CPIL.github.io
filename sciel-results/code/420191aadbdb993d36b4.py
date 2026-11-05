"""Non-stationary linear contextual bandit with global model and discontinuity detection."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Global Linear Bandit with Discontinuity Detection.
    
    Strategy:
    1. Global Model: Instead of maintaining a separate linear model for each arm,
       we maintain a single global model for the reward function. The context is
       augmented with one-hot arm indicators to allow the model to capture
       arm-specific biases and interactions. This increases the effective sample
       size per parameter and improves generalization, especially in early stages
       or when the number of rounds is limited.
    2. Discontinuity Detection (DD): We track the prediction error (residual)
       for each update. If the cumulative error exceeds a threshold (indicating
       a shift in the underlying weights), we reset the model to a prior state
       (identity matrix, zero vector) to quickly adapt to the new regime.
       This is more responsive to sudden changes than exponential forgetting,
       which can suffer from lag.
    3. Optimism: We use UCB (Upper Confidence Bound) for exploration.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.d_aug = dim + n_arms  # Augmented dimension: context + one-hot arm
        
        # Global Model Parameters
        self.lam = 1.0
        self.A = np.eye(self.d_aug) * self.lam
        self.b = np.zeros(self.d_aug)
        
        # Exploration and Detection Parameters
        self.alpha = 2.0  # Exploration constant
        self.error_threshold = 5.0  # Threshold for triggering reset
        self.error_window = 5  # Number of recent errors to consider for reset
        
        # State tracking
        self.t = 0
        self.recent_errors = []
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0
        
        # Cache for inverse A to speed up selection
        self.A_inv = np.eye(self.d_aug)

    def _augment_context(self, context, arm):
        """
        Augment context with one-hot encoding of the arm.
        context: (d,)
        arm: int
        returns: (d + n_arms,)
        """
        x_aug = np.zeros(self.d_aug)
        x_aug[:self.d] = context
        x_aug[self.d + arm] = 1.0
        return x_aug

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            # Fallback: pick arm with highest mean prediction
            scores = []
            for a in range(self.n):
                x_aug = self._augment_context(context, a)
                try:
                    theta = self.A_inv @ self.b
                    mu = x_aug @ theta
                    scores.append(mu)
                except:
                    scores.append(0.0)
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            x_aug = self._augment_context(context, a)
            try:
                # Mean prediction
                theta = self.A_inv @ self.b
                mu = x_aug @ theta
                
                # Variance
                # var = x^T A^{-1} x
                var = x_aug @ (self.A_inv @ x_aug)
                if var < 0:
                    var = 0.0
                uc = self.alpha * np.sqrt(var)
                
                scores.append(mu + uc)
            except np.linalg.LinAlgError:
                scores.append(0.0)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1
        
        # Deadline guard for heavy computation
        if time.time() > self.deadline:
            # Minimal update to avoid crashes, but skip complex detection
            x_aug = self._augment_context(context, arm)
            self.A += np.outer(x_aug, x_aug)
            self.b += reward * x_aug
            # Update inverse using Sherman-Morrison (approximate if failed)
            try:
                self.A_inv = np.linalg.inv(self.A)
            except:
                pass
            return

        # 1. Discontinuity Detection
        # Calculate prediction error for the current observation
        x_aug = self._augment_context(context, arm)
        try:
            theta = self.A_inv @ self.b
            pred = x_aug @ theta
            error = reward - pred
        except:
            error = 0.0

        self.recent_errors.append(error)
        if len(self.recent_errors) > self.error_window:
            self.recent_errors.pop(0)
        
        # Check if cumulative absolute error exceeds threshold
        if len(self.recent_errors) >= self.error_window:
            cumulative_error = np.sum(np.abs(self.recent_errors))
            if cumulative_error > self.error_threshold:
                # Reset model to prior
                self.A = np.eye(self.d_aug) * self.lam
                self.b = np.zeros(self.d_aug)
                self.A_inv = np.eye(self.d_aug)
                self.recent_errors = []
                # Optionally reduce alpha slightly after reset to stabilize?
                # Or keep it high to re-explore. Let's keep it high.
                self.alpha = 2.0

        # 2. Update Model
        # Standard linear update
        self.A += np.outer(x_aug, x_aug)
        self.b += reward * x_aug
        
        # Update inverse efficiently using Sherman-Morrison-Woodbury
        # A_new = A_old + x x^T
        # A_new^{-1} = A_old^{-1} - (A_old^{-1} x x^T A_old^{-1}) / (1 + x^T A_old^{-1} x)
        try:
            A_inv_x = self.A_inv @ x_aug
            denom = 1.0 + x_aug @ A_inv_x
            if denom > 1e-10:
                self.A_inv -= np.outer(A_inv_x, A_inv_x) / denom
            else:
                # Fallback to direct inversion if numerical instability
                self.A_inv = np.linalg.inv(self.A)
        except np.linalg.LinAlgError:
            self.A_inv = np.linalg.inv(self.A)

        # 3. Adaptive Alpha (Optional)
        # Decay alpha slightly as time passes, but maintain a floor
        # This helps shift from exploration to exploitation in stable periods
        self.alpha = max(1.0, self.alpha * 0.9995)
# EVOLVE-BLOCK-END

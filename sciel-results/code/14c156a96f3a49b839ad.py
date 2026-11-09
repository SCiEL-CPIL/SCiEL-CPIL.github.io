"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Global Non-stationarity Detection with Adaptive Forgetting.
    
    Strategy:
    1. Maintain a single global tracking error signal based on the residuals of the
       selected arm in the most recent rounds.
    2. This global error drives an adaptive forgetting factor (gamma) for all arms.
       When the environment is stable (low error), gamma is close to 1 (slow forgetting).
       When the environment is changing (high error), gamma decreases (fast forgetting).
    3. This is a structural change from local per-arm adaptation: it allows the policy
       to globally detect non-stationarity and adjust its memory window across all arms
       simultaneously, which is more robust to changes that affect multiple arms.
    4. Use a standard LinUCB-style update with the adaptive gamma.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0  # Safety margin

        # Hyperparameters
        self.alpha = 1.5
        self.reg = 1.0
        
        # Adaptive forgetting parameters
        self.gamma_min = 0.98
        self.gamma_max = 0.9995
        self.error_window = 50  # Number of recent residuals to track
        
        # Initialize models for each arm
        self.A_inv = [np.eye(self.d) / self.reg for _ in range(n_arms)]
        self.b = [np.zeros(self.d) for _ in range(n_arms)]
        
        # Track recent residuals for global non-stationarity detection
        self.residual_history = []
        self.current_gamma = self.gamma_max
        
        # Round counter
        self.round = 0

    def _compute_adaptive_gamma(self):
        """
        Compute the adaptive forgetting factor based on recent tracking error.
        High error -> low gamma (fast forgetting)
        Low error -> high gamma (slow forgetting)
        """
        if len(self.residual_history) < 10:
            return self.gamma_max
        
        # Use the most recent residuals
        recent_res = self.residual_history[-self.error_window:]
        mean_sq_error = np.mean(np.square(recent_res))
        
        # Map error to gamma using a smooth function
        # error_scale controls how sensitive the adaptation is
        error_scale = 0.5
        normalized_error = 1.0 / (1.0 + np.exp(-mean_sq_error / error_scale))
        
        # Interpolate between gamma_min and gamma_max
        self.current_gamma = self.gamma_min + (self.gamma_max - self.gamma_min) * (1.0 - normalized_error)
        
        return self.current_gamma

    def _update_model(self, A_inv, b, context, reward, gamma):
        """
        Update the Bayesian linear regression parameters with a forgetting factor.
        """
        x = context
        y = reward
        
        # Update b
        b_new = gamma * b + y * x
        
        # Update A_inv using Sherman-Morrison-Woodbury
        Ax = A_inv @ x
        denom = gamma + x @ Ax
        
        A_inv_new = (1.0 / gamma) * A_inv
        correction = np.outer(Ax, Ax) / denom
        A_inv_new -= (1.0 / gamma) * correction
        
        # Symmetrize to prevent numerical drift
        A_inv_new = (A_inv_new + A_inv_new.T) / 2.0
        
        return A_inv_new, b_new

    def _get_value_and_uncertainty(self, arm, context):
        """
        Compute the estimated mean and uncertainty for an arm.
        """
        x = context
        A_inv = self.A_inv[arm]
        b = self.b[arm]

        # Mean estimate
        theta = A_inv @ b
        mean_val = theta @ x
        
        # Uncertainty
        var = x @ A_inv @ x
        std = np.sqrt(var)
        
        return mean_val, std

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            return int(np.random.randint(self.n))
        
        # Compute adaptive gamma based on recent residuals
        gamma = self._compute_adaptive_gamma()
        
        scores = []
        for a in range(self.n):
            mean_val, uncertainty = self._get_value_and_uncertainty(a, context)
            # UCB-style score
            score = mean_val + self.alpha * uncertainty
            scores.append(score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Get the current gamma (already computed in select, but recompute for consistency)
        gamma = self._compute_adaptive_gamma()
        
        # Update the model for the selected arm
        self.A_inv[arm], self.b[arm] = self._update_model(
            self.A_inv[arm], self.b[arm], context, reward, gamma
        )
        
        # Track the residual for non-stationarity detection
        # Compute the predicted reward and the residual
        x = context
        theta = self.A_inv[arm] @ self.b[arm]
        predicted_reward = theta @ x
        residual = reward - predicted_reward
        
        self.residual_history.append(residual)
        if len(self.residual_history) > self.error_window:
            self.residual_history.pop(0)
        
        self.round += 1

# EVOLVE-BLOCK-END

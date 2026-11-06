"""Non-stationary linear contextual bandit with adaptive forgetting and exploration."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    LinUCB with adaptive forgetting factor and adaptive exploration.
    
    - Forgetting factor decreases (faster forgetting) when prediction error is high,
      to track non-stationary weight shifts.
    - Exploration coefficient increases when volatility is high.
    - Uses a single global model augmented with arm-specific bias terms for
      better sample efficiency and cross-arm correlation exploitation.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Global model: A_g (d x d), b_g (d,)
        self.A_g = np.eye(dim) * 1.0
        self.b_g = np.zeros(dim)
        
        # Arm-specific bias terms: one scalar per arm
        self.bias = np.zeros(n_arms)
        # Track bias variance for exploration
        self.bias_A = np.ones(n_arms)  # effective sample count per arm for bias
        self.bias_b = np.zeros(n_arms)  # sum of rewards per arm (for bias estimation)
        
        # Adaptive parameters
        self.base_gamma = 0.995       # base forgetting factor
        self.min_gamma = 0.90         # minimum (fastest forgetting)
        self.base_alpha = 1.0         # base exploration
        self.max_alpha = 3.0          # max exploration
        
        # Error tracking
        self.ema_error = 0.0          # exponential moving average of squared prediction error
        self.error_beta = 0.95        # EMA decay for error tracking
        self.n_obs = 0
        self._start_time = time.time()
        self._deadline = 115.0        # safety margin before 120s limit

    def _get_adaptive_params(self):
        """Compute adaptive gamma and alpha based on recent prediction error."""
        err = min(self.ema_error, 10.0)
        
        # Forgetting factor: lower error -> closer to base_gamma (slow forgetting)
        # Higher error -> lower gamma (faster forgetting)
        gamma = self.base_gamma - 0.04 * err
        gamma = max(self.min_gamma, min(self.base_gamma, gamma))
        
        # Exploration: higher error -> more exploration
        alpha = self.base_alpha + 0.5 * err
        alpha = min(self.max_alpha, alpha)
        
        return gamma, alpha

    def _predict(self, context, arm):
        """Predict reward for given context and arm."""
        context = np.asarray(context, dtype=np.float64)
        try:
            theta_g = np.linalg.solve(self.A_g, self.b_g)
            pred_g = theta_g @ context
        except np.linalg.LinAlgError:
            theta_g = np.linalg.lstsq(self.A_g, self.b_g, rcond=None)[0]
            pred_g = theta_g @ context
        
        # Bias estimate for this arm
        if self.bias_A[arm] > 0:
            bias_est = self.bias_b[arm] / self.bias_A[arm]
        else:
            bias_est = 0.0
        
        return pred_g + bias_est

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)
        
        # Deadline guard: if close to timeout, use pure exploitation
        if time.time() - self._start_time > self._deadline:
            scores = []
            for a in range(self.n):
                try:
                    theta_g = np.linalg.solve(self.A_g, self.b_g)
                    pred_g = theta_g @ context
                except np.linalg.LinAlgError:
                    theta_g = np.linalg.lstsq(self.A_g, self.b_g, rcond=None)[0]
                    pred_g = theta_g @ context
                bias_est = self.bias_b[a] / self.bias_A[a] if self.bias_A[a] > 0 else 0.0
                scores.append(pred_g + bias_est)
            return int(np.argmax(scores))
        
        gamma, alpha = self._get_adaptive_params()
        
        scores = []
        for a in range(self.n):
            try:
                theta_g = np.linalg.solve(self.A_g, self.b_g)
                pred_g = theta_g @ context
                
                # UCB bonus via Cholesky on global model
                L = np.linalg.cholesky(self.A_g)
                y = np.linalg.solve(L, context)
                bonus_global = alpha * np.sqrt(np.dot(y, y))
                
                # Bias uncertainty
                if self.bias_A[a] > 0:
                    bias_est = self.bias_b[a] / self.bias_A[a]
                    bias_var = 1.0 / self.bias_A[a]
                else:
                    bias_est = 0.0
                    bias_var = 1.0
                bonus_bias = alpha * np.sqrt(bias_var)
                
                scores.append(pred_g + bias_est + bonus_global + bonus_bias)
            except np.linalg.LinAlgError:
                theta_g = np.linalg.lstsq(self.A_g, self.b_g, rcond=None)[0]
                pred_g = theta_g @ context
                inv_A = np.linalg.pinv(self.A_g)
                bonus_global = alpha * np.sqrt(context @ inv_A @ context)
                
                if self.bias_A[a] > 0:
                    bias_est = self.bias_b[a] / self.bias_A[a]
                    bias_var = 1.0 / self.bias_A[a]
                else:
                    bias_est = 0.0
                    bias_var = 1.0
                bonus_bias = alpha * np.sqrt(bias_var)
                
                scores.append(pred_g + bias_est + bonus_global + bonus_bias)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        self.n_obs += 1
        
        # Compute prediction error for this arm
        pred = self._predict(context, arm)
        squared_error = (reward - pred) ** 2
        
        # Update EMA of error
        self.ema_error = self.error_beta * self.ema_error + (1 - self.error_beta) * squared_error
        
        # Get adaptive forgetting factor
        gamma, _ = self._get_adaptive_params()
        
        # Update global model with forgetting
        self.A_g = gamma * self.A_g + np.outer(context, context)
        self.b_g = gamma * self.b_g + reward * context
        
        # Update arm-specific bias with forgetting
        self.bias_A[arm] = gamma * self.bias_A[arm] + 1.0
        self.bias_b[arm] = gamma * self.bias_b[arm] + reward
# EVOLVE-BLOCK-END

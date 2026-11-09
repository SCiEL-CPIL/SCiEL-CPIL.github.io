"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Policy:
    """
    Multi-timescale sliding window LinUCB for non-stationary linear contextual bandits.
    
    Maintains two parallel estimators:
    - Short-window model: rapid detection of recent weight shifts
    - Long-window model: stable baseline estimation
    
    Selection uses a combined UCB that prioritizes short-window adaptation
    while using long-window for regularization/confidence.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Short window: fast adaptation
        self.gamma_s = 0.95  # forgetting factor for short window
        self.A_s = [np.eye(dim) for _ in range(n_arms)]
        self.b_s = [np.zeros(dim) for _ in range(n_arms)]
        
        # Long window: stable baseline
        self.gamma_l = 0.99  # forgetting factor for long window
        self.A_l = [np.eye(dim) for _ in range(n_arms)]
        self.b_l = [np.zeros(dim) for _ in range(n_arms)]
        
        # UCB parameters
        self.alpha_s = 1.0   # exploration weight for short window
        self.alpha_l = 0.5   # exploration weight for long window (regularization)
        
        # Adaptive exploration: track recent rewards to adjust alpha
        self.recent_rewards = []
        self.reward_window_size = 20
        self.base_alpha = 1.0

    def _get_theta(self, A, b):
        """Compute theta = A^{-1} b with regularization."""
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.zeros(self.d)

    def _ucb_score(self, context, A_s, b_s, A_l, b_l, alpha_s, alpha_l):
        """
        Compute combined UCB score using both short and long window models.
        
        The short window drives exploration for rapid adaptation,
        while the long window provides a stable baseline for regularization.
        """
        theta_s = self._get_theta(A_s, b_s)
        theta_l = self._get_theta(A_l, b_l)
        
        # Predicted reward from each model
        pred_s = theta_s @ context
        pred_l = theta_l @ context
        
        # Confidence widths from each model
        try:
            width_s = alpha_s * np.sqrt(context @ np.linalg.solve(A_s, context))
        except np.linalg.LinAlgError:
            width_s = alpha_s * np.sqrt(self.d)
        
        try:
            width_l = alpha_l * np.sqrt(context @ np.linalg.solve(A_l, context))
        except np.linalg.LinAlgError:
            width_l = alpha_l * np.sqrt(self.d)
        
        # Combined estimate: blend predictions, with short window weighted more
        # for responsiveness to non-stationarity
        blend = 0.7 * pred_s + 0.3 * pred_l
        
        # Combined confidence: use max of the two widths to be conservative,
        # but weight short window more for adaptation
        combined_width = max(width_s, width_l)
        
        return blend + combined_width

    def _adaptive_alpha(self):
        """
        Adapt exploration rate based on recent reward variance.
        Higher variance suggests more non-stationarity -> more exploration.
        """
        if len(self.recent_rewards) < 5:
            return self.base_alpha
        
        recent = self.recent_rewards[-self.reward_window_size:]
        variance = np.var(recent)
        
        # Scale alpha with variance: more variance -> more exploration
        # Normalize variance to get a reasonable alpha range
        alpha = self.base_alpha * (1.0 + 0.5 * np.clip(variance, 0, 2))
        return np.clip(alpha, 0.5, 3.0)

    def select(self, context):
        """Select arm based on combined multi-timescale UCB."""
        alpha_s = self._adaptive_alpha()
        
        scores = []
        for a in range(self.n):
            score = self._ucb_score(
                context,
                self.A_s[a], self.b_s[a],
                self.A_l[a], self.b_l[a],
                alpha_s, self.alpha_l
            )
            scores.append(score)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        """Update both short and long window models."""
        # Update short window (fast forgetting)
        self.A_s[arm] = self.gamma_s * self.A_s[arm] + np.outer(context, context)
        self.b_s[arm] = self.gamma_s * self.b_s[arm] + reward * context
        
        # Update long window (slow forgetting)
        self.A_l[arm] = self.gamma_l * self.A_l[arm] + np.outer(context, context)
        self.b_l[arm] = self.gamma_l * self.b_l[arm] + reward * context
        
        # Track recent rewards for adaptive exploration
        self.recent_rewards.append(reward)
        if len(self.recent_rewards) > self.reward_window_size * 2:
            self.recent_rewards = self.recent_rewards[-self.reward_window_size:]
# EVOLVE-BLOCK-END

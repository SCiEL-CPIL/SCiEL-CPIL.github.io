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
        self.gamma_s = 0.93  # faster forgetting for quicker adaptation
        self.A_s = [np.eye(dim) * 0.1 for _ in range(n_arms)]
        self.b_s = [np.zeros(dim) for _ in range(n_arms)]
        
        # Long window: stable baseline
        self.gamma_l = 0.98  # slower forgetting for stability
        self.A_l = [np.eye(dim) * 0.1 for _ in range(n_arms)]
        self.b_l = [np.zeros(dim) for _ in range(n_arms)]
        
        # UCB parameters
        self.alpha_s = 1.2   # higher exploration weight for short window
        self.alpha_l = 0.3   # lower exploration weight for long window (regularization only)
        
        # Adaptive exploration: track recent rewards to adjust alpha
        self.recent_rewards = []
        self.reward_window_size = 15
        self.base_alpha = 1.2
        
        # Track model disagreement for adaptive exploration
        self.disagreement_history = []
        self.disagreement_window = 10

    def _get_theta(self, A, b):
        """Compute theta = A^{-1} b with regularization."""
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.zeros(self.d)

    def _ucb_score(self, context, A_s, b_s, A_l, b_l, alpha_s, alpha_l):
        """
        Compute UCB score prioritizing short-window model for rapid adaptation.
        
        The short window drives both prediction and exploration for rapid adaptation,
        while the long window provides only regularization through its confidence width.
        """
        theta_s = self._get_theta(A_s, b_s)
        theta_l = self._get_theta(A_l, b_l)
        
        # Predicted reward from short window (primary)
        pred_s = theta_s @ context
        
        # Confidence width from short window (primary exploration)
        try:
            width_s = alpha_s * np.sqrt(context @ np.linalg.solve(A_s, context))
        except np.linalg.LinAlgError:
            width_s = alpha_s * np.sqrt(self.d)
        
        # Confidence width from long window (regularization only)
        try:
            width_l = alpha_l * np.sqrt(context @ np.linalg.solve(A_l, context))
        except np.linalg.LinAlgError:
            width_l = alpha_l * np.sqrt(self.d)
        
        # Use short window prediction as primary estimate
        # Add a small regularization term from long window to prevent overfitting
        pred_l = theta_l @ context
        regularized_pred = 0.8 * pred_s + 0.2 * pred_l
        
        # Primary UCB from short window, plus small regularization from long window
        score = regularized_pred + width_s + 0.1 * width_l
        
        return score

    def _adaptive_alpha(self):
        """
        Adapt exploration rate based on model disagreement and reward variance.
        Higher disagreement or variance suggests more non-stationarity -> more exploration.
        """
        if len(self.recent_rewards) < 5:
            return self.base_alpha
        
        recent = self.recent_rewards[-self.reward_window_size:]
        variance = np.var(recent)
        
        # Use disagreement history if available
        if len(self.disagreement_history) >= 3:
            recent_disagreement = np.mean(self.disagreement_history[-self.disagreement_window:])
            # Combine variance and disagreement signals
            alpha = self.base_alpha * (1.0 + 0.3 * np.clip(variance, 0, 2) + 0.5 * np.clip(recent_disagreement, 0, 1))
        else:
            alpha = self.base_alpha * (1.0 + 0.3 * np.clip(variance, 0, 2))
        
        return np.clip(alpha, 0.8, 2.5)

    def select(self, context):
        """Select arm based on short-window UCB with adaptive exploration."""
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
        
        best_arm = int(np.argmax(scores))
        
        # Track model disagreement for adaptive exploration
        if len(self.recent_rewards) >= 3:
            # Compute prediction difference between short and long window for selected arm
            theta_s = self._get_theta(self.A_s[best_arm], self.b_s[best_arm])
            theta_l = self._get_theta(self.A_l[best_arm], self.b_l[best_arm])
            pred_diff = abs(theta_s @ context - theta_l @ context)
            self.disagreement_history.append(pred_diff)
            if len(self.disagreement_history) > self.disagreement_window * 2:
                self.disagreement_history = self.disagreement_history[-self.disagreement_window:]
        
        return best_arm

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

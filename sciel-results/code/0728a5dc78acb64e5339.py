"""Non-stationary linear contextual bandit policy with per-arm adaptive forgetting."""
import numpy as np
import time


class Policy:
    """
    LinUCB with per-arm adaptive forgetting factor and robust scaling.
    
    Key improvements over the seed:
    1. Per-arm adaptive lambda (forgetting rate) based on reward variance (MAD-based).
    2. Smoother exploration bonus decay.
    3. Deadline guard to avoid timeout.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.deadline = time.time() + 115  # Safety margin before 120s kill

        # Per-arm adaptive parameters
        self.gamma = np.full(n_arms, 0.99)  # Base forgetting factor
        self.gamma_min = 0.95
        self.gamma_max = 0.999

        # Per-arm reward tracking for adaptive lambda
        self.rewards = [np.array([], dtype=np.float64) for _ in range(n_arms)]
        self.reward_window = 20  # Window size for MAD computation

        # Regularization
        self.lam = 1.0

        # Per-arm design matrices and reward vectors
        self.A = [self.lam * np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

        # Exploration coefficient
        self.alpha = 1.0
        self.alpha_decay = 0.9999  # Slow decay

    def _compute_adaptive_gamma(self, arm):
        """Compute adaptive forgetting factor based on reward variability."""
        if len(self.rewards[arm]) < 5:
            return self.gamma[arm]

        # Use recent rewards for MAD-based robust scaling
        recent = self.rewards[arm][-self.reward_window:]
        median = np.median(recent)
        mad = np.median(np.abs(recent - median))

        if mad < 1e-6:
            # Low variability -> can forget slower
            self.gamma[arm] = min(self.gamma_max, self.gamma[arm] * 1.001)
        else:
            # High variability -> forget faster
            # Scale gamma inversely with MAD
            scale = 1.0 / (1.0 + mad * 0.5)
            self.gamma[arm] = max(self.gamma_min, min(self.gamma_max, self.gamma[arm] * scale))

        return self.gamma[arm]

    def _get_theta(self, arm):
        """Compute theta estimate for an arm."""
        inv_A = np.linalg.inv(self.A[arm])
        return inv_A @ self.b[arm]

    def _ucb_score(self, arm, context):
        """Compute UCB score for an arm."""
        inv_A = np.linalg.inv(self.A[arm])
        theta = inv_A @ self.b[arm]
        mean = theta @ context
        # Exploration bonus with adaptive scaling
        variance = context @ inv_A @ context
        bonus = self.alpha * np.sqrt(variance)
        return mean + bonus

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            # Return a default arm to avoid timeout
            return 0

        self.t += 1
        context = np.asarray(context, dtype=np.float64)

        scores = []
        for a in range(self.n):
            try:
                score = self._ucb_score(a, context)
            except np.linalg.LinAlgError:
                score = 0.0
            scores.append(score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Deadline guard
        if time.time() > self.deadline:
            return

        context = np.asarray(context, dtype=np.float64)

        # Update reward history for adaptive gamma
        self.rewards[arm] = np.append(self.rewards[arm], reward)
        if len(self.rewards[arm]) > self.reward_window * 2:
            self.rewards[arm] = self.rewards[arm][-self.reward_window:]

        # Compute adaptive forgetting factor
        gamma = self._compute_adaptive_gamma(arm)

        # Update design matrix and reward vector with forgetting
        self.A[arm] = gamma * self.A[arm] + np.outer(context, context)
        self.b[arm] = gamma * self.b[arm] + reward * context

        # Ensure A remains positive definite by adding small regularization
        self.A[arm] += 1e-6 * np.eye(self.d)

        # Decay exploration coefficient slowly
        self.alpha = max(0.5, self.alpha * self.alpha_decay)

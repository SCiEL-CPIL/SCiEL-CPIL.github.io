"""Non-stationary linear contextual bandit with sliding window and Thompson Sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import collections


class Policy:
    """
    Non-stationary linear contextual bandit using:
    1. Adaptive sliding window (EWMA-based non-stationarity detection).
    2. Thompson Sampling with Cholesky-based sampling.
    3. Global context-augmented model blended with per-arm models.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0

        # Adaptive window parameters
        self.min_window = 10
        self.max_window = 60
        self.window_size = 30
        self.ewma_alpha = 0.1  # learning rate for error tracking
        self.ewma_error = 0.0
        self.ewma_error_sq = 0.0

        # Ridge regularization
        self.lam = 1.0

        # Per-arm sliding window
        self.queues = [collections.deque(maxlen=self.max_window) for _ in range(n_arms)]
        self.A = [self.lam * np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

        # Global model: feature dim = dim + n_arms (context + one-hot arm)
        self.d_global = dim + n_arms
        self.A_global = self.lam * np.eye(self.d_global)
        self.b_global = np.zeros(self.d_global)
        self.global_window = collections.deque(maxlen=self.max_window)

        # Precompute Cholesky caches
        self._chol_cache = [None] * n_arms
        self._chol_cache_global = None

    def _build_augmented_feature(self, arm, context):
        """Create augmented feature: [context, one-hot arm]."""
        feat = np.zeros(self.d_global)
        feat[:self.d] = context
        feat[self.d + arm] = 1.0
        return feat

    def _update_arm_stats(self, arm, context, reward):
        """Update per-arm sliding window statistics."""
        q = self.queues[arm]
        if len(q) == self.window_size:
            old_ctx, old_reward = q[0]
            self.A[arm] -= np.outer(old_ctx, old_ctx)
            self.b[arm] -= old_reward * old_ctx
        q.append((context, reward))
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
        self._chol_cache[arm] = None

    def _update_global_stats(self, arm, context, reward):
        """Update global sliding window statistics."""
        feat = self._build_augmented_feature(arm, context)
        if len(self.global_window) == self.window_size:
            old_feat, old_reward = self.global_window[0]
            self.A_global -= np.outer(old_feat, old_feat)
            self.b_global -= old_reward * old_feat
        self.global_window.append((feat, reward))
        self.A_global += np.outer(feat, feat)
        self.b_global += reward * feat
        self._chol_cache_global = None

    def _adapt_window_size(self, context, arm, reward):
        """Adapt window size based on prediction error."""
        # Compute prediction error for the pulled arm
        try:
            A = self.A[arm]
            b = self.b[arm]
            theta = np.linalg.solve(A, b)
            pred = theta @ context
            error = reward - pred
        except np.linalg.LinAlgError:
            error = 0.0

        # Update EWMA of error
        self.ewma_error = (1 - self.ewma_alpha) * self.ewma_error + self.ewma_alpha * error
        self.ewma_error_sq = (1 - self.ewma_alpha) * self.ewma_error_sq + self.ewma_alpha * (error ** 2)

        # Estimate variance
        var = max(self.ewma_error_sq - self.ewma_error ** 2, 1e-6)
        std = np.sqrt(var)

        # Adapt window: larger error -> smaller window (faster adaptation)
        # Normalize error relative to a scale
        if std > 1.0:
            self.window_size = max(self.min_window, self.window_size - 2)
        elif std < 0.3:
            self.window_size = min(self.max_window, self.window_size + 1)

        # Ensure queues reflect new window size
        for a in range(self.n):
            q = self.queues[a]
            while len(q) > self.window_size:
                old_ctx, old_reward = q.popleft()
                self.A[a] -= np.outer(old_ctx, old_ctx)
                self.b[a] -= old_reward * old_ctx

        # Trim global window
        while len(self.global_window) > self.window_size:
            old_feat, old_reward = self.global_window.popleft()
            self.A_global -= np.outer(old_feat, old_feat)
            self.b_global -= old_reward * old_feat

    def _get_theta_and_covariance(self, arm):
        """Compute posterior mean and covariance for an arm."""
        A = self.A[arm]
        b = self.b[arm]
        try:
            theta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            theta = np.zeros(self.d)
        try:
            cov = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            cov = np.eye(self.d)
        return theta, cov

    def _get_global_theta_and_covariance(self):
        """Compute posterior mean and covariance for the global model."""
        A = self.A_global
        b = self.b_global
        try:
            theta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            theta = np.zeros(self.d_global)
        try:
            cov = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            cov = np.eye(self.d_global)
        return theta, cov

    def select(self, context):
        """Select an arm using blended Thompson Sampling."""
        context = np.asarray(context, dtype=np.float64)

        # Deadline check: greedy fallback
        if time.time() > self.deadline:
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                theta, _ = self._get_theta_and_covariance(a)
                score = theta @ context
                if score > best_score:
                    best_score = score
                    best_arm = a
            return int(best_arm)

        # Thompson Sampling with blending
        samples = []
        for a in range(self.n):
            theta, cov = self._get_theta_and_covariance(a)
            sampled_theta = theta + np.random.multivariate_normal(np.zeros(self.d), cov)
            pred_reward = sampled_theta @ context
            samples.append(pred_reward)

        # Blend with global model
        try:
            theta_g, cov_g = self._get_global_theta_and_covariance()
            sampled_theta_g = theta_g + np.random.multivariate_normal(np.zeros(self.d_global), cov_g)
            global_preds = []
            for a in range(self.n):
                feat = self._build_augmented_feature(a, context)
                global_preds.append(sampled_theta_g @ feat)
            # Blend: 70% per-arm, 30% global
            blended = 0.7 * np.array(samples) + 0.3 * np.array(global_preds)
            samples = blended.tolist()
        except Exception:
            pass

        return int(np.argmax(samples))

    def update(self, context, arm, reward):
        """Update the policy with a new observation."""
        context = np.asarray(context, dtype=np.float64)
        self.t += 1
        self._adapt_window_size(context, arm, reward)
        self._update_arm_stats(arm, context, reward)
        self._update_global_stats(arm, context, reward)
# EVOLVE-BLOCK-END

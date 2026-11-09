"""Non-stationary linear contextual bandit policy with multi-timescale adaptive forgetting and UCB exploration."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Multi-timescale non-stationary LinUCB with:
    - Short-window model for rapid detection of weight shifts
    - Long-window model for stable baseline estimation
    - Adaptive blending based on detected non-stationarity
    - Efficient linear algebra via solve instead of explicit inverse
    - Deadline guard for safety
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.dim = dim

        # Short-window model (fast adaptation)
        self.A_short = [np.eye(dim) * 0.05 for _ in range(n_arms)]
        self.b_short = [np.zeros(dim) for _ in range(n_arms)]
        self.gamma_short = 0.92  # Fast forgetting

        # Long-window model (stable baseline)
        self.A_long = [np.eye(dim) * 0.1 for _ in range(n_arms)]
        self.b_long = [np.zeros(dim) for _ in range(n_arms)]
        self.gamma_long = 0.995  # Slow forgetting

        # Exploration parameter: adaptive
        self.alpha_base = 1.8
        self.alpha = 1.8
        self.alpha_decay = 0.9997

        # Tracking for non-stationarity detection
        self.reward_history = [[] for _ in range(n_arms)]
        self.max_history_len = 50

        # Per-arm EMA of rewards to detect shifts
        self.reward_ema = [0.0 for _ in range(n_arms)]
        self.reward_ema_alpha = 0.3

        # Per-arm reward change tracking for drift detection
        self.prev_reward = [None for _ in range(n_arms)]
        self.reward_change_ema = [0.0 for _ in range(n_arms)]

        # Per-arm count for adaptive alpha
        self.arm_counts = [0] * n_arms

        # Blending weight: 0 = fully long-window, 1 = fully short-window
        self.blend_weight = 0.0
        self.blend_target = 0.0

        # Time tracking
        self.t = 0
        self.start_time = time.time()
        self.deadline = 145.0  # Safety margin before 150s limit

    def _is_time_up(self):
        """Check if we're approaching the deadline."""
        return (time.time() - self.start_time) > self.deadline

    def _compute_ucb_scores(self, A_list, b_list, context, alpha):
        """Compute UCB scores for a given set of LinUCB models."""
        scores = np.zeros(self.n)
        max_unc = 0.0
        for a in range(self.n):
            A = A_list[a] + np.eye(self.d) * 1e-6
            try:
                theta = np.linalg.solve(A, b_list[a])
                y = np.linalg.solve(A, context)
                unc_sq = context @ y
                unc = np.sqrt(max(0.0, unc_sq))
            except np.linalg.LinAlgError:
                theta = np.zeros(self.d)
                unc = 0.0

            mean = theta @ context
            scores[a] = mean + alpha * unc
            if unc > max_unc:
                max_unc = unc
        return scores, max_unc

    def select(self, context):
        self.t += 1

        # Deadline guard: if we're close to timeout, use greedy with best known arm
        if self._is_time_up():
            # Use the arm with highest recent average reward
            best_arm = 0
            best_mean = -np.inf
            for a in range(self.n):
                if self.reward_history[a]:
                    mean_r = np.mean(self.reward_history[a][-10:])
                    if mean_r > best_mean:
                        best_mean = mean_r
                        best_arm = a
            return int(best_arm)

        # Adaptive blending: detect non-stationarity to adjust blend weight
        max_change = 0.0
        for a in range(self.n):
            if self.reward_change_ema[a] > max_change:
                max_change = self.reward_change_ema[a]

        # Higher max change -> higher blend weight (more short-window)
        # Use smooth sigmoid-like mapping
        blend_target = 1.0 / (1.0 + np.exp(-8.0 * (max_change - 0.15)))
        self.blend_target = max(0.0, min(1.0, blend_target))
        # Smoothly adapt blend weight
        self.blend_weight = 0.8 * self.blend_weight + 0.2 * self.blend_target

        # Adaptive alpha: decay over time but boost if uncertainty is high
        self.alpha = max(0.5, self.alpha * self.alpha_decay)

        context = np.asarray(context, dtype=np.float64)

        # Compute scores from both models
        scores_short, max_unc_short = self._compute_ucb_scores(self.A_short, self.b_short, context, self.alpha)
        scores_long, max_unc_long = self._compute_ucb_scores(self.A_long, self.b_long, context, self.alpha)

        # Blend scores: short-window prioritized for adaptation, long-window for stability
        # Use convex combination with adaptive weights
        w = self.blend_weight
        scores = w * scores_short + (1 - w) * scores_long

        # If overall uncertainty is high, boost exploration
        max_unc = w * max_unc_short + (1 - w) * max_unc_long
        if max_unc > 1.0:
            self.alpha = min(3.0, self.alpha * 1.1)

        # Clip to reasonable range to avoid extreme values
        scores = np.clip(scores, -10.0, 10.0)

        # Small random tie-breaking for stability
        if np.allclose(scores, scores[0]):
            return int(np.random.randint(self.n))

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        # Update per-arm reward change tracking
        if self.prev_reward[arm] is not None:
            change = abs(reward - self.prev_reward[arm])
            self.reward_change_ema[arm] = 0.4 * change + 0.6 * self.reward_change_ema[arm]
        self.prev_reward[arm] = reward

        # Update EMA
        if self.reward_history[arm]:
            self.reward_ema[arm] = self.reward_ema_alpha * reward + (1 - self.reward_ema_alpha) * self.reward_ema[arm]
        else:
            self.reward_ema[arm] = reward

        # Update short-window model (fast adaptation)
        self.A_short[arm] = self.gamma_short * self.A_short[arm] + np.outer(context, context)
        self.b_short[arm] = self.gamma_short * self.b_short[arm] + reward * context

        # Update long-window model (stable baseline)
        self.A_long[arm] = self.gamma_long * self.A_long[arm] + np.outer(context, context)
        self.b_long[arm] = self.gamma_long * self.b_long[arm] + reward * context

        # Maintain reward history for volatility detection
        self.reward_history[arm].append(reward)
        if len(self.reward_history[arm]) > self.max_history_len:
            self.reward_history[arm].pop(0)

        # Increment arm count
        self.arm_counts[arm] += 1

        # Periodic renormalization to prevent numerical issues
        if self.t % 100 == 0:
            for a in range(self.n):
                # Check short-window model
                diag_short = np.diag(self.A_short[a])
                min_diag_short = np.min(diag_short)
                if min_diag_short < 1e-2:
                    self.A_short[a] += np.eye(self.d) * (1e-2 - min_diag_short + 1e-6)
                # Check long-window model
                diag_long = np.diag(self.A_long[a])
                min_diag_long = np.min(diag_long)
                if min_diag_long < 1e-2:
                    self.A_long[a] += np.eye(self.d) * (1e-2 - min_diag_long + 1e-6)
# EVOLVE-BLOCK-END

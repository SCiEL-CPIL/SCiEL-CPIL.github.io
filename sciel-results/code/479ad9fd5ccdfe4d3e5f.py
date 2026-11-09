"""Non-stationary linear contextual bandit policy with adaptive forgetting and UCB exploration."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary LinUCB with:
    - Adaptive forgetting factor based on reward volatility
    - Per-arm exponential moving average for baseline detection
    - Time-aware regularization to handle drift
    - Efficient linear algebra via solve instead of explicit inverse
    - Deadline guard for safety
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.dim = dim

        # Core matrices for LinUCB
        # Smaller initial regularization → higher initial uncertainty → better early exploration
        self.A = [np.eye(dim) * 0.01 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

        # Forgetting factor: per-arm, start moderate, adapt based on perceived non-stationarity
        self.base_gamma = 0.99
        self.gamma = 0.99
        self.per_arm_gamma = [0.99] * n_arms
        # Per-arm drift score: smoothed measure of how much rewards are shifting
        self.drift_score = [0.0] * n_arms

        # Exploration parameter: adaptive
        self.alpha_base = 1.8
        self.alpha = 1.8
        self.alpha_decay = 0.9997

        # Tracking for non-stationarity detection
        self.reward_history = [[] for _ in range(n_arms)]
        self.max_history_len = 30

        # Per-arm EMA of rewards to detect shifts (faster response)
        self.reward_ema = [0.0 for _ in range(n_arms)]
        self.reward_ema_alpha = 0.40

        # Per-arm reward change tracking for drift detection (faster response)
        self.prev_reward = [None for _ in range(n_arms)]
        self.reward_change_ema = [0.0 for _ in range(n_arms)]
        # Recent reward window for more robust drift detection
        self.recent_rewards = [[] for _ in range(n_arms)]
        self.recent_window = 12

        # Per-arm count for adaptive alpha
        self.arm_counts = [0] * n_arms

        # Time tracking
        self.t = 0
        self.start_time = time.time()
        self.deadline = 145.0  # Safety margin before 150s limit

    def _is_time_up(self):
        """Check if we're approaching the deadline."""
        return (time.time() - self.start_time) > self.deadline

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

        # Adaptive forgetting: per-arm drift detection with smoother response
        # Update per-arm gamma based on each arm's own drift signal
        for a in range(self.n):
            # Combine change-based drift with window-based variance
            change = self.reward_change_ema[a]
            normalized_change = min(change / 0.25, 1.0)

            # Also check variance in recent rewards as a drift signal
            window_var_signal = 0.0
            if len(self.recent_rewards[a]) >= 4:
                recent = np.array(self.recent_rewards[a])
                # Standardized deviation from mean of recent window
                if len(recent) >= 4:
                    std = np.std(recent)
                    mean_r = np.mean(recent)
                    # If rewards are spread out, it may indicate non-stationarity
                    window_var_signal = min(std / (0.15 + abs(mean_r) * 0.1), 1.0)

            combined_drift = 0.6 * normalized_change + 0.4 * window_var_signal
            self.drift_score[a] = 0.7 * self.drift_score[a] + 0.3 * combined_drift

            # Higher drift -> lower gamma (faster forgetting)
            gamma_target = 0.999 - 0.07 * (self.drift_score[a] / (1.0 + self.drift_score[a]))
            gamma_target = max(0.88, min(0.999, gamma_target))
            # Smoothly adapt per-arm gamma
            self.per_arm_gamma[a] = 0.70 * self.per_arm_gamma[a] + 0.30 * gamma_target

        # Global gamma is the minimum (most aggressive forgetting) for safety
        self.gamma = min(self.per_arm_gamma)

        # Adaptive alpha: decay over time but boost if uncertainty is high
        self.alpha = max(0.5, self.alpha * self.alpha_decay)

        context = np.asarray(context, dtype=np.float64)

        scores = np.zeros(self.n)
        max_unc = 0.0
        for a in range(self.n):
            # Use solve for numerical stability and efficiency
            A = self.A[a] + np.eye(self.d) * 1e-6
            try:
                # Solve A @ theta = b instead of computing inverse
                theta = np.linalg.solve(A, self.b[a])
                # For uncertainty, we need x^T A^{-1} x
                # Solve A @ y = x, then uncertainty = sqrt(x^T y)
                y = np.linalg.solve(A, context)
                unc_sq = context @ y
                unc = np.sqrt(max(0.0, unc_sq))
            except np.linalg.LinAlgError:
                theta = np.zeros(self.d)
                unc = 0.0

            mean = theta @ context
            scores[a] = mean + self.alpha * unc
            if unc > max_unc:
                max_unc = unc

        # If overall uncertainty is high, boost exploration
        if max_unc > 0.8:
            self.alpha = min(3.0, self.alpha * 1.15)

        # Clip to reasonable range to avoid extreme values
        scores = np.clip(scores, -10.0, 10.0)

        # Small random tie-breaking for stability
        if np.allclose(scores, scores[0]):
            return int(np.random.randint(self.n))

        return int(np.argmax(scores))

    def _adaptive_update_params(self, arm, reward, context):
        """Update internal parameters based on observed reward and context."""
        pass

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        # Update per-arm reward change tracking with faster adaptation
        if self.prev_reward[arm] is not None:
            change = abs(reward - self.prev_reward[arm])
            # Faster EMA to respond to recent changes
            self.reward_change_ema[arm] = 0.55 * change + 0.45 * self.reward_change_ema[arm]
        self.prev_reward[arm] = reward

        # Update EMA with faster response to shifts
        if self.reward_history[arm]:
            self.reward_ema[arm] = 0.40 * reward + 0.60 * self.reward_ema[arm]
        else:
            self.reward_ema[arm] = reward

        # Maintain recent reward window for drift detection
        self.recent_rewards[arm].append(reward)
        if len(self.recent_rewards[arm]) > self.recent_window:
            self.recent_rewards[arm].pop(0)

        # Detect if we're in a shift: compare current reward to recent EMA
        # If reward is far from EMA, boost forgetting for this specific arm
        ema_diff = abs(reward - self.reward_ema[arm])
        if ema_diff > 0.35:
            # Strong signal of non-stationarity: temporarily boost forgetting for this arm
            local_gamma = max(0.87, self.per_arm_gamma[arm] - 0.05)
        else:
            local_gamma = self.per_arm_gamma[arm]

        # Apply forgetting and update with per-arm gamma
        self.A[arm] = local_gamma * self.A[arm] + np.outer(context, context)
        self.b[arm] = local_gamma * self.b[arm] + reward * context

        # Maintain reward history for volatility detection
        self.reward_history[arm].append(reward)
        if len(self.reward_history[arm]) > self.max_history_len:
            self.reward_history[arm].pop(0)

        # Update drift score with current observation
        if self.prev_reward[arm] is not None:
            change = abs(reward - self.prev_reward[arm])
            normalized_change = min(change / 0.25, 1.0)
            window_var_signal = 0.0
            if len(self.recent_rewards[arm]) >= 4:
                recent = np.array(self.recent_rewards[arm])
                std = np.std(recent)
                mean_r = np.mean(recent)
                window_var_signal = min(std / (0.15 + abs(mean_r) * 0.1), 1.0)
            combined_drift = 0.6 * normalized_change + 0.4 * window_var_signal
            self.drift_score[arm] = 0.7 * self.drift_score[arm] + 0.3 * combined_drift

        # Increment arm count
        self.arm_counts[arm] += 1

        # Periodic renormalization to prevent numerical issues (cheaper check)
        if self.t % 100 == 0:
            for a in range(self.n):
                # Quick check: if diagonal elements are too small, add regularization
                diag = np.diag(self.A[a])
                min_diag = np.min(diag)
                if min_diag < 1e-2:
                    self.A[a] += np.eye(self.d) * (1e-2 - min_diag + 1e-6)
# EVOLVE-BLOCK-END

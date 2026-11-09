"""Non-stationary linear contextual bandit: model-and-optimize (global ridge value
model fit directly against the reward objective) with self-tuning forgetting."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Global ridge value model (fit to reward) + per-arm ridge models for
    exploration, with drift-adaptive forgetting and a hard wall-clock guard."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t0 = time.time()
        self.deadline = 115.0          # safety margin vs. the 150 s kill
        self.elapsed = 0.0
        self.deadline_hit = False

        # Global linear value model fit DIRECTLY against the reward objective.
        self.ridge = 1.0
        self.G = np.eye(dim) * self.ridge          # gram (covariance) of contexts
        self.f = np.zeros(dim)                      # context-reward cross term
        self.theta = np.zeros(dim)                  # value model: context -> reward
        self.theta_var = 1.0                        # (regularized) variance of theta

        # Per-arm ridge models for the bandit's exploration term.
        self.A = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

        self.forgetting = 0.995      # base forget rate (tuned from inspiration 1)
        self.f_min = 0.90
        self.f_max = 0.998
        self.alpha = 1.0             # exploration coefficient
        self.last_pred = 0.0
        self.last_reward = 0.0
        self.count = 0

        # Change-detection state for adaptive forgetting
        self.err_history = []
        self.err_window = 15
        self.change_threshold = 1.3

    def _check_deadline(self):
        if self.deadline_hit:
            return
        now = time.time()
        self.elapsed = now - self.t0
        if self.elapsed >= self.deadline:
            self.deadline_hit = True

    def select(self, context):
        self._check_deadline()
        ctx = np.asarray(context, dtype=float)

        if self.deadline_hit:
            # Pure exploitation: return the argmax of the current value model.
            return int(np.argmax(self.theta @ ctx))

        # Value model (the answer) used as the exploitation baseline.
        self._update_global()

        scores = np.empty(self.n)
        for a in range(self.n):
            inv = np.linalg.inv(self.A[a])
            theta_a = inv @ self.b[a]
            scores[a] = theta_a @ ctx + self.alpha * np.sqrt(ctx @ inv @ ctx)

        # Blend the global value model with the per-arm LinUCB score
        blend = 0.5 * (self.theta @ ctx + self.theta_var * np.linalg.norm(ctx) ** 2)
        base = 0.5 * scores + blend
        return int(np.argmax(base))

    def update(self, context, arm, reward):
        self._check_deadline()
        ctx = np.asarray(context, dtype=float)
        reward = float(reward)

        if self.deadline_hit:
            return  # stop heavy work; still a no-op update (interface preserved)

        # --- drift-adaptive forgetting with change detection ---
        if self.count >= 1:
            err = abs(reward - self.last_pred)
            self.err_history.append(err)
            if len(self.err_history) > self.err_window:
                self.err_history.pop(0)
            self.last_reward = reward

            # Detect change: if recent error is significantly higher than baseline
            if len(self.err_history) >= 5:
                recent_err = np.mean(self.err_history[-5:])
                baseline_err = np.mean(self.err_history[:-5]) if len(self.err_history) > 5 else np.mean(self.err_history)
                change_detected = (recent_err > self.change_threshold * max(baseline_err, 0.01))
            else:
                change_detected = False

            # Large error or detected change -> accelerate forgetting
            if change_detected:
                target = self.f_min  # aggressive forgetting during change
            else:
                target = self.f_min + (self.f_max - self.f_min) * (0.3 ** (err / 1.5 + 1e-6))

            self.forgetting += 0.15 * (target - self.forgetting)
            self.forgetting = min(max(self.forgetting, self.f_min), self.f_max)
        else:
            self.last_reward = reward

        # --- per-arm ridge update with forgetting ---
        f = self.forgetting
        self.A[arm] = f * self.A[arm] + np.outer(ctx, ctx)
        self.b[arm] = f * self.b[arm] + reward * ctx

        # --- global value model update (direct fit to reward) ---
        self.G = f * self.G + np.outer(ctx, ctx)
        self.f = f * self.f + reward * ctx
        self._update_global()

        # remember predicted reward for the next drift signal
        self.last_pred = self.theta @ ctx
        self.count += 1

    def _update_global(self):
        # theta = (G + ridge*I)^-1 f ; variance tracks the inverse gram.
        inv = np.linalg.inv(self.G + self.ridge * np.eye(self.d))
        self.theta = inv @ self.f
        # diagonal trace-based variance proxy for exploration magnitude
        self.theta_var = 0.5 * np.mean(np.abs(np.diag(inv)))
# EVOLVE-BLOCK-END

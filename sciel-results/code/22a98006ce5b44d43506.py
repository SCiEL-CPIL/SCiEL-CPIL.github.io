# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Adaptive LinUCB with variance-driven forgetting and efficient solves."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.lam = 1.0
        self.A = [np.eye(dim) * self.lam for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

        # Per-arm running stats for reward variance
        self.count = [0] * n_arms
        self.mean_r = [0.0] * n_arms
        self.m2 = [0.0] * n_arms          # running sum of squared deviations

        # Base parameters
        self.beta_min = 0.90              # fast forgetting under high variance
        self.beta_max = 0.999             # slow forgetting under low variance
        self.alpha_base = 1.0             # base exploration
        self.var_scale = 2.0              # variance threshold scaling

        # Time guard
        self.start_time = time.time()
        self.deadline = self.start_time + 112.0  # safety margin before 120s

    def _safe(self):
        return time.time() >= self.deadline

    def _adaptive_beta(self):
        # Global reward variance across arms (using pooled stats)
        total_n = sum(self.count)
        if total_n < 2:
            return 0.99
        # Pooled mean
        pooled_mean = sum(self.mean_r[a] * self.count[a] for a in range(self.n)) / total_n
        pooled_m2 = sum(self.m2[a] + self.count[a] * (self.mean_r[a] - pooled_mean) ** 2
                        for a in range(self.n))
        var = max(pooled_m2 / max(total_n - 1, 1), 1e-8)
        # Map variance to beta: high var -> low beta
        # Normalize: var in [0, var_scale^2] maps to [beta_max, beta_min]
        t = min(var / (self.var_scale ** 2), 1.0)
        beta = self.beta_max - t * (self.beta_max - self.beta_min)
        return beta

    def _adaptive_alpha(self):
        # Exploration grows with non-stationarity (low beta) and shrinks with confidence
        beta = self._adaptive_beta()
        nonstat = 1.0 - beta  # 0..0.1
        # Confidence: average observations per arm
        avg_n = max(sum(self.count) / self.n, 1.0)
        conf = 1.0 / np.sqrt(avg_n)
        # alpha: base scaled by (base exploration) + non-stationarity boost
        alpha = self.alpha_base * (0.5 + conf) + 2.0 * nonstat
        return alpha

    def select(self, context):
        if self._safe():
            # Deadline guard: return argmax of current point estimates (no exploration)
            scores = []
            for a in range(self.n):
                try:
                    theta = np.linalg.solve(self.A[a], self.b[a])
                    scores.append(theta @ context)
                except np.linalg.LinAlgError:
                    scores.append(0.0)
            return int(np.argmax(scores))

        alpha = self._adaptive_alpha()
        scores = []
        for a in range(self.n):
            try:
                # Solve A theta = b via Cholesky for speed/stability
                L = np.linalg.cholesky(self.A[a])
                theta = np.linalg.solve(L.T, np.linalg.solve(L, self.b[a]))
                # Variance term
                v = np.linalg.solve(L, context)
                var_term = float(v @ v)
                scores.append(theta @ context + alpha * np.sqrt(max(var_term, 0.0)))
            except np.linalg.LinAlgError:
                # Fallback to pinv-like via lstsq
                try:
                    theta = np.linalg.lstsq(self.A[a], self.b[a], rcond=None)[0]
                    scores.append(theta @ context)
                except Exception:
                    scores.append(0.0)
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if self._safe():
            # Still update minimally so state isn't corrupted, but skip heavy work
            self.A[arm] = self.A[arm] + np.outer(context, context)
            self.b[arm] = self.b[arm] + reward * context
            self._update_stats(arm, reward)
            return

        beta = self._adaptive_beta()
        # Forgetting: decay old info, add new
        self.A[arm] = beta * self.A[arm] + np.outer(context, context)
        self.b[arm] = beta * self.b[arm] + reward * context
        self._update_stats(arm, reward)

    def _update_stats(self, arm, reward):
        n = self.count[arm] + 1
        self.count[arm] = n
        delta = reward - self.mean_r[arm]
        self.mean_r[arm] += delta / n
        self.m2[arm] += delta * (reward - self.mean_r[arm])
# EVOLVE-BLOCK-END

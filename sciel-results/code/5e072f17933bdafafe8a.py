# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Non-stationary LinUCB with residual-driven adaptive process noise and decaying alpha."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.base_gamma = 0.90          # more aggressive base forgetting
        self.alpha0 = 1.5               # initial exploration bonus
        self.alpha_decay = 0.9995       # per-round decay of alpha
        self.alpha_min = 0.3            # floor for exploration
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        # EMA of squared residuals (innovation) per arm, used to modulate process noise
        self.res_ema = [0.0 for _ in range(n_arms)]
        self.res_ema_beta = 0.95        # EMA smoothing for residual magnitude
        self.res_q_base = 0.05          # base process-noise variance
        self.res_q_max = 0.5            # cap on process-noise variance
        self.t = 0
        self.alpha = self.alpha0
        self.t0 = time.time()
        self.deadline = 145.0           # safety margin under 150s kill

    def _check_deadline(self):
        return (time.time() - self.t0) < self.deadline

    def select(self, context):
        scores = []
        for a in range(self.n):
            try:
                inv = np.linalg.inv(self.A[a])
                theta = inv @ self.b[a]
                pred = theta @ context
                var = context @ inv @ context
                bonus = self.alpha * np.sqrt(var)
                scores.append(pred + bonus)
            except np.linalg.LinAlgError:
                scores.append(np.random.randn())
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Compute residual (innovation) for this arm
        try:
            inv = np.linalg.inv(self.A[arm])
            theta = inv @ self.b[arm]
            pred = theta @ context
        except np.linalg.LinAlgError:
            pred = 0.0
        residual = reward - pred
        sq_res = residual * residual

        # Update EMA of squared residual
        self.res_ema[arm] = self.res_ema_beta * self.res_ema[arm] + (1 - self.res_ema_beta) * sq_res

        # Derive adaptive process-noise variance from residual EMA with tanh saturation
        # Larger residual -> larger Q (more trust in new data, faster forgetting)
        q = self.res_q_base + (self.res_q_max - self.res_q_base) * np.tanh(self.res_ema[arm] / 1.0)

        # Adaptive forgetting: base_gamma shrunk by q (more forgetting when drift detected)
        gamma = self.base_gamma - q
        # Clamp gamma to a reasonable range
        if gamma < 0.7:
            gamma = 0.7
        if gamma > 0.99:
            gamma = 0.99

        # Decay existing statistics
        self.A[arm] = gamma * self.A[arm]
        self.b[arm] = gamma * self.b[arm]

        # Add new observation
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context

        # Decay exploration bonus over time (reduce over-exploration in later rounds)
        self.alpha = max(self.alpha_min, self.alpha * self.alpha_decay)
        self.t += 1
# EVOLVE-BLOCK-END

"""Per-Arm RLS linear bandit with context-adaptive forgetting factor."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Per-Arm RLS with context-adaptive forgetting factor."""

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t0 = time.time()
        self.deadline = self.t0 + 115.0   # safety margin before 120s limit

        # RLS state per arm
        self.P = [np.eye(dim) / 1.0 for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]

        # Adaptive forgetting params
        self.lam_base = 0.985
        self.gamma = 0.002          # scales norm^2
        self.lam_max = 0.9995
        self.lam_min = 0.90
        self.alpha = 0.01           # RLS step (regularization in P update)
        self.beta = 1.0             # step-size multiplier
        self.norm_w = 0.0           # EMA of context norm^2
        self.norm_beta = 0.95

    def _elapsed(self):
        return time.time() - self.t0

    def select(self, context):
        # Deadline guard: if we are too close to the limit, fall back to
        # the greedy argmax (cheap) to avoid any slow path.
        if self._elapsed() > self.deadline:
            scores = np.array([self.theta[a] @ context for a in range(self.n)])
            return int(np.argmax(scores))

        x = np.asarray(context, dtype=float)
        scores = np.empty(self.n)
        for a in range(self.n):
            theta = self.theta[a]
            P = self.P[a]
            pred = theta @ x
            # RLS-style variance: x^T P x (P is the covariance, ~1/lambda^t * A^{-1})
            var = x @ (P @ x)
            scores[a] = pred + 0.7 * np.sqrt(max(var, 0.0))
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        x = np.asarray(context, dtype=float)
        # Track context norm^2 (EMA) to adapt the forgetting factor.
        x2 = float(x @ x)
        self.norm_w = self.norm_beta * self.norm_w + (1.0 - self.norm_beta) * x2

        # Context-adaptive forgetting factor: larger norm -> faster forgetting.
        lam = self.lam_base + self.gamma * self.norm_w
        lam = min(max(lam, self.lam_min), self.lam_max)

        P = self.P[arm]
        theta = self.theta[arm]

        # RLS update:
        # K = P x / (lam + x^T P x)
        Px = P @ x
        denom = lam + float(x @ Px)
        K = Px / denom
        pred = float(theta @ x)
        err = reward - pred
        theta = theta + K * err
        # P = (P - K x^T P) / lam
        P = (P - np.outer(K, x) @ P) / lam

        # Numerical safety: keep P symmetric positive-definite-ish.
        P = 0.5 * (P + P.T)

        self.P[arm] = P
        self.theta[arm] = theta
# EVOLVE-BLOCK-END

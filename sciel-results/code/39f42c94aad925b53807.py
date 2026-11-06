"""
Non-stationary linear contextual bandit policy using Recursive Least Squares (RLS).
Combines robust IQR-based error scaling with logistic forgetting mapping and
adaptive UCB exploration. The structural core is RLS with explicit inverse
covariance (P) for O(d^2) updates; the refinement uses a smooth logistic
mapping from normalized prediction error to the forgetting factor, plus a
hyperbolic decay schedule for exploration.
"""

import numpy as np
import time
from collections import deque


class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim

        # RLS: P is the inverse covariance (precision) matrix per arm
        # theta is the weight vector per arm
        # Initialize P to large identity (high uncertainty) and theta to zero
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]

        # Forgetting factor parameters
        self.lam_min = 0.80   # Fast forgetting during high error
        self.lam_max = 0.995  # Slow forgetting during low error

        # Error tracking for adaptive forgetting
        self.sq_errors = [0.0] * n_arms  # EMA of squared errors per arm
        self.sq_error_alpha = 0.3        # EMA decay for error tracking

        # Robust error scale via IQR of recent squared errors
        self.error_samples = [deque(maxlen=200) for _ in range(n_arms)]
        self.error_scale = [1.0] * n_arms

        # Exploration: UCB with adaptive alpha
        self.alpha_base = 1.5
        self.arm_counts = [0] * n_arms
        self.total_rounds = 0

        # Deadline guard
        self._start_time = time.time()
        self._deadline = self._start_time + 140.0

    def _update_error_scale(self, arm, sq_err):
        """
        Update robust error scale using IQR of recent squared errors.
        """
        samples = self.error_samples[arm]
        samples.append(sq_err)
        if len(samples) >= 20:
            arr = np.array(samples)
            q75 = np.percentile(arr, 75)
            q25 = np.percentile(arr, 25)
            iqr = q75 - q25
            new_scale = max(iqr, 1e-4)
            # Smooth update to avoid abrupt changes
            self.error_scale[arm] = 0.9 * self.error_scale[arm] + 0.1 * new_scale

    def _get_lambda(self, arm):
        """
        Compute adaptive forgetting factor using a logistic (sigmoid) mapping
        from normalized squared prediction error.
        High error -> low lambda (fast forgetting)
        Low error  -> high lambda (slow forgetting)
        """
        err = self.sq_errors[arm]
        err_scale = self.error_scale[arm]
        if err_scale < 1e-6:
            err_scale = 1e-6
        normalized = err / err_scale

        # Logistic mapping: t in [0,1], centered at normalized=1
        # t = 1/(1+exp(-k*(x - x0)))
        k = 10.0
        x0 = 1.0
        t = 1.0 / (1.0 + np.exp(-k * (normalized - x0)))
        t = np.clip(t, 0.0, 1.0)

        # High t (high error) -> low lambda
        lam = self.lam_max - t * (self.lam_max - self.lam_min)
        return np.clip(lam, self.lam_min, self.lam_max)

    def _get_alpha(self):
        """
        Adaptive exploration parameter with hyperbolic decay.
        """
        if self.total_rounds < 30:
            return self.alpha_base + 1.0

        decay = 1.0 / (1.0 + self.total_rounds / 200.0)
        return self.alpha_base * decay + 0.5

    def select(self, context):
        # Deadline check
        if time.time() > self._deadline:
            return int(np.random.randint(self.n))

        context = np.asarray(context, dtype=np.float64)
        alpha = self._get_alpha()
        scores = []

        for a in range(self.n):
            # Mean prediction
            mu = self.theta[a] @ context

            # Uncertainty: sqrt(context^T P context)
            p_context = self.P[a] @ context
            var = context @ p_context
            std = np.sqrt(max(0.0, var))

            # UCB score
            score = mu + alpha * std
            scores.append(score)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)

        # Deadline check
        if time.time() > self._deadline:
            return

        self.total_rounds += 1
        self.arm_counts[arm] += 1

        # Compute prediction error
        pred = self.theta[arm] @ context
        error = reward - pred
        sq_err = error * error

        # Update EMA of squared error for this arm
        self.sq_errors[arm] = (1.0 - self.sq_error_alpha) * self.sq_errors[arm] + self.sq_error_alpha * sq_err

        # Update robust error scale
        self._update_error_scale(arm, sq_err)

        # Get adaptive forgetting factor
        lam = self._get_lambda(arm)

        # RLS update
        P = self.P[arm]
        theta = self.theta[arm]

        Px = P @ context
        denom = lam + context @ Px
        if denom < 1e-12:
            denom = 1e-12

        K = Px / denom

        # Update theta
        self.theta[arm] = theta + K * error

        # Update P: P = (P - K x^T P) / lam
        self.P[arm] = (P - np.outer(K, context) @ P) / lam

        # Ensure P remains symmetric positive definite
        P_new = self.P[arm]
        P_new = (P_new + P_new.T) * 0.5
        P_new += 1e-6 * np.eye(self.d)
        self.P[arm] = P_new

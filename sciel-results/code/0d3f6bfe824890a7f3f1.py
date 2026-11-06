"""Bayesian Linear Bandit with Sharp Event-Driven Forgetting and Thompson Sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Thompson Sampling for non-stationary linear bandits with event-driven adaptive forgetting.
    
    Key design:
    - Bayesian posterior N(theta, V) per arm, updated via Sherman-Morrison.
    - Cholesky factors maintained for fast sampling in select().
    - Event-driven forgetting: when the normalized residual (z-score) exceeds a threshold,
      apply strong decay to V (rapid forgetting). Otherwise, apply mild decay.
    - This directly targets non-stationarity by rapidly discarding stale models only when
      prediction errors indicate a regime shift.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim

        # Posterior parameters per arm
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.V = [np.eye(dim) for _ in range(n_arms)]
        # Cholesky factors L where V = L @ L.T
        self.L = [np.eye(dim) for _ in range(n_arms)]

        # Forgetting parameters
        self.strong_decay = 0.5   # Strong forgetting on regime shift
        self.mild_decay = 0.99    # Mild forgetting otherwise
        self.z_threshold = 2.0    # Z-score threshold for triggering strong forgetting

        # Time tracking
        self.start_time = time.time()
        self.deadline = 140.0

    def _update_cholesky(self, V_new):
        """Compute Cholesky factor with jitter for numerical stability."""
        try:
            L_new = np.linalg.cholesky(V_new)
        except np.linalg.LinAlgError:
            jitter = 1e-6 * np.eye(V_new.shape[0])
            try:
                L_new = np.linalg.cholesky(V_new + jitter)
            except np.linalg.LinAlgError:
                # Last resort: use identity scaled by max eigenvalue
                eigvals = np.linalg.eigvalsh(V_new)
                max_eig = max(np.max(eigvals), 1e-4)
                L_new = np.sqrt(max_eig) * np.eye(V_new.shape[0])
        return L_new

    def select(self, context):
        # Deadline guard
        if time.time() - self.start_time > self.deadline:
            return int(np.random.randint(self.n))

        context = np.asarray(context, dtype=np.float64)

        scores = np.empty(self.n)
        for a in range(self.n):
            # Thompson Sampling: sample from N(theta, V)
            try:
                L = self.L[a]
                z = np.random.randn(self.d)
                theta_sample = self.theta[a] + L @ z
            except Exception:
                try:
                    L = np.linalg.cholesky(self.V[a])
                    z = np.random.randn(self.d)
                    theta_sample = self.theta[a] + L @ z
                except np.linalg.LinAlgError:
                    theta_sample = self.theta[a]

            scores[a] = theta_sample @ context

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Deadline guard
        if time.time() - self.start_time > self.deadline:
            return

        context = np.asarray(context, dtype=np.float64)

        # Compute prediction and residual
        pred = self.theta[arm] @ context
        residual = reward - pred

        # Compute prediction variance (uncertainty)
        var = context @ (self.V[arm] @ context)
        std = np.sqrt(max(var, 1e-8))

        # Z-score
        z_score = abs(residual / std)

        # Event-driven forgetting: strong if z-score exceeds threshold
        if z_score > self.z_threshold:
            decay = self.strong_decay
        else:
            decay = self.mild_decay

        # Apply forgetting: V_new = V_old / decay (equivalent to V_inv *= decay)
        V = self.V[arm] / decay

        # Bayesian update via Sherman-Morrison
        x = context
        Vx = V @ x
        denom = x @ Vx + 1.0
        V_new = V - np.outer(Vx, Vx) / denom

        # Update theta
        self.theta[arm] = self.theta[arm] + (V_new @ x) * residual

        # Store updated matrices
        self.V[arm] = V_new
        self.L[arm] = self._update_cholesky(V_new)
# EVOLVE-BLOCK-END

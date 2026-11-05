# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit policy using Thompson Sampling.
    Maintains per-arm posterior Gaussian beliefs over weight vectors.
    Uses adaptive covariance inflation driven by standardized innovation magnitude
    to handle non-stationarity. This replaces deterministic UCB with probabilistic
    arm selection, naturally balancing exploration and exploitation.
    
    Key structural changes from UCB:
    - Thompson Sampling: sample theta from posterior, select argmax of sampled rewards
    - Adaptive covariance inflation: P grows when innovations are large (non-stationarity)
    - Kalman-style update with adaptive process noise Q
    - Soft reset mechanism for severe regime changes
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self._start_time = time.time()
        self._deadline = 112.0  # Safety margin before 150s limit

        # Per-arm posterior: theta (mean), P (covariance)
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.count = [0 for _ in range(n_arms)]

        # Observation noise variance
        self._R = 0.1

        # Adaptive process noise parameters
        self._innov_ema = [0.0 for _ in range(n_arms)]
        self._innov_alpha = 0.12
        self._base_Q = 3e-4
        self._Q_max = 0.04

        # Soft reset mechanism
        self._consec_large = [0 for _ in range(n_arms)]
        self._reset_threshold = 3
        self._reset_cov_scale = 5.0

        # Cholesky factors for efficient sampling (precompute when possible)
        self._P_cholesky = [None for _ in range(n_arms)]
        self._cholesky_dirty = [True for _ in range(n_arms)]

    def _adaptive_Q(self, arm):
        """Compute adaptive process noise based on EMA of squared innovations."""
        if self.count[arm] < 3:
            return self._base_Q
        ema_val = self._innov_ema[arm]
        scaled = np.tanh(ema_val / 0.4)
        q = self._base_Q + (self._Q_max - self._base_Q) * scaled
        return q

    def _maybe_reset(self, arm, innov_sq):
        """If too many consecutive large innovations, reset covariance."""
        if innov_sq > 4.0:
            self._consec_large[arm] += 1
        else:
            self._consec_large[arm] = max(0, self._consec_large[arm] - 1)

        if self._consec_large[arm] >= self._reset_threshold:
            self.P[arm] = np.eye(self.d) * self._reset_cov_scale
            self._consec_large[arm] = 0
            self._cholesky_dirty[arm] = True

    def _ensure_cholesky(self, arm):
        """Ensure Cholesky factor is up to date for the given arm."""
        if self._cholesky_dirty[arm]:
            try:
                # Add small jitter for numerical stability
                P = self.P[arm] + 1e-8 * np.eye(self.d)
                self._P_cholesky[arm] = np.linalg.cholesky(P)
                self._cholesky_dirty[arm] = False
            except np.linalg.LinAlgError:
                # Fallback: use diagonal approximation
                self._P_cholesky[arm] = np.diag(np.sqrt(np.maximum(np.diag(self.P[arm]), 1e-8)))
                self._cholesky_dirty[arm] = False

    def _sample_theta(self, arm):
        """Sample theta from N(theta_mean, P) using Cholesky decomposition."""
        self._ensure_cholesky(arm)
        L = self._P_cholesky[arm]
        z = np.random.randn(self.d)
        return self.theta[arm] + L @ z

    def select(self, context):
        """Select arm via Thompson Sampling: sample from posterior, pick argmax."""
        if time.time() - self._start_time > self._deadline:
            # Fallback to greedy
            ctx = np.asarray(context, dtype=np.float64)
            scores = [self.theta[a] @ ctx for a in range(self.n)]
            return int(np.argmax(scores))

        ctx = np.asarray(context, dtype=np.float64)
        best_arm = 0
        best_score = -np.inf

        for a in range(self.n):
            # Sample theta from posterior
            theta_sample = self._sample_theta(a)
            score = theta_sample @ ctx
            if score > best_score:
                best_score = score
                best_arm = a

        return int(best_arm)

    def update(self, context, arm, reward):
        """Update posterior with new observation using Kalman filter with adaptive Q."""
        self.t += 1
        a = arm
        x = np.asarray(context, dtype=np.float64)
        r = reward

        # Deadline guard: minimal update
        if time.time() - self._start_time > self._deadline:
            self.count[a] += 1
            return

        # 1. Predict: add process noise
        Q = self._adaptive_Q(a)
        P_pred = self.P[a] + Q * np.eye(self.d)

        # 2. Calculate innovation
        y = r - self.theta[a] @ x

        # 3. Kalman update
        Px = P_pred @ x
        denom = x @ Px + self._R
        if denom < 1e-12:
            denom = 1e-12
        K = Px / denom

        # Update theta
        self.theta[a] = self.theta[a] + K * y

        # Update P (Joseph form for numerical stability)
        I = np.eye(self.d)
        IKxT = I - np.outer(K, x)
        P_new = IKxT @ P_pred @ IKxT.T + np.outer(K, K) * self._R

        # Ensure symmetry
        self.P[a] = (P_new + P_new.T) / 2.0
        self._cholesky_dirty[a] = True

        # Track prediction error for adaptive Q
        innov_sq = (y * y) / denom
        self._innov_ema[a] = (1.0 - self._innov_alpha) * self._innov_ema[a] + self._innov_alpha * innov_sq

        # Check for reset
        self._maybe_reset(a, innov_sq)

        self.count[a] += 1
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit policy using Thompson Sampling
    with inverse-covariance (information form) representation.
    
    Structural change from parent: Uses inverse covariance (V = P^{-1}) and
    information vector (b = P^{-1} theta) instead of maintaining P directly.
    This enables O(d^2) updates instead of O(d^3) Cholesky decompositions,
    and more stable posterior sampling via the information form.
    
    Combined with:
    - Adaptive Kalman-style process noise (Q) driven by innovation EMA
    - Joseph-form covariance update for numerical stability
    - Soft reset mechanism for severe regime changes
    - Decaying exploration bonus on top of Thompson samples
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self._start_time = time.time()
        self._deadline = 112.0

        # Information form: V = P^{-1}, b = P^{-1} * theta
        # Initialize with P = 10*I, so V = 0.1*I, b = 0
        self.V = [np.eye(dim) * 0.1 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
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

        # Exploration bonus parameters (decaying)
        self._alpha_base = 1.0
        self._alpha_min = 0.2

    def _get_theta_and_P(self, arm):
        """Recover theta and P from information form V and b."""
        V = self.V[arm]
        b = self.b[arm]
        # P = V^{-1}, theta = V^{-1} b = P * b
        # Use solve for numerical stability: theta = V^{-1} b
        try:
            theta = np.linalg.solve(V, b)
        except np.linalg.LinAlgError:
            theta = b / np.max(np.diag(V), 1e-8)
        # P = V^{-1}
        try:
            P = np.linalg.inv(V)
        except np.linalg.LinAlgError:
            P = np.eye(self.d) / np.max(np.diag(V), 1e-8)
        return theta, P

    def _adaptive_Q(self, arm):
        """Compute adaptive process noise based on EMA of squared innovations."""
        if self.count[arm] < 3:
            return self._base_Q
        ema_val = self._innov_ema[arm]
        scaled = np.tanh(ema_val / 0.4)
        q = self._base_Q + (self._Q_max - self._base_Q) * scaled
        return q

    def _maybe_reset(self, arm, innov_sq):
        """If too many consecutive large innovations, reset to uniform prior."""
        if innov_sq > 4.0:
            self._consec_large[arm] += 1
        else:
            self._consec_large[arm] = max(0, self._consec_large[arm] - 1)

        if self._consec_large[arm] >= self._reset_threshold:
            # Reset to initial prior: P = reset_cov_scale * I
            # So V = (1/reset_cov_scale) * I, b = 0
            self.V[arm] = np.eye(self.d) / self._reset_cov_scale
            self.b[arm] = np.zeros(self.d)
            self._consec_large[arm] = 0

    def select(self, context):
        """Select arm via Thompson Sampling with decaying exploration bonus."""
        if time.time() - self._start_time > self._deadline:
            # Fallback to greedy
            ctx = np.asarray(context, dtype=np.float64)
            scores = []
            for a in range(self.n):
                theta, _ = self._get_theta_and_P(a)
                scores.append(theta @ ctx)
            return int(np.argmax(scores))

        ctx = np.asarray(context, dtype=np.float64)
        best_arm = 0
        best_score = -np.inf

        # Decaying exploration coefficient
        alpha_t = max(self._alpha_min, self._alpha_base * (1.0 + self.t) ** (-0.15))

        for a in range(self.n):
            theta, P = self._get_theta_and_P(a)
            
            # Thompson sample: theta_sample ~ N(theta, P)
            try:
                L = np.linalg.cholesky(P + 1e-8 * np.eye(self.d))
                z = np.random.randn(self.d)
                theta_sample = theta + L @ z
            except np.linalg.LinAlgError:
                # Fallback: use diagonal approximation
                diag = np.diag(P)
                stds = np.sqrt(np.maximum(diag, 1e-8))
                theta_sample = theta + stds * np.random.randn(self.d)
            
            # Base score from Thompson sample
            base_score = theta_sample @ ctx
            
            # Add exploration bonus: alpha_t * std of prediction
            Px = P @ ctx
            var = ctx @ Px
            if var < 0:
                var = 0.0
            std = np.sqrt(var)
            
            # Scale exploration by count to reduce it as we learn
            count_factor = 1.0 / (1.0 + np.sqrt(self.count[a]))
            score = base_score + alpha_t * std * count_factor
            
            if score > best_score:
                best_score = score
                best_arm = a

        return int(best_arm)

    def update(self, context, arm, reward):
        """Update posterior using information-form Kalman filter with adaptive Q."""
        self.t += 1
        a = arm
        x = np.asarray(context, dtype=np.float64)
        r = reward

        # Deadline guard: minimal update
        if time.time() - self._start_time > self._deadline:
            self.count[a] += 1
            return

        # Get current theta and P
        theta, P = self._get_theta_and_P(a)

        # 1. Predict: add process noise Q*I to P
        Q = self._adaptive_Q(a)
        P_pred = P + Q * np.eye(self.d)

        # 2. Calculate innovation
        y = r - theta @ x

        # 3. Kalman gain
        Px = P_pred @ x
        denom = x @ Px + self._R
        if denom < 1e-12:
            denom = 1e-12
        K = Px / denom

        # Update theta
        theta_new = theta + K * y

        # Update P using Joseph form
        I = np.eye(self.d)
        IKxT = I - np.outer(K, x)
        P_new = IKxT @ P_pred @ IKxT.T + np.outer(K, K) * self._R

        # Ensure symmetry
        P_new = (P_new + P_new.T) / 2.0

        # Convert back to information form: V = P^{-1}, b = V * theta
        try:
            V_new = np.linalg.inv(P_new)
        except np.linalg.LinAlgError:
            V_new = np.eye(self.d) / np.max(np.diag(P_new), 1e-8)
        
        b_new = V_new @ theta_new

        self.V[a] = V_new
        self.b[a] = b_new

        # Track prediction error for adaptive Q
        innov_sq = (y * y) / denom
        self._innov_ema[a] = (1.0 - self._innov_alpha) * self._innov_ema[a] + self._innov_alpha * innov_sq

        # Check for reset
        self._maybe_reset(a, innov_sq)

        self.count[a] += 1
# EVOLVE-BLOCK-END

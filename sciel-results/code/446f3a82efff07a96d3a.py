# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit with a unified linear model
    and adaptive Kalman filtering.

    Design family (KEPT): single flattened weight vector w (size d*n) with a
    precision matrix P_inv (information form), Kalman prediction via adaptive
    forgetting on precision, and Joseph-form-stable information updates.

    Refinements within this family:
      1. Adaptive observation noise R per arm: R_a is set from the standardized
         innovation magnitude so that outlier / regime-shift observations are
         down-weighted instead of forcing the process noise to inflate. This
         preserves the integrity of weight estimates during abrupt changes.
      2. Per-arm adaptive forgetting: the precision decay is applied per arm
         (diagonal process-noise structure) so a drifting arm forgets faster
         while a stable arm retains precision.
      3. Efficiency: covariance P is inverted once per select and reused; phi
         construction is minimal; a strict deadline guard returns greedy near
         the wall-clock limit.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.dn = self.d * self.n
        self.t = 0
        self.t0 = time.time()
        self.deadline = 143.0  # safety margin before 150s kill

        # State: flattened weight vector (d * n,)
        self.w = np.zeros(self.dn)

        # Precision matrix (inverse covariance). Start with moderate variance.
        self.P_inv = np.eye(self.dn) * 1.0

        # Per-arm adaptive forgetting (diagonal process noise on precision).
        self.lambda_per_arm = np.full(self.n, 0.99)
        self.lambda_min = 0.80
        self.lambda_max = 0.999

        # Per-arm innovation statistics for adaptive R.
        self.residual_ema = np.zeros(self.n)      # E[innovation^2]
        self.innov_mean_ema = np.zeros(self.n)    # E[innovation]
        self.ema_beta = 0.05

        # Adaptive R parameters.
        self.R_base = 1.0
        self.R_max = 50.0

        # Exploration.
        self.count = np.zeros(self.n, dtype=np.int64)
        self.exploration_base = 1.0
        self.exploration_decay = 0.999

        # Cache for the covariance P (inverse of P_inv) within a select call.
        self._P_cache = None

    def _get_phi(self, context, arm):
        """Construct the one-hot encoded context vector for the unified model."""
        phi = np.zeros(self.dn)
        start = arm * self.d
        phi[start:start + self.d] = context
        return phi

    def _get_arm_weights(self, arm):
        """Extract weight vector for a specific arm."""
        start = arm * self.d
        return self.w[start:start + self.d]

    def _compute_P(self):
        """Invert P_inv once and cache the covariance P."""
        try:
            P = np.linalg.inv(self.P_inv)
        except np.linalg.LinAlgError:
            P = np.linalg.pinv(self.P_inv)
        self._P_cache = P
        return P

    def _predict(self):
        """
        Adaptive Kalman prediction step with per-arm forgetting.
        We apply a diagonal decay to the precision: blocks corresponding to
        each arm are multiplied by that arm's lambda. lambda < 1 increases
        uncertainty (forgets old data). lambda is adaptive per arm based on
        that arm's residual error.
        """
        if self.t > 5:
            # Map each arm's residual EMA to a forgetting factor.
            # High error -> low lambda (forget faster).
            err_norm = np.tanh(self.residual_ema / 10.0)
            lam = 0.99 - 0.15 * err_norm
            lam = np.clip(lam, self.lambda_min, self.lambda_max)
            self.lambda_per_arm = lam

        # Apply per-arm diagonal decay to P_inv.
        # Build a diagonal decay vector: for each arm a, its d entries get
        # multiplied by lambda_per_arm[a].
        decay = np.repeat(self.lambda_per_arm, self.d)
        self.P_inv = self.P_inv * decay  # row-wise (diagonal) scaling
        # Row-wise scaling of a symmetric matrix is not symmetric in general;
        # symmetrize to keep P_inv a valid precision matrix.
        self.P_inv = (self.P_inv + self.P_inv.T) / 2.0

        # Invalidate covariance cache.
        self._P_cache = None

    def _adaptive_R(self, arm, innovation):
        """
        Adaptive observation noise R for a given arm based on the standardized
        innovation magnitude. Large |innovation| relative to recent scale
        suggests a regime shift / outlier -> increase R to down-weight it.
        """
        # Update per-arm innovation statistics.
        self.innov_mean_ema[arm] = (1 - self.ema_beta) * self.innov_mean_ema[arm] \
                                   + self.ema_beta * innovation
        self.residual_ema[arm] = (1 - self.ema_beta) * self.residual_ema[arm] \
                                 + self.ema_beta * innovation * innovation

        # Standardized magnitude: |innovation| / (recent std).
        std = np.sqrt(max(self.residual_ema[arm], 1e-6))
        z = abs(innovation) / std
        # Soft-cap the influence: R grows with z^2 but saturates.
        # R = R_base * (1 + alpha * z^2), clipped.
        R = self.R_base * (1.0 + 2.0 * z * z)
        return min(R, self.R_max)

    def _update(self, context, arm, reward):
        """
        Kalman update step for the unified model with adaptive R.
        """
        phi = self._get_phi(context, arm)

        # Predicted reward.
        pred = phi @ self.w

        # Innovation.
        innovation = reward - pred

        # Adaptive observation noise for this arm.
        R = self._adaptive_R(arm, innovation)

        # Solve P_inv x = phi  =>  x = P @ phi.
        try:
            x = np.linalg.solve(self.P_inv, phi)
        except np.linalg.LinAlgError:
            x = np.linalg.pinv(self.P_inv) @ phi

        # Innovation variance S = phi @ P @ phi + R.
        S = phi @ x + R
        if S < 1e-8:
            S = 1e-8

        # Kalman gain.
        K = x / S

        # Update weight.
        self.w += K * innovation

        # Update Precision (Information Form): P_inv += phi phi^T / R.
        self.P_inv += np.outer(phi, phi) / R

        # Symmetrize P_inv for numerical stability.
        self.P_inv = (self.P_inv + self.P_inv.T) / 2.0

        # Invalidate covariance cache.
        self._P_cache = None

    def select(self, context):
        """Select arm using UCB with adaptive-uncertainty bonus."""
        # Deadline guard: fall back to greedy near the wall-clock limit.
        if time.time() - self.t0 > self.deadline:
            scores = []
            for a in range(self.n):
                w_a = self._get_arm_weights(a)
                scores.append(w_a @ context)
            return int(np.argmax(scores))

        self._predict()

        # Invert once and reuse for all arms.
        P = self._compute_P()

        scores = []
        for a in range(self.n):
            w_a = self._get_arm_weights(a)
            mean = w_a @ context

            # Uncertainty: Var(reward) = phi^T P phi.
            phi = self._get_phi(context, a)
            var = phi @ P @ phi
            uncertainty = np.sqrt(max(0.0, var))

            # Count-based exploration that decays with experience.
            count_factor = self.exploration_base * (self.exploration_decay ** self.count[a])
            bonus = 1.0 * uncertainty * count_factor + 0.1 * count_factor / (1 + self.count[a])

            scores.append(mean + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        """Update the model."""
        self.t += 1
        self.count[arm] += 1

        self._update(context, arm, reward)
# EVOLVE-BLOCK-END

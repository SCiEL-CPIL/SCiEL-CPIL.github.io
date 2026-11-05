"""Per-Arm Linear Kalman Filter with Adaptive Observation Noise and Innovation-Gated Process Noise."""
# EVOLVE-BLOCK-START
import numpy as np


class Policy:
    """
    Non-stationary linear contextual bandit policy using a per-arm
    Linear Kalman Filter with adaptive observation noise variance and
    innovation-gated process noise.

    Key design choices:
    1. Random-walk process noise Q (small base) to allow weight drift.
    2. Adaptive observation noise R: when the standardized innovation z is
       large, R is inflated to down-weight potentially corrupted/outlier
       observations rather than letting them corrupt the state estimate.
    3. Innovation-gated process noise: when z is large we *also* modestly
       increase Q for the next prediction step, but only up to a cap, so
       the filter can track regime shifts without being hijacked by a
       single outlier.
    4. UCB exploration with a slowly decaying alpha to balance exploration
       and exploitation over the horizon.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim

        # Per-arm Kalman filter states
        self.P = [np.eye(dim) * 50.0 for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]

        # Base process noise (random walk)
        self.Q_base = 0.02 * np.eye(dim)
        # Per-arm process noise (starts at base)
        self.Q = [self.Q_base.copy() for _ in range(n_arms)]
        # Max process noise multiplier when a regime shift is detected
        self.Q_max_mult = 5.0

        # Base observation noise
        self.R_base = 0.8

        # UCB exploration scaling (decays over time)
        self.alpha_init = 2.0
        self.alpha_decay = 0.9995
        self.alpha = self.alpha_init

        # Per-arm innovation statistics for adaptive R
        # We keep a running EMA of |z| to normalize the adaptive gain
        self.z_ema = [0.0 for _ in range(n_arms)]
        self.z_ema_beta = 0.9

        # Count of updates per arm (for alpha decay)
        self.t = 0

    def select(self, context):
        """
        Select arm using UCB-style exploration based on Kalman filter uncertainty.
        """
        context = np.asarray(context, dtype=np.float64)
        scores = np.zeros(self.n)

        for a in range(self.n):
            mu = self.theta[a] @ context
            var = context @ self.P[a] @ context
            std = np.sqrt(max(var, 0.0))
            scores[a] = mu + self.alpha * std

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        """
        Update the Kalman filter for the selected arm.
        """
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        a = arm
        self.t += 1

        # Decay exploration
        self.alpha *= self.alpha_decay

        # --- Prediction Step ---
        # Use the Q that was set at the end of the previous update
        self.P[a] = self.P[a] + self.Q[a]

        # --- Update Step ---
        predicted_reward = self.theta[a] @ context
        nu = reward - predicted_reward

        Pxx = context @ self.P[a] @ context
        S_base = Pxx + self.R_base
        std_S_base = np.sqrt(max(S_base, 1e-12))

        z = abs(nu) / std_S_base if std_S_base > 1e-12 else 0.0

        # Update running EMA of |z|
        self.z_ema[a] = (1.0 - self.z_ema_beta) * z + self.z_ema_beta * self.z_ema[a]

        # Adaptive R: inflate when z is large relative to its own history
        # Use a smooth tanh-based gating for more stable adaptation
        threshold = max(self.z_ema[a] + 0.5, 1.0)
        excess = max(0.0, z - threshold)
        # Smooth gate: tanh-based to avoid sharp transitions
        gate = np.tanh(0.5 * excess)
        R_adaptive = self.R_base * (1.0 + 3.0 * gate ** 2)

        # Cap R_adaptive to avoid completely ignoring the observation
        R_adaptive = min(R_adaptive, self.R_base * 10.0)

        S = Pxx + R_adaptive

        if S > 1e-12:
            K = self.P[a] @ context / S
        else:
            K = np.zeros(self.d)

        self.theta[a] = self.theta[a] + K * nu

        # Joseph form update
        I_Kx = np.eye(self.d) - np.outer(K, context)
        self.P[a] = I_Kx @ self.P[a] @ I_Kx.T + R_adaptive * np.outer(K, K)
        self.P[a] = (self.P[a] + self.P[a].T) / 2.0

        # Cheap PSD check: ensure trace is positive and add jitter if needed
        # Only do full eigenvalue check every 50 updates for this arm
        if self.t % 50 == 0:
            eigvals = np.linalg.eigvalsh(self.P[a])
            if np.min(eigvals) < 1e-8:
                self.P[a] += 1e-6 * np.eye(self.d)
        else:
            # Cheap check: if diagonal has negative values, add jitter
            if np.any(np.diag(self.P[a]) < 1e-10):
                self.P[a] += 1e-6 * np.eye(self.d)

        # --- Set Q for next prediction step based on this innovation ---
        # If z is large, the regime may have shifted; increase Q for next step
        # but cap it to prevent runaway
        if z > threshold:
            q_mult = min(1.0 + 0.5 * excess, self.Q_max_mult)
        else:
            q_mult = 1.0
        self.Q[a] = self.Q_base * q_mult
# EVOLVE-BLOCK-END

"""Per-Arm State-Space Model with Regime-Switching Process Noise."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit using a per-arm Kalman filter
    with a lightweight hidden Markov model (HMM) to switch between
    'stable' (low process noise) and 'volatile' (high process noise) modes.
    
    This distinguishes gradual drift from abrupt structural breaks.
    """

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.t_start = time.time()
        self.deadline = 115.0  # safety margin before 120s limit

        # Per-arm state: theta (mean), P (covariance)
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]

        # HMM parameters
        # States: 0 = stable, 1 = volatile
        # Transition probabilities
        self.trans = np.array([
            [0.995, 0.005],  # from stable
            [0.05, 0.95]     # from volatile
        ])
        # Initial state probabilities
        self.pi = np.array([0.9, 0.1])

        # Per-arm current HMM state probability
        self.state_prob = [self.pi.copy() for _ in range(n_arms)]

        # Process noise parameters
        self.Q_stable = 0.001 * np.eye(dim)
        self.Q_volatile = 0.5 * np.eye(dim)

        # Observation noise (assumed known)
        self.R = 1.0

        # Exploration constant
        self.alpha = 1.5

        # Track recent innovations for adaptive threshold
        self.innov_history = [[] for _ in range(n_arms)]
        self.max_history = 50

        # Counters
        self.step = 0

    def _predict(self, arm):
        """Kalman prediction step with regime-dependent process noise."""
        # Compute effective process noise based on state probability
        p_stable = self.state_prob[arm][0]
        p_volatile = self.state_prob[arm][1]
        Q_eff = p_stable * self.Q_stable + p_volatile * self.Q_volatile

        # Update covariance
        self.P[arm] = self.P[arm] + Q_eff

    def _update_hmm(self, arm, innovation, innovation_var):
        """Update HMM state probabilities based on innovation magnitude."""
        # Standardized innovation
        if innovation_var > 1e-10:
            z = abs(innovation) / np.sqrt(innovation_var)
        else:
            z = abs(innovation)

        # Likelihood of innovation under each regime
        # Stable regime: small innovations expected
        # Volatile regime: larger innovations expected
        # Use a simple Gaussian likelihood with different scales
        sigma_stable = 1.0
        sigma_volatile = 3.0

        log_lik_stable = -0.5 * (z / sigma_stable) ** 2
        log_lik_volatile = -0.5 * (z / sigma_volatile) ** 2

        # Normalize to probabilities
        exp_stable = np.exp(log_lik_stable)
        exp_volatile = np.exp(log_lik_volatile)
        total = exp_stable + exp_volatile
        if total < 1e-300:
            obs_prob = np.array([0.5, 0.5])
        else:
            obs_prob = np.array([exp_stable / total, exp_volatile / total])

        # Bayes update
        prior = self.state_prob[arm]
        posterior = prior * obs_prob
        s = posterior.sum()
        if s > 1e-300:
            self.state_prob[arm] = posterior / s
        else:
            self.state_prob[arm] = obs_prob

    def select(self, context):
        # Deadline guard
        if time.time() - self.t_start > self.deadline:
            # Return best arm based on current mean predictions
            scores = [self.theta[a] @ context for a in range(self.n)]
            return int(np.argmax(scores))

        self.step += 1

        # Prediction step for all arms
        for a in range(self.n):
            self._predict(a)

        # Compute scores with UCB exploration
        scores = []
        for a in range(self.n):
            # Predicted reward
            pred = self.theta[a] @ context
            # Uncertainty
            var = context @ self.P[a] @ context
            if var < 0:
                var = 0.0
            # Adaptive exploration: scale by confidence in stable regime
            p_stable = self.state_prob[a][0]
            # More exploration when volatile (less confident)
            alpha_eff = self.alpha * (1.0 + 0.5 * (1.0 - p_stable))
            ucb = pred + alpha_eff * np.sqrt(var)
            scores.append(ucb)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Deadline guard
        if time.time() - self.t_start > self.deadline:
            return

        # Kalman update step
        # Innovation
        pred = self.theta[arm] @ context
        innovation = reward - pred

        # Innovation variance
        S = context @ self.P[arm] @ context + self.R
        if S < 1e-10:
            S = 1e-10

        # Gain
        K = (self.P[arm] @ context) / S

        # Update state
        self.theta[arm] = self.theta[arm] + K * innovation
        self.P[arm] = self.P[arm] - np.outer(K, context)

        # Ensure symmetry
        self.P[arm] = 0.5 * (self.P[arm] + self.P[arm].T)

        # Update HMM
        self._update_hmm(arm, innovation, S)

        # Maintain innovation history
        self.innov_history[arm].append(innovation)
        if len(self.innov_history[arm]) > self.max_history:
            self.innov_history[arm].pop(0)

        # Decay HMM state probabilities toward prior if no recent large innovations
        # This helps recover from false volatile detections
        recent_innov = self.innov_history[arm]
        if len(recent_innov) > 5:
            avg_abs_innov = np.mean([abs(x) for x in recent_innov[-10:]])
            if avg_abs_innov < 0.5:
                # Gently pull back toward stable
                self.state_prob[arm] = 0.95 * self.state_prob[arm] + 0.05 * self.pi
# EVOLVE-BLOCK-END

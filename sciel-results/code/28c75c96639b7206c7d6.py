# EVOLVE-BLOCK-START
import time
import numpy as np


class Policy:
    """Per-arm Kalman filter with regime-switching process noise.

    Each arm maintains two Kalman filter states: a 'stable' mode with low
    process noise and a 'volatile' mode with high process noise. A lightweight
    HMM tracks the probability of being in each mode, switching based on
    innovation magnitude. This explicitly distinguishes gradual drift from
    abrupt structural breaks.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.start_time = time.time()
        self.deadline = 115.0

        # Dual-state Kalman filters per arm
        self.theta_stable = [np.zeros(dim) for _ in range(n_arms)]
        self.theta_volatile = [np.zeros(dim) for _ in range(n_arms)]
        self.P_stable = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.P_volatile = [np.eye(dim) * 10.0 for _ in range(n_arms)]
        self.count = [0 for _ in range(n_arms)]

        # Process noise parameters for each regime
        self.q_stable = 0.02
        self.q_volatile = 0.5
        self.jitter = 1e-6

        # HMM parameters
        self.p_stay_stable = 0.95    # P(stay stable | stable)
        self.p_stay_volatile = 0.85  # P(stay volatile | volatile)
        self.p_switch_to_volatile = 0.05  # P(volatile | stable)
        self.p_switch_to_stable = 0.15    # P(stable | volatile)

        # Innovation statistics for regime detection
        self.innov_ema_stable = [0.0] * n_arms
        self.innov_ema_volatile = [0.0] * n_arms
        self.ema_alpha = 0.3
        self.regime_prob = [0.5] * n_arms  # P(volatile) per arm

        # UCB exploration parameters
        self.u_scale = 1.5
        self.count_scale = 0.4
        self.warmup_steps = 40
        self.step_count = 0

        # Huber robustification
        self.huber_delta = 1.5

    def _check_deadline(self):
        return (time.time() - self.start_time) > self.deadline

    def _adaptive_u(self):
        if self.step_count < self.warmup_steps:
            return self.u_scale * 1.7
        factor = 1.0 / (1.0 + self.step_count / 800.0)
        return self.u_scale * max(factor, 0.4)

    def _huber_innov(self, innov, std):
        if std <= 0:
            return innov
        ratio = abs(innov) / std
        if ratio <= self.huber_delta:
            return innov
        sign = 1.0 if innov > 0 else -1.0
        return sign * self.huber_delta * std

    def _predict(self, a, mode, q):
        if mode == 'stable':
            self.P_stable[a] = self.P_stable[a] + q * np.eye(self.d) + self.jitter
        else:
            self.P_volatile[a] = self.P_volatile[a] + q * np.eye(self.d) + self.jitter

    def _update(self, a, ctx, reward, mode):
        if mode == 'stable':
            P = self.P_stable[a]
            theta = self.theta_stable[a]
        else:
            P = self.P_volatile[a]
            theta = self.theta_volatile[a]

        pred = float(theta @ ctx)
        innov_raw = reward - pred
        var = float(ctx @ P @ ctx) + self.jitter
        std = float(np.sqrt(var))
        innov = self._huber_innov(innov_raw, std)
        K = P @ ctx / var
        new_theta = theta + (K * innov)
        IK = np.eye(self.d) - np.outer(K, ctx)
        new_P = IK @ P @ IK.T
        new_P = 0.5 * (new_P + new_P.T)
        np.fill_diagonal(new_P, np.maximum(np.diag(new_P), 0.0))

        if mode == 'stable':
            self.theta_stable[a] = new_theta
            self.P_stable[a] = new_P
        else:
            self.theta_volatile[a] = new_theta
            self.P_volatile[a] = new_P

        self.count[a] += 1
        return innov_raw

    def _update_regime(self, a, innov_raw, ctx):
        """Update regime probability based on innovation and context."""
        P_stab = self.P_stable[a]
        P_vola = self.P_volatile[a]
        
        # Compute predicted variance in each regime
        var_stab = float(ctx @ P_stab @ ctx) + self.jitter
        var_vola = float(ctx @ P_vola @ ctx) + self.jitter
        
        # Likelihood of observation under each regime
        # Assuming Gaussian likelihood with unit observation noise
        log_lik_stab = -0.5 * (innov_raw ** 2) / (var_stab + 1.0)
        log_lik_vola = -0.5 * (innov_raw ** 2) / (var_vola + 1.0)
        
        # Bayesian update of regime probability
        p_vola = self.regime_prob[a]
        p_stab = 1.0 - p_vola
        
        # Predicted probabilities
        p_vola_pred = p_stab * self.p_switch_to_volatile + p_vola * self.p_stay_volatile
        p_stab_pred = p_stab * self.p_stay_stable + p_vola * self.p_switch_to_stable
        
        # Posterior probabilities
        p_vola_post = p_vola_pred * np.exp(log_lik_vola)
        p_stab_post = p_stab_pred * np.exp(log_lik_stab)
        total = p_vola_post + p_stab_post
        if total > 0:
            self.regime_prob[a] = p_vola_post / total
        else:
            self.regime_prob[a] = 0.5

    def select(self, context):
        ctx = np.asarray(context, dtype=float)
        if self._check_deadline():
            # Use blended estimate
            scores = []
            for a in range(self.n):
                p_vola = self.regime_prob[a]
                p_stab = 1.0 - p_vola
                mean = p_stab * float(self.theta_stable[a] @ ctx) + \
                       p_vola * float(self.theta_volatile[a] @ ctx)
                scores.append(mean)
            return int(np.argmax(scores))

        self.step_count += 1
        u = self._adaptive_u()

        best = -np.inf
        best_arm = 0
        for a in range(self.n):
            p_vola = self.regime_prob[a]
            p_stab = 1.0 - p_vola
            
            # Blended mean and covariance
            mean = p_stab * float(self.theta_stable[a] @ ctx) + \
                   p_vola * float(self.theta_volatile[a] @ ctx)
            
            # Blended uncertainty (simplified)
            var_stab = float(ctx @ self.P_stable[a] @ ctx)
            var_vola = float(ctx @ self.P_volatile[a] @ ctx)
            blended_var = p_stab * var_stab + p_vola * var_vola
            std = float(np.sqrt(max(blended_var, 0.0)))
            
            c = self.count[a]
            bonus = u * std / (1.0 + self.count_scale * np.sqrt(c))
            if c == 0:
                bonus += 1.0
            score = mean + bonus
            if score > best:
                best = score
                best_arm = a
        return int(best_arm)

    def update(self, context, arm, reward):
        ctx = np.asarray(context, dtype=float)
        if self._check_deadline():
            return

        self.t += 1
        p_vola = self.regime_prob[arm]
        
        # Update both regimes
        innov_stab = self._update(arm, ctx, reward, 'stable')
        innov_vola = self._update(arm, ctx, reward, 'volatile')
        
        # Update regime probability
        # Use the innovation from the more likely regime
        if p_vola > 0.5:
            innov_for_regime = innov_vola
        else:
            innov_for_regime = innov_stab
        
        self._update_regime(arm, innov_for_regime, ctx)
        
        # Apply process noise prediction for next step
        if p_vola > 0.5:
            self._predict(arm, 'volatile', self.q_volatile)
        else:
            self._predict(arm, 'stable', self.q_stable)
# EVOLVE-BLOCK-END

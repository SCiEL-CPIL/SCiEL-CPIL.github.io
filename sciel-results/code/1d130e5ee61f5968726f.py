"""
Non-stationary linear contextual bandit with Bayesian Online Changepoint Detection (BOCPD).

Combines the structural idea of BOCPD (explicitly tracking regime age and resetting state on
changepoints) with the refinement machinery of a Kalman filter (covariance-based uncertainty
and stable matrix updates).

- Per-arm Kalman filter state: theta (mean), P (covariance).
- Per-arm BOCPD: posterior over age t in {0, 1, ..., max_age}.
- Hazard rate h adapts based on prediction error (dual-timescale adaptation).
- On high probability of age=0 (changepoint), reset the arm's Kalman state.
- UCB exploration using sqrt(context^T P context).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.dim = dim
        self.start_time = time.time()
        self.deadline = 118.0
        self.round = 0

        # Per-arm Kalman states
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Initialize P large to allow fast initial learning
        self.P = [np.eye(dim) * 10.0 for _ in range(n_arms)]

        # BOCPD parameters
        self.max_age = 50
        self.hazard_base = 0.02
        self.hazard_min = 0.005
        self.hazard_max = 0.2

        # Per-arm hazard rate (adaptive)
        self.hazard = [self.hazard_base] * n_arms

        # Per-arm age posterior: p(t) for t=0..max_age
        # t=0 means "just changed"
        self.age_posterior = [np.zeros(self.max_age + 1) for _ in range(n_arms)]
        for a in range(n_arms):
            self.age_posterior[a][0] = 1.0  # Start with age 0

        # Per-arm running stats for age prediction (mean and variance of rewards per age)
        # We maintain a running mean and variance for each age bin.
        # Since max_age is small, this is cheap.
        # For each age t, we store:
        #   count[t], sum_r[t], sum_r2[t]
        # But we only observe one reward per round for one arm.
        # We update the stats for the *current* age of that arm.
        self.age_count = [np.zeros(self.max_age + 1) for _ in range(n_arms)]
        self.age_sum = [np.zeros(self.max_age + 1) for _ in range(n_arms)]
        self.age_sum2 = [np.zeros(self.max_age + 1) for _ in range(n_arms)]

        # Prediction error tracking for hazard adaptation
        self.pred_error_ema = np.zeros(n_arms)
        self.error_alpha = 0.1

        # Exploration
        self.alpha = 1.5
        self.alpha_min = 0.5
        self.alpha_max = 3.0

        # Fallback
        self.arm_mean_reward = [0.0] * n_arms
        self.arm_count = [0] * n_arms
        self.best_arm = 0
        self.best_reward = -np.inf
        self.recent_rewards = [[] for _ in range(n_arms)]
        self.recent_reward_window = 20

        # Observation noise
        self.R = 1.0

    def _check_deadline(self):
        return time.time() - self.start_time < self.deadline

    def _adapt_hazard(self, arm, pred_error):
        """Adapt hazard rate based on prediction error."""
        self.pred_error_ema[arm] = (1 - self.error_alpha) * self.pred_error_ema[arm] + \
                                   self.error_alpha * abs(pred_error)
        e = self.pred_error_ema[arm]
        # Map error to hazard: higher error -> higher hazard (more likely to be in a new regime)
        if e < 0.5:
            h = self.hazard_base
        elif e < 2.0:
            h = self.hazard_base + (0.1 - self.hazard_base) * (e - 0.5) / 1.5
        else:
            h = 0.1 + (self.hazard_max - 0.1) * min(1.0, (e - 2.0) / 5.0)
        self.hazard[arm] = max(self.hazard_min, min(self.hazard_max, h))

    def _update_age_posterior(self, arm, reward, context):
        """Update the age posterior for a given arm."""
        h = self.hazard[arm]
        post = self.age_posterior[arm]
        
        # Prediction step: age increases by 1 with probability (1-h), stays at 0 with probability h
        # p_pred(t=0) = h * sum(p_old)
        # p_pred(t=k) = (1-h) * p_old(t=k-1) for k>=1
        p_pred = np.zeros(self.max_age + 1)
        p_pred[0] = h
        p_pred[1:] = (1 - h) * post[:-1]
        
        # Likelihood: P(reward | age=t, context, params)
        # We use the Kalman prediction for each age? No, that's expensive.
        # Instead, we use the running stats for each age bin.
        # Likelihood is Gaussian: N(reward; mean_t, var_t + R)
        # If count[t] is small, use a prior (e.g., mean=0, var=large)
        
        log_lik = np.zeros(self.max_age + 1)
        for t in range(self.max_age + 1):
            c = self.age_count[arm][t]
            if c < 1:
                # Prior: assume mean 0, var 10
                mu = 0.0
                var = 10.0 + self.R
            else:
                mu = self.age_sum[arm][t] / c
                var_s = self.age_sum2[arm][t] / c - mu**2
                if var_s < 0:
                    var_s = 0.0
                # Add observation noise and shrinkage for small c
                var = var_s + self.R + 1.0 / (c + 1)
            
            # Log likelihood of reward given (mu, var)
            # -0.5 * log(2*pi*var) - 0.5 * (reward-mu)^2 / var
            log_lik[t] = -0.5 * np.log(2 * np.pi * var) - 0.5 * (reward - mu) ** 2 / var
        
        # Update posterior: p_post(t) = p_pred(t) * exp(log_lik(t))
        # Use log-sum-exp for stability
        log_p_pred = np.log(p_pred + 1e-10)
        log_post = log_p_pred + log_lik
        max_log = np.max(log_post)
        log_post -= max_log
        weights = np.exp(log_post)
        weights /= np.sum(weights)
        
        self.age_posterior[arm] = weights
        
        # Update running stats for the *current* age (the one with highest posterior? or just the one we used for prediction?)
        # Standard BOCPD updates stats for all ages, but that's expensive.
        # Here, we update stats for the age that is most likely, or simply increment the age we just observed.
        # Actually, in BOCPD, the "age" is the number of steps since the last changepoint.
        # The reward we just observed belongs to the current regime.
        # We should update the stats for the age that we believe is current.
        # Let's update the stats for the age with the highest posterior probability.
        current_age = int(np.argmax(weights))
        self.age_count[arm][current_age] += 1
        self.age_sum[arm][current_age] += reward
        self.age_sum2[arm][current_age] += reward ** 2

    def _maybe_reset(self, arm):
        """Reset Kalman state if changepoint is likely."""
        # If probability of age=0 is high, reset
        p_change = self.age_posterior[arm][0]
        if p_change > 0.5:
            # Reset
            self.theta[arm] = np.zeros(self.d)
            self.P[arm] = np.eye(self.d) * 10.0
            # Reset age stats? Maybe keep them for context, but reset the posterior to age 0
            self.age_posterior[arm] = np.zeros(self.max_age + 1)
            self.age_posterior[arm][0] = 1.0
            # Reset running stats for this arm to avoid stale data
            self.age_count[arm] = np.zeros(self.max_age + 1)
            self.age_sum[arm] = np.zeros(self.max_age + 1)
            self.age_sum2[arm] = np.zeros(self.max_age + 1)

    def select(self, context):
        self.round += 1
        context = np.asarray(context, dtype=np.float64).flatten()

        if not self._check_deadline():
            scores = []
            for a in range(self.n):
                scores.append(self.theta[a] @ context)
            return int(np.argmax(scores))

        # Adaptive alpha
        r = self.round
        if r < 30:
            self.alpha = self.alpha_max
        elif r < 100:
            self.alpha = self.alpha_max - (self.alpha_max - 1.5) * (r - 30) / 70.0
        elif r < 300:
            self.alpha = 1.5 - (1.5 - 1.0) * (r - 100) / 200.0
        else:
            self.alpha = max(self.alpha_min, 1.0 - 0.3 * (r - 300) / 1000.0)

        scores = np.zeros(self.n)
        for a in range(self.n):
            mean = self.theta[a] @ context
            # Uncertainty
            P_ctx = self.P[a] @ context
            var_term = context @ P_ctx
            if var_term < 0:
                var_term = 0.0
            scores[a] = mean + self.alpha * np.sqrt(var_term)
        
        # Small random tiebreaker
        scores += np.random.normal(0, 0.05, self.n)
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64).flatten()
        reward = float(reward)

        # Track for fallback
        self.arm_count[arm] += 1
        self.recent_rewards[arm].append(reward)
        if len(self.recent_rewards[arm]) > self.recent_reward_window:
            self.recent_rewards[arm].pop(0)
        self.arm_mean_reward[arm] = np.mean(self.recent_rewards[arm]) if self.recent_rewards[arm] else 0.0
        if self.arm_count[arm] >= 3:
            if self.arm_mean_reward[arm] > self.best_reward:
                self.best_reward = self.arm_mean_reward[arm]
                self.best_arm = arm

        # Prediction error
        pred = self.theta[arm] @ context
        pred_error = reward - pred

        # Adapt hazard
        self._adapt_hazard(arm, pred_error)

        # Update age posterior
        self._update_age_posterior(arm, reward, context)

        # Maybe reset
        self._maybe_reset(arm)

        # Kalman Filter Update
        # Prediction step: theta_pred = theta, P_pred = P + Q
        # We use a fixed small Q to account for noise/drift between observations
        Q = 0.01 * np.eye(self.d)
        P_pred = self.P[arm] + Q

        # Update step
        # Innovation
        innovation = pred_error
        # Innovation Covariance
        S = context @ P_pred @ context + self.R
        # Kalman Gain
        if S > 1e-10:
            K = P_pred @ context / S
        else:
            K = np.zeros(self.d)
        
        # State Update
        self.theta[arm] = self.theta[arm] + K * innovation
        
        # Covariance Update (Joseph form)
        I_KH = np.eye(self.d) - np.outer(K, context)
        self.P[arm] = I_KH @ P_pred @ I_KH.T + np.outer(K, K) * self.R
        self.P[arm] = 0.5 * (self.P[arm] + self.P[arm].T)
# EVOLVE-BLOCK-END

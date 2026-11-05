"""Non-stationary linear contextual bandit with smoothed adaptive forgetting and UCB exploration."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit with:
    - Per-arm discounted least-squares with smoothed adaptive forgetting
    - UCB exploration with error-driven scaling
    - Ridge regularization for numerical stability
    - Deadline guard
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = 145.0  # Safety margin before 150s kill

        # Per-arm state
        self.V = [np.eye(dim) * 1.0 for _ in range(n_arms)]  # inverse covariance
        self.b = [np.zeros(dim) for _ in range(n_arms)]       # weighted sum of rewards*x
        self.counts = [0] * n_arms
        self.theta = [np.zeros(dim) for _ in range(n_arms)]   # cached theta = V @ b

        # Global adaptive parameters
        self.gamma_base = 0.95     # Base forgetting factor
        self.gamma_min = 0.80
        self.gamma_max = 0.995
        self.lambda_ridge = 0.1    # Ridge regularization

        # Exploration parameters
        self.C_base = 2.0
        self.C_min = 0.5
        self.C_max = 5.0
        self.round_num = 0

        # Per-arm error tracking for adaptive gamma
        # We track EWMA of squared prediction errors to estimate drift/noise
        self.arm_err_ewma = [0.0 for _ in range(n_arms)]
        self.arm_err_ewma_alpha = 0.1  # Smoothing factor for error tracking
        
        # Baseline error estimation (long-term average)
        self.arm_err_baseline = [None] * n_arms
        self.arm_err_baseline_window = 50
        self.arm_err_history = [[] for _ in range(n_arms)]

    def _check_deadline(self):
        return (time.time() - self.start_time) < self.deadline

    def _get_C(self, arm, normalized_error):
        """
        Exploration coefficient.
        Scales with the estimated error/drift level.
        """
        c = self.counts[arm]
        t = self.round_num
        
        # Base decay over time
        decay = 1.0 / (1.0 + t / 200.0)
        C_time = self.C_min + (self.C_base - self.C_min) * decay
        
        # Error-driven scaling: if the model is performing poorly (high error),
        # we should explore more to find the new optimal arm.
        # normalized_error is in [0, 1] roughly, scaled by baseline.
        error_scale = 1.0 + 1.5 * min(normalized_error, 2.0)
        
        C_arm = C_time * error_scale
        
        # Ensure strong exploration for new arms
        if c < 3:
            C_arm = max(C_arm, self.C_max * 0.8)
            
        return min(C_arm, self.C_max)

    def _adaptive_gamma(self, arm, current_err_sq):
        """
        Adapt forgetting factor based on smoothed error signal.
        """
        # Update EWMA of squared error
        alpha = self.arm_err_ewma_alpha
        if self.arm_err_ewma[arm] == 0.0 and self.counts[arm] < 5:
             # Initialize with current value if new
             self.arm_err_ewma[arm] = current_err_sq
        else:
             self.arm_err_ewma[arm] = (1 - alpha) * self.arm_err_ewma[arm] + alpha * current_err_sq
        
        # Maintain a history of recent errors for baseline calculation
        self.arm_err_history[arm].append(current_err_sq)
        if len(self.arm_err_history[arm]) > self.arm_err_baseline_window:
            self.arm_err_history[arm].pop(0)
            
        # Calculate baseline (long-term average of recent window)
        if len(self.arm_err_history[arm]) >= 10:
            baseline = np.mean(self.arm_err_history[arm])
            if self.arm_err_baseline[arm] is None:
                self.arm_err_baseline[arm] = baseline
            # Smoothly update baseline to track long-term trends
            self.arm_err_baseline[arm] = 0.95 * self.arm_err_baseline[arm] + 0.05 * baseline
        else:
            baseline = self.arm_err_ewma[arm] if self.arm_err_ewma[arm] > 0 else 1e-6

        # Determine gamma
        gamma = self.gamma_base
        
        if baseline > 1e-8:
            # Ratio of current smoothed error to baseline
            ratio = self.arm_err_ewma[arm] / baseline
            
            # If current error is higher than baseline, the environment has changed
            # or the model is stale. Decrease gamma (forget faster).
            if ratio > 1.2:
                # Scale the reduction based on how much higher the error is
                reduction = 0.15 * min(ratio - 1.0, 3.0)
                gamma -= reduction
            elif ratio < 0.8:
                # If error is lower, the model is stable. Increase gamma (forget slower).
                increase = 0.05 * (1.0 - ratio)
                gamma += increase

        gamma = max(self.gamma_min, min(self.gamma_max, gamma))
        return gamma

    def _update_theta(self, arm):
        """Update cached theta for arm."""
        self.theta[arm] = self.V[arm] @ self.b[arm]

    def select(self, context):
        context = np.asarray(context, dtype=np.float64)

        # Deadline guard
        if not self._check_deadline():
            # Greedy fallback
            scores = []
            for a in range(self.n):
                scores.append(self.theta[a] @ context)
            return int(np.argmax(scores))

        scores = []
        for a in range(self.n):
            mu = self.theta[a] @ context
            
            # Calculate normalized error for exploration scaling
            # Use the EWMA error relative to a global scale or baseline
            if self.arm_err_baseline[a] is not None and self.arm_err_baseline[a] > 1e-8:
                norm_err = self.arm_err_ewma[a] / self.arm_err_baseline[a]
            else:
                norm_err = 1.0
            
            c_val = self._get_C(a, norm_err)
            
            # UCB bonus: c * sqrt(x^T V x)
            Vx = self.V[a] @ context
            var_term = context @ Vx
            if var_term < 0:
                var_term = 0.0
            bonus = c_val * np.sqrt(var_term)
            scores.append(mu + bonus)

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        self.round_num += 1

        # Track prediction error for adaptive gamma
        pred = self.theta[arm] @ context
        pred_err = reward - pred
        pred_err_sq = pred_err * pred_err
        
        # Adaptive forgetting
        gamma = self._adaptive_gamma(arm, pred_err_sq)

        # Update inverse covariance with forgetting + ridge
        # V_new = (gamma * V_old^{-1} + x x^T)^{-1}
        # Using Sherman-Morrison:
        # V_new = (1/gamma) * V_old - (1/gamma) * (V_old x x^T V_old) / (gamma + x^T V_old x)
        
        V_old = self.V[arm]
        Vx = V_old @ context
        xVx = context @ Vx  # scalar
        denom = gamma + xVx

        if denom > 1e-10:
            V_new = (1.0 / gamma) * V_old - (1.0 / gamma) * np.outer(Vx, Vx) / denom
        else:
            # Fallback to regularized identity if singular
            V_new = np.eye(self.d) * (1.0 / self.lambda_ridge)

        # Symmetrize for numerical stability
        V_new = 0.5 * (V_new + V_new.T)

        # Clip: ensure positive semi-definite by adding small ridge if needed
        # Check diagonal
        diag = np.diag(V_new)
        if np.any(diag < 1e-8) or np.any(~np.isfinite(diag)):
            V_new = np.eye(self.d) * (1.0 / self.lambda_ridge)
            V_new = 0.5 * (V_new + V_new.T)

        self.V[arm] = V_new

        # Update b with forgetting
        self.b[arm] = gamma * self.b[arm] + reward * context

        # Update theta
        self._update_theta(arm)
        
        # Increment count after update
        self.counts[arm] += 1
# EVOLVE-BLOCK-END

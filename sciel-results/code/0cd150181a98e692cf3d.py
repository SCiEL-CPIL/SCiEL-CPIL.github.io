# EVOLVE-BLOCK-START
import time
import numpy as np


class Policy:
    """
    Per-Arm Non-Stationary Linear Bandit with Adaptive Regularized Least Squares (ARLS).
    
    This policy replaces the standard Kalman Filter with a sliding-window recursive 
    least squares (RLS) approach with adaptive forgetting.
    
    Core Structural Changes:
    1. State Representation: Maintains A (inverse Hessian) and b (moment vector) per arm.
       theta = A^-1 * b.
    2. Non-Stationarity: Uses a forgetting factor lambda_t that adapts based on 
       innovation magnitude. High innovation -> low lambda (fast forgetting).
    3. Robustness: Huber-loss clipping on innovations to prevent outlier corruption.
    4. Exploration: Adaptive UCB based on the diagonal of A^-1 (variance).
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.start_time = time.time()
        self.deadline = 115.0  # Safety margin before the 120s kill

        # ARLS State:
        # A_a is the inverse Hessian matrix (regularized sum of x x^T)
        # b_a is the moment vector (regularized sum of x r)
        # theta_a = A_a^-1 b_a
        self.A = [np.eye(dim) * 1.0 for _ in range(n_arms)]  # Regularization alpha=1.0
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        
        # Forgetting factor parameters
        self.lambda_min = 0.7   # Fast forgetting (short memory)
        self.lambda_max = 0.98  # Slow forgetting (long memory)
        self.lambda_base = 0.90 # Default forgetting factor
        
        # Innovation tracking for adaptive lambda
        self.innov_ema = [0.0] * n_arms
        self.innov_ema_alpha = 0.2 # Slow adaptation of lambda to avoid oscillation
        self.innov_threshold_low = 0.5  # Below this, use lambda_max
        self.innov_threshold_high = 5.0 # Above this, use lambda_min
        
        # Huber robustification
        self.huber_delta = 2.5
        
        # UCB Exploration
        self.u_scale = 1.5
        self.count_scale = 0.4
        self.warmup_steps = 20
        self.jitter = 1e-6
        self.step_count = 0
        self.count = [0] * n_arms

    def _check_deadline(self):
        return (time.time() - self.start_time) > self.deadline

    def _compute_adaptive_lambda(self, arm, innov_abs):
        """
        Compute the forgetting factor lambda for the current step.
        Low innovation -> High lambda (keep history).
        High innovation -> Low lambda (forget history).
        """
        # Update EMA of absolute innovation
        self.innov_ema[arm] = (1.0 - self.innov_ema_alpha) * self.innov_ema[arm] + \
                               self.innov_ema_alpha * innov_abs
        ema_val = self.innov_ema[arm]
        
        # Map EMA to lambda range
        # If ema < low, lambda = lambda_max
        # If ema > high, lambda = lambda_min
        # Linear interpolation in between
        if ema_val <= self.innov_threshold_low:
            return self.lambda_max
        elif ema_val >= self.innov_threshold_high:
            return self.lambda_min
        else:
            # Interpolate
            ratio = (ema_val - self.innov_threshold_low) / (self.innov_threshold_high - self.innov_threshold_low)
            return self.lambda_max - ratio * (self.lambda_max - self.lambda_min)

    def _huber_innov(self, innov, std):
        """
        Robustify innovation using Huber loss logic.
        """
        if std <= 0:
            return innov
        ratio = abs(innov) / std
        if ratio <= self.huber_delta:
            return innov
        sign = 1.0 if innov > 0 else -1.0
        return sign * self.huber_delta * std

    def _update_arsls(self, arm, ctx, reward):
        """
        Perform the ARLS update step.
        1. Compute prediction and innovation.
        2. Determine adaptive lambda.
        3. Decay A and b by lambda.
        4. Update A and b with new observation.
        5. Solve for theta.
        """
        A = self.A[arm]
        b = self.b[arm]
        theta = self.theta[arm]
        
        # 1. Prediction and Innovation
        pred = float(theta @ ctx)
        innov_raw = reward - pred
        
        # Estimate variance for Huber clipping
        # Approximate variance of prediction using diagonal of A^-1 (if A is well-conditioned)
        # For robustness, we use a fixed std estimate based on recent innovation EMA if A^-1 is hard to compute
        # However, we can compute A^-1 ctx for the gain, so let's use that for std estimation if needed.
        # Actually, for Huber, we need the std of the innovation. 
        # In RLS, the innovation variance is approx 1 + x^T (A^-1) x ? No, that's for Bayesian.
        # In standard RLS, the residual variance is not directly tracked in the same way.
        # We will use the EMA of squared innovations as a proxy for std.
        std_proxy = np.sqrt(max(self.innov_ema[arm], 1.0))
        
        innov_robust = self._huber_innov(innov_raw, std_proxy)
        
        # 2. Adaptive Forgetting
        lambda_t = self._compute_adaptive_lambda(arm, abs(innov_raw))
        
        # 3. Decay
        A = A * lambda_t
        b = b * lambda_t
        
        # 4. Update with new observation
        # A <- A + x x^T
        # b <- b + x * r
        # Using outer product
        A += np.outer(ctx, ctx)
        b += ctx * innov_robust
        
        # Add jitter to A for numerical stability
        A += self.jitter * np.eye(self.d)
        
        # 5. Solve for theta
        # theta = A^-1 b
        try:
            theta_new = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if singular
            theta_new = np.linalg.pinv(A) @ b
            
        self.A[arm] = A
        self.b[arm] = b
        self.theta[arm] = theta_new
        self.count[arm] += 1
        
        return innov_raw

    def _adaptive_u(self):
        """Adapt exploration coefficient."""
        if self.step_count < self.warmup_steps:
            return self.u_scale * 1.5
        factor = 1.0 / (1.0 + self.step_count / 1000.0)
        return self.u_scale * max(factor, 0.5)

    def select(self, context):
        ctx = np.asarray(context, dtype=np.float64)
        
        # Deadline guard
        if self._check_deadline():
            return int(np.argmax([float(t @ ctx) for t in self.theta]))

        self.step_count += 1
        u = self._adaptive_u()

        best = -np.inf
        best_arm = 0
        
        for a in range(self.n):
            theta = self.theta[a]
            mean = float(theta @ ctx)
            
            # Calculate uncertainty
            # In RLS, the variance of the prediction is x^T (A^-1) x
            # We need A^-1. Since we store A, we can compute the gain K = A^-1 x / (1 + x^T A^-1 x) ?
            # No, in standard RLS, the prediction variance is x^T P x where P = A^-1.
            # Computing full inverse is O(d^3). Computing x^T A^-1 x is O(d^2) if we solve A z = x.
            
            try:
                # Solve A z = x to get z = A^-1 x
                z = np.linalg.solve(self.A[a], ctx)
                # Variance of prediction is x^T z
                var_pred = float(ctx @ z)
                std = np.sqrt(max(var_pred, 0.0))
            except np.linalg.LinAlgError:
                std = 1.0 # Fallback uncertainty

            c = self.count[a]
            # UCB Bonus
            # Standard UCB uses std. We dampen it by count.
            bonus = u * std / (1.0 + self.count_scale * np.sqrt(c))
            
            if c == 0:
                bonus += 2.0 # Encourage exploration of unseen arms

            score = mean + bonus
            if score > best:
                best = score
                best_arm = a
                
        return int(best_arm)

    def update(self, context, arm, reward):
        ctx = np.asarray(context, dtype=np.float64)
        reward = float(reward)

        if self._check_deadline():
            return

        self.t += 1
        self._update_arsls(arm, ctx, reward)

        # Periodic numerical check
        if self.step_count % 100 == 0:
            self._stabilize()

    def _stabilize(self):
        """Check for numerical instability and reset if necessary."""
        for a in range(self.n):
            A = self.A[a]
            theta = self.theta[a]
            if not np.all(np.isfinite(A)) or not np.all(np.isfinite(theta)):
                # Reset to initial state
                self.A[a] = np.eye(self.d) * 1.0
                self.b[a] = np.zeros(self.d)
                self.theta[a] = np.zeros(self.d)
                self.count[a] = 0
                self.innov_ema[a] = 0.0
# EVOLVE-BLOCK-END

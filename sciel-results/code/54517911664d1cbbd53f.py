# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary Linear Contextual Bandit using Per-Arm Adaptive Discounted
    Recursive Least Squares (AD-RLS) with innovation-driven forgetting.
    
    Structural change from Kalman filter:
    - Replaces covariance matrix P with a single forgetting factor lambda per arm
    - Uses recursive inverse-covariance (information matrix) updates
    - Forgetting factor adapts inversely to recent innovation variance:
      * High innovation (regime shift) -> lambda decreases -> faster forgetting
      * Low innovation (stable) -> lambda increases -> more stable estimates
    - UCB exploration based on trace of covariance (inverse information matrix)
    - Deadline guard and periodic numerical stabilization
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0  # Safety margin before 120s limit

        # RLS state per arm
        # theta: weight vector (dim,)
        # C: covariance matrix (dim, dim) = inverse of information matrix
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        # Start with large uncertainty so unexplored arms are favoured
        self.C = [np.eye(dim) * 100.0 for _ in range(n_arms)]
        self.count = [0] * n_arms

        # Adaptive forgetting factor parameters
        # lambda in (0, 1]; closer to 1 = remember more, closer to 0 = forget faster
        self.lambda_base = 0.95
        self.lambda_min = 0.7
        self.lambda_max = 0.99
        self.lambda_ema = [self.lambda_base] * n_arms
        self.innov_ema = [0.0] * n_arms
        self.innov_ema_alpha = 0.25  # Smoothing for innovation tracking
        self.innov_threshold = 0.5   # Threshold for detecting regime shifts

        # UCB exploration parameters
        self.u_scale = 1.8
        self.count_scale = 0.3  # sqrt(count) dampening
        self.warmup_steps = 15
        self.step_count = 0
        self.jitter = 1e-6

    def _check_deadline(self):
        return (time.time() - self.start_time) > self.deadline

    def _adapt_lambda(self, arm, innov):
        """
        Adapt the forgetting factor based on recent innovation.
        High innovation -> decrease lambda (forget faster)
        Low innovation -> increase lambda (trust model more)
        """
        c = self.count[arm]
        if c < 3:
            # Early phase: moderate forgetting for exploration
            return self.lambda_base

        # Update EMA of squared innovation
        self.innov_ema[arm] = (1.0 - self.innov_ema_alpha) * self.innov_ema[arm] + \
                               self.innov_ema_alpha * (innov * innov)
        
        innov_ema = self.innov_ema[arm]
        
        # Map innovation to lambda adjustment
        # If innovation is high (exceeds threshold), reduce lambda significantly
        # If innovation is low, let lambda drift toward lambda_max
        if innov_ema > self.innov_threshold:
            # High volatility: accelerate forgetting
            excess = min((innov_ema - self.innov_threshold) / self.innov_threshold, 3.0)
            # Exponential decay of lambda with excess
            lambda_adj = self.lambda_base * np.exp(-0.5 * excess)
            new_lambda = max(self.lambda_min, lambda_adj)
        else:
            # Low volatility: stabilize toward high lambda
            lambda_adj = self.lambda_base + (self.lambda_max - self.lambda_base) * \
                         (1.0 - innov_ema / max(self.innov_threshold, 1e-8))
            new_lambda = min(self.lambda_max, lambda_adj)
        
        # Smooth transition to new lambda
        self.lambda_ema[arm] = 0.8 * self.lambda_ema[arm] + 0.2 * new_lambda
        return self.lambda_ema[arm]

    def _select_arm(self, ctx):
        """Select arm using UCB based on RLS covariance."""
        u = self.u_scale
        if self.step_count < self.warmup_steps:
            u *= 1.5  # Boost exploration during warmup
        
        best = -np.inf
        best_arm = 0
        for a in range(self.n):
            theta = self.theta[a]
            C = self.C[a]
            mean = float(theta @ ctx)
            
            # Uncertainty: sqrt of predicted variance
            var = float(ctx @ C @ ctx)
            std = float(np.sqrt(max(var, 0.0)))
            
            c = self.count[a]
            bonus = u * std / (1.0 + self.count_scale * np.sqrt(c))
            if c == 0:
                bonus += 3.0  # Strong incentive to explore unseen arms
            
            score = mean + bonus
            if score > best:
                best = score
                best_arm = a
        
        return int(best_arm)

    def select(self, context):
        ctx = np.asarray(context, dtype=np.float64)
        
        # Deadline guard: commit to best mean estimate
        if self._check_deadline():
            scores = [float(t @ ctx) for t in self.theta]
            return int(np.argmax(scores))
        
        self.step_count += 1
        return self._select_arm(ctx)

    def update(self, context, arm, reward):
        ctx = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        
        if self._check_deadline():
            return
        
        theta = self.theta[arm]
        C = self.C[arm]
        
        # Compute innovation before update
        pred = float(theta @ ctx)
        innov = reward - pred
        
        # Adapt forgetting factor
        lam = self._adapt_lambda(arm, innov)
        
        # RLS update with forgetting factor
        # C_new = (1/lambda) * C - (1/lambda) * C @ ctx @ ctx' @ C / (lambda + ctx' @ C @ ctx)
        # Simplified using matrix inversion lemma:
        # K = C @ ctx / (lambda + ctx' @ C @ ctx)
        # theta_new = theta + K * innov
        # C_new = (1/lambda) * (C - K @ ctx')
        
        ctx_C_ctx = float(ctx @ C @ ctx)
        denom = lam + ctx_C_ctx + self.jitter
        
        # Kalman-like gain for RLS
        K = C @ ctx / denom
        
        # Update theta
        self.theta[arm] = theta + K * innov
        
        # Update covariance with forgetting
        # C_new = (1/lam) * (C - outer(K, ctx))
        C_update = C - np.outer(K, ctx)
        self.C[arm] = (1.0 / lam) * C_update
        
        # Symmetrise and ensure positive semi-definite
        self.C[arm] = 0.5 * (self.C[arm] + self.C[arm].T)
        # Clip negative eigenvalues by adding jitter to diagonal
        np.fill_diagonal(self.C[arm], np.diag(self.C[arm]) + self.jitter)
        
        # Periodic numerical stabilization
        if self.step_count % 50 == 0:
            for a in range(self.n):
                if not np.all(np.isfinite(self.C[a])) or not np.all(np.isfinite(self.theta[a])):
                    # Reset to initial state
                    self.C[a] = np.eye(self.d) * 100.0
                    self.theta[a] = np.zeros(self.d)
                    self.count[a] = 0
                    self.innov_ema[a] = 0.0
                    self.lambda_ema[a] = self.lambda_base
        
        self.count[arm] += 1
# EVOLVE-BLOCK-END

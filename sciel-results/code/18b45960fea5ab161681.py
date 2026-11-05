"""Bayesian Linear Bandit with Adaptive Mahalanobis Forgetting."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Thompson Sampling for non-stationary linear bandits.
    
    Uses a Bayesian framework where the posterior over weights is Gaussian.
    The precision matrix (inverse covariance) and precision-weighted mean are maintained.
    An adaptive forgetting mechanism is triggered by large Mahalanobis residuals,
    which indicates a potential regime shift in the underlying weights.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Initialize posterior parameters in precision form
        # A = V^{-1}, b = V^{-1} mu
        # Prior: mu = 0, V = I  =>  A = I, b = 0
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # For adaptive forgetting, we track the last reward and context per arm
        self.last_reward = [None] * n_arms
        self.last_context = [None] * n_arms
        
        # Forgetting factor parameters
        self.min_forgetting = 0.90  # Strongest forgetting (keep 10% of old info)
        self.max_forgetting = 1.0   # No forgetting
        
        # Time tracking for deadline guard
        self.start_time = time.time()
        self.deadline = self.start_time + 145.0  # 5s safety margin from 150s limit

    def _get_mean_and_covariance(self, arm):
        """Compute mean and covariance from A and b."""
        try:
            inv_A = np.linalg.inv(self.A[arm])
            mu = inv_A @ self.b[arm]
            V = inv_A
        except np.linalg.LinAlgError:
            # Fallback: add jitter
            jitter = 1e-6 * np.eye(self.d)
            inv_A = np.linalg.inv(self.A[arm] + jitter)
            mu = inv_A @ self.b[arm]
            V = inv_A
        return mu, V

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            # Return the arm with the highest mean reward
            scores = []
            for a in range(self.n):
                mu, _ = self._get_mean_and_covariance(a)
                scores.append(np.dot(mu, context))
            return int(np.argmax(scores))

        context = np.asarray(context, dtype=float)
        
        # Thompson Sampling: sample from posterior
        samples = []
        for a in range(self.n):
            mu, V = self._get_mean_and_covariance(a)
            
            # Sample theta from N(mu, V)
            try:
                # Use Cholesky decomposition for sampling
                L = np.linalg.cholesky(V + 1e-9 * np.eye(self.d))
                z = np.random.randn(self.d)
                theta = mu + L @ z
            except np.linalg.LinAlgError:
                # Fallback if Cholesky fails
                theta = mu + np.random.randn(self.d) * 0.1
            
            samples.append(np.dot(theta, context))
        
        return int(np.argmax(samples))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=float)
        
        # 1. Compute Mahalanobis distance of the current residual
        #    to determine the forgetting factor for this update.
        mu, V = self._get_mean_and_covariance(arm)
        pred = np.dot(mu, context)
        residual = reward - pred
        
        try:
            # Mahalanobis distance: sqrt(residual^T V^{-1} residual)
            # V^{-1} is self.A[arm]
            md2 = residual * (self.A[arm] @ context * residual) # This is wrong, A @ context is not the residual scaling
            # Correct: md2 = residual^2 * (V^{-1})_{ii} is for 1D.
            # For multivariate: md2 = residual^T V^{-1} residual
            # But residual is a scalar. The prediction error variance is context^T V context.
            # The standardized residual is residual / sqrt(context^T V context).
            # Let's use the standardized residual as a proxy for Mahalanobis distance of the point.
            var_pred = context @ V @ context
            if var_pred > 1e-9:
                md = abs(residual) / np.sqrt(var_pred)
            else:
                md = abs(residual)
        except Exception:
            md = abs(residual)
        
        # 2. Determine forgetting factor
        #    If md is large, the model is surprised. We should forget more.
        #    Map md to a forgetting factor between min_forgetting and max_forgetting.
        #    Use a smooth function: alpha = max_forgetting - (max_forgetting - min_forgetting) * (1 - exp(-md / 2.0))
        #    If md=0, alpha=1.0 (no forgetting).
        #    If md->inf, alpha->min_forgetting.
        
        alpha = self.max_forgetting - (self.max_forgetting - self.min_forgetting) * (1 - np.exp(-md / 2.0))
        
        # 3. Update A and b with forgetting
        #    A_new = alpha * A_old + context @ context
        #    b_new = alpha * b_old + reward * context
        
        self.A[arm] = alpha * self.A[arm] + np.outer(context, context)
        self.b[arm] = alpha * self.b[arm] + reward * context
        
        # 4. Store last reward and context for potential future use (not strictly needed if we compute md online)
        self.last_reward[arm] = reward
        self.last_context[arm] = context

# EVOLVE-BLOCK-END

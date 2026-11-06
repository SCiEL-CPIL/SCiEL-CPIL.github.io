"""Non-stationary linear contextual bandit with Thompson Sampling and adaptive decay."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary Linear Contextual Bandit using Thompson Sampling.
    
    Strategy:
    1. Maintain a recency-weighted posterior (exponential forgetting) for each arm.
    2. Use Thompson Sampling for action selection to balance exploration/exploitation.
    3. Adaptive decay rate based on prediction error to better track non-stationarity.
    4. Use Cholesky decomposition for efficient sampling when possible.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        
        # Initialize posterior parameters for each arm
        # A: precision matrix (inverse covariance)
        # b: cross-covariance vector
        self.A = [np.eye(dim) * 1.0 for _ in range(n_arms)]  # Regularized
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Adaptive decay parameters
        self.base_decay = 0.98  # Lower base decay for faster adaptation
        self.min_decay = 0.95
        self.max_decay = 0.999
        self.current_decays = [self.base_decay] * n_arms
        
        # Track prediction errors for adaptive decay
        self.pred_errors = [0.0] * n_arms
        self.error_ema = [0.0] * n_arms
        
        # Counter for steps
        self.step = 0
        
        # Pre-compute Cholesky factors
        self.cholesky = [None] * n_arms
        self._update_cholesky()

    def _update_cholesky(self):
        """Update Cholesky factorization for all arms."""
        for a in range(self.n):
            try:
                # Add small regularization for numerical stability
                L = np.linalg.cholesky(self.A[a] + 1e-8 * np.eye(self.d))
                self.cholesky[a] = L
            except np.linalg.LinAlgError:
                # Fallback if Cholesky fails
                self.cholesky[a] = None

    def _sample_theta(self, arm):
        """Sample theta from the posterior for a given arm using Thompson Sampling."""
        # Posterior mean: theta_hat = A^{-1} b
        # Posterior covariance: A^{-1}
        # We can sample by: theta = theta_hat + A^{-1/2} z, where z ~ N(0, I)
        
        A = self.A[arm]
        b = self.b[arm]
        
        try:
            # Compute posterior mean
            theta_hat = np.linalg.solve(A, b)
            
            # Sample from N(theta_hat, A^{-1})
            # Use Cholesky: A^{-1} = (L^{-1})^T L^{-1}
            # So A^{-1/2} = L^{-1} (if A = L L^T, then A^{-1} = L^{-T} L^{-1})
            # Actually: if A = L @ L.T, then A^{-1} = inv(L.T) @ inv(L)
            # For sampling: x = theta_hat + inv(L.T) @ z where z ~ N(0,I)
            
            if self.cholesky[arm] is not None:
                L = self.cholesky[arm]
                # A = L @ L.T, so A^{-1} = L^{-T} @ L^{-1}
                # Sample: theta = theta_hat + L^{-T} @ z
                z = np.random.randn(self.d)
                # Solve L.T @ v = z, then theta = theta_hat + v
                v = np.linalg.solve(L.T, z)
                theta = theta_hat + v
            else:
                # Fallback: direct inverse
                inv_A = np.linalg.inv(A)
                theta_hat = inv_A @ b
                z = np.random.randn(self.d)
                # Need square root of covariance
                eigvals, eigvecs = np.linalg.eigh(inv_A)
                eigvals = np.maximum(eigvals, 0)  # Ensure non-negative
                sqrt_cov = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                theta = theta_hat + sqrt_cov @ z
        except Exception:
            # Final fallback
            theta = np.zeros(self.d)
        
        return theta

    def select(self, context):
        """Select an arm using Thompson Sampling."""
        context = np.asarray(context, dtype=np.float64)
        self.step += 1
        
        # Check deadline: if we've been running too long, exploit greedily
        elapsed = time.time() - self.start_time
        if elapsed > 115.0:  # Safety margin before 120s limit
            # Greedy selection based on current estimates
            scores = []
            for a in range(self.n):
                try:
                    theta = np.linalg.solve(self.A[a], self.b[a])
                    scores.append(theta @ context)
                except Exception:
                    scores.append(0.0)
            return int(np.argmax(scores))
        
        # Thompson Sampling: sample theta for each arm and pick best
        sampled_scores = []
        for a in range(self.n):
            theta = self._sample_theta(a)
            sampled_scores.append(theta @ context)
        
        return int(np.argmax(sampled_scores))

    def update(self, context, arm, reward):
        """Update the posterior for the selected arm with observed reward."""
        context = np.asarray(context, dtype=np.float64)
        reward = float(reward)
        
        # Compute prediction error for adaptive decay
        try:
            theta_hat = np.linalg.solve(self.A[arm], self.b[arm])
            predicted = theta_hat @ context
            error = reward - predicted
        except Exception:
            error = reward
        
        # Update error EMA
        self.pred_errors[arm] = error
        alpha = 0.1  # EMA smoothing
        self.error_ema[arm] = (1 - alpha) * self.error_ema[arm] + alpha * (error ** 2)
        
        # Adaptive decay: higher error -> lower decay (faster forgetting)
        # Map error magnitude to decay rate
        error_mag = np.sqrt(self.error_ema[arm])
        # Normalize error to [0, 1] range approximately
        normalized_error = min(error_mag / 2.0, 1.0)  # Cap at 1.0
        # Interpolate between max_decay and min_decay
        decay = self.max_decay - normalized_error * (self.max_decay - self.min_decay)
        decay = max(self.min_decay, min(self.max_decay, decay))
        self.current_decays[arm] = decay
        
        # Update posterior with forgetting
        decay = self.current_decays[arm]
        self.A[arm] = decay * self.A[arm] + np.outer(context, context)
        self.b[arm] = decay * self.b[arm] + reward * context
        
        # Update Cholesky factorization
        try:
            L = np.linalg.cholesky(self.A[arm] + 1e-8 * np.eye(self.d))
            self.cholesky[arm] = L
        except np.linalg.LinAlgError:
            self.cholesky[arm] = None
# EVOLVE-BLOCK-END

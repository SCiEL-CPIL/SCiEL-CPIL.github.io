"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Adaptive Discounted Thompson Sampling for Non-Stationary Linear Bandits.
    
    Uses a Gaussian posterior over arm weights with an adaptive discount factor
    that responds to prediction errors, allowing faster adaptation when the
    environment changes.
    """

    def __init__(self, n_arms, dim):
        self.n_arms = n_arms
        self.dim = dim
        
        # Initialize sufficient statistics for each arm
        # A[a] is the precision matrix (inverse covariance)
        # b[a] is the sufficient statistic for the mean
        self.A = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Adaptive discount factor
        self.discount = 0.95
        self.min_discount = 0.8
        self.max_discount = 0.999
        self.error_history = []
        self.error_window = 10
        self.error_threshold = 1.0
        
        # Precompute Cholesky decompositions for efficiency
        self._cache_chol = [None] * n_arms
        self._cache_time = [0.0] * n_arms
        self._start_time = time.time()
        self._deadline = self._start_time + 110  # Safety margin

    def _get_chol(self, a):
        """Get or compute Cholesky decomposition with caching."""
        current_time = time.time()
        if current_time > self._deadline:
            # If we're close to deadline, use a simpler approximation
            return np.linalg.cholesky(self.A[a])
        
        # Check if cache is valid (simplified: recompute every few calls)
        if self._cache_chol[a] is None or current_time - self._cache_time[a] > 0.001:
            try:
                self._cache_chol[a] = np.linalg.cholesky(self.A[a])
                self._cache_time[a] = current_time
            except np.linalg.LinAlgError:
                # Add small regularization if matrix is not positive definite
                self.A[a] += 1e-6 * np.eye(self.dim)
                self._cache_chol[a] = np.linalg.cholesky(self.A[a])
                self._cache_time[a] = current_time
        return self._cache_chol[a]

    def _adapt_discount(self, arm, context, reward):
        """Adapt discount factor based on prediction error."""
        # Compute prediction for the selected arm
        try:
            chol = self._get_chol(arm)
            # Solve for theta: A @ theta = b
            # Using Cholesky: L @ L^T @ theta = b
            # theta = L^-T @ L^-1 @ b
            L = chol
            theta = np.linalg.solve(L.T, np.linalg.solve(L, self.b[arm]))
            prediction = theta @ context
        except Exception:
            prediction = 0.0
            
        error = abs(reward - prediction)
        self.error_history.append(error)
        
        # Keep only recent errors
        if len(self.error_history) > self.error_window:
            self.error_history.pop(0)
        
        # Adapt discount: higher errors -> lower discount (faster forgetting)
        if len(self.error_history) >= 3:
            avg_error = np.mean(self.error_history)
            if avg_error > self.error_threshold:
                # Environment seems non-stationary, forget faster
                self.discount = max(self.min_discount, self.discount * 0.95)
            elif avg_error < 0.3:
                # Environment seems stable, can retain more history
                self.discount = min(self.max_discount, self.discount * 1.01)

    def select(self, context):
        """Select arm using Thompson Sampling."""
        if time.time() > self._deadline:
            # Emergency fallback: return best arm based on current estimates
            scores = []
            for a in range(self.n_arms):
                try:
                    theta = np.linalg.solve(self.A[a], self.b[a])
                    scores.append(theta @ context)
                except Exception:
                    scores.append(0.0)
            return int(np.argmax(scores))
        
        scores = []
        for a in range(self.n_arms):
            try:
                chol = self._get_chol(a)
                L = chol
                # Solve for theta: A @ theta = b
                theta = np.linalg.solve(L.T, np.linalg.solve(L, self.b[a]))
                # Sample from posterior: theta ~ N(theta_hat, (L^-1)^T @ L^-1)
                # theta_sample = theta_hat + L^-T @ z where z ~ N(0, I)
                z = np.random.randn(self.dim)
                theta_sample = theta + np.linalg.solve(L, z)
                scores.append(theta_sample @ context)
            except Exception:
                # Fallback to mean estimate
                try:
                    theta = np.linalg.solve(self.A[a], self.b[a])
                    scores.append(theta @ context)
                except Exception:
                    scores.append(0.0)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        """Update the model with observed reward."""
        # Adapt discount factor first
        self._adapt_discount(arm, context, reward)
        
        # Apply discount to existing statistics
        self.A[arm] *= self.discount
        self.b[arm] *= self.discount
        
        # Add new observation
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
        
        # Invalidate cache for this arm
        self._cache_chol[arm] = None
# EVOLVE-BLOCK-END

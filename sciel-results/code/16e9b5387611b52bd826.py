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
        self.beta = 0.95
        self.beta_min = 0.8
        self.beta_max = 0.999
        self.beta_alpha = 0.01
        
        # For adaptive beta: track recent squared prediction errors
        self.recent_errors = []
        self.error_window = 20
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = 140.0  # 140s to be safe within 150s limit
        
        # Precompute Cholesky for speed
        self._chol = [np.linalg.cholesky(self.A[a]) for a in range(n_arms)]

    def _update_chol(self, arm):
        """Update Cholesky decomposition for arm."""
        try:
            self._chol[arm] = np.linalg.cholesky(self.A[arm])
        except np.linalg.LinAlgError:
            # Add small regularization if not positive definite
            self.A[arm] += 1e-6 * np.eye(self.dim)
            try:
                self._chol[arm] = np.linalg.cholesky(self.A[arm])
            except np.linalg.LinAlgError:
                self._chol[arm] = np.eye(self.dim)

    def select(self, context):
        # Check deadline
        if time.time() > self.deadline:
            # Return best arm based on current means
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n_arms):
                try:
                    theta = np.linalg.solve(self.A[a], self.b[a])
                    score = theta @ context
                    if score > best_score:
                        best_score = score
                        best_arm = a
                except np.linalg.LinAlgError:
                    pass
            return best_arm

        scores = []
        for a in range(self.n_arms):
            # Compute posterior mean and variance using Cholesky
            try:
                L = self._chol[a]
                # Solve L x = b[a]
                x = np.linalg.solve(L, self.b[a])
                theta = np.linalg.solve(L.T, x)
                
                # Variance: context^T A^{-1} context
                # A^{-1} = L^{-T} L^{-1}
                y = np.linalg.solve(L, context)
                variance = np.dot(y, y)
                std = np.sqrt(max(variance, 0.0))
                
                # Thompson sampling: sample from posterior
                z = np.random.normal(0, 1, self.dim)
                sample = theta + std * z
                
                scores.append(sample)
            except np.linalg.LinAlgError:
                scores.append(0.0)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Check deadline
        if time.time() > self.deadline:
            return
        
        # Compute prediction error for adaptive beta
        try:
            L = self._chol[arm]
            x = np.linalg.solve(L, self.b[arm])
            theta = np.linalg.solve(L.T, x)
            pred = theta @ context
            error = reward - pred
            sq_error = error * error
            
            # Update recent errors
            self.recent_errors.append(sq_error)
            if len(self.recent_errors) > self.error_window:
                self.recent_errors.pop(0)
            
            # Adaptive beta: higher variance -> lower beta (faster forgetting)
            if len(self.recent_errors) >= 5:
                mean_err = np.mean(self.recent_errors)
                # Map error magnitude to beta adjustment
                # High error -> lower beta (more forgetting)
                target_beta = self.beta_max - (self.beta_max - self.beta_min) * min(1.0, mean_err / 2.0)
                self.beta = (1 - self.beta_alpha) * self.beta + self.beta_alpha * target_beta
                self.beta = np.clip(self.beta, self.beta_min, self.beta_max)
        except np.linalg.LinAlgError:
            pass

        # Update sufficient statistics with discount
        self.A[arm] = self.beta * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.beta * self.b[arm] + reward * context
        
        # Update Cholesky
        self._update_chol(arm)
# EVOLVE-BLOCK-END

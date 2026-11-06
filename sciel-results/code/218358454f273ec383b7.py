"""Non-stationary linear contextual bandit with adaptive Thompson Sampling."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Non-stationary linear contextual bandit using Thompson Sampling
    with adaptive forgetting (decay) per arm.
    """
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Per-arm precision matrices and cross-moment vectors
        self.A = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Adaptive decay factors per arm
        # Start with long memory
        self.betas = [0.99 for _ in range(n_arms)]
        self.beta_min = 0.80
        self.beta_max = 0.99
        
        # Parameters for adaptive decay
        self.decay_step = 0.05  # How much to reduce beta on shock
        self.grow_step = 0.002  # How much to increase beta on stability
        
        # Tracking
        self.t = 0
        self.start_time = time.time()
        self.deadline = 145.0  # Safety margin
        
        # Cache for Cholesky factors to speed up selection? 
        # Since we update only one arm per step, we can cache inverses for non-updated arms.
        # But Thompson Sampling requires sampling from N(theta, Sigma).
        # We need A_inv and Cholesky of A_inv (or A for sampling via L).
        # Let's just compute on the fly for simplicity first, optimize if too slow.
        # With dim likely small (context dim) and n_arms=5, it should be fast.

    def _sample_theta(self, arm_idx, context):
        """
        Sample theta from the posterior N(theta, Sigma) for the given arm.
        Returns the sampled theta vector.
        """
        A = self.A[arm_idx]
        b = self.b[arm_idx]
        
        # Compute theta_hat = A^{-1} b
        try:
            # Use Cholesky for stability
            L = np.linalg.cholesky(A + 1e-6 * np.eye(self.d))
            # A^{-1} b = L^{-T} L^{-1} b
            v = np.linalg.solve(L.T, b)
            theta_hat = np.linalg.solve(L, v)
            
            # Sample from N(theta_hat, A^{-1})
            # We need to sample z ~ N(0, A^{-1})
            # A^{-1} = L^{-T} L^{-1}
            # z = L^{-T} u, where u ~ N(0, I)
            u = np.random.normal(0, 1, size=self.d)
            # Solve L w = u -> w = L^{-1} u
            w = np.linalg.solve(L, u)
            # z = L^{-T} w
            z = np.linalg.solve(L.T, w)
            
            return theta_hat + z
            
        except np.linalg.LinAlgError:
            # Fallback to pinv if Cholesky fails (should be rare with identity init)
            A_inv = np.linalg.pinv(A)
            theta_hat = A_inv @ b
            # Covariance is A_inv
            # Sample from N(theta_hat, A_inv)
            # Eigen decomposition or Cholesky of A_inv
            try:
                L_inv = np.linalg.cholesky(A_inv + 1e-6 * np.eye(self.d))
                u = np.random.normal(0, 1, size=self.d)
                z = L_inv @ u
                return theta_hat + z
            except:
                # Last resort: diagonal sampling
                theta_sample = theta_hat + np.random.normal(0, 1, size=self.d) * np.sqrt(np.diag(A_inv))
                return theta_sample

    def select(self, context):
        """
        Select arm using Thompson Sampling.
        """
        # Deadline check
        if time.time() - self.start_time > self.deadline:
            # Greedy fallback: pick arm with highest mean prediction
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                try:
                    A_inv = np.linalg.pinv(self.A[a])
                    theta = A_inv @ self.b[a]
                    mu = theta @ context
                    if mu > best_score:
                        best_score = mu
                        best_arm = a
                except:
                    pass
            return best_arm

        scores = []
        for a in range(self.n):
            # Sample theta
            theta_sample = self._sample_theta(a, context)
            # Score is theta_sample^T context
            score = theta_sample @ context
            scores.append(score)
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        """
        Update the model for the selected arm.
        """
        self.t += 1
        
        # 1. Compute prediction and residual using CURRENT model (before update)
        # We need the mean and variance for the adaptive decay logic
        A = self.A[arm]
        b = self.b[arm]
        
        try:
            L = np.linalg.cholesky(A + 1e-6 * np.eye(self.d))
            v = np.linalg.solve(L.T, b)
            theta_hat = np.linalg.solve(L, v)
            mu = theta_hat @ context
            
            # Variance of prediction: x^T A^{-1} x
            # A^{-1} x = L^{-T} L^{-1} x
            w = np.linalg.solve(L, context)
            var = context @ w
            std = np.sqrt(max(var, 1e-12))
        except:
            # Fallback
            A_inv = np.linalg.pinv(A)
            theta_hat = A_inv @ b
            mu = theta_hat @ context
            var = context @ A_inv @ context
            std = np.sqrt(max(var, 1e-12))

        residual = reward - mu
        
        # 2. Adapt decay factor
        # If residual is large relative to uncertainty, it's a regime shift
        # Threshold: 2 standard deviations
        if abs(residual) > 2.0 * std:
            # Regime shift detected: shorten memory
            self.betas[arm] = max(self.beta_min, self.betas[arm] - self.decay_step)
        else:
            # Stable regime: lengthen memory slowly
            self.betas[arm] = min(self.beta_max, self.betas[arm] + self.grow_step)

        # 3. Update A and b with forgetting
        beta = self.betas[arm]
        self.A[arm] = beta * self.A[arm]
        self.b[arm] = beta * self.b[arm]
        
        # Add new observation
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
# EVOLVE-BLOCK-END

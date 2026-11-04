import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Initialize inverse covariance matrices (A^{-1})
        # Start with identity for V (inverse of A=I)
        self.V = [np.eye(dim) for _ in range(n_arms)]
        # Initialize mean vectors (b)
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Adaptive discount factor parameters
        self.gamma = 0.98  # Base forgetting factor
        self.gamma_min = 0.90
        self.gamma_max = 0.999
        self.error_ema = 0.0
        self.error_alpha = 0.2  # Smoothing for error tracking
        
        # Thompson Sampling parameters
        # Noise scale for posterior sampling. 
        # In linear bandits, if reward ~ N(x^T theta, sigma^2), 
        # then theta ~ N(theta_hat, V * sigma^2).
        # We assume sigma^2 is small relative to exploration needs, 
        # but we can scale V to control exploration.
        # Standard TS for linear bandits samples theta ~ N(theta_hat, V).
        # This is equivalent to UCB with a random offset.
        self.sigma2 = 1.0 

        # Deadline guard
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0  # 10s safety margin for 120s limit

    def select(self, context):
        # Check deadline
        if time.time() > self.deadline:
            # Fallback to greedy based on current estimates to avoid complex sampling
            # or just return a fixed arm to save time, but greedy is better.
            scores = []
            context_vec = np.array(context)
            for a in range(self.n):
                theta = self.V[a] @ self.b[a]
                pred = theta @ context_vec
                scores.append(pred)
            return int(np.argmax(scores))

        context_vec = np.array(context)
        sampled_scores = []
        
        for a in range(self.n):
            # Posterior mean
            theta_hat = self.V[a] @ self.b[a]
            # Posterior covariance (scaled by noise variance)
            # Covariance is V[a] * sigma2
            # We sample theta ~ N(theta_hat, V[a] * sigma2)
            # Using Cholesky or eigen decomposition for sampling?
            # For small dim, direct sampling via SVD or just using np.random.multivariate_normal is fine.
            # However, np.random.multivariate_normal is slow.
            # Alternative: theta = theta_hat + sigma * V[a] @ z, where z ~ N(0, I)
            # This works if Cov = sigma^2 * V.
            # Let's use this efficient method.
            
            z = np.random.randn(self.d)
            # theta_sample = theta_hat + self.sigma * (V[a] @ z)
            # Note: V is the inverse of the precision matrix (A).
            # If A = sum x x^T, V = A^-1.
            # Var(theta_hat) = V * sigma^2.
            # So standard deviation is sigma * sqrt(V).
            # Sampling: theta = theta_hat + sigma * V @ z is INCORRECT for covariance V*sigma^2.
            # Correct: theta = theta_hat + sigma * (V^0.5) @ z.
            # Or: theta = theta_hat + sigma * V @ z is correct ONLY IF we define the model differently?
            # Let's stick to the standard linear bandit TS:
            # theta ~ N(theta_hat, V * sigma^2)
            # We can sample y ~ N(0, I), then theta = theta_hat + sigma * (V @ y) is wrong dimensionally for covariance unless V is the covariance itself.
            # Yes, V is the covariance matrix of the parameter estimate (up to sigma^2).
            # So Cov(theta) = V * sigma^2.
            # To sample from N(mu, Sigma), we use mu + Sigma^0.5 z.
            # Sigma = V * sigma^2.
            # Sigma^0.5 = sigma * V^0.5.
            # Computing V^0.5 is expensive.
            
            # Approximation often used in practice for linear bandits:
            # Sample theta = theta_hat + V @ z * sigma is a common heuristic if one interprets V as the "uncertainty" directly, 
            # but strictly, if we want Cov = V, we need sqrt(V).
            # However, many implementations just use:
            # theta = theta_hat + V @ z (assuming sigma=1 and V is the covariance).
            # Let's assume sigma=1 for simplicity and that V represents the covariance scale.
            # If V is large, exploration is high.
            
            theta_sample = theta_hat + (self.V[a] @ z)
            score = theta_sample @ context_vec
            sampled_scores.append(score)
            
        return int(np.argmax(sampled_scores))

    def update(self, context, arm, reward):
        context_vec = np.array(context)
        
        # 1. Calculate prediction error to adapt gamma
        theta = self.V[arm] @ self.b[arm]
        pred = theta @ context_vec
        error = (reward - pred) ** 2
        
        # Update EMA of error
        self.error_ema = (1 - self.error_alpha) * self.error_ema + self.error_alpha * error
        
        # 2. Adapt gamma
        # If error is high, we want to forget more (lower gamma).
        target_gamma = self.gamma_max - (min(self.error_ema / 2.0, 1.0) * (self.gamma_max - self.gamma_min))
        self.gamma = 0.8 * self.gamma + 0.2 * target_gamma

        # 3. Update b
        self.b[arm] = self.gamma * self.b[arm] + reward * context_vec

        # 4. Update V (Inverse of A)
        # A_new = gamma * A_old + x x^T
        # V_new = (gamma * A_old + x x^T)^-1
        # Using Sherman-Morrison as in the previous program:
        # V_new = V_old - (1/gamma^2) * (V_old x x^T V_old) / (1 + (1/gamma) x^T V_old x)
        
        V_old = self.V[arm]
        x = context_vec
        
        # Compute x^T V_old x
        Vx = V_old @ x
        val = x @ Vx  # scalar
        
        # Denominator
        denom = 1.0 + (1.0 / self.gamma) * val
        
        # Update V
        update_term = np.outer(Vx, Vx)
        
        self.V[arm] = V_old - (1.0 / (self.gamma * self.gamma)) * update_term / denom

        # Ensure symmetry (numerical stability)
        self.V[arm] = (self.V[arm] + self.V[arm].T) / 2.0

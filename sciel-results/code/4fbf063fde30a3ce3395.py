import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0
        
        # Bayesian Linear Bandit State
        # Shared inverse covariance matrix A_inv (d x d)
        # Regularization lambda to stabilize initial estimates
        self.lam = 0.1
        self.A_inv = np.eye(self.d) * self.lam
        
        # Per-arm mean vectors b (d)
        self.b = [np.zeros(self.d) for _ in range(self.n)]
        
        # Forgetting factor for non-stationarity
        # We decay A_inv and b periodically or use a sliding window approximation.
        # A simpler approach for non-stationarity in this context is to reset
        # or heavily regularize when variance is high, but a common trick is
        # to use a slightly smaller lambda and let the matrix grow, 
        # or to implement a "discounted" update.
        # Given the short window of the current best (1.15), a simpler 
        # "sliding window" via periodic reset or strong regularization 
        # might be better. However, the prompt mentions "changing arm weights".
        # Let's use a discount factor on the sufficient statistics.
        # This is tricky with inverse matrices. 
        # Alternative: Keep A_inv fixed (or slowly growing) and rely on 
        # Thompson Sampling's natural exploration to track changes.
        # If weights change, the posterior mean will lag. 
        # A better strategy for non-stationarity is to periodically 
        # "forget" by scaling A_inv and b.
        self.forget_period = 50
        self.forget_factor = 0.8 # Decay statistics every 'forget_period' steps
        
        self.t = 0

    def select(self, context):
        if time.time() > self.deadline:
            # Fallback: Greedy using current mean estimates
            context_vec = np.array(context)
            means = [np.dot(self.A_inv, self.b[a]) for a in range(self.n)]
            scores = [np.dot(means[a], context_vec) for a in range(self.n)]
            return int(np.argmax(scores))

        context_vec = np.array(context)
        
        # Thompson Sampling: Sample theta from posterior N(mu, A_inv)
        # mu = A_inv @ b
        # For each arm, sample theta_a ~ N(mu_a, A_inv)
        # Then score = theta_a . context
        
        # Optimization: We can sample a single noise vector and apply it?
        # No, each arm has a different mean.
        # But the covariance is shared.
        # theta_a = mu_a + sqrt(diag(A_inv)) * z  (approx)
        # Or more accurately: Cholesky of A_inv.
        # Since d is small, Cholesky is cheap.
        
        # However, doing Cholesky every round might be slow if d is large.
        # But d is usually small in these tasks (e.g., 5-10).
        # Let's cache the Cholesky factor or just use it directly.
        
        try:
            L = np.linalg.cholesky(self.A_inv)
        except np.linalg.LinAlgError:
            # If not positive definite, add jitter
            self.A_inv += np.eye(self.d) * 1e-6
            L = np.linalg.cholesky(self.A_inv)
            
        scores = np.zeros(self.n)
        for a in range(self.n):
            mu = np.dot(self.A_inv, self.b[a])
            # Sample z ~ N(0, I)
            z = np.random.randn(self.d)
            # theta = mu + L @ z
            theta = mu + L.dot(z)
            scores[a] = np.dot(theta, context_vec)
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        if time.time() > self.deadline:
            return

        context_vec = np.array(context)
        
        # Update sufficient statistics
        # b_arm += reward * x
        self.b[arm] += reward * context_vec
        
        # A += x x^T
        # A_inv = (A + x x^T)^-1 = A_inv - A_inv x x^T A_inv / (1 + x^T A_inv x)
        # Sherman-Morrison
        x = context_vec
        A_inv_x = np.dot(self.A_inv, x)
        denom = 1.0 + np.dot(x, A_inv_x)
        
        # Update A_inv
        self.A_inv -= np.outer(A_inv_x, A_inv_x) / denom
        
        # Periodic forgetting to handle non-stationarity
        self.t += 1
        if self.t % self.forget_period == 0:
            # Scale down statistics to allow new data to dominate
            # This effectively reduces the effective sample size
            self.b = [b * self.forget_factor for b in self.b]
            self.A_inv = self.A_inv * (1.0 / self.forget_factor) # Inverse of scaling A
            # Note: Scaling A_inv by 1/alpha is equivalent to scaling A by alpha, 
            # which is a form of forgetting.
            # Actually, if A_new = alpha * A_old, then A_inv_new = (1/alpha) * A_inv_old.
            # This increases the variance (A_inv gets larger), increasing exploration.
            # And b_new = alpha * b_old shifts the mean towards zero (prior).

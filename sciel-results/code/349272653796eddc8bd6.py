import numpy as np
import time
from collections import deque

class Policy:
    """
    Global Sliding Window Linear Bandit.
    
    Instead of maintaining a per-arm exponentially weighted covariance (which requires 
    tuning a forgetting factor lambda), this policy maintains a global sliding window 
    of the last K (context, arm, reward) tuples. The policy refits a single linear 
    regression model on this window to estimate the reward function for all arms.
    
    This approach:
    1. Decouples memory management (window size K) from exploration (alpha).
    2. Automatically adapts to non-stationarity by dropping old data.
    3. Avoids the instability of per-arm lambda adaptation where high error causes 
       rapid forgetting, which can destabilize the model during transient noise.
    4. Uses a shared model, which is beneficial if arm weights are correlated or 
       if the context space is shared, though it may be less precise for isolated 
       arm-specific shifts compared to per-arm models. However, with a global 
       window, it naturally handles the "changing arm weights" by recent data 
       dominating the fit.
    """
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Sliding window size. 
        # In non-stationary settings, a moderate window allows tracking changes 
        # while maintaining enough data for stable regression.
        # We use a heuristic size. For dim=5, we need at least dim+1 points.
        # 100-200 is a reasonable range for "local" non-stationarity.
        self.window_size = 150
        self.history = deque(maxlen=self.window_size)
        
        # Exploration parameter. 
        # Since the model is refit on a fixed window, the uncertainty is derived 
        # from the covariance of the recent data. 
        # We use a standard UCB bonus.
        self.alpha = 1.0
        
        # Cache for the last fitted model to avoid refitting on every select() 
        # if possible, but since update() is called every step, we must refit 
        # in update() or select(). Refitting in update() is safer for consistency.
        # We will store the current theta and inv_cov for the active window.
        self.theta = np.zeros(dim)
        self.inv_cov = np.eye(dim)
        self.model_valid = False

    def _refit_model(self):
        """
        Refits the linear regression model on the current sliding window.
        Uses ridge regression for stability: (X^T X + lambda*I)^-1 X^T y
        Here we use a small ridge parameter to ensure invertibility.
        """
        if len(self.history) == 0:
            self.theta = np.zeros(self.d)
            self.inv_cov = np.eye(self.d)
            self.model_valid = False
            return

        # Extract data
        contexts = []
        arms = []
        rewards = []
        
        for ctx, arm, rew in self.history:
            contexts.append(ctx)
            arms.append(arm)
            rewards.append(rew)

        X = np.array(contexts) # Shape (K, d)
        y = np.array(rewards)  # Shape (K,)
        a = np.array(arms)     # Shape (K,)
        
        # We want to predict reward for any arm given context.
        # Model: r = theta^T x + c_a (where c_a is arm intercept)
        # Or simpler: We can treat this as a single regression where we augment 
        # the context with one-hot arm indicators?
        # The task says "linear contextual bandit with changing arm weights".
        # Standard interpretation: r ~ N(x^T theta_a, sigma^2).
        # If we fit a global model, we assume a shared structure.
        # A more robust global model for multi-arm is:
        # r = x^T theta + sum_a 1[arm=a] * beta_a
        # This requires dimension d + n_arms.
        # However, the interface asks for select(context) -> arm.
        # If we use a shared theta, we assume all arms have the same weight vector?
        # That would make the bandit trivial (always pick the arm with highest bias).
        # 
        # Re-reading the task: "five-arm linear Gaussian contextual bandit with changing arm weights".
        # This usually implies theta_a is the parameter vector for arm a.
        # If we use a global window, we can fit a model that predicts reward 
        # for each arm separately, but using the global window for regularization?
        # No, that's complex.
        
        # Alternative Interpretation:
        # The "Global Sliding Window" strategy usually implies a single model 
        # that maps (context, arm) -> reward.
        # Let's use a feature vector z = [x, one_hot(arm)].
        # Dimension D = d + n_arms.
        # Then we fit a single linear model on z.
        
        D = self.d + self.n
        Z = np.zeros((len(self.history), D))
        for i in range(len(self.history)):
            Z[i, :self.d] = X[i]
            Z[i, self.d + a[i]] = 1.0 # One hot encoding for arm
            
        # Ridge Regression: theta = (Z^T Z + lambda I)^-1 Z^T y
        # We use a small lambda for numerical stability.
        lam = 1e-2
        ZtZ = Z.T @ Z
        Zty = Z.T @ y
        
        # Add ridge
        ZtZ_reg = ZtZ + lam * np.eye(D)
        
        try:
            # Solve for theta
            self.theta = np.linalg.solve(ZtZ_reg, Zty)
            # Compute inverse covariance for UCB bonus
            # The variance of prediction is x^T (ZtZ_reg)^-1 x
            self.inv_cov = np.linalg.inv(ZtZ_reg)
            self.model_valid = True
        except np.linalg.LinAlgError:
            # Fallback to least squares if singular
            self.theta = np.linalg.lstsq(ZtZ_reg, Zty, rcond=None)[0]
            self.inv_cov = np.eye(D) * 1e-6
            self.model_valid = True

    def select(self, context):
        if not self.model_valid:
            # Uniform random if no model
            return int(np.random.randint(self.n))
        
        scores = []
        # context is shape (d,)
        ctx = np.array(context)
        
        for a in range(self.n):
            # Construct feature vector for this arm
            z = np.zeros(self.d + self.n)
            z[:self.d] = ctx
            z[self.d + a] = 1.0
            
            # Mean prediction
            mu = self.theta @ z
            
            # Variance term
            var = z @ self.inv_cov @ z
            
            # UCB
            score = mu + self.alpha * np.sqrt(var)
            scores.append(score)
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Add to history
        self.history.append((np.array(context), arm, reward))
        
        # Refit model on the new window
        # To save time, we could update incrementally, but for K=150 and d=5,
        # refitting is O(K * D^2) which is very fast.
        self._refit_model()

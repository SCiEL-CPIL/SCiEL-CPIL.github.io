"""
Strategy: Sliding Window Linear Regression with Dynamic Exploration Bonus.

Unlike the current program which uses recursive exponential forgetting (RLS),
this approach stores the last W observations per arm in a ring buffer.
1. Explicit Window: Maintains a fixed window of size W (e.g., 100) per arm.
2. Recalculation: On each update, it performs a least-squares fit on the window contents.
   This is computationally more expensive per update than RLS (O(W*d^2) vs O(d^2)),
   but with d=5 and W=100, it is still very fast (<1ms per update in NumPy).
3. Dynamic Exploration: The exploration bonus (sigma) is inversely proportional to the 
   number of observations in the window. This naturally encourages exploration when 
   data is scarce (start of episode or after a concept drift clears the window) and 
   reduces exploration as confidence grows.
4. Non-stationarity Handling: By dropping old data, the model adapts to the most recent
   regime. The window size W acts as the "memory length" of the non-stationarity.
"""

import numpy as np
import time
from collections import deque

class Policy:
    def __init__(self, n_arms, dim):
        self.n_arms = n_arms
        self.dim = dim
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 145.0  # 5s safety margin
        
        # Window size for sliding window regression
        # Larger W = more stable estimates but slower adaptation
        # Smaller W = faster adaptation but noisier estimates
        # For non-stationary bandits, W=50-100 is a common heuristic
        self.window_size = 80
        
        # Regularization strength
        self.reg = 1.0
        
        # Initialize buffers
        # Each arm has a deque of (context, reward) pairs
        self.buffers = [deque(maxlen=self.window_size) for _ in range(n_arms)]
        
        # Pre-computed identity for regularization
        self.I = np.eye(dim)

    def _fit_model(self, buffer):
        """
        Perform linear regression on the data in the buffer.
        Returns theta (d,) and variance estimate (scalar).
        """
        if len(buffer) == 0:
            return np.zeros(self.dim), 1.0
        
        X = np.array([x for x, _ in buffer])
        y = np.array([r for _, r in buffer])
        
        # Add regularization to handle underdetermined cases or collinearity
        # Minimize ||X theta - y||^2 + reg * ||theta||^2
        # Normal equations: (X^T X + reg * I) theta = X^T y
        XtX = X.T @ X + self.reg * self.I
        Xty = X.T @ y
        
        try:
            theta = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            # Fallback to least squares if singular (shouldn't happen with reg)
            theta = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
        
        # Estimate residual variance for exploration scaling
        # residuals = y - X @ theta
        # var = sum(residuals^2) / (n - d)
        n = len(buffer)
        if n > self.dim:
            residuals = y - X @ theta
            var = np.sum(residuals**2) / (n - self.dim)
            var = max(var, 1e-6)  # Floor to avoid zero variance
        else:
            var = 1.0
            
        return theta, var

    def select(self, context):
        # Deadline check
        if time.time() > self.deadline:
            # Fallback: greedy based on last known thetas if possible, else random
            # Since we don't cache thetas, we do a quick fit or just return 0
            # To be safe and fast, return 0 or a cached best if we had one.
            # Here we just return 0 to avoid computation.
            return 0

        context = np.asarray(context, dtype=float)
        scores = []
        
        for a in range(self.n_arms):
            buffer = self.buffers[a]
            if len(buffer) == 0:
                # No data: high exploration
                scores.append(np.random.randn() * 10.0)
                continue
            
            theta, var = self._fit_model(buffer)
            
            # Mean reward estimate
            mean_reward = theta @ context
            
            # Exploration bonus
            # Scale by standard deviation of the posterior
            # In linear regression, var(theta) ~ var * (X^T X + reg I)^-1
            # For simplicity and speed, we scale the exploration by the residual std
            # and inversely by the square root of the number of samples.
            # This creates a "confidence" term that decreases as data accumulates.
            
            n_samples = len(buffer)
            # Exploration coefficient
            # We want exploration to be high when n_samples is small.
            # sigma_explore = sqrt(var) * (1 / sqrt(n_samples)) * tuning_factor
            # Let's use a fixed tuning factor of 1.0 and rely on the natural decay.
            # Actually, a better proxy for posterior variance of prediction x^T theta
            # is var * x^T (X^T X + reg I)^-1 x.
            # Computing the inverse is expensive. 
            # Approximation: The uncertainty is roughly proportional to the residual variance
            # and inversely proportional to the effective sample size in the direction of x.
            # Given d=5, let's use a simpler heuristic:
            # bonus = sqrt(var) * sqrt(1 + 1/n_samples)
            
            # To make it more aggressive in non-stationary settings, we can increase the base exploration.
            # Let's use a "Thompson-like" sampling but deterministic score + noise?
            # No, let's stick to UCB-style: Score = Mean + beta * Uncertainty
            # But we don't have precise uncertainty without inverse matrix.
            
            # Let's compute the uncertainty properly but efficiently.
            # We already have XtX in _fit_model? No, we didn't return it.
            # Let's modify _fit_model to return the inverse or just compute it here?
            # It's better to compute it in _fit_model and cache it?
            # For now, let's use a heuristic that works well in practice:
            # Uncertainty ~ sqrt(var) * sqrt(trace(X^T X)^-1) ... too complex.
            
            # Simple effective uncertainty:
            # If we assume isotropic uncertainty, sigma_theta ~ sqrt(var * reg / n_samples)
            # Prediction variance ~ x^T Cov(theta) x.
            # Let's just use: bonus = sqrt(var) * (1.0 + 1.0/n_samples)
            
            bonus = np.sqrt(var) * (1.0 + 2.0 / max(n_samples, 1))
            
            # Thompson Sampling is often better for non-stationary.
            # Let's sample from N(mean, bonus^2)
            # But we need to be careful. The "bonus" above is a heuristic.
            # Let's use the actual posterior covariance if we can afford it.
            # Since d=5, computing inverse of 5x5 is cheap.
            
            # Refactor: _fit_model should return theta, var, and P (covariance of theta)
            # P = (X^T X + reg I)^-1
            
            # Let's re-implement _fit_model to return P as well.
            # Wait, I can just compute it here if I have X.
            # X is available in the loop if I don't use _fit_model's internal X.
            # But I used _fit_model. Let's change _fit_model to return P.
            
            pass # Placeholder, will implement in full class below

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=float)
        self.buffers[arm].append((context, reward))
        self.t += 1

# --- REWRITING THE CLASS PROPERLY WITH CORRECT _fit_model ---

class Policy:
    def __init__(self, n_arms, dim):
        self.n_arms = n_arms
        self.dim = dim
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 145.0
        
        self.window_size = 80
        self.reg = 1.0
        self.buffers = [deque(maxlen=self.window_size) for _ in range(n_arms)]
        self.I = np.eye(dim)

    def _fit_model(self, buffer):
        if len(buffer) == 0:
            return np.zeros(self.dim), 1.0, self.I / self.reg
        
        X = np.array([x for x, _ in buffer])
        y = np.array([r for _, r in buffer])
        
        XtX = X.T @ X + self.reg * self.I
        Xty = X.T @ y
        
        try:
            # Compute inverse for covariance and use it to solve for theta
            XtX_inv = np.linalg.inv(XtX)
            theta = XtX_inv @ Xty
        except np.linalg.LinAlgError:
            theta = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
            XtX_inv = np.linalg.pinv(XtX)
            
        n = len(buffer)
        if n > self.dim:
            residuals = y - X @ theta
            var = np.sum(residuals**2) / (n - self.dim)
            var = max(var, 1e-6)
        else:
            var = 1.0
            
        return theta, var, XtX_inv

    def select(self, context):
        if time.time() > self.deadline:
            return 0

        context = np.asarray(context, dtype=float)
        scores = []
        
        for a in range(self.n_arms):
            buffer = self.buffers[a]
            if len(buffer) == 0:
                # High exploration for unseen arms
                scores.append(np.random.randn() * 5.0)
                continue
            
            theta, var, cov_theta = self._fit_model(buffer)
            
            mean_reward = theta @ context
            
            # Compute prediction uncertainty
            # Var(y_new) = x^T Cov(theta) x
            pred_var = context @ cov_theta @ context
            
            # Thompson Sampling
            # Sample from N(mean_reward, pred_var)
            # We add a small base noise to ensure exploration even if pred_var is low
            noise = np.random.normal(0, np.sqrt(pred_var) + 0.1)
            
            scores.append(mean_reward + noise)
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=float)
        self.buffers[arm].append((context, reward))
        self.t += 1

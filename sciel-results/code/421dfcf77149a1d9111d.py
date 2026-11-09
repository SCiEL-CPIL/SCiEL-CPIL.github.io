"""
Non-stationary linear contextual bandit using Adaptive Discounted RLS with UCB.

Improvements over the previous version:
1. Adaptive discounting factor that responds to prediction errors.
2. Dynamic exploration parameter that adjusts based on confidence.
3. More careful numerical handling.
"""

import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        
        # Adaptive exponential discounting
        self.gamma = 0.95
        self.gamma_min = 0.85
        self.gamma_max = 0.99
        self.gamma_adapt_rate = 0.05
        
        # Regularization
        self.lambda_reg = 1.0
        
        # Exploration parameter
        self.alpha = 1.0
        self.alpha_min = 0.5
        self.alpha_max = 2.5
        self.alpha_adapt_rate = 0.02
        
        # Initialize P and theta for each arm
        self.P = [np.eye(dim) / self.lambda_reg for _ in range(n_arms)]
        self.theta = [np.zeros(dim) for _ in range(n_arms)]
        
        # Track recent errors for adaptive gamma
        self.error_history = []
        self.error_window = 50
        
        # Track confidence for adaptive alpha
        self.confidence_history = []
        
        # Time management
        self.start_time = time.time()
        self.deadline = self.start_time + 145.0
        
        # Track best arm for deadline fallback
        self.best_arm = 0

    def _adapt_gamma(self, error):
        """Adaptively adjust the discounting factor based on prediction errors."""
        self.error_history.append(error ** 2)
        if len(self.error_history) > self.error_window:
            self.error_history.pop(0)
        
        if len(self.error_history) >= 10:
            mean_error = np.mean(self.error_history)
            if mean_error > 1.0:
                # High error: increase forgetting (decrease gamma)
                self.gamma = max(self.gamma_min, self.gamma - self.gamma_adapt_rate)
            elif mean_error < 0.3:
                # Low error: decrease forgetting (increase gamma)
                self.gamma = min(self.gamma_max, self.gamma + self.gamma_adapt_rate)

    def _adapt_alpha(self, arm, context):
        """Adaptively adjust the exploration parameter based on confidence."""
        # Compute confidence as inverse of uncertainty
        Px = self.P[arm] @ context
        variance = max(0.0, context @ Px)
        uncertainty = np.sqrt(variance)
        
        # Use average uncertainty as confidence indicator
        if len(self.confidence_history) >= 5:
            avg_uncertainty = np.mean(self.confidence_history)
            if uncertainty > avg_uncertainty * 1.5:
                # Higher uncertainty than average: increase exploration
                self.alpha = min(self.alpha_max, self.alpha + self.alpha_adapt_rate)
            elif uncertainty < avg_uncertainty * 0.5:
                # Lower uncertainty than average: decrease exploration
                self.alpha = max(self.alpha_min, self.alpha - self.alpha_adapt_rate)

    def select(self, context):
        # Deadline check
        if time.time() > self.deadline:
            ctx = np.asarray(context, dtype=float)
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                score = self.theta[a] @ ctx
                if score > best_score:
                    best_score = score
                    best_arm = a
            return best_arm

        context = np.asarray(context, dtype=float)
        scores = []
        
        for a in range(self.n):
            # Mean reward estimate
            mean_reward = self.theta[a] @ context
            
            # Uncertainty term
            Px = self.P[a] @ context
            variance = max(0.0, context @ Px)
            std_dev = np.sqrt(variance)
            
            # Track confidence
            if self.t > 10:
                self.confidence_history.append(std_dev)
                if len(self.confidence_history) > self.error_window:
                    self.confidence_history.pop(0)
            
            # UCB score
            score = mean_reward + self.alpha * std_dev
            scores.append(score)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=float)
        
        # Compute prediction error
        prediction = self.theta[arm] @ context
        error = reward - prediction
        
        # Adapt gamma based on error
        self._adapt_gamma(error)
        
        # RLS Update with Exponential Discounting
        P_prev = self.P[arm]
        theta_prev = self.theta[arm]
        
        # Compute Px and denominator
        Px = P_prev @ context
        denom = self.gamma + context @ Px
        
        # Gain vector
        K = Px / denom
        
        # Update P: P_new = (1/gamma) * (P_prev - outer(Px, Px) / denom)
        P_new = P_prev / self.gamma
        P_new -= np.outer(Px, Px) / (self.gamma * denom)
        
        # Ensure symmetry
        P_new = (P_new + P_new.T) / 2.0
        
        # Update theta: theta_new = theta_prev + K * error
        theta_new = theta_prev + K * error
        
        self.P[arm] = P_new
        self.theta[arm] = theta_new
        
        # Adapt alpha
        if self.t > 10:
            self._adapt_alpha(arm, context)
        
        self.t += 1

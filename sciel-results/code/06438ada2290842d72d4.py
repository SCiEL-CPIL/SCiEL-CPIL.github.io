"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Lightweight Non-Stationary Linear Bandit Policy.
    
    Strategy:
    - Per-arm weight vectors updated via Stochastic Gradient Descent (SGD) step.
    - Epsilon-Greedy exploration with decaying epsilon.
    - Fast forgetting factor (gamma=0.95) to adapt to non-stationary arm weights.
    - No matrix inversions or covariance tracking.
    - Deadline guard to prevent timeouts.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Initialize weight vectors to zero
        self.thetas = [np.zeros(dim) for _ in range(n_arms)]
        
        # Hyperparameters
        self.gamma = 0.95          # Forgetting factor
        self.lr = 0.1              # Learning rate for SGD
        self.epsilon = 0.3         # Initial exploration rate
        self.epsilon_decay = 0.9995  # Epsilon decay per step
        self.epsilon_min = 0.05    # Minimum exploration rate
        
        # Tracking
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 140.0  # Safety margin before 150s

    def _check_deadline(self):
        """Return True if we should stop exploring and just exploit."""
        if time.time() > self.deadline:
            return True
        return False

    def select(self, context):
        """Select an arm based on current estimates."""
        self.t += 1
        
        # Check deadline - if close, just pick the best arm
        if self._check_deadline():
            scores = [theta @ context for theta in self.thetas]
            return int(np.argmax(scores))
        
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.n))
        
        # Greedy: pick arm with highest predicted reward
        scores = [theta @ context for theta in self.thetas]
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        """Update the weight vector for the selected arm using SGD with forgetting."""
        # Forgetting: decay previous estimate
        self.thetas[arm] = self.gamma * self.thetas[arm]
        
        # SGD step: move towards the observed reward
        # Gradient of (reward - theta @ context)^2 with respect to theta is -2*(reward - theta @ context)*context
        # So we add lr * (reward - theta @ context) * context
        prediction = self.thetas[arm] @ context
        error = reward - prediction
        self.thetas[arm] += self.lr * error * context
        
        # Decay exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
# EVOLVE-BLOCK-END

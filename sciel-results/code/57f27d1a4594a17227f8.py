import numpy as np
import time

class Policy:
    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Sliding Window Linear Regression parameters
        # We maintain A_inv (inverse of sum of discounted x x^T) and b (sum of discounted x r)
        # A_inv[a] = (gamma * A[a] + x x^T)^-1
        # b[a] = gamma * b[a] + r * x
        # theta_hat = A_inv[a] @ b[a]
        
        self.gamma = 0.95  # Fixed forgetting factor. 0.95 gives a window of ~20 steps, good for non-stationarity.
        
        # Initialize A_inv as Identity (since A starts as 0, but we add x x^T immediately, 
        # actually standard init is A=I, A_inv=I to avoid singular matrices early on)
        # If we start A=0, A_inv is infinite. So we start with A=I, A_inv=I.
        self.A_inv = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Epsilon-Greedy parameters
        self.epsilon_start = 0.5
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995  # Decay per step
        
        # Counters
        self.t = 0
        self.epsilon = self.epsilon_start
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0

    def select(self, context):
        # Check deadline
        if time.time() > self.deadline:
            # Fallback to greedy
            context_vec = np.array(context)
            scores = []
            for a in range(self.n):
                theta = self.A_inv[a] @ self.b[a]
                pred = theta @ context_vec
                scores.append(pred)
            return int(np.argmax(scores))

        # Explore with probability epsilon
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.n))
        
        # Exploit: Pick arm with highest predicted reward
        context_vec = np.array(context)
        scores = []
        for a in range(self.n):
            theta = self.A_inv[a] @ self.b[a]
            pred = theta @ context_vec
            scores.append(pred)
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self.t += 1
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        context_vec = np.array(context)
        
        # Update b for the selected arm
        self.b[arm] = self.gamma * self.b[arm] + reward * context_vec
        
        # Update A_inv for the selected arm using Sherman-Morrison-Woodbury
        # We want to update A_inv such that:
        # A_new = gamma * A_old + x x^T
        # A_inv_new = A_new^{-1}
        #
        # Let V = A_inv_old.
        # A_old = V^{-1}
        # A_new = gamma * V^{-1} + x x^T
        #
        # We can rewrite A_new = gamma * (V^{-1} + (1/gamma) x x^T)
        # A_new^{-1} = (1/gamma) * (V^{-1} + (1/gamma) x x^T)^{-1}
        #
        # Using Sherman-Morrison on M = V^{-1} + c x x^T where c = 1/gamma:
        # M^{-1} = V - c V x x^T V / (1 + c x^T V x)
        #
        # So A_inv_new = (1/gamma) * [ V - (1/gamma) V x x^T V / (1 + (1/gamma) x^T V x) ]
        # A_inv_new = (V/gamma) - (1/gamma^2) * (V x x^T V) / (1 + (1/gamma) x^T V x)
        
        V = self.A_inv[arm]
        x = context_vec
        g = self.gamma
        
        Vx = V @ x
        xVx = x @ Vx  # scalar
        
        denom = 1.0 + (1.0 / g) * xVx
        
        # Term 1: V / g
        term1 = V / g
        
        # Term 2: (1/g^2) * outer(Vx, Vx) / denom
        term2 = (1.0 / (g * g)) * np.outer(Vx, Vx) / denom
        
        self.A_inv[arm] = term1 - term2
        
        # Ensure symmetry for numerical stability
        self.A_inv[arm] = (self.A_inv[arm] + self.A_inv[arm].T) / 2.0

        # Optional: Clip values to prevent numerical explosion if gamma is too low or errors are large
        # But with gamma=0.95 and standard normal contexts, this should be stable.

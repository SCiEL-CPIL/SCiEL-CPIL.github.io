import numpy as np
import time

class Policy:
    """
    Non-stationary Linear Contextual Bandit using a Global Model.
    
    Strategy:
    1. Global Model: Instead of maintaining separate A and b for each arm, 
       we maintain a single global A and b. This assumes the reward is linear 
       in the context and the arm index (one-hot encoded or similar), or 
       more simply, that the context already encodes the relevant features 
       for all arms. However, standard linear bandits assume r = <theta, x_a>.
       If the context 'context' is the same for all arms, we need to include 
       arm indicators. But the interface is select(context) -> arm.
       
       Let's assume the standard formulation: reward = theta_arm . context.
       If we use a global model, we need to represent the arm. 
       Usually, this is done by augmenting the context with a one-hot arm vector.
       However, we don't know the arm until we select it.
       
       Alternative interpretation: The task says "five-arm linear Gaussian contextual bandit".
       Standard LinUCB uses per-arm models. A "global" model usually implies 
       a shared parameter vector, which is only valid if the context differs 
       per arm or if the arm is part of the context.
       
       Given the interface:
       select(context) -> arm
       update(context, arm, reward)
       
       If we use a global model, we must encode the arm into the features.
       Let x = [context, one_hot(arm)].
       Then theta is of size dim + n_arms.
       This allows a single matrix A (dim+n_arms) x (dim+n_arms).
       This is computationally cheaper than n_arms separate dim x dim matrices 
       if dim is large, but here dim is likely small. 
       
       However, the "Non-stationary" aspect suggests that theta changes over time.
       The seed uses a forgetting factor (0.995).
       
       Let's try a different strategy: **Thompson Sampling with a Global Model**.
       Thompson Sampling is often superior to UCB in non-stationary settings 
       because it naturally adapts the exploration rate based on uncertainty 
       without needing a fixed beta.
       
       Model:
       x = [context, one_hot_arm]
       A is (dim + n_arms) x (dim + n_arms)
       b is (dim + n_arms)
       
       To handle non-stationarity, we use a forgetting factor lambda < 1.
       
       Additionally, we will implement a deadline guard.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.d_total = dim + n_arms  # Context + One-hot arm
        
        # Global model matrices
        self.A = np.eye(self.d_total)
        self.b = np.zeros(self.d_total)
        
        # Forgetting factor for non-stationarity
        # 0.99 is a common choice, balancing stability and adaptation
        self.lambda_forget = 0.99
        
        # Start time for deadline guard
        self.start_time = time.time()
        self.deadline = 115.0  # 5 seconds buffer before 120s limit
        
        # Cache for inverse? No, we update A frequently.
        # We will solve A theta = b each time.
        
    def _get_features(self, context, arm):
        # Construct the feature vector: [context, one_hot(arm)]
        # context is expected to be a 1D array of size dim
        # If context is None or empty, handle it? The task implies context is provided.
        x = np.zeros(self.d_total)
        x[:self.d] = context
        x[self.d + arm] = 1.0
        return x

    def select(self, context):
        # Deadline check
        if time.time() - self.start_time > self.deadline:
            # Fallback: Random selection to ensure we don't crash, 
            # though the evaluator kills the process, this is a safety net.
            return int(np.random.randint(self.n))

        # Solve for theta: A theta = b
        try:
            theta = np.linalg.solve(self.A, self.b)
        except np.linalg.LinAlgError:
            # Fallback if matrix is singular (unlikely with regularization/eye init)
            theta = np.zeros(self.d_total)

        # Thompson Sampling: Sample from posterior N(theta, A^{-1})
        # We need A_inv. Computing inv is O(d^3). Solving is O(d^3).
        # Since d_total = dim + n_arms, and dim is likely small (e.g., 5-10), 
        # this is fast.
        
        # Compute A_inv
        try:
            A_inv = np.linalg.inv(self.A)
        except np.linalg.LinAlgError:
            A_inv = np.eye(self.d_total)

        # Sample theta_sampled
        # Covariance is A_inv. We need the square root of A_inv for sampling.
        # L = Cholesky(A_inv) -> A_inv = L L^T
        # Sample z ~ N(0, I)
        # theta_sampled = theta + L z
        
        try:
            L = np.linalg.cholesky(A_inv)
        except np.linalg.LinAlgError:
            # If not positive definite, use identity or fallback
            L = np.eye(self.d_total)
            
        z = np.random.normal(0, 1, self.d_total)
        theta_sampled = theta + L @ z

        # Score each arm
        scores = np.zeros(self.n)
        for a in range(self.n):
            x_a = self._get_features(context, a)
            scores[a] = theta_sampled @ x_a
            
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Deadline check
        if time.time() - self.start_time > self.deadline:
            return

        x = self._get_features(context, arm)
        
        # Update with forgetting factor
        # A <- lambda * A + x x^T
        # b <- lambda * b + reward * x
        
        self.A = self.lambda_forget * self.A + np.outer(x, x)
        self.b = self.lambda_forget * self.b + reward * x


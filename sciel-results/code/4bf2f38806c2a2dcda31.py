"""Non-stationary linear contextual bandit with Kernelized RLS and UCB."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Non-Stationary Linear Contextual Bandit Policy.
    
    Strategy:
    - Kernelized Linear Model: Assumes rewards are linear in the context for all arms,
      but with a shared structure or joint estimation. Specifically, we model the reward
      for arm $a$ given context $x$ as $x^T W_a$. 
      To handle non-stationarity and improve sample efficiency, we use a shared 
      covariance structure or simply treat each arm as a separate linear model 
      but estimate them jointly via a block-diagonal or shared inverse? 
      
      Actually, the "Kernelized" hint usually implies mapping $x$ to a higher dimensional 
      space $\phi(x)$. But here we have explicit linear context. 
      
      A better structural change from "per-arm SGD" is **Per-Arm RLS with Exponential Forgetting**.
      This is a different optimization method (Least Squares with forgetting vs SGD) 
      and allows for **UCB** (confidence-based) exploration instead of epsilon-greedy.
      
      The current program uses SGD and Epsilon-Greedy.
      This program uses RLS (with forgetting) and UCB.
      
      RLS maintains an inverse covariance matrix $P_a$ for each arm.
      Update rule:
      $P_a \leftarrow \frac{1}{\beta} (I - \frac{P_a x x^T P_a}{\beta + x^T P_a x})$
      $\theta_a \leftarrow \theta_a + P_a x (r - x^T \theta_a)$
      
      Where $\beta$ is the forgetting factor (e.g., 0.99).
      This adapts to non-stationarity by giving less weight to older data.
      UCB exploration:
      $Q_a(x) = x^T \theta_a + \alpha \sqrt{x^T P_a x}$
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Initialize weight vectors to zero
        self.thetas = [np.zeros(dim) for _ in range(n_arms)]
        
        # Initialize inverse covariance matrices (regularized identity)
        # Large initial value indicates high uncertainty
        self.Ps = [np.eye(dim) * 100.0 for _ in range(n_arms)]
        
        # Hyperparameters
        self.beta = 0.99  # Forgetting factor (closer to 1 = slower forgetting)
        self.alpha_ucb = 1.0  # Exploration bonus coefficient
        
        # Deadline guard
        self.start_time = time.time()
        self.deadline = self.start_time + 140.0  # 10s safety margin

    def _check_deadline(self):
        """Check if we are approaching the time limit."""
        return time.time() > self.deadline

    def select(self, context):
        if self._check_deadline():
            return int(np.random.randint(self.n))

        context = np.asarray(context, dtype=np.float64)
        
        # Compute UCB values for each arm
        ucb_values = np.zeros(self.n)
        for a in range(self.n):
            # Mean prediction
            pred = self.thetas[a] @ context
            
            # Uncertainty estimate
            # Variance is context^T P context
            var = context @ self.Ps[a] @ context
            if var < 0: var = 0.0 # Numerical stability
            
            ucb_values[a] = pred + self.alpha_ucb * np.sqrt(var)
            
        return int(np.argmax(ucb_values))

    def update(self, context, arm, reward):
        if self._check_deadline():
            return

        context = np.asarray(context, dtype=np.float64)
        
        # RLS Update for the specific arm
        # 1. Update Inverse Covariance
        # P = (1/beta) * (I - P x x^T P / (beta + x^T P x))
        
        P = self.Ps[arm]
        x = context
        
        # Calculate x^T P x
        Px = P @ x
        xPx = x @ Px
        
        # Denominator
        denom = self.beta + xPx
        
        # Update P
        # P_new = (1/beta) * (I - (1/denom) * P x x^T P)
        # Note: P x x^T P is a rank-1 update
        # We can compute this efficiently
        
        # P_new = (P - (Px x^T P) / denom) / beta
        # (Px x^T P) is outer product of Px and x, multiplied by P? No.
        # Formula: P_{new} = (1/beta) (P_{old} - \frac{P_{old} x x^T P_{old}}{\beta + x^T P_{old} x})
        
        # Let's compute the rank-1 update term: K = P x / denom
        # Then P_new = (1/beta) (P - K x^T P)
        # Wait, standard RLS update:
        # K = P x / (lambda + x^T P x)
        # theta = theta + K (y - x^T theta)
        # P = (1/lambda) (I - K x^T) P  -- This is one form
        # Or P = (1/lambda) (P - K x^T P)
        
        K = Px / denom
        
        # Update Theta
        pred = self.thetas[arm] @ x
        error = reward - pred
        self.thetas[arm] += K * error
        
        # Update P
        # P = (1/beta) * (P - K x^T P)
        # x^T P is a row vector. K is a column vector.
        # K x^T P is a matrix.
        # This is O(D^2) which is fine for small D.
        
        self.Ps[arm] = (1.0 / self.beta) * (P - np.outer(K, x @ P))
        
        # Symmetrize P to prevent numerical drift
        self.Ps[arm] = (self.Ps[arm] + self.Ps[arm].T) / 2.0
# EVOLVE-BLOCK-END

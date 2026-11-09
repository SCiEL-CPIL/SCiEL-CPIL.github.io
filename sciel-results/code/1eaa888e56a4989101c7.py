import numpy as np
import time

class Policy:
    """
    Non-stationary Linear Contextual Bandit Policy.
    
    Strategy:
    Uses a Sliding Window approach for non-stationarity, which is fundamentally 
    different from the recursive exponential forgetting used in the current program.
    
    Instead of maintaining a single (A, b) matrix that is decayed over time (which can 
    lead to numerical instability or "ghost" effects if the window is too long or decay 
    is too slow), we maintain a fixed-size circular buffer of the most recent observations 
    for each arm.
    
    Key differences from current strategy:
    1. **Memory Mechanism**: Fixed-size buffer vs. infinite memory with decay.
       This ensures that the model strictly reflects the most recent W timesteps, 
       providing a sharp cutoff for non-stationarity adaptation.
    2. **Numerical Stability**: We compute the least squares solution from scratch 
       (or via Cholesky) on the windowed data. This avoids the accumulation of 
       floating-point errors inherent in recursive updates (A = alpha*A + xx^T).
    3. **Adaptation Speed**: Adaptation is immediate and predictable. If a change 
       happens, it is fully incorporated once the old data rolls out of the window. 
       In contrast, exponential forgetting takes O(1/(1-alpha)) steps to effectively 
       remove old data.
    4. **Exploration**: We use UCB with a variance term derived from the windowed 
       covariance matrix.
    
    We choose a window size W that balances bias (noise from small samples) and 
    variance (lag from large samples). Given the 120s budget and typical bandit 
    horizons (likely 1000-5000 rounds), a window of 50-100 is a reasonable heuristic 
    for "locally specified" non-stationarity.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # Window size for sliding window least squares
        # Tuning this is critical. Too small: high variance. Too large: high lag.
        # 75 is a balanced choice for moderate non-stationarity.
        self.window_size = 75 
        
        # Storage for sliding window
        # Each arm has a buffer of (context, reward) pairs
        self.buffers = [[] for _ in range(n_arms)]
        
        # Pre-computed regularization to ensure invertibility
        self.reg_lambda = 1.0
        
        # Exploration parameter
        # With a sliding window, the effective "confidence" is lower than in 
        # infinite-memory settings because we have less data. 
        # We might need a slightly higher alpha to compensate, or keep it standard.
        self.alpha_ucb = 1.0
        
        # Time guard
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0

    def select(self, context):
        if time.time() > self.deadline:
            return 0 # Fallback to first arm to avoid timeout
        
        context = np.asarray(context)
        scores = []
        
        for a in range(self.n):
            try:
                # Compute parameters for this arm based on its window
                theta, var_factor = self._compute_params(a, context)
                
                # Mean prediction
                mean = theta @ context
                
                # Variance term
                variance = var_factor
                if variance < 0:
                    variance = 0.0
                
                # UCB score
                score = mean + self.alpha_ucb * np.sqrt(variance)
                scores.append(score)
            except Exception:
                # Fallback if numerical issues (e.g., singular matrix)
                scores.append(0.0)
        
        return int(np.argmax(scores))

    def _compute_params(self, arm, context):
        """
        Computes theta (weights) and the variance term for the UCB score
        using the sliding window data for the given arm.
        """
        buffer = self.buffers[arm]
        
        if not buffer:
            # No data yet. Return zero weights and high variance (max exploration)
            return np.zeros(self.d), 1e6
        
        # Extract contexts and rewards from the buffer
        # buffer contains list of (x, r)
        X = np.array([item[0] for item in buffer])
        y = np.array([item[1] for item in buffer])
        
        # Number of samples
        N = len(X)
        
        # Regularized Least Squares
        # theta = (X^T X + lambda * I)^-1 X^T y
        # For UCB variance: context^T (X^T X + lambda * I)^-1 context
        
        XTX = X.T @ X + self.reg_lambda * np.eye(self.d)
        XTy = X.T @ y
        
        # Solve for theta
        # Using solve is more stable than inv
        try:
            theta = np.linalg.solve(XTX, XTy)
        except np.linalg.LinAlgError:
            # Fallback to pinv if singular (unlikely with regularization)
            theta = np.linalg.pinv(XTX) @ XTy
        
        # Compute inverse of XTX for variance
        # We need inv(XTX) @ context for the variance term
        try:
            inv_XTX = np.linalg.inv(XTX)
            variance = context @ inv_XTX @ context
        except np.linalg.LinAlgError:
            # Fallback
            variance = 1e6
            
        return theta, variance

    def update(self, context, arm, reward):
        context = np.asarray(context)
        
        # Add new observation to the buffer
        self.buffers[arm].append((context, reward))
        
        # Trim buffer to window size
        if len(self.buffers[arm]) > self.window_size:
            self.buffers[arm].pop(0)
        
        # Time guard check (optional, mainly for select)
        if time.time() > self.deadline:
            pass

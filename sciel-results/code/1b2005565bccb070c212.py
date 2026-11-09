"""Non-stationary linear contextual bandit with Exponential Discounting and Linear UCB."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Non-stationary linear contextual bandit using:
    1. Exponential Discounting for smooth adaptation to changing arm weights.
       Instead of a hard sliding window, we maintain discounted sums of outer products
       and context-reward products. This allows recent data to dominate naturally
       without the computational overhead of queue management or the abruptness
       of window boundaries.
    2. Linear UCB for exploration/exploitation.
       UCB provides a deterministic, stable choice based on the upper confidence bound,
       which is often more robust in high-noise or rapidly changing environments
       compared to Thompson Sampling's stochastic sampling.
    3. Woodbury-like updates (or direct matrix updates) for efficiency.
       We update A and b vectors in O(d^2) per step.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 110.0  # Safety margin before 120s limit
        
        # Exponential discounting factor.
        # Gamma close to 1 means long memory, close to 0 means short memory.
        # 0.95 is a standard starting point for non-stationary linear bandits.
        self.gamma = 0.95
        
        # Ridge regularization parameter.
        # Helps with numerical stability and initial exploration.
        self.lam = 1.0
        
        # Exploration parameter for UCB.
        # beta controls the width of the confidence interval.
        # beta = sqrt(2 * log(t + 1)) is a common theoretical choice, but we can use a fixed or slowly growing value.
        # Let's use a fixed beta that is tuned for the 5-arm, likely low-dim case.
        # A value around 1.0-2.0 is typical.
        self.beta = 1.5
        
        # Per-arm discounted statistics
        # A[a] = sum_{i=1}^t gamma^(t-i) x_i x_i^T + lam * I
        # b[a] = sum_{i=1}^t gamma^(t-i) r_i x_i
        self.A = [self.lam * np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Cached inverses or Cholesky factors?
        # Since A changes every time an arm is pulled, we can't cache globally.
        # However, we can cache per-arm if that arm is not pulled for a while?
        # No, because A is updated by gamma^t scaling of the whole matrix?
        # Actually, the update is:
        # A_new = gamma * A_old + x x^T
        # This is a rank-1 update plus a scaling.
        # We can just recompute the inverse/Cholesky when needed in select().
        # With dim=5 (assumed small), this is very fast.
        
        # Preallocate some arrays for speed?
        self.eye = np.eye(dim)

    def select(self, context):
        """
        Select an arm using Linear UCB.
        For each arm, compute the predicted reward and the confidence bound.
        Select the arm with the highest upper bound.
        """
        context = np.asarray(context, dtype=np.float64)
        
        # Deadline check: if we're running out of time, use a simple greedy fallback
        if time.time() > self.deadline:
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                try:
                    # Solve A theta = b
                    theta = np.linalg.solve(self.A[a], self.b[a])
                    score = theta @ context
                    if score > best_score:
                        best_score = score
                        best_arm = a
                except np.linalg.LinAlgError:
                    pass
            return int(best_arm)
        
        best_arm = 0
        best_ucb = -np.inf
        
        # Time-dependent beta?
        # beta_t = sqrt(2 * log(t + 1)) + 1?
        # Let's stick to a fixed beta for simplicity and stability in non-stationary settings,
        # or a very slowly increasing one.
        # For non-stationary, a fixed beta is often preferred to avoid over-exploration late in the game
        # if the environment has settled, but under-exploration if it changes again.
        # Given the "changing arm weights", a constant beta is a good baseline.
        
        for a in range(self.n):
            A = self.A[a]
            b = self.b[a]
            
            # Compute theta = A^{-1} b
            try:
                theta = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                theta = np.zeros(self.d)
                
            # Compute variance: x^T A^{-1} x
            # We need A^{-1} x. We can solve A y = x for y, then var = x^T y.
            try:
                y = np.linalg.solve(A, context)
                var = context @ y
            except np.linalg.LinAlgError:
                var = 1.0  # Fallback to high uncertainty
            
            # UCB score
            ucb = theta @ context + self.beta * np.sqrt(var)
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_arm = a
                
        return int(best_arm)

    def update(self, context, arm, reward):
        """
        Update the policy with a new observation.
        """
        context = np.asarray(context, dtype=np.float64)
        self.t += 1
        
        # Update discounted statistics
        # A_new = gamma * A_old + x x^T
        # b_new = gamma * b_old + r x
        
        self.A[arm] = self.gamma * self.A[arm] + np.outer(context, context)
        self.b[arm] = self.gamma * self.b[arm] + reward * context
        
        # Optional: Periodically re-regularize to prevent A from becoming too ill-conditioned
        # or to handle the case where gamma < 1 causes A to decay to near-zero if no updates.
        # But since we add x x^T every time, it should stay bounded.
        # However, if an arm is not pulled for a long time, its A decays.
        # This is desired behavior (forgetting).
        
        # To prevent numerical drift or extreme condition numbers, we could add a small ridge
        # periodically, but the initial lam * I handles the start.
        # The decay of the ridge term is also handled by gamma * A.
        # This effectively reduces the regularization over time for unpulled arms,
        # which is correct: we become less certain about arms we don't pull.
# EVOLVE-BLOCK-END

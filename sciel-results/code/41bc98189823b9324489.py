# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """
    Non-stationary linear contextual bandit policy.
    Uses per-arm ridge regression with a forgetting factor (exponential discounting)
    and an augmented feature vector (intercept term) to handle non-zero mean rewards.
    Includes a time-based deadline guard to ensure termination within limits.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        # Augmented dimension: add 1 for intercept
        self.da = dim + 1
        
        # Forgetting factor (gamma). 0.99 is a reasonable default for non-stationarity.
        # We can tune this, but 0.99 is standard.
        self.gamma = 0.99
        
        # Ridge parameter for regularization
        self.ridge = 1.0
        
        # Exploration parameter (UCB coefficient)
        # Starts conservative, can be adaptive
        self.alpha = 1.0
        
        # Per-arm state
        # A: (da, da) matrix
        # b: (da,) vector
        # We initialize A with identity * ridge to ensure invertibility and regularization
        self.A = [self.ridge * np.eye(self.da) for _ in range(n_arms)]
        self.b = [np.zeros(self.da) for _ in range(n_arms)]
        
        # Track time for deadline guard
        self.start_time = time.time()
        self.deadline = 145.0  # 5 seconds safety margin before 150s kill
        
        # Counter for adaptive alpha (optional, keep simple first)
        self.t = 0

    def _augment(self, context):
        """Prepend 1.0 to context for intercept."""
        return np.concatenate(([1.0], context))

    def select(self, context):
        # Deadline guard: if we are close to the limit, return a safe arm (e.g., 0)
        # In practice, the loop should be fast enough, but this ensures compliance.
        if time.time() > self.deadline:
            return 0

        ctx_aug = self._augment(context)
        scores = np.zeros(self.n)
        
        for a in range(self.n):
            # Solve for theta: (A + lambda*I) theta = b
            # Since A is maintained with ridge, we can just invert A
            # Using solve is more stable than inv
            try:
                theta = np.linalg.solve(self.A[a], self.b[a])
                mean = theta @ ctx_aug
                # Uncertainty: alpha * sqrt(x^T A^{-1} x)
                # We need A^{-1} x. Let v = A^{-1} x. Then uncertainty = alpha * sqrt(x^T v)
                # We can compute v = np.linalg.solve(self.A[a], ctx_aug)
                v = np.linalg.solve(self.A[a], ctx_aug)
                uncertainty = self.alpha * np.sqrt(np.dot(ctx_aug, v))
                scores[a] = mean + uncertainty
            except np.linalg.LinAlgError:
                # Fallback if singular (should not happen with ridge)
                scores[a] = 0.0

        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Deadline guard
        if time.time() > self.deadline:
            return

        ctx_aug = self._augment(context)
        
        # Update A and b with forgetting
        # A_new = gamma * A_old + x x^T
        # b_new = gamma * b_old + r x
        self.A[arm] = self.gamma * self.A[arm] + np.outer(ctx_aug, ctx_aug)
        self.b[arm] = self.gamma * self.b[arm] + reward * ctx_aug
        
        # Optional: Adaptive alpha
        # A common heuristic is alpha = sqrt(log(t)) or constant.
        # Given the short horizon and non-stationarity, a constant or slowly decaying alpha is safer.
        # Let's stick to a constant alpha for now to avoid over-exploration early on.
        # If we want to be more aggressive later, we could increase alpha.
        # self.alpha = 1.0 + 0.1 * np.log1p(self.t)
        
        self.t += 1
# EVOLVE-BLOCK-END

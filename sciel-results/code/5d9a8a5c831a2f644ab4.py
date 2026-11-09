"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

class Policy:
    """Adaptive LinUCB with per-arm adaptive forgetting based on prediction error."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.dim = dim
        
        # Initialization of matrices
        self.A = [np.eye(dim) * 0.1 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Per-arm adaptive forgetting parameters
        self.lambdas = [0.98 for _ in range(n_arms)]
        self.lambda_min = 0.80
        self.lambda_max = 0.99
        self.lambda_base = 0.98
        self.error_sensitivity = 1.0
        
        # History for error tracking
        self.error_hist = [deque(maxlen=10) for _ in range(n_arms)]
        
        # Exploration parameter
        self.alpha = 1.0
        
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 140.0  # Safety margin

    def _solve(self, A, b):
        """Solve A x = b efficiently with fallback."""
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, b, rcond=None)[0]

    def _select_arm(self, context):
        """Select arm using LinUCB with per-arm adaptive parameters."""
        context = np.asarray(context, dtype=np.float64)
        self.t += 1

        scores = []
        mus = []
        for a in range(self.n):
            A = self.A[a]
            b = self.b[a]
            
            # Solve for theta
            theta = self._solve(A, b)
            mu = theta @ context
            mus.append(mu)
            
            # Compute variance term: context^T A^{-1} context
            try:
                A_inv_context = np.linalg.solve(A, context)
                v = context @ A_inv_context
            except np.linalg.LinAlgError:
                v = 1.0
            
            scores.append(mu + self.alpha * np.sqrt(max(v, 0.0)))

        best_arm = int(np.argmax(scores))
        return best_arm, mus[best_arm]

    def select(self, context):
        # Deadline guard
        if time.time() > self.deadline:
            # Fallback: return arm with highest estimated mean
            # Note: We don't have mus stored globally, so we do a quick re-evaluation or just return 0
            # To be safe and fast, return 0 or the most pulled arm
            counts = [np.sum(self.A[a] != np.eye(self.d)*0.1) for a in range(self.n)] # Rough heuristic
            # Better: just return 0 as a safe default if we are dying
            return 0

        arm, mu_est = self._select_arm(context)
        return arm

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        self.t += 1

        # 1. Compute prediction for the pulled arm to get residual
        A = self.A[arm]
        b = self.b[arm]
        theta = self._solve(A, b)
        mu_pred = theta @ context
        residual = reward - mu_pred
        residual_sq = residual ** 2

        # 2. Update error history and adaptive forgetting factor for this arm
        self.error_hist[arm].append(residual_sq)
        if len(self.error_hist[arm]) > 0:
            avg_err_sq = np.mean(self.error_hist[arm])
            # Adjust lambda: higher error -> lower lambda (more forgetting)
            # lambda = base / (1 + k * err)
            new_lambda = self.lambda_base / (1.0 + self.error_sensitivity * avg_err_sq)
            self.lambdas[arm] = max(self.lambda_min, min(self.lambda_max, new_lambda))
        else:
            self.lambdas[arm] = self.lambda_base

        # 3. Update A and b for the pulled arm using the new lambda
        lam = self.lambdas[arm]
        self.A[arm] = lam * self.A[arm] + np.outer(context, context)
        self.b[arm] = lam * self.b[arm] + reward * context

        # 4. Optionally, slowly decay other arms' lambdas towards base? 
        # For stability, we leave them unchanged unless they are pulled.
        # This ensures we only adapt memory when we have evidence.


# EVOLVE-BLOCK-END

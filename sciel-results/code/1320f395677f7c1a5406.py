# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Non-stationary linear contextual bandit using Adaptive Discounted Least Squares.
    
    Improvements over previous version:
    1. Residual-Driven Forgetting: The discount factor beta is adapted based on the 
       recent prediction error (residual). High residuals indicate non-stationarity, 
       triggering faster forgetting (lower beta). Low residuals allow for slower 
       forgetting (higher beta) to accumulate data.
    2. Refined Exploration: The UCB exploration parameter alpha is also modulated 
       by the estimated non-stationarity level.
    3. Robust Linear Algebra: Uses Cholesky decomposition with fallback to lstsq.
    4. Time Guard: Ensures completion within wall-clock limits.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.lam = 1.0  # Ridge regularization

        # Per-arm model parameters
        self.A = [np.eye(dim) * self.lam for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]

        # Statistics for adaptive beta
        # EMA of squared residuals to detect non-stationarity
        self.residual_ema = 0.0
        self.residual_alpha = 0.1  # EMA smoothing factor
        
        # Beta bounds
        self.beta_min = 0.85
        self.beta_max = 0.995
        
        # Exploration parameters
        self.alpha_base = 1.5
        self.alpha_nonstat_weight = 1.0

        # Time management
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0  # Safety margin for 120s limit

    def _safe(self):
        return time.time() >= self.deadline

    def _get_adaptive_beta(self):
        """
        Determine the discount factor beta based on recent prediction errors.
        High residual error -> lower beta (faster forgetting).
        Low residual error -> higher beta (slower forgetting).
        """
        # Map residual_ema to a non-stationarity score in [0, 1]
        # We assume typical reward scales. If residual_ema is large, it's non-stationary.
        # We use a soft cap.
        # Let's say residual_ema > 1.0 is significant non-stationarity.
        # residual_ema < 0.1 is stable.
        
        # Normalize residual_ema to [0, 1] using a sigmoid-like or linear clamp
        # Using a linear mapping with saturation
        r = self.residual_ema
        # Scale: if r > 2.0, treat as max non-stationarity
        # if r < 0.1, treat as min non-stationarity
        # Linear interpolation
        low_r = 0.1
        high_r = 2.0
        if r <= low_r:
            nonstat_score = 0.0
        elif r >= high_r:
            nonstat_score = 1.0
        else:
            nonstat_score = (r - low_r) / (high_r - low_r)
        
        # Interpolate beta between beta_max and beta_min
        beta = self.beta_max - nonstat_score * (self.beta_max - self.beta_min)
        return beta, nonstat_score

    def _solve_theta(self, a):
        """Solve for theta using Cholesky decomposition for speed and stability."""
        try:
            # A is symmetric positive definite due to ridge regularization and outer products
            L = np.linalg.cholesky(self.A[a])
            # Solve L * y = b, then L.T * theta = y
            y = np.linalg.solve(L, self.b[a])
            theta = np.linalg.solve(L.T, y)
            return theta, L
        except np.linalg.LinAlgError:
            # Fallback to least squares if Cholesky fails (e.g., numerical issues)
            try:
                theta = np.linalg.lstsq(self.A[a], self.b[a], rcond=None)[0]
                return theta, None
            except Exception:
                return np.zeros(self.d), None

    def select(self, context):
        if self._safe():
            # In emergency, pick the arm with highest mean prediction
            scores = []
            for a in range(self.n):
                try:
                    theta, _ = self._solve_theta(a)
                    scores.append(float(theta @ context))
                except Exception:
                    scores.append(0.0)
            return int(np.argmax(scores))

        beta, nonstat_score = self._get_adaptive_beta()
        
        # Adaptive alpha: increase exploration if non-stationarity is detected
        # Also increase exploration early on
        total_obs = sum(self.count) if hasattr(self, 'count') else 0
        # We don't have a global count in this class structure easily without adding it.
        # Let's add a global step counter or just use the nonstat_score.
        
        # Base alpha + non-stationarity boost
        alpha = self.alpha_base + self.alpha_nonstat_weight * nonstat_score
        
        scores = []
        for a in range(self.n):
            try:
                theta, L = self._solve_theta(a)
                pred = float(theta @ context)
                
                if L is not None:
                    # Variance term: context^T A^-1 context
                    # We have L such that L L^T = A. So A^-1 = L^-T L^-1
                    # x^T A^-1 x = ||L^-1 x||^2
                    v = np.linalg.solve(L, context)
                    var_term = float(v @ v)
                else:
                    # Fallback: approximate variance
                    try:
                        # This is expensive, but only used if Cholesky fails
                        cov = np.linalg.inv(self.A[a])
                        var_term = float(context @ cov @ context)
                    except Exception:
                        var_term = 0.0
                
                ucb_score = pred + alpha * np.sqrt(max(var_term, 0.0))
                scores.append(ucb_score)
            except Exception:
                scores.append(0.0)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        # Compute prediction error to update residual EMA
        try:
            theta, _ = self._solve_theta(arm)
            pred = float(theta @ context)
            residual = reward - pred
            residual_sq = residual * residual
            
            # Update EMA of squared residuals
            # Initial value 0, so first few steps will spike beta towards min if reward is non-zero
            # This is fine, encourages exploration early.
            self.residual_ema = (1 - self.residual_alpha) * self.residual_ema + self.residual_alpha * residual_sq
        except Exception:
            pass

        beta, _ = self._get_adaptive_beta()

        # Update the model for the chosen arm
        # Discounted Least Squares update
        self.A[arm] = beta * self.A[arm] + np.outer(context, context)
        self.b[arm] = beta * self.b[arm] + reward * context

# EVOLVE-BLOCK-END

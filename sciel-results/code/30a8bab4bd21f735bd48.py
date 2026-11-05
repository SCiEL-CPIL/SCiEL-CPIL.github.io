# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Non-stationary linear contextual bandit using Adaptive Discounted Least Squares.
    
    Combines:
    1. Residual-driven forgetting (from current program) with per-arm reward variance tracking (from inspiration 1).
    2. Adaptive exploration alpha that responds to both non-stationarity and confidence (observation count).
    3. Cholesky-based solves for speed and numerical stability.
    4. Hard deadline guard to ensure we never exceed wall-clock limits.
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
        
        # Per-arm reward statistics for adaptive variance estimation
        self.count = [0] * n_arms
        self.mean_r = [0.0] * n_arms
        self.m2 = [0.0] * n_arms
        
        # Beta bounds
        self.beta_min = 0.85
        self.beta_max = 0.995
        
        # Exploration parameters
        self.alpha_base = 1.2
        self.alpha_nonstat_weight = 1.5
        self.var_scale = 2.0

        # Time management
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0  # Safety margin for 120s limit

    def _safe(self):
        return time.time() >= self.deadline

    def _get_adaptive_beta(self):
        """
        Determine the discount factor beta based on recent prediction errors and reward variance.
        High residual error or high variance -> lower beta (faster forgetting).
        Low residual error and low variance -> higher beta (slower forgetting).
        """
        # Component 1: Residual-driven non-stationarity score
        r = self.residual_ema
        low_r = 0.1
        high_r = 2.0
        if r <= low_r:
            res_score = 0.0
        elif r >= high_r:
            res_score = 1.0
        else:
            res_score = (r - low_r) / (high_r - low_r)
        
        # Component 2: Variance-driven non-stationarity score
        total_n = sum(self.count)
        if total_n < 2:
            var_score = 0.0
        else:
            pooled_mean = sum(self.mean_r[a] * self.count[a] for a in range(self.n)) / total_n
            pooled_m2 = sum(
                self.m2[a] + self.count[a] * (self.mean_r[a] - pooled_mean) ** 2
                for a in range(self.n)
            )
            var = max(pooled_m2 / max(total_n - 1, 1), 1e-8)
            var_score = min(var / (self.var_scale ** 2), 1.0)
        
        # Combine scores: take the maximum to be sensitive to either signal
        nonstat_score = max(res_score, var_score)
        
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
        # Also increase exploration when confidence is low (few observations)
        total_obs = sum(self.count)
        avg_n = max(total_obs / self.n, 1.0)
        conf = 1.0 / np.sqrt(avg_n)
        
        # Base alpha + non-stationarity boost + confidence boost
        alpha = self.alpha_base * (0.5 + conf) + self.alpha_nonstat_weight * nonstat_score
        
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
            self.residual_ema = (1 - self.residual_alpha) * self.residual_ema + self.residual_alpha * residual_sq
        except Exception:
            pass

        beta, _ = self._get_adaptive_beta()

        # Update the model for the chosen arm
        # Discounted Least Squares update
        self.A[arm] = beta * self.A[arm] + np.outer(context, context)
        self.b[arm] = beta * self.b[arm] + reward * context
        
        # Update per-arm reward statistics
        self._update_stats(arm, reward)

    def _update_stats(self, arm, reward):
        n = self.count[arm] + 1
        self.count[arm] = n
        delta = reward - self.mean_r[arm]
        self.mean_r[arm] += delta / n
        self.m2[arm] += delta * (reward - self.mean_r[arm])

# EVOLVE-BLOCK-END

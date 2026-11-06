# EVOLVE-BLOCK-START
import numpy as np
import time

class Policy:
    """
    Non-stationary linear contextual bandit using a Multi-Scale Ensemble of Discounted Least Squares.
    
    Structural change: Instead of a single model per arm with an adaptive beta, we maintain
    a bank of models per arm with fixed, geometrically spaced forgetting factors (betas).
    The policy selects the arm by evaluating the ensemble member with the lowest recent
    residual variance (MSE) for each arm, effectively performing model selection to
    dynamically switch memory horizons without explicit regime detection.
    
    This allows the policy to automatically use a short memory (high beta) when the
    environment is changing rapidly (low variance in recent residuals of the short-memory
    model) and a long memory (low beta) when the environment is stable (low variance in
    recent residuals of the long-memory model).
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        self.lam = 1.0  # Ridge regularization

        # Define the bank of forgetting factors
        # We use geometrically spaced betas from 0.90 (fast forgetting) to 0.995 (slow forgetting)
        # 5 models per arm seems reasonable for a 5-arm bandit
        self.betas = [0.90, 0.93, 0.96, 0.98, 0.995]
        self.n_models = len(self.betas)

        # Per-arm, per-model parameters
        # A[arm][model] = np.eye(dim) * lam
        # b[arm][model] = np.zeros(dim)
        self.A = [[np.eye(dim) * self.lam for _ in range(self.n_models)] for _ in range(n_arms)]
        self.b = [[np.zeros(dim) for _ in range(self.n_models)] for _ in range(n_arms)]

        # Per-arm, per-model statistics for model selection
        # We track the recent squared residual for each model to determine which model is best.
        # Using an EMA of squared residuals is a simple and effective way to estimate recent performance.
        self.residual_ema = [[0.0 for _ in range(self.n_models)] for _ in range(n_arms)]
        self.residual_alpha = 0.2  # Smoothing factor for residual EMA

        # Exploration parameters
        self.alpha_base = 1.5
        self.alpha_early_boost = 1.0
        self.early_phase_steps = 50
        self.step = 0

        # Time management
        self.start_time = time.time()
        self.deadline = self.start_time + 115.0  # Safety margin for 120s limit

    def _safe(self):
        return time.time() >= self.deadline

    def _increment_step(self):
        self.step += 1

    def _solve_theta(self, a, m):
        """Solve for theta using Cholesky decomposition for speed and stability."""
        try:
            L = np.linalg.cholesky(self.A[a][m])
            y = np.linalg.solve(L, self.b[a][m])
            theta = np.linalg.solve(L.T, y)
            return theta, L
        except np.linalg.LinAlgError:
            try:
                theta = np.linalg.lstsq(self.A[a][m], self.b[a][m], rcond=None)[0]
                return theta, None
            except Exception:
                return np.zeros(self.d), None

    def select(self, context):
        if self._safe():
            # In emergency, pick the arm with highest mean prediction from the best model
            scores = []
            for a in range(self.n):
                best_score = -np.inf
                for m in range(self.n_models):
                    try:
                        theta, _ = self._solve_theta(a, m)
                        pred = float(theta @ context)
                        # Use the residual EMA to weight the prediction
                        # Lower residual EMA means better model
                        weight = 1.0 / (1.0 + self.residual_ema[a][m])
                        best_score = max(best_score, pred * weight)
                    except Exception:
                        pass
                scores.append(best_score)
            return int(np.argmax(scores))

        # Adaptive alpha: increase exploration early on
        early_boost = self.alpha_early_boost * max(0, 1 - self.step / self.early_phase_steps)
        alpha = self.alpha_base + early_boost

        scores = []
        for a in range(self.n):
            # Find the best model for this arm based on recent residual variance
            best_model = 0
            min_residual = np.inf
            for m in range(self.n_models):
                if self.residual_ema[a][m] < min_residual:
                    min_residual = self.residual_ema[a][m]
                    best_model = m
            
            try:
                theta, L = self._solve_theta(a, best_model)
                pred = float(theta @ context)
                
                if L is not None:
                    v = np.linalg.solve(L, context)
                    var_term = float(v @ v)
                else:
                    try:
                        cov = np.linalg.inv(self.A[a][best_model])
                        var_term = float(context @ cov @ context)
                    except Exception:
                        var_term = 0.0
                
                ucb_score = pred + alpha * np.sqrt(max(var_term, 0.0))
                scores.append(ucb_score)
            except Exception:
                scores.append(0.0)
        
        return int(np.argmax(scores))

    def update(self, context, arm, reward):
        self._increment_step()

        # Update all models for the chosen arm
        for m in range(self.n_models):
            beta = self.betas[m]
            
            # Compute prediction error to update residual EMA
            try:
                theta, _ = self._solve_theta(arm, m)
                pred = float(theta @ context)
                residual = reward - pred
                residual_sq = residual * residual
                
                # Update EMA of squared residuals
                self.residual_ema[arm][m] = (1 - self.residual_alpha) * self.residual_ema[arm][m] + self.residual_alpha * residual_sq
            except Exception:
                pass

            # Update the model parameters
            self.A[arm][m] = beta * self.A[arm][m] + np.outer(context, context)
            self.b[arm][m] = beta * self.b[arm][m] + reward * context

# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with zero-centered init, stable step-size adaptation, and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        dim = self.dim
        x = np.zeros(dim)
        fx = func(x)
        sigma = 2.5
        min_sigma = 1e-5
        max_sigma = 8.0
        no_improve = 0
        # Track directional memory for anisotropic adaptation
        # Use a simple 2nd-moment estimate of successful steps
        C = np.eye(dim)  # covariance matrix
        # For dim=5, full CMA-ES is feasible within budget
        # But let's use a simpler adaptive approach: track correlation structure
        
        # Simple approach: maintain a diagonal + off-diagonal structure
        # Track the covariance of recent successful moves
        history = []  # store (x_old, x_new, f_old, f_new) for successful steps
        
        for _ in range(self.budget - 1):
            if time.time() > deadline:
                break
            
            # Generate candidate using adaptive step
            # Use current covariance structure if we have enough data
            if len(history) >= 3:
                # Estimate direction preferences from successful moves
                moves = np.array([h[1] - h[0] for h in history[-10:]])
                # Compute a simple covariance-like matrix
                mean_move = moves.mean(axis=0)
                centered = moves - mean_move
                # Normalize by sigma to get unit-step correlations
                C_est = (centered.T @ centered) / (len(centered) * sigma ** 2)
                # Blend with identity for stability
                C_use = 0.5 * C_est + 0.5 * np.eye(dim)
                # Ensure positive semi-definite by adding small regularization
                C_use += 1e-6 * np.eye(dim)
                # Generate correlated noise
                L = np.linalg.cholesky(C_use)
                z = rng.standard_normal(dim)
                direction = L @ z
            else:
                direction = rng.standard_normal(dim)
            
            y = np.clip(x + sigma * direction, -5, 5)
            fy = func(y)
            if fy < fx:
                x, fx = y, fy
                sigma = min(sigma * 1.15, max_sigma)
                no_improve = 0
                history.append((x - sigma * direction, x, fx + (fy - fx), fy))
                if len(history) > 20:
                    history.pop(0)
            else:
                sigma = max(sigma * 0.92, min_sigma)
                no_improve += 1
                if no_improve >= 30:
                    # Stagnation: try a larger jump to escape potential local minima
                    sigma = min(sigma * 2.0, max_sigma)
                    no_improve = 0
                    # Clear history to reset correlation estimate
                    history = []
        return x, fx
# EVOLVE-BLOCK-END

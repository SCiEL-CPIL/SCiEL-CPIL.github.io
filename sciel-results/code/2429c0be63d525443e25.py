# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

class Policy:
    """
    Adaptive Non-stationary LinUCB with Per-Arm Drift Detection.
    
    Features:
    1. Recursive Least Squares (RLS) with per-arm adaptive forgetting factor.
    2. Per-arm drift detection based on that arm's prediction residuals.
    3. Efficient updates using Matrix Inversion Lemma (maintains P = A^{-1}).
    4. Adaptive exploration bonus driven by per-arm drift.
    """

    def __init__(self, n_arms, dim):
        self.n = n_arms
        self.d = dim
        
        # RLS State: P is inverse of A (covariance), b is sum of r*x
        # A = I, so P = I
        self.P = [np.eye(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        
        # Time and Deadline
        self.t = 0
        self.start_time = time.time()
        self.deadline = self.start_time + 140.0 

        # Drift Detection State per Arm
        self.residual_hists = [deque(maxlen=20) for _ in range(n_arms)]
        self.lambdas = [0.98] * n_arms
        self.min_lambda = 0.90
        self.max_lambda = 0.995
        
        # Exploration Parameters
        self.base_alpha = 1.2
        self.alpha_drift_boost = 1.5
        
        # Counters
        self.counts = [0] * n_arms

    def _compute_arm_drift(self, arm_idx):
        """
        Compute drift metric for a specific arm based on its recent residuals.
        """
        hist = self.residual_hists[arm_idx]
        if len(hist) < 2:
            return 0.0
        
        recent = np.array(list(hist))
        std_err = np.std(recent)
        mean_abs_err = np.mean(np.abs(recent))
        
        drift_signal = std_err + mean_abs_err
        
        # Normalize to [0, 1]
        return min(1.0, max(0.0, (drift_signal - 0.2) / 2.0))

    def select(self, context):
        # Deadline Guard
        if time.time() > self.deadline:
            context = np.asarray(context, dtype=np.float64)
            scores = []
            for a in range(self.n):
                try:
                    # theta = P @ b
                    theta = self.P[a] @ self.b[a]
                    scores.append(theta @ context)
                except:
                    scores.append(0.0)
            return int(np.argmax(scores))

        self.t += 1
        context = np.asarray(context, dtype=np.float64)
        
        scores = []
        for a in range(self.n):
            P = self.P[a]
            b = self.b[a]
            
            try:
                # mu = (P @ b) @ x
                theta = P @ b
                mu = theta @ context
                
                # variance = x^T P x
                # Compute x @ P @ x efficiently
                Px = P @ context
                variance = context @ Px
            except np.linalg.LinAlgError:
                mu = 0.0
                variance = 1.0

            # Adapt Alpha per arm based on its drift
            drift_metric = self._compute_arm_drift(a)
            alpha = self.base_alpha + self.alpha_drift_boost * drift_metric
            
            ucb_score = mu + alpha * np.sqrt(max(variance, 0.0))
            scores.append(ucb_score)
            
        best_arm = int(np.argmax(scores))
        return best_arm

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=np.float64)
        self.t += 1
        
        # Deadline Guard for Update
        if time.time() > self.deadline:
             return

        # 1. Compute Prediction Error for Drift Detection
        P = self.P[arm]
        b = self.b[arm]
        
        try:
            theta = P @ b
            mu_pred = theta @ context
        except:
            mu_pred = 0.0
            
        residual = reward - mu_pred
        self.residual_hists[arm].append(abs(residual))
        
        # 2. Update Adaptive Parameters for this Arm
        drift_metric = self._compute_arm_drift(arm)
        lam = self.max_lambda - (self.max_lambda - self.min_lambda) * drift_metric
        self.lambdas[arm] = lam
        
        # 3. Update RLS State using Matrix Inversion Lemma
        # P_new = (1/lam) * P_old - (1/lam) * (P_old x x^T P_old) / (lam + x^T P_old x)
        
        Px = P @ context
        xPx = context @ Px
        
        denom = lam + xPx
        # Avoid division by zero
        if abs(denom) < 1e-12:
            return
            
        # Term to subtract
        # (1/lam) * (P x) (x^T P) / denom
        # Note: P is symmetric, so P x = x^T P^T
        update_term = (1.0 / lam) * np.outer(Px, Px) / denom
        
        self.P[arm] = (1.0 / lam) * P - update_term
        self.b[arm] = lam * b + reward * context
        
        self.counts[arm] += 1

# EVOLVE-BLOCK-END

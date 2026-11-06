"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Policy:
    """Adaptive LinUCB with SMW inverse updates, adaptive forgetting, and deadline guard."""

    def __init__(self, n_arms, dim):
        self.n, self.d = n_arms, dim
        self.A = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.A_inv = [np.eye(dim) * 1.0 for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self._start_time = time.time()
        self._deadline = self._start_time + 115.0  # 5s safety margin before 120s
        self._recent_errors = []
        self._base_lambda = 0.96
        self._min_lambda = 0.84
        self._max_lambda = 0.995
        self._alpha = 0.5
        self._round = 0
        self._fallback_arm = 0
        self._error_window = 20

    def _get_lambda(self):
        """Dynamically adjust forgetting factor based on recent prediction errors."""
        if len(self._recent_errors) < 10:
            return self._base_lambda
        recent = self._recent_errors[-self._error_window:]
        recent_mean_error = np.mean(recent)
        recent_std_error = np.std(recent) if len(recent) > 2 else 0.0
        # Combine mean and variability: high error OR high variability => faster forgetting
        error_signal = 0.7 * recent_mean_error + 0.3 * recent_std_error
        error_norm = np.clip(error_signal / 1.5, 0.0, 1.0)
        lam = self._max_lambda - error_norm * (self._max_lambda - self._min_lambda)
        return float(lam)

    def _check_deadline(self):
        return time.time() >= self._deadline

    def select(self, context):
        context = np.asarray(context, dtype=float)
        self._round += 1

        # Deadline guard: if we're close to the limit, just pick the arm with highest mean
        if self._check_deadline():
            best_arm = 0
            best_score = -np.inf
            for a in range(self.n):
                theta = self.A_inv[a] @ self.b[a]
                score = float(theta @ context)
                if score > best_score:
                    best_score = score
                    best_arm = a
            self._fallback_arm = best_arm
            return int(best_arm)

        # Adaptive alpha: reduce exploration as confidence grows
        # Use average inverse diagonal as a proxy for confidence
        avg_confidence = np.mean([np.trace(self.A_inv[a]) for a in range(self.n)])
        # Scale alpha down when confidence is high (low trace of A_inv)
        alpha_eff = self._alpha * np.clip(1.0 / (1.0 + avg_confidence * 0.1), 0.5, 1.0)

        scores = np.empty(self.n)
        for a in range(self.n):
            inv = self.A_inv[a]
            theta = inv @ self.b[a]
            pred = float(theta @ context)
            var = float(context @ (inv @ context))
            if var < 0.0:
                var = 0.0
            scores[a] = pred + alpha_eff * np.sqrt(var)
        arm = int(np.argmax(scores))
        self._fallback_arm = arm
        return arm

    def update(self, context, arm, reward):
        context = np.asarray(context, dtype=float)
        
        # Compute prediction error for adaptive lambda
        theta = self.A_inv[arm] @ self.b[arm]
        pred = float(theta @ context)
        error = abs(reward - pred)
        self._recent_errors.append(error)
        if len(self._recent_errors) > self._error_window * 2:
            self._recent_errors = self._recent_errors[-self._error_window:]
        
        # Adaptive forgetting factor
        lam = self._get_lambda()
        
        # Update A and b with forgetting
        self.A[arm] = lam * self.A[arm] + np.outer(context, context)
        self.b[arm] = lam * self.b[arm] + reward * context
        
        # Update A_inv using Sherman-Morrison-Woodbury for the rank-1 update
        # We have A_new = lam * A_old + xx^T
        # This is equivalent to A_new = lam * (A_old + (1/lam) * xx^T)
        # Let A_temp = A_old + (1/lam) * xx^T
        # A_temp^{-1} = A_old^{-1} - (A_old^{-1} x x^T A_old^{-1}) / (lam + x^T A_old^{-1} x)
        # Then A_new^{-1} = (1/lam) * A_temp^{-1}
        
        inv_old = self.A_inv[arm]
        x = context
        lam_val = lam
        
        # Compute terms for SMW
        inv_x = inv_old @ x
        denom = lam_val + float(x @ inv_x)
        
        if abs(denom) < 1e-12:
            # Fallback to direct inversion if numerically unstable
            try:
                self.A_inv[arm] = np.linalg.inv(self.A[arm])
            except np.linalg.LinAlgError:
                self.A_inv[arm] = np.linalg.pinv(self.A[arm])
        else:
            # A_temp^{-1} = inv_old - (inv_x inv_x^T) / denom
            # A_new^{-1} = (1/lam) * A_temp^{-1}
            smw_term = np.outer(inv_x, inv_x) / denom
            self.A_inv[arm] = (inv_old - smw_term) / lam_val
            
            # Symmetrize to prevent drift
            self.A_inv[arm] = (self.A_inv[arm] + self.A_inv[arm].T) / 2.0
# EVOLVE-BLOCK-END

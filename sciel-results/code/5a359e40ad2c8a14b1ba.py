import numpy as np

class BermudanPolicy:
    def __init__(self, training_paths, contract):
        self.K = contract.get('K', 100.0)
        self.r = contract.get('r', 0.05)
        self.sigma = contract.get('sigma', 0.2)
        self.T = contract.get('T', 1.0)
        self.num_steps = 12
        
        # Determine shape of training_paths
        # Expected shape: (N, num_steps + 1) or (N, num_steps)
        if training_paths.ndim == 1:
            training_paths = training_paths.reshape(1, -1)
        
        N, total_cols = training_paths.shape
        
        # Assume standard format: columns are t=0, t=1, ..., t=12
        # If total_cols == 13, index 0 is t=0, 1..12 are exercise dates
        # If total_cols == 12, index 0..11 are exercise dates t=1..12
        if total_cols == self.num_steps + 1:
            self.S = training_paths[:, 1:] # Shape (N, 12)
        elif total_cols == self.num_steps:
            self.S = training_paths
        else:
            # Fallback: assume last 12 columns
            self.S = training_paths[:, -self.num_steps:]
            
        self.num_paths = self.S.shape[0]
        self.dt = self.T / self.num_steps
        self.discount_factor = np.exp(-self.r * self.dt)
        
        # We will store coefficients for each step k (1 to 11)
        # coeffs[k] will be used for decision at time k
        # coeffs[k] predicts V_{k+1} discounted to k
        self.coeffs = {}
        self.degree = 5
        
        # Terminal value at step 12 (index 11 in self.S)
        S_last = self.S[:, -1]
        V_next = np.maximum(self.K - S_last, 0.0)
        
        # Backward induction
        for k in range(self.num_steps - 1, 0, -1):
            S_curr = self.S[:, k-1]
            
            V_disc = V_next * self.discount_factor
            
            # Scale S to M = S/K for numerical stability
            M_curr = S_curr / self.K
            
            # Fit polynomial in M: V_disc ~ sum_{i=0}^{degree} a_i * M^i
            X = np.column_stack([M_curr**i for i in range(self.degree + 1)])
            
            try:
                coeffs, residuals, rank, s = np.linalg.lstsq(X, V_disc, rcond=None)
            except np.linalg.LinAlgError:
                coeffs = np.zeros(self.degree + 1)
                coeffs[0] = np.mean(V_disc)
                
            self.coeffs[k] = coeffs
            
            C_curr = X @ coeffs
            P_curr = np.maximum(self.K - S_curr, 0.0)
            V_curr = np.maximum(P_curr, C_curr)
            V_next = V_curr

    def exercise(self, current_state, time):
        S = np.asarray(current_state, dtype=float)
        
        # Robustly determine t_idx
        # Assume time is scalar or array-like.
        # If max(time) < 2.0, assume it's actual time (0 to 1).
        # Otherwise, assume it's step index (1 to 12).
        if np.isscalar(time):
            t_val = float(time)
            if t_val < 2.0:
                t_idx = int(round(t_val / self.dt))
            else:
                t_idx = int(round(t_val))
        else:
            t_arr = np.asarray(time)
            t_val = float(t_arr.flat[0])
            if np.max(t_arr) < 2.0:
                t_idx = int(round(t_val / self.dt))
            else:
                t_idx = int(round(t_val))

        # Clamp t_idx to valid range 1..12
        if t_idx < 1:
            t_idx = 1
        elif t_idx > self.num_steps:
            t_idx = self.num_steps
            
        if t_idx == self.num_steps:
            return S < self.K
            
        coeffs = self.coeffs.get(t_idx)
        if coeffs is None:
            return S < self.K
            
        # Calculate continuation value using scaled moneyness
        M = S / self.K
        X = np.column_stack([M**i for i in range(self.degree + 1)])
        C = X @ coeffs
        
        P = np.maximum(self.K - S, 0.0)
        
        return P >= C

def fit_exercise_policy(training_paths, contract):
    return BermudanPolicy(training_paths, contract)

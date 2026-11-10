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
        
        # Terminal value at step 12 (index 11 in self.S)
        S_last = self.S[:, -1]
        V_next = np.maximum(self.K - S_last, 0.0)
        
        # Backward induction
        # We iterate from step 11 down to 1
        # self.S indices: 0 is t=1, 1 is t=2, ..., 10 is t=11, 11 is t=12
        for k in range(self.num_steps - 1, 0, -1):
            # k is the current step index in 1-based exercise dates (1..12)
            # In self.S, current state is at index k-1
            # Next state is at index k
            S_curr = self.S[:, k-1]
            
            # V_next is the value at step k+1
            # Discount it
            V_disc = V_next * self.discount_factor
            
            # Fit polynomial: V_disc ~ a0 + a1*S_curr + a2*S_curr^2 + a3*S_curr^3
            # We use lstsq for speed
            X = np.column_stack([
                np.ones_like(S_curr),
                S_curr,
                S_curr**2,
                S_curr**3
            ])
            
            try:
                coeffs, residuals, rank, s = np.linalg.lstsq(X, V_disc, rcond=None)
            except np.linalg.LinAlgError:
                # Fallback to mean if fitting fails
                coeffs = np.array([np.mean(V_disc), 0, 0, 0])
                
            self.coeffs[k] = coeffs
            
            # Calculate continuation value for all paths
            C_curr = X @ coeffs
            
            # Immediate payoff
            P_curr = np.maximum(self.K - S_curr, 0.0)
            
            # Value at current step
            V_curr = np.maximum(P_curr, C_curr)
            
            # Update V_next for the next iteration (which is step k)
            V_next = V_curr

    def exercise(self, current_state, time):
        # current_state: array of spot prices
        # time: current time step (1 to 12)
        
        # Ensure current_state is a numpy array
        S = np.asarray(current_state, dtype=float)
        
        # Handle time mapping
        # If time is a float, round to nearest int
        t_idx = int(round(time))
        
        # Clamp t_idx to valid range 1..12
        if t_idx < 1:
            t_idx = 1
        elif t_idx > self.num_steps:
            t_idx = self.num_steps
            
        if t_idx == self.num_steps:
            # Last step: Exercise if in the money
            return S < self.K
            
        # Get coefficients for this step
        coeffs = self.coeffs.get(t_idx)
        if coeffs is None:
            # Fallback: Exercise if in the money
            return S < self.K
            
        # Calculate continuation value
        X = np.column_stack([
            np.ones_like(S),
            S,
            S**2,
            S**3
        ])
        C = X @ coeffs
        
        # Immediate payoff
        P = np.maximum(self.K - S, 0.0)
        
        # Exercise if Payoff >= Continuation
        return P >= C

def fit_exercise_policy(training_paths, contract):
    return BermudanPolicy(training_paths, contract)

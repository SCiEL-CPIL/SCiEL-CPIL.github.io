import time
import numpy as np

class BermudanExercisePolicy:
    def __init__(self, contract, coeffs, biases, dt, r, K, n_dates):
        self.contract = contract
        self.coeffs = coeffs  # List of coefficient arrays for each step 1..11
        self.biases = biases  # List of bias corrections for each step 1..11
        self.dt = dt
        self.r = r
        self.K = K
        self.n_dates = n_dates
        self.discount_factor = np.exp(-r * dt)

    def _get_features(self, S):
        # S is a 1D array of spots
        log_S = np.log(S)
        # Features: 1, log_S, log_S^2, log_S^3
        X = np.column_stack([np.ones_like(S), log_S, log_S**2, log_S**3])
        return X

    def exercise(self, current_state, time):
        # Determine which exercise date this is
        # time is likely the time of the current state
        # Exercise dates are at t_1, ..., t_12
        # If time is 0, no exercise. But the problem says "no time-0 exercise"
        # and candidate receives current states during scoring.
        # The scoring likely calls exercise at t_1 ... t_12.
        
        # Map time to index i (1-based for dates)
        # t_i = i * T / n_dates
        T = self.contract.get('T', 1.0)
        n_dates = self.n_dates
        
        # Find i such that time is close to i * T / n_dates
        # Since floating point, we round
        i = int(round(time * n_dates / T))
        
        # Clamp i to valid range 1..11 (since t_12 is terminal, we don't decide exercise there in the same way, 
        # but actually at t_12 we just take payoff. The policy decides at t_1..t_11 whether to exercise 
        # or continue. At t_12, it's always "exercise" in the sense of realizing the payoff, 
        # but the function returns boolean. If called at t_12, we should return True (exercise) 
        # because there is no continuation.
        
        if i >= self.n_dates:
            # At the last date, always exercise (realize payoff)
            return np.ones(len(current_state), dtype=bool)
        
        if i < 1:
            # Before first exercise date
            return np.zeros(len(current_state), dtype=bool)

        # i is now 1..11
        # Get coefficients for step i
        # coeffs is stored for steps 1..11
        # Index in coeffs list: i-1
        beta = self.coeffs[i-1]
        bias = self.biases[i-1]
        
        X = self._get_features(current_state)
        # Predicted continuation value
        C_pred = X @ beta
        C_adj = C_pred + bias
        
        # Immediate payoff
        P = np.maximum(self.K - current_state, 0.0)
        
        # Discounted continuation value
        C_disc = self.discount_factor * C_adj
        
        # Exercise if Payoff >= Discounted Continuation
        return P >= C_disc

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # Parse contract parameters
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)
    
    n_paths = training_paths.shape[0]
    n_steps = n_dates
    
    dt = T / n_steps
    discount_factor = np.exp(-r * dt)
    
    # training_paths shape: (n_paths, n_dates + 1) or (n_dates + 1, n_paths)?
    # Usually (n_paths, n_times). Let's assume (n_paths, n_dates + 1) where index 0 is t=0.
    # If shape is (n_dates+1, n_paths), we transpose.
    if training_paths.shape[0] < n_dates + 1 and training_paths.shape[1] > n_dates:
        training_paths = training_paths.T
        
    # Check shape
    if training_paths.shape[1] < n_dates + 1:
        # Might be just exercise dates?
        # Problem says "12 monthly exercise dates". 
        # Usually paths include t=0 to t=T.
        # Let's assume shape is (n_paths, n_dates + 1).
        pass

    # Initialize value at the last date (t_12)
    # At t_12, the value is the intrinsic value of the put
    S_end = training_paths[:, -1]
    V_next = np.maximum(K - S_end, 0.0)
    
    coeffs = []
    biases = []
    
    # Backward induction from t_11 down to t_1
    # t_i corresponds to column i in training_paths (if 0 is t_0)
    # t_1 is col 1, ..., t_11 is col 11, t_12 is col 12
    for i in range(n_dates - 1, 0, -1):
        if time.time() > deadline:
            break
            
        S_i = training_paths[:, i]
        
        # Immediate payoff at t_i
        P_i = np.maximum(K - S_i, 0.0)
        
        # Features for regression
        log_S = np.log(S_i)
        X = np.column_stack([np.ones_like(S_i), log_S, log_S**2, log_S**3])
        
        # Regress V_next on X
        # Solve least squares: X beta = V_next
        # Use np.linalg.lstsq for stability
        beta, residuals, rank, s = np.linalg.lstsq(X, V_next, rcond=None)
        
        # Predicted continuation value on training paths
        C_pred_train = X @ beta
        
        # Bias correction: mean of (True Continuation - Predicted Continuation)
        # Note: V_next is the value at t_{i+1}. The continuation value at t_i is E[V_{i+1} | S_i].
        # The regression estimates this expectation.
        # The bias is the difference between the actual V_next (which is a random variable) 
        # and its conditional expectation estimate.
        # Wait, the standard LS bias correction uses the difference between the 
        # realized next value and the predicted next value on the *same* paths.
        bias = np.mean(V_next - C_pred_train)
        
        # Adjusted continuation value for training paths (for calculating V_i)
        # We use the adjusted value to determine the optimal exercise decision on training paths
        # to generate the "true" V_i for the next step of the backward induction.
        # However, for the final policy, we will use the bias-corrected prediction.
        
        C_adj_train = C_pred_train + bias
        C_disc_train = discount_factor * C_adj_train
        
        # Value at t_i on training paths
        V_i = np.maximum(P_i, C_disc_train)
        
        # Store coefficients and bias for this step i
        coeffs.insert(0, beta) # Insert at front so index 0 is step 1
        biases.insert(0, bias)
        
        # Update V_next for next iteration
        V_next = V_i
        
    # If we broke out early, we might not have all coeffs. 
    # But we must return a valid policy.
    # If we didn't finish, we can pad with simple heuristic or last known coeffs.
    # Given 12 steps and fast numpy, it should finish.
    
    # Create the policy object
    # coeffs[0] corresponds to step 1, coeffs[1] to step 2, etc.
    policy = BermudanExercisePolicy(contract, coeffs, biases, dt, r, K, n_dates)
    
    return policy

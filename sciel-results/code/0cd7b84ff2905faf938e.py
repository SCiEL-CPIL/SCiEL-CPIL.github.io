"""Bermudan put exercise policy via Longstaff-Schwartz Monte Carlo method."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    # Extract parameters from contract
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)

    # Exercise dates: 12 monthly, no time-0
    dt = T / n_exercise
    exercise_times = np.arange(1, n_exercise + 1) * dt

    # Get training paths shape: (n_paths, n_steps)
    # training_paths should be the full paths including all time steps
    n_paths, n_steps = training_paths.shape

    # Verify n_steps matches n_exercise + 1 (time 0 to T)
    # Actually, for Bermudan with 12 exercise dates and no time-0 exercise,
    # we need paths at times t_1, t_2, ..., t_12
    # training_paths likely has shape (n_paths, n_exercise) or (n_paths, n_exercise + 1)
    
    # Let's assume training_paths gives us the states at each exercise date
    # If it has n_steps == n_exercise, those are the exercise dates
    # If it has n_steps == n_exercise + 1, the first column is t=0
    
    if n_steps == n_exercise:
        # Paths at exercise dates t_1, ..., t_n
        exercise_states = training_paths
    elif n_steps == n_exercise + 1:
        # Paths at t_0, t_1, ..., t_n; use t_1 to t_n
        exercise_states = training_paths[:, 1:]
    else:
        # Fallback: use last n_exercise columns
        exercise_states = training_paths[:, -n_exercise:]

    n_paths = exercise_states.shape[0]

    # Discount factors
    discount = np.exp(-r * dt)

    # Payoff at maturity (last exercise date)
    payoff_maturity = np.maximum(K - exercise_states[:, -1], 0.0)

    # We'll do backward induction using regression
    # For each exercise date from last to first, we estimate the continuation value
    # using features and regression, then decide to exercise or continue

    # Feature functions for regression
    def make_features(S, t_idx):
        """Create feature matrix for regression at time index t_idx."""
        n = S.shape[0]
        # Standard features for Bermudan options
        X = np.column_stack([
            np.ones(n),
            S,
            S**2,
            np.log(S + 1e-10),
            np.log(S + 1e-10)**2,
            np.maximum(K - S, 0.0),  # intrinsic value
            S * np.log(S + 1e-10),
        ])
        return X

    # Store continuation values for each exercise date
    # cont_values[i] = estimated continuation value at exercise date i
    # We compute backward from maturity

    # Initialize: at maturity, continuation value is 0 (must exercise or expire)
    # Actually, at the last exercise date, if you don't exercise, the option expires
    # So continuation value at last date is 0, and optimal value is intrinsic

    # For Longstaff-Schwartz:
    # V_n = max(intrinsic_n, E[discount * V_{n+1} | S_n])
    # We estimate E[discount * V_{n+1} | S_n] via regression

    # Start from the second-to-last exercise date going backward
    # We need to store the estimated continuation values at each date

    # Let's use a simpler approach: iterative backward induction with regression
    # At each step, we have the "future value" which is the estimated value from the next step
    # We regress the discounted future value on current features to get continuation value

    # Initialize future value at maturity: this is the payoff if we hold to maturity
    # But for Bermudan, we can exercise at any date, so we need to be careful

    # Standard LS approach for Bermudan:
    # 1. Simulate paths
    # 2. At maturity, V_N = payoff_N
    # 3. For i = N-1, ..., 1:
    #    a. For paths that are "alive" (not yet exercised in the optimal policy), 
    #       regress discount * V_{i+1} on features at time i
    #    b. continuation_value_i = predicted value from regression
    #    c. V_i = max(intrinsic_i, continuation_value_i)
    #    d. Mark paths where intrinsic_i > continuation_value_i as exercised

    # For simplicity and speed, let's use all paths in regression (not just alive ones)
    # This is a common simplification

    # Store values at each exercise date
    values = np.zeros((n_paths, n_exercise))
    
    # At maturity (last index)
    values[:, -1] = payoff_maturity

    # Backward induction
    for i in range(n_exercise - 2, -1, -1):
        if time.time() > deadline - 5.0:
            # Not enough time, break and use what we have
            break
        
        # Intrinsic value at date i
        intrinsic = np.maximum(K - exercise_states[:, i], 0.0)
        
        # Target: discounted value from next date
        target = discount * values[:, i + 1]
        
        # Features at date i
        X = make_features(exercise_states[:, i], i)
        
        # OLS regression
        try:
            # Use lstsq for numerical stability
            coeffs, _, _, _ = np.linalg.lstsq(X, target, rcond=None)
            continuation = X @ coeffs
        except Exception:
            # Fallback: use simple average
            continuation = np.full(n_paths, np.mean(target))
        
        # Optimal value at date i
        values[:, i] = np.maximum(intrinsic, continuation)
    
    # Now we need to construct the exercise policy
    # The policy should exercise when intrinsic > continuation value
    # We need to store the continuation values (or the decision) at each date

    # Reconstruct the exercise decisions
    # We need to know, at each date, whether to exercise given the current state
    # The continuation value is a function of the state, so we need to store the regression coefficients
    # or the continuation value function

    # For the exercise function, we'll store the regression coefficients at each date
    # and evaluate the continuation value at query time

    reg_coeffs = [None] * n_exercise
    
    # Recompute backward to store coefficients
    for i in range(n_exercise - 2, -1, -1):
        if time.time() > deadline - 3.0:
            break
        
        intrinsic = np.maximum(K - exercise_states[:, i], 0.0)
        target = discount * values[:, i + 1]
        X = make_features(exercise_states[:, i], i)
        
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, target, rcond=None)
            reg_coeffs[i] = coeffs
        except Exception:
            # Fallback: constant continuation value
            reg_coeffs[i] = np.array([np.mean(target)])  # Only intercept
        
        values[:, i] = np.maximum(intrinsic, X @ reg_coeffs[i])

    # For the last date, no regression needed (must exercise or expire)
    # reg_coeffs[-1] remains None

    # Create the exercise policy object
    class ExercisePolicy:
        def __init__(self, coeffs, K, r, dt, n_exercise):
            self.coeffs = coeffs
            self.K = K
            self.r = r
            self.dt = dt
            self.n_exercise = n_exercise

        def _make_features(self, S):
            """Create feature matrix for a single state or array of states."""
            if np.isscalar(S):
                S = np.array([S])
            n = S.shape[0]
            X = np.column_stack([
                np.ones(n),
                S,
                S**2,
                np.log(S + 1e-10),
                np.log(S + 1e-10)**2,
                np.maximum(self.K - S, 0.0),
                S * np.log(S + 1e-10),
            ])
            return X

        def exercise(self, current_state, time_idx):
            """
            Determine whether to exercise at the given time index.
            
            Parameters:
            -----------
            current_state : array-like
                Current spot price(s). Can be scalar or array.
            time_idx : int
                Index of the current exercise date (0-based, where 0 is the first exercise date).
            
            Returns:
            --------
            boolean array indicating whether to exercise at each state.
            """
            if np.isscalar(current_state):
                S = np.array([current_state])
                scalar_input = True
            else:
                S = np.asarray(current_state)
                scalar_input = False
            
            if time_idx >= self.n_exercise:
                # Beyond last exercise date, cannot exercise
                result = np.zeros(S.shape, dtype=bool)
                return result[0] if scalar_input else result
            
            if time_idx == self.n_exercise - 1:
                # Last exercise date: exercise if intrinsic > 0
                result = S < self.K
                return result[0] if scalar_input else result
            
            # Check if we have coefficients for this date
            if self.coeffs[time_idx] is None:
                # No regression available, use intrinsic value only
                result = S < self.K
                return result[0] if scalar_input else result
            
            # Compute continuation value
            X = self._make_features(S)
            continuation = X @ self.coeffs[time_idx]
            
            # Intrinsic value
            intrinsic = np.maximum(self.K - S, 0.0)
            
            # Exercise if intrinsic >= continuation value
            result = intrinsic >= continuation
            
            return result[0] if scalar_input else result

    return ExercisePolicy(reg_coeffs, K, r, dt, n_exercise)
# EVOLVE-BLOCK-END

"""
Bermudan exercise-policy research: LSMC-based backward induction.

Strategy:
- Use Least-Squares Monte Carlo (LSMC) with polynomial features to estimate
  continuation values at each exercise date.
- Backward induction from the last exercise date to the first.
- At each step, fit a regression of the discounted future value (or terminal payoff)
  on the current spot price using polynomial features (degree 2-4).
- The exercise decision is: exercise if intrinsic payoff >= fitted continuation value.

This adapts to the specific training paths and handles continuous state evaluation
smoothly. The regression captures the optimal boundary's shape directly from data.
"""

import time
import numpy as np
from numpy.polynomial import polynomial as P

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using LSMC backward induction.
    
    Parameters:
    - training_paths: array of shape (n_paths, n_steps+1) with GBM path values
    - contract: dict with keys 'K', 'r', 'sigma', 'T', 'n_dates', 'n_paths', etc.
    
    Returns:
    - object with .exercise(current_state, time) method returning boolean vector
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin: 10s before 150s kill
    
    # Extract parameters from contract
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)
    n_paths = training_paths.shape[0]
    
    # Training paths shape: (n_paths, n_steps + 1) where n_steps = n_dates
    # Exercise dates are at indices 1, 2, ..., n_dates (no time-0 exercise)
    # Path values at time t are at column t (if paths start at t=0)
    
    # Determine the number of columns in training_paths
    n_cols = training_paths.shape[1]
    
    # Exercise dates correspond to columns 1 through n_dates (1-indexed)
    # We need to figure out which columns correspond to exercise dates
    # Typically, if there are n_dates exercise dates evenly spaced over [0, T],
    # and the path has n_dates+1 points (including t=0), then columns 1..n_dates are exercise dates
    
    # Assume training_paths has shape (n_paths, n_dates + 1)
    # Column 0 is t=0, columns 1..n_dates are exercise dates
    
    # Extract spot prices at each exercise date
    # S[t] for t = 1, ..., n_dates
    S = np.zeros((n_dates, n_paths))
    for t in range(1, n_dates + 1):
        if t < n_cols:
            S[t-1] = training_paths[:, t]
        else:
            S[t-1] = training_paths[:, -1]
    
    # Time step for discounting
    dt = T / n_dates
    
    # Discount factors
    discount = np.exp(-r * dt)
    
    # Initialize continuation values (to be computed backward)
    # cont_val[t] will hold the continuation value at date t for each path
    # We'll store fitted functions for each date
    
    # Polynomial degree for regression
    poly_degree = 3
    
    # Storage for fitted coefficients at each date
    # fitted_funcs[t] will be a callable that takes S and returns continuation value estimate
    fitted_funcs = [None] * n_dates
    
    # Terminal payoff at last exercise date (date n_dates)
    # For a put: max(K - S, 0)
    
    # Backward induction: start from the last exercise date
    # At date n_dates (index n_dates-1 in our 0-indexed array), the value is just the payoff
    # For dates n_dates-1 down to 1, we fit regression
    
    # Let's work with 0-indexed dates: date 0 is first exercise, date n_dates-1 is last
    
    # At the last exercise date, the "future value" is just the payoff itself
    # But we need to fit a function for the continuation at the second-to-last date
    
    # Standard LSMC:
    # 1. At the last exercise date, compute payoff: payoff_T = max(K - S_T, 0)
    # 2. For t = T-1 down to 1:
    #    a. Compute "future value" for paths that would continue: discount * (payoff if exercised at t+1 or later)
    #    b. But in LSMC, we use the fitted continuation from the next step
    #    c. Fit regression of the "continuation value" (which is the discounted expected future payoff) on S_t
    
    # Let me restructure:
    # cont_value[t] = estimated continuation value at date t (before exercise decision)
    # At date n_dates-1 (last): cont_value = max(K - S, 0)  [no future, so continuation = intrinsic]
    # Actually, at the last date, you always exercise if in the money, so value = intrinsic
    # 
    # For date t (0-indexed, t < n_dates-1):
    #   The continuation value is E[discount * V_{t+1} | S_t]
    #   where V_{t+1} is the value at date t+1 (which is max(intrinsic, continuation))
    
    # Let's compute backward:
    # values[t] = value at date t for each path (after optimal exercise decision)
    
    values = np.zeros((n_dates, n_paths))
    
    # Last date: always exercise if in the money
    values[-1] = np.maximum(K - S[-1], 0.0)
    
    # For earlier dates, we need to fit regression
    # We'll store the fitted regression functions
    reg_funcs = [None] * n_dates
    
    # Work backward from date n_dates-2 to date 0
    for t in range(n_dates - 2, -1, -1):
        if time.time() > deadline - 5.0:
            # Not enough time, use simple heuristic for remaining dates
            break
        
        # The "target" for regression at date t is the discounted value at date t+1
        # But we need to account for the fact that at date t+1, the holder can exercise
        # So the target is discount * values[t+1]
        
        target = discount * values[t + 1]
        
        # Fit polynomial regression: target ~ poly(S_t)
        S_t = S[t]
        
        # Use polynomial features
        # Create design matrix
        X = np.ones((n_paths, poly_degree + 1))
        for d in range(1, poly_degree + 1):
            X[:, d] = S_t ** d
        
        # Solve least squares
        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(X, target, rcond=None)
        except np.linalg.LinAlgError:
            # Fallback to constant
            coeffs = np.array([np.mean(target)] + [0.0] * poly_degree)
        
        # Store the fitted function
        def make_func(coeffs, degree):
            def func(s):
                result = np.zeros_like(s, dtype=float)
                for d in range(degree + 1):
                    result += coeffs[d] * (s ** d)
                return result
            return func
        
        reg_funcs[t] = make_func(coeffs, poly_degree)
        
        # Compute continuation value for each path at date t
        cont_val = reg_funcs[t](S_t)
        
        # Intrinsic payoff at date t
        intrinsic = np.maximum(K - S_t, 0.0)
        
        # Value at date t: max(intrinsic, continuation)
        values[t] = np.maximum(intrinsic, cont_val)
    
    # Now create the exercise policy object
    class ExercisePolicy:
        def __init__(self, reg_funcs, K, n_dates, S_train):
            self.reg_funcs = reg_funcs
            self.K = K
            self.n_dates = n_dates
            self.S_train = S_train
            # Precompute some statistics for fallback
            self.mean_S = [np.mean(S_train[t]) for t in range(n_dates)]
            self.std_S = [np.std(S_train[t]) for t in range(n_dates)]
        
        def exercise(self, current_state, time):
            """
            Determine whether to exercise at each path.
            
            Parameters:
            - current_state: array of current spot prices (one per path)
            - time: current time index (0-indexed exercise date, 0 to n_dates-1)
            
            Returns:
            - boolean array: True if should exercise
            """
            t = int(time)
            if t < 0 or t >= self.n_dates:
                # Out of range, default to no exercise except at last date
                if t == self.n_dates - 1:
                    return np.maximum(self.K - current_state, 0.0) > 0
                return np.zeros(len(current_state), dtype=bool)
            
            # Intrinsic payoff
            intrinsic = np.maximum(self.K - current_state, 0.0)
            
            # If at the last exercise date, always exercise if in the money
            if t == self.n_dates - 1:
                return intrinsic > 0
            
            # Get continuation value from fitted regression
            if self.reg_funcs[t] is not None:
                cont_val = self.reg_funcs[t](current_state)
                # Exercise if intrinsic >= continuation
                return intrinsic >= cont_val
            else:
                # Fallback: simple heuristic
                # Exercise if deep in the money
                moneyness = (self.K - current_state) / self.K
                return moneyness > 0.1  # Exercise if >10% in the money
    
    policy = ExercisePolicy(reg_funcs, K, n_dates, S)
    return policy

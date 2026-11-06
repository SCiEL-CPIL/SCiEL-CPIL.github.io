"""Bermudan exercise-policy: Longstaff-Schwartz regression-based backward induction."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from numpy.polynomial import polynomial as P


class BermudanExercisePolicy:
    def __init__(self, thresholds, discount_factors, K):
        """
        thresholds: array of shape (n_dates,) where thresholds[i] is the exercise threshold
                    for exercise date i (0-indexed, corresponding to dates t_1, ..., t_12)
        discount_factors: array of shape (n_dates,) for discounting from each date to t=0
        K: strike price
        """
        self.thresholds = np.asarray(thresholds, dtype=np.float64)
        self.discount_factors = np.asarray(discount_factors, dtype=np.float64)
        self.K = K

    def exercise(self, current_state, time_index):
        """
        Return a boolean vector indicating whether to exercise at time_index for each path.
        current_state: array of spot prices, shape (n_paths,)
        time_index: int, 0-based index of the exercise date (0 = first exercise date at t_1)
        """
        if time_index < 0 or time_index >= len(self.thresholds):
            return np.zeros(len(current_state), dtype=bool)
        # Exercise if S < K and S < threshold (i.e., in-the-money and below threshold)
        # Actually for a put, we exercise when S is low enough.
        # The threshold represents the value below which it's optimal to exercise.
        return (current_state < self.K) & (current_state < self.thresholds[time_index])


def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using Longstaff-Schwartz regression-based
    backward induction on the training paths.
    
    training_paths: array of shape (n_paths, n_dates + 1) where column 0 is S0 and
                    columns 1..n_dates are the spot prices at each monthly exercise date.
    contract: dict with keys S0, K, r, sigma, T, n_dates
    """
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    # Parse contract parameters
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)
    n_paths = training_paths.shape[0]
    dt = T / n_dates

    # Discount factors from each exercise date to t=0
    # Exercise dates are at t_1 = dt, t_2 = 2*dt, ..., t_12 = 12*dt = T
    # Discount factor from t_i to t_0 is exp(-r * t_i) = exp(-r * i * dt)
    dates = np.arange(1, n_dates + 1) * dt
    discount_factors = np.exp(-r * dates)

    # Terminal payoff at t_T (last exercise date)
    # For a put: max(K - S_T, 0)
    S_terminal = training_paths[:, n_dates]
    terminal_payoff = np.maximum(K - S_terminal, 0.0)

    # Backward induction: for each exercise date from last to first,
    # compute the continuation value and determine exercise threshold.
    # 
    # At each date i, we have:
    # - Immediate exercise value: max(K - S_i, 0)
    # - Continuation value: E[discounted value from next date onward | S_i]
    # 
    # We use regression to estimate the continuation value as a function of S_i.
    # 
    # The optimal policy: exercise if immediate value > continuation value.
    # For a put, this is equivalent to: exercise if S_i < threshold_i.

    # We'll store the continuation value estimates and thresholds for each date.
    # Start from the last date: continuation value = 0 (no more exercise opportunities after T)
    # Actually, at the last exercise date (t_12 = T), the continuation value is 0
    # because there are no more exercise opportunities. So we always exercise if in-the-money.
    
    # Let's work backward from date n_dates-1 (0-indexed) to date 0.
    # continuation_values[i] = estimated continuation value at date i given S_i
    # We'll use polynomial regression to estimate E[continuation | S_i]
    
    # For simplicity and speed, we'll use a piecewise approach:
    # For each date, we compute the optimal threshold by finding the point where
    # immediate exercise value equals the expected discounted continuation value.
    
    # Let's use a more direct approach:
    # At each date i (0-indexed, corresponding to t_{i+1}), we have spot prices S_i.
    # The continuation value is the expected discounted value from date i+1 onward.
    # We estimate this using regression on the training paths.
    
    # Initialize: continuation value at the last date is 0 (no future exercise)
    # So at the last date, we exercise if S < K (in-the-money put).
    
    # For earlier dates, we need to estimate E[discounted payoff from i+1 to T | S_i].
    
    # Let's use a simple approach: for each date, we'll compute the continuation value
    # as the mean of the discounted optimal values from the next date, conditioned on S_i.
    # We'll use binning or regression to estimate this conditional expectation.
    
    # For efficiency, let's use polynomial regression with a few terms.
    
    # Store thresholds for each date
    thresholds = np.zeros(n_dates, dtype=np.float64)
    
    # Start from the last date (index n_dates - 1)
    # At the last date, continuation value = 0, so exercise if S < K
    thresholds[n_dates - 1] = K  # Exercise if S < K at the last date
    
    # Work backward from date n_dates-2 to date 0
    # For each date i, we need to estimate the continuation value C(S_i) = E[discounted value from i+1 | S_i]
    
    # For the continuation value, we use the fact that at date i+1, the optimal value is:
    # V(S_{i+1}) = max(K - S_{i+1}, 0) if S_{i+1} < threshold_{i+1}, else E[discounted V(S_{i+2}) | S_{i+1}]
    
    # This is complex. Let's use a simpler approximation:
    # At each date i, we estimate the continuation value as a function of S_i using regression.
    # The target for regression at date i is the discounted optimal value from date i+1 onward.
    
    # Let's compute the optimal value at each date from the training paths using the thresholds we've determined.
    
    # Optimal value at the last date:
    V = np.maximum(K - S_terminal, 0.0)  # V at date n_dates-1 (last)
    
    # Work backward
    for i in range(n_dates - 2, -1, -1):
        # Check time
        if time.time() > deadline:
            break
        
        # S_i is the spot price at date i (0-indexed), which is training_paths[:, i+1]
        S_i = training_paths[:, i + 1]
        
        # Immediate exercise value at date i
        immediate = np.maximum(K - S_i, 0.0)
        
        # Continuation value: discounted expected value from date i+1
        # The discount from date i to date i+1 is exp(-r * dt)
        disc = np.exp(-r * dt)
        
        # The continuation value is disc * E[V_{i+1} | S_i]
        # We estimate E[V_{i+1} | S_i] using regression.
        
        # For simplicity, we'll use a simple binning approach:
        # Divide S_i into bins and compute the mean of V_{i+1} in each bin.
        # Then for a given S_i, interpolate to get the conditional expectation.
        
        # Actually, let's use a polynomial fit for speed and smoothness.
        # Fit V_{i+1} as a polynomial in S_i.
        
        # Use a low-degree polynomial to avoid overfitting
        degree = 3
        try:
            # Fit polynomial: V_{i+1} ≈ poly(S_i)
            coeffs = np.polyfit(S_i, V, degree)
            # Evaluate the polynomial at S_i to get the estimated conditional expectation
            cont_est = np.polyval(coeffs, S_i)
            # Ensure non-negative
            cont_est = np.maximum(cont_est, 0.0)
            cont_value = disc * cont_est
        except Exception:
            # Fallback: use mean
            cont_value = disc * np.mean(V)
        
        # Optimal value at date i: max(immediate, continuation)
        V = np.maximum(immediate, cont_value)
        
        # Determine threshold: the point where immediate = continuation
        # For a put, immediate = K - S for S < K, 0 for S >= K
        # We want to find S* such that K - S* = cont_value(S*)
        # This is implicit. Let's find it numerically by scanning.
        
        # Create a fine grid of S values
        S_grid = np.linspace(0, 2 * K, 1000)
        immediate_grid = np.maximum(K - S_grid, 0.0)
        
        # Evaluate continuation value on the grid
        try:
            cont_grid = disc * np.polyval(coeffs, S_grid)
            cont_grid = np.maximum(cont_grid, 0.0)
        except Exception:
            cont_grid = disc * np.mean(V)
        
        # Find where immediate >= continuation
        # Exercise if immediate > continuation, i.e., K - S > cont(S) for S < K
        # The threshold is the largest S where immediate >= continuation
        
        # Compute the difference
        diff = immediate_grid - cont_grid
        # Find where diff >= 0
        exercise_mask = diff >= 0
        # The threshold is the maximum S where we exercise
        if np.any(exercise_mask):
            threshold = S_grid[exercise_mask].max()
        else:
            threshold = 0.0  # Never exercise
        
        # Clamp threshold to [0, K]
        threshold = np.clip(threshold, 0.0, K)
        thresholds[i] = threshold
    
    # If we broke out of the loop early due to time, fill remaining thresholds with K
    for i in range(n_dates - 1, -1, -1):
        if thresholds[i] == 0.0 and i < n_dates - 1:
            thresholds[i] = K
        if time.time() > deadline:
            break
    
    return BermudanExercisePolicy(thresholds, discount_factors, K)
# EVOLVE-BLOCK-END

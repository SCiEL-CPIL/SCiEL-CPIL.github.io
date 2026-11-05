"""Bermudan put exercise policy via Monte Carlo Backward Induction with Regression."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import brentq

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using Monte Carlo backward induction 
    with polynomial regression for continuation values.
    
    Returns an object with .exercise(current_state, time) method that
    returns a boolean vector indicating whether to exercise at each path.
    """
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin before 120s kill

    # Extract parameters from contract
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)
    
    # training_paths shape: (n_paths, n_times) where n_times = n_exercise + 1 (includes t=0)
    # We only care about exercise dates 1..n_exercise.
    # Indices in training_paths: 0 is t=0, 1 is t=1/12, ..., 12 is t=1.
    
    n_paths = training_paths.shape[0]
    
    # Precompute discount factors for each step
    dt = T / n_exercise
    discount = np.exp(-r * dt)
    
    # We will store the fitted continuation value models for each exercise date
    # Model: V_cont(S) = discount * P(log(S))
    # We fit P(log(S)) to the optimal values of the next step.
    
    # Start from the last exercise date
    # At t = n_exercise (index n_exercise in paths), the value is the payoff.
    S_last = training_paths[:, n_exercise]
    V_next = np.maximum(K - S_last, 0.0)
    
    boundaries = [0.0] * n_exercise
    # Store the polynomial coefficients for each step (from n_exercise-1 down to 0)
    # poly_coeffs[i] corresponds to the continuation value at step i+1 (0-indexed step in 1..n_exercise)
    # Let's align indices:
    # Exercise date k (1-indexed) corresponds to column k in training_paths.
    # Continuation value at date k depends on values at date k+1.
    
    # We fit a polynomial of degree 4 in log(S)
    degree = 4
    
    # We will iterate backwards from k = n_exercise - 1 down to 0
    # k is the index in the 'boundaries' list (0-indexed), corresponding to date k+1
    # The value V_next currently holds the optimal values for date k+2.
    # We need to fit a function to approximate E[V_{k+2} | S_{k+1}].
    
    # Note: V_next is the value at the *next* date.
    # We regress V_next on S at the *current* date (which is training_paths[:, k+1]).
    
    for k in range(n_exercise - 1, -1, -1):
        if time.time() > deadline:
            # If we run out of time, use current best estimates
            # Fill remaining boundaries with 0 or K/2
            for j in range(k + 1):
                if boundaries[j] == 0.0: # Check if uninitialized (though we init to 0)
                    boundaries[j] = K * 0.5
            break
            
        S_curr = training_paths[:, k + 1]
        
        # Filter out paths where S_curr is extremely small or zero to avoid log issues
        # In GBM, S > 0 always, but numerically it might be very small.
        valid_mask = S_curr > 1e-8
        if np.sum(valid_mask) < degree + 1:
            # Not enough data, fallback to simple threshold
            boundaries[k] = K * 0.8
            continue
            
        S_curr_valid = S_curr[valid_mask]
        V_next_valid = V_next[valid_mask]
        
        # Fit polynomial in log(S)
        log_S_curr = np.log(S_curr_valid)
        
        # Use np.polynomial.polynomial.polyfit
        # We want to fit V_next_valid = sum(c_i * (log_S)^i)
        # This approximates the conditional expectation E[V_{next} | S_{curr}]
        try:
            coeffs = np.polynomial.polynomial.polyfit(log_S_curr, V_next_valid, deg=degree)
        except Exception:
            # Fallback to linear if polyfit fails
            coeffs = np.polyfit(log_S_curr, V_next_valid, 1)
            # Convert from standard polyfit (highest power first) to polynomial module (lowest first)
            coeffs = coeffs[::-1]
            
        # Define the continuation value function
        # V_cont(S) = discount * P(log(S))
        # We need to find the boundary S* where Payoff(S) = V_cont(S)
        # Payoff(S) = K - S for S < K, 0 otherwise.
        # For S < K: K - S = discount * P(log(S))
        # f(S) = K - S - discount * P(log(S)) = 0
        
        def f(S):
            if S <= 0:
                return K
            if S >= K:
                return -discount * np.polynomial.polynomial.polyval(np.log(S), coeffs)
            # Calculate P(log(S))
            val = np.polynomial.polynomial.polyval(np.log(S), coeffs)
            return K - S - discount * val
            
        # Find the root in [1e-6, K]
        # f(0+) = K > 0
        # f(K) = -discount * P(log(K)) which is likely negative if continuation value is positive
        # We want the root where f crosses from positive to negative?
        # If S is very small, Payoff is large, V_cont is small. f > 0. Exercise is optimal.
        # If S is close to K, Payoff is small, V_cont is larger. f < 0. Hold is optimal.
        # So we look for a root in (0, K).
        
        try:
            # Use brentq. Need a sign change.
            # Check f at a small epsilon
            S_low = 1e-6
            S_high = K - 1e-6
            
            f_low = f(S_low)
            f_high = f(S_high)
            
            if f_low * f_high > 0:
                # No sign change, might be always exercise or always hold
                # If f_low > 0 and f_high > 0, always exercise? (Unlikely for put near K)
                # If f_low < 0 and f_high < 0, always hold?
                # Check the sign at S = K/2
                S_mid = K / 2.0
                f_mid = f(S_mid)
                
                if f_low > 0 and f_mid > 0:
                    # Exercise everywhere?
                    boundaries[k] = K
                elif f_mid < 0 and f_high < 0:
                    # Hold everywhere?
                    boundaries[k] = 0.0
                else:
                    # Find root in [S_low, S_mid] or [S_mid, S_high]
                    if f_low * f_mid < 0:
                        boundaries[k] = brentq(f, S_low, S_mid, maxiter=50)
                    else:
                        boundaries[k] = brentq(f, S_mid, S_high, maxiter=50)
            else:
                boundaries[k] = brentq(f, S_low, S_high, maxiter=50)
                
        except Exception:
            # Fallback: use the max S in training data where exercise was optimal
            # We don't have the optimal exercise mask yet, but we can estimate
            # Let's just use a heuristic: K * 0.8
            boundaries[k] = K * 0.8
            
        # Update V_next for the next iteration (k-1)
        # V_next becomes the optimal value at step k+1
        # V_opt(S_curr) = max(Payoff(S_curr), V_cont(S_curr))
        
        # We only need V_opt for the paths that will be used in the next regression.
        # The next regression will use S at step k and V at step k+1.
        # So V_next should be the optimal values at step k+1 for ALL paths.
        
        # Calculate V_cont for all paths at step k+1
        log_S_all = np.log(np.maximum(S_curr, 1e-8))
        V_cont_all = discount * np.polynomial.polynomial.polyval(log_S_all, coeffs)
        Payoff_all = np.maximum(K - S_curr, 0.0)
        
        V_next = np.maximum(Payoff_all, V_cont_all)
        
    # Ensure boundaries are monotonically non-decreasing (earlier dates -> lower/equal threshold)
    # For a put, as time progresses, the region of exercise expands (threshold increases).
    # boundaries[0] is t=1/12, boundaries[11] is t=1.
    # So boundaries[k] <= boundaries[k+1]
    for k in range(n_exercise - 1):
        if boundaries[k] > boundaries[k + 1]:
            boundaries[k] = boundaries[k + 1]
            
    # Cap boundaries at K
    for k in range(n_exercise):
        if boundaries[k] > K:
            boundaries[k] = K
        if boundaries[k] < 0:
            boundaries[k] = 0.0

    class ExercisePolicy:
        def __init__(self, K, boundaries, n_exercise, T):
            self.K = K
            self.boundaries = boundaries
            self.n_exercise = n_exercise
            self.T = T
        
        def exercise(self, current_state, time):
            """
            current_state: array of S values (shape: (n_paths,))
            time: current time t
            Returns boolean array: True if should exercise at this path
            """
            n_paths = len(current_state)
            result = np.zeros(n_paths, dtype=bool)
            
            # Determine which exercise date we're at based on time
            # time is given as the current time; find the closest exercise date
            best_ex_idx = -1
            best_dist = float('inf')
            for i in range(1, self.n_exercise + 1):
                t_ex = i / self.n_exercise
                dist = abs(time - t_ex)
                if dist < best_dist:
                    best_dist = dist
                    best_ex_idx = i - 1  # 0-indexed
            
            if best_ex_idx >= 0 and best_ex_idx < len(self.boundaries):
                threshold = self.boundaries[best_ex_idx]
                # Exercise if S <= threshold
                result = (current_state <= threshold)
            
            return result
    
    policy = ExercisePolicy(K, boundaries, n_exercise, T)
    return policy

# EVOLVE-BLOCK-END

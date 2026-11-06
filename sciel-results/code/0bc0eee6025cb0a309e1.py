"""Bermudan exercise policy using Longstaff-Schwartz regression-based backward induction."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from numpy.polynomial import polynomial as P

def fit_exercise_policy(training_paths, contract):
    """
    Fits a Bermudan exercise policy using Longstaff-Schwartz method.
    
    Args:
        training_paths: Array of shape (N, n_steps) containing simulated spot prices.
                        Typically n_steps includes the initial time or just exercise dates.
                        Based on "12 monthly exercise dates (no time-0 exercise)", 
                        we assume training_paths has 12 columns corresponding to t_1..t_12.
                        If it has 13, we ignore the first.
        contract: Dictionary containing contract parameters (S0, K, r, sigma, T, n_dates).
    
    Returns:
        An object with an `exercise(current_state, time)` method.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Extract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)
    
    # Determine time step
    dt = T / n_dates
    discount_factor = np.exp(-r * dt)

    # Prepare data
    paths = np.asarray(training_paths)
    if paths.ndim != 2:
        paths = paths.reshape(-1, n_dates)
        
    if paths.shape[1] > n_dates:
        paths = paths[:, 1:n_dates+1]
    elif paths.shape[1] < n_dates:
        pass

    N_paths = paths.shape[0]
    
    # Storage for regression coefficients for each time step
    poly_coeffs = [None] * n_dates
    
    # Value at the last exercise date (index n_dates-1)
    S_last = paths[:, n_dates-1]
    V_last = np.maximum(K - S_last, 0.0)
    
    current_values = V_last.copy()
    
    # Loop backward from n_dates-2 to 0
    for i in range(n_dates - 2, -1, -1):
        if time.time() > deadline:
            break
            
        S_i = paths[:, i]
        
        # The target for regression is the discounted value of the next step
        target = discount_factor * current_values
        
        # Fit polynomial to S_i
        # Use degree 4 for good balance of fit and stability
        # We use in-the-money subset for regression to improve LS accuracy
        itm_mask = (K - S_i) > 0
        if np.sum(itm_mask) > 20:
            S_i_fit = S_i[itm_mask]
            target_fit = target[itm_mask]
        else:
            S_i_fit = S_i
            target_fit = target
            
        try:
            coeffs = P.polyfit(S_i_fit, target_fit, deg=4)
        except Exception:
            try:
                coeffs = P.polyfit(S_i_fit, target_fit, deg=3)
            except Exception:
                coeffs = P.polyfit(S_i_fit, target_fit, deg=1)
                
        poly_coeffs[i] = coeffs
        
        # Calculate estimated continuation value for all paths at step i
        C_i = P.polyval(S_i, coeffs)
        
        # Immediate payoff at step i
        Payoff_i = np.maximum(K - S_i, 0.0)
        
        # Optimal value at step i is max(Payoff_i, C_i)
        V_i = np.maximum(Payoff_i, C_i)
        
        # Set current_values to V_i for the next iteration (i-1)
        current_values = V_i
        
    # Create the policy object
    class BermudanExercisePolicy:
        def __init__(self, coeffs, K, r, dt, n_dates):
            self.coeffs = coeffs
            self.K = K
            self.r = r
            self.dt = dt
            self.n_dates = n_dates
            self.discount_factor = np.exp(-r * dt)

        def exercise(self, current_state, time):
            """
            Determines whether to exercise the option.
            
            Args:
                current_state: Array of spot prices at the current time.
                time: The index of the current exercise date (0-based, 0..n_dates-1).
            
            Returns:
                Boolean array indicating whether to exercise (True) or hold (False).
            """
            # If it's the last date, always exercise if in the money
            if time >= self.n_dates - 1:
                return np.maximum(self.K - current_state, 0.0) > 0.0
            
            # Get coefficients for this time step
            if self.coeffs[time] is None:
                return current_state < self.K * 0.95
            
            # Calculate estimated continuation value
            C = P.polyval(current_state, self.coeffs[time])
            
            # Calculate immediate payoff
            Payoff = np.maximum(self.K - current_state, 0.0)
            
            # Exercise if Payoff > Continuation Value
            return Payoff > C + 1e-9

    return BermudanExercisePolicy(poly_coeffs, K, r, dt, n_dates)

# EVOLVE-BLOCK-END

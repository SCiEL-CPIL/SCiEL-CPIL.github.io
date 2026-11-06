"""Rewritten program: threshold-based Bermudan put exercise policy via backward induction."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import math

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using backward induction on training paths.
    
    Args:
        training_paths: array of shape (n_paths, n_dates) with simulated spot prices
        contract: dict with contract parameters
    
    Returns:
        An object with exercise(current_state, time) method returning boolean vector
    """
    start_time = time.time()
    deadline = start_time + 105.0  # Safety margin before 120s limit
    
    # Extract parameters from contract
    K = contract.get('K', contract.get('strike', 100.0))
    r = contract.get('r', contract.get('rate', 0.05))
    sigma = contract.get('sigma', contract.get('vol', 0.2))
    T = contract.get('T', contract.get('maturity', 1.0))
    n_dates = contract.get('n_dates', contract.get('n_exercise', 12))
    
    # Ensure n_dates is 12 (monthly, no time-0)
    if n_dates is None:
        n_dates = 12
    
    # Training paths shape: (n_paths, n_dates) - these are the spot prices at each exercise date
    if training_paths is None or len(training_paths) == 0:
        # Fallback: simple threshold policy
        return _SimpleThresholdPolicy(K, r, sigma, T, n_dates)
    
    paths = np.asarray(training_paths, dtype=np.float64)
    n_paths = paths.shape[0]
    
    # The paths contain spot prices at exercise dates 1..n_dates
    # We need to determine exercise decisions at each date
    
    # Strategy: Use backward induction with a grid of spot values
    # For each date t (from last to first), compute the value function V_t(S)
    # V_t(S) = max(K - S, exp(-r*dt) * E[V_{t+1}(S_{t+1}) | S_t = S])
    
    # Since we have GBM dynamics, S_{t+1} = S_t * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
    # We can compute the expectation analytically or via Monte Carlo
    
    dt = T / n_dates
    
    # Create a dense log-spaced grid for the value function
    # Log-spaced grid gives better resolution across the range, especially for low S values
    n_grid = 800
    S_min = 0.01 * K
    S_max = 3.0 * K
    # Use log-spaced grid for better resolution across orders of magnitude
    S_grid = np.geomspace(S_min, S_max, n_grid)
    
    # Backward induction
    # V[n_dates] is the terminal payoff: max(K - S, 0)
    V_next = np.maximum(K - S_grid, 0.0)
    
    # Store thresholds: at each date t, exercise if S < threshold_t
    thresholds = np.zeros(n_dates)
    
    # Precompute quadrature (same for all steps)
    # Use higher-order Gauss-Hermite quadrature for more accurate integration
    n_quad = 80
    z_quad, w_quad = np.polynomial.hermite.hermgauss(n_quad)
    z_norm = z_quad * np.sqrt(2.0)
    w_norm = w_quad / np.sqrt(np.pi)
    
    drift = (r - 0.5 * sigma * sigma) * dt
    vol_step = sigma * np.sqrt(dt)
    disc_factor = np.exp(-r * dt)
    
    # Precompute the log of the grid for faster interpolation in log-space
    log_S_grid = np.log(S_grid)
    
    # Work backwards from date n_dates-1 down to 0
    for t_idx in range(n_dates - 1, -1, -1):
        # Compute continuation value: exp(-r*dt) * E[V_next(S_{t+1}) | S = S_grid]
        # S_next = S * exp(drift + vol_step * Z)
        # In log-space: log(S_next) = log(S) + drift + vol_step * Z
        log_S_next = log_S_grid[:, np.newaxis] + drift + vol_step * z_norm[np.newaxis, :]
        
        # Convert back to linear space for interpolation
        S_next = np.exp(log_S_next)
        
        # Interpolate V_next at S_next
        S_next_clipped = np.clip(S_next, S_min, S_max)
        V_interp = np.interp(S_next_clipped, S_grid, V_next)
        
        # Weighted sum for expectation
        cont_value = disc_factor * np.sum(w_norm[np.newaxis, :] * V_interp, axis=1)
        
        # Immediate payoff
        immediate = np.maximum(K - S_grid, 0.0)
        
        # Value function at this date
        V_current = np.maximum(immediate, cont_value)
        
        # Find threshold: largest S where immediate >= cont_value
        diff = immediate - cont_value
        valid_mask = S_grid < K
        if np.any(valid_mask):
            diff_valid = diff[valid_mask]
            pos_indices = np.where(diff_valid >= 0)[0]
            if len(pos_indices) > 0:
                last_pos = pos_indices[-1]
                # Refine threshold with local bisection using the value function
                lo = S_grid[last_pos]
                if last_pos + 1 < len(S_grid) and S_grid[last_pos + 1] < K:
                    hi = S_grid[last_pos + 1]
                else:
                    hi = K
                
                # Bisection refinement: evaluate diff at midpoints
                def compute_diff_at(S_val):
                    """Compute immediate - continuation at a single S value."""
                    log_S_val = np.log(S_val)
                    log_S_next_single = log_S_val + drift + vol_step * z_norm
                    S_next_single = np.exp(log_S_next_single)
                    S_next_single_clipped = np.clip(S_next_single, S_min, S_max)
                    V_interp_single = np.interp(S_next_single_clipped, S_grid, V_next)
                    cont_single = disc_factor * np.sum(w_norm * V_interp_single)
                    imm_single = max(K - S_val, 0.0)
                    return imm_single - cont_single
                
                # Bisection: 12 iterations for ~1e-4 relative precision
                for _ in range(12):
                    mid = 0.5 * (lo + hi)
                    diff_mid = compute_diff_at(mid)
                    if diff_mid >= 0:
                        lo = mid
                    else:
                        hi = mid
                
                threshold = 0.5 * (lo + hi)
            else:
                threshold = 0.0
        else:
            threshold = 0.0
        
        thresholds[t_idx] = threshold
        V_next = V_current
        
        # Check time
        if time.time() > deadline - 5.0:
            break
    
    # Create policy object
    class ExercisePolicy:
        def __init__(self, thresholds, K):
            self.thresholds = thresholds
            self.K = K
        
        def exercise(self, current_state, time_idx):
            """
            Determine whether to exercise at given time index.
            
            Args:
                current_state: array of spot prices
                time_idx: integer index of the exercise date (0-based, 0 = first exercise date)
            
            Returns:
                Boolean array: True if exercise
            """
            if time_idx < 0 or time_idx >= len(self.thresholds):
                return np.zeros(len(current_state), dtype=bool)
            
            threshold = self.thresholds[time_idx]
            return np.asarray(current_state) < threshold
    
    return ExercisePolicy(thresholds, K)


def _SimpleThresholdPolicy(K, r, sigma, T, n_dates):
    """Fallback simple threshold policy."""
    dt = T / n_dates
    # Rough heuristic: threshold increases toward K as we approach maturity
    # At the last date, threshold is close to K
    # At the first date, threshold is lower
    thresholds = np.zeros(n_dates)
    for i in range(n_dates):
        # Time to maturity from this date
        t_remaining = (n_dates - i) * dt
        # Simple heuristic: threshold = K * (1 - exp(-alpha * t_remaining))
        # Where alpha is tuned
        alpha = 2.0
        thresholds[i] = K * (1.0 - np.exp(-alpha * t_remaining))
    
    class ExercisePolicy:
        def __init__(self, thresholds, K):
            self.thresholds = thresholds
            self.K = K
        
        def exercise(self, current_state, time_idx):
            if time_idx < 0 or time_idx >= len(self.thresholds):
                return np.zeros(len(current_state), dtype=bool)
            threshold = self.thresholds[time_idx]
            return np.asarray(current_state) < threshold
    
    return ExercisePolicy(thresholds, K)
# EVOLVE-BLOCK-END

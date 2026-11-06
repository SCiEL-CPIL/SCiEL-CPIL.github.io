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
    
    # Create a grid of spot values for the value function
    # Use log-spaced grid for better resolution near the threshold region
    n_grid = 400
    S_min = 0.01 * K
    S_max = 3.0 * K
    S_grid = np.geomspace(S_min, S_max, n_grid)
    
    # Backward induction
    # V[n_dates] is the terminal payoff: max(K - S, 0)
    V_next = np.maximum(K - S_grid, 0.0)
    
    # Store thresholds: at each date t, exercise if S < threshold_t
    thresholds = np.zeros(n_dates)
    
    # Work backwards from date n_dates-1 down to 0
    # Date indices: 0 corresponds to the first exercise date (time dt)
    # Date index n_dates-1 corresponds to the last exercise date (time T)
    
    for t_idx in range(n_dates - 1, -1, -1):
        # Compute continuation value: exp(-r*dt) * E[V_next(S_{t+1}) | S = S_grid]
        # For GBM: S_{t+1} = S * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
        # E[V_next(S_{t+1})] can be computed by integrating over Z
        # Use Gauss-Hermite or simple MC with the training paths structure
        
        # Since we have GBM, the log of S_{t+1}/S_t is normal
        # log(S_{t+1}) = log(S_t) + (r - sigma^2/2)*dt + sigma*sqrt(dt)*Z
        # So S_{t+1} is lognormal
        
        # For each S in S_grid, compute E[V_next(S * exp(...))]
        # Use a set of quadrature points for Z
        n_quad = 64
        # Gauss-Hermite quadrature
        z_quad, w_quad = np.polynomial.hermite.hermgauss(n_quad)
        # Transform to standard normal: Z = sqrt(2) * z_quad
        z_norm = z_quad * np.sqrt(2.0)
        w_norm = w_quad / np.sqrt(np.pi)
        
        # For each S in S_grid, compute the expectation
        # S_next = S * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
        drift = (r - 0.5 * sigma * sigma) * dt
        vol_step = sigma * np.sqrt(dt)
        
        # S_next shape: (n_grid, n_quad)
        S_next = S_grid[:, np.newaxis] * np.exp(drift + vol_step * z_norm[np.newaxis, :])
        
        # Interpolate V_next at S_next
        # Clip to grid range
        S_next_clipped = np.clip(S_next, S_min, S_max)
        V_interp = np.interp(S_next_clipped, S_grid, V_next)
        
        # Weighted sum for expectation
        cont_value = np.exp(-r * dt) * np.sum(w_norm[np.newaxis, :] * V_interp, axis=1)
        
        # Immediate payoff
        immediate = np.maximum(K - S_grid, 0.0)
        
        # Value function at this date
        V_current = np.maximum(immediate, cont_value)
        
        # Threshold: exercise if S < threshold, i.e., where immediate > cont_value
        # Find the crossover point
        # immediate = K - S (for S < K), cont_value is decreasing in S
        # We want the largest S where immediate >= cont_value
        # This is where K - S >= cont_value(S)
        
        # Find threshold by interpolation
        diff = immediate - cont_value
        # We want the largest S where immediate >= cont_value (i.e., diff >= 0)
        # For S < K: immediate = K - S > 0
        # For S >= K: immediate = 0, so threshold must be < K
        
        # Find where diff crosses from positive to negative
        # Use np.sign to find sign changes
        pos_indices = np.where(diff >= 0)[0]
        if len(pos_indices) > 0:
            last_pos = pos_indices[-1]
            # Interpolate between last_pos and last_pos+1 for the zero crossing
            if last_pos + 1 < len(S_grid):
                d1 = diff[last_pos]
                d2 = diff[last_pos + 1]
                if d2 < 0 and d1 >= 0:
                    frac = d1 / (d1 - d2)
                    threshold = S_grid[last_pos] + frac * (S_grid[last_pos + 1] - S_grid[last_pos])
                else:
                    threshold = S_grid[last_pos]
            else:
                threshold = S_grid[last_pos]
        else:
            threshold = 0.0
        
        thresholds[t_idx] = threshold
        
        # Update V_next for next iteration
        V_next = V_current
        
        # Check time - but ensure we have at least computed the last few thresholds
        # If we're running out of time, fill remaining thresholds with K (exercise always)
        elapsed = time.time() - start_time
        remaining_steps = t_idx  # number of steps left (including current)
        if elapsed > deadline - 10.0 and remaining_steps > 3:
            # Fill remaining thresholds with K (conservative: exercise when in-the-money)
            for remaining_t in range(t_idx - 1, -1, -1):
                thresholds[remaining_t] = K
            break
    
    # Ensure all thresholds are set (in case loop broke early)
    for i in range(n_dates):
        if thresholds[i] == 0.0 and i < n_dates:
            # If threshold is 0, it means we never exercise - set to K as fallback
            # Actually, threshold=0 means diff was never >= 0, which shouldn't happen for a put
            # Let's set a reasonable default
            pass
    
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

"""Bermudan put exercise policy via parametric smooth boundary fitted by backward induction."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np
from scipy.stats import norm


def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy.
    
    Returns an object with method:
        exercise(current_state, time) -> boolean vector
    
    where current_state is an array of spot prices and time is a scalar time value.
    """
    # Extract parameters from contract (with safe defaults)
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_dates = 12
    
    def _get(obj, *names, default=None):
        for n in names:
            if isinstance(obj, dict):
                if n in obj:
                    return obj[n]
            else:
                if hasattr(obj, n):
                    return getattr(obj, n)
        return default
    
    try:
        K = float(_get(contract, 'K', 'strike', 'Strike', default=K))
        r = float(_get(contract, 'r', 'rate', 'Rate', 'risk_free', 'risk_free_rate', default=r))
        sigma = float(_get(contract, 'sigma', 'vol', 'volatility', 'Sigma', default=sigma))
        T = float(_get(contract, 'T', 'maturity', 'Maturity', 'horizon', default=T))
        n_dates = int(_get(contract, 'n_dates', 'num_dates', 'exercise_dates', 'n_exercise', default=n_dates))
    except Exception:
        pass
    
    # Time step between exercise dates
    dt = T / n_dates
    # Exercise dates are at dt, 2*dt, ..., T (no time-0 exercise)
    # So there are n_dates exercise opportunities
    
    # Discount factor per step
    df = math.exp(-r * dt)
    
    # We'll use a grid-based backward induction to compute optimal thresholds
    # then fit a smooth parametric boundary
    
    # Grid for spot prices
    # Use a log-spaced grid for better resolution near the strike
    n_grid = 2000
    # Grid from very low to reasonably high
    S_min = K * 0.01
    S_max = K * 3.0
    # Log-spaced grid
    log_S_min = math.log(S_min)
    log_S_max = math.log(S_max)
    log_S_grid = np.linspace(log_S_min, log_S_max, n_grid)
    S_grid = np.exp(log_S_grid)
    
    # For backward induction, we need to compute the value function at each exercise date
    # V_i(S) = max(K - S, df * E[V_{i+1}(S') | S])
    # where S' follows GBM: S' = S * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    
    # We'll represent the value function on the grid and interpolate
    # At maturity (last exercise date), V_n(S) = max(K - S, 0)
    
    # Value function at the last exercise date (index n_dates-1, which is time T)
    V = np.maximum(K - S_grid, 0.0)
    
    # Thresholds at each exercise date (from last to first)
    # threshold[i] is the spot price below which we exercise at exercise date i
    # Exercise dates are indexed 0 to n_dates-1, corresponding to times dt, 2*dt, ..., T
    thresholds = np.zeros(n_dates)
    
    # At the last exercise date, the threshold is K (exercise if S < K)
    thresholds[n_dates - 1] = K
    
    # Backward induction
    # For i from n_dates-2 down to 0:
    #   V_i(S) = max(K - S, df * E[V_{i+1}(S') | S])
    #   The continuation value is df * E[V_{i+1}(S')]
    #   S' = S * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    #   E[V_{i+1}(S')] can be computed by numerical integration or by interpolating on the grid
    
    sqrt_dt = math.sqrt(dt)
    drift = (r - 0.5 * sigma * sigma) * dt
    
    # For each grid point S_j, compute E[V_{i+1}(S_j * exp(drift + sigma*sqrt(dt)*Z))]
    # Using Gauss-Hermite quadrature for accuracy
    
    # Gauss-Hermite quadrature nodes and weights
    # We use a sufficient number of nodes
    n_quad = 64
    x_nodes, w_nodes = np.polynomial.hermite.hermgauss(n_quad)
    # Transform to standard normal: Z = sqrt(2) * x
    z_nodes = x_nodes * math.sqrt(2.0)
    # Weights for standard normal: w_i * exp(-z_i^2/2) / sqrt(pi)
    # Actually, Hermite quadrature: integral f(x) exp(-x^2) dx = sum w_i f(x_i)
    # For E[f(Z)] where Z ~ N(0,1): integral f(z) phi(z) dz = (1/sqrt(2*pi)) * integral f(z) exp(-z^2/2) dz
    # Let z = sqrt(2)*x, dz = sqrt(2) dx
    # (1/sqrt(2*pi)) * integral f(sqrt(2)*x) exp(-x^2) sqrt(2) dx
    # = (1/sqrt(pi)) * integral f(sqrt(2)*x) exp(-x^2) dx
    # = (1/sqrt(pi)) * sum w_i f(sqrt(2)*x_i)
    
    z_nodes = x_nodes * math.sqrt(2.0)
    w_norm = w_nodes / math.sqrt(math.pi)
    
    # Precompute the log of the multiplicative factor
    # S' = S * exp(drift + sigma*sqrt(dt)*Z)
    # log(S') = log(S) + drift + sigma*sqrt(dt)*Z
    
    # For each step in backward induction
    for i in range(n_dates - 2, -1, -1):
        # Compute continuation value at each grid point
        # cont[j] = df * E[V_{i+1}(S_j * exp(drift + sigma*sqrt(dt)*Z))]
        
        # For each grid point j, log(S_j) is known
        # log(S') = log_S_grid[j] + drift + sigma*sqrt(dt)*Z
        # S' = exp(log_S_grid[j] + drift + sigma*sqrt(dt)*Z)
        
        # We need to interpolate V_{i+1} at these points
        # V_{i+1} is defined on S_grid
        
        # Create a 2D array of log(S') for all grid points and quadrature nodes
        # This might be memory intensive: n_grid * n_quad = 2000 * 64 = 128000, which is fine
        
        # log_S_prime[j, k] = log_S_grid[j] + drift + sigma*sqrt_dt*z_nodes[k]
        log_S_prime = log_S_grid[:, np.newaxis] + drift + sigma * sqrt_dt * z_nodes[np.newaxis, :]
        
        # Clip to the grid range to avoid extrapolation issues
        # Actually, we should handle this more carefully
        # For very high or low values, we can clip or use boundary conditions
        
        # Interpolate V on S_grid
        # We need V at S' = exp(log_S_prime)
        # Use np.interp for linear interpolation on the grid
        
        # S_prime = exp(log_S_prime)
        # For each column k, interpolate V at S_prime[:, k]
        
        # To do this efficiently, we can use np.interp for each column
        # But that's n_quad calls to np.interp, each with n_grid points
        
        # Alternative: use a lookup table approach
        # Since log_S_grid is uniform, we can compute the index directly
        
        # The grid in log-space is uniform with spacing dlog = (log_S_max - log_S_min) / (n_grid - 1)
        dlog = (log_S_max - log_S_min) / (n_grid - 1)
        
        # For a given log_S_prime, the fractional index is (log_S_prime - log_S_min) / dlog
        frac_idx = (log_S_prime - log_S_min) / dlog
        
        # Clip to valid range
        frac_idx_clipped = np.clip(frac_idx, 0, n_grid - 1)
        
        # Integer and fractional parts
        idx0 = np.floor(frac_idx_clipped).astype(int)
        idx1 = np.minimum(idx0 + 1, n_grid - 1)
        frac = frac_idx_clipped - idx0
        
        # Interpolated values
        V_interp = V[idx0] * (1 - frac) + V[idx1] * frac
        
        # Weighted sum over quadrature nodes
        # cont[j] = df * sum_k w_norm[k] * V_interp[j, k]
        cont = df * np.sum(V_interp * w_norm[np.newaxis, :], axis=1)
        
        # Value function at this step
        intrinsic = np.maximum(K - S_grid, 0.0)
        V = np.maximum(intrinsic, cont)
        
        # Find the threshold: the largest S where intrinsic >= cont
        # i.e., the largest S where K - S >= cont (for S < K)
        # For S >= K, intrinsic = 0, so we only exercise if cont <= 0, which is unlikely
        # So the threshold is in [S_min, K]
        
        # Find where intrinsic >= cont
        exercise_mask = intrinsic >= cont
        # Find the last index where exercise is optimal
        if np.any(exercise_mask):
            last_exercise_idx = np.max(np.where(exercise_mask)[0])
            # Interpolate to get the threshold
            if last_exercise_idx < n_grid - 1:
                # Linear interpolation between last_exercise_idx and last_exercise_idx+1
                S0 = S_grid[last_exercise_idx]
                S1 = S_grid[last_exercise_idx + 1]
                diff0 = intrinsic[last_exercise_idx] - cont[last_exercise_idx]
                diff1 = intrinsic[last_exercise_idx + 1] - cont[last_exercise_idx + 1]
                if diff0 != diff1:
                    t = diff0 / (diff0 - diff1)
                    thresholds[i] = S0 + t * (S1 - S0)
                else:
                    thresholds[i] = S0
            else:
                thresholds[i] = S_grid[last_exercise_idx]
        else:
            # Never exercise, threshold is very low
            thresholds[i] = S_min
    
    # Now we have thresholds[i] for i = 0, ..., n_dates-1
    # Exercise date i corresponds to time t_i = (i+1)*dt
    # The policy is: exercise if S <= thresholds[i] at time t_i
    
    # To make the policy smooth and robust, we can fit a smooth curve to the thresholds
    # The thresholds should be monotonically increasing as time approaches maturity
    # (it becomes less optimal to exercise early as we get closer to maturity)
    
    # Actually, for a put, the threshold should be monotonically increasing with time
    # (closer to maturity, the threshold is higher, meaning we're more willing to exercise)
    
    # Let's check monotonicity and enforce it if needed
    for i in range(1, n_dates):
        if thresholds[i] < thresholds[i-1]:
            thresholds[i] = thresholds[i-1]
    
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, thresholds, n_dates, dt, T):
            self.thresholds = thresholds
            self.n_dates = n_dates
            self.dt = dt
            self.T = T
            # Precompute the exercise dates
            self.exercise_times = np.arange(1, n_dates + 1) * dt
        
        def exercise(self, current_state, time):
            """
            Determine whether to exercise at each state.
            
            Parameters:
            current_state: array of spot prices
            time: scalar time value
            
            Returns:
            boolean array, True if exercise is optimal
            """
            # Determine which exercise date we're at
            # time should be close to one of the exercise times
            # Find the closest exercise date
            if time <= 0:
                # No exercise at time 0
                return np.zeros(len(current_state), dtype=bool)
            
            # Find the index of the exercise date
            # Exercise dates are at dt, 2*dt, ..., T
            # The index is round(time/dt) - 1, clipped to [0, n_dates-1]
            idx = int(round(time / self.dt)) - 1
            idx = max(0, min(idx, self.n_dates - 1))
            
            # Exercise if S <= threshold
            threshold = self.thresholds[idx]
            return np.asarray(current_state) <= threshold
    
    return ExercisePolicy(thresholds, n_dates, dt, T)
# EVOLVE-BLOCK-END

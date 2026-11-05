"""Bermudan exercise-policy research: direct backward-induction approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import sys


def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using backward induction on a fine grid.
    
    The approach:
    1. Extract contract parameters (K, r, sigma, T, n_steps, etc.)
    2. Build a fine discretization grid for the underlying asset price
    3. Perform backward induction from maturity to the first exercise date
    4. At each exercise date, compare immediate exercise payoff (K - S) with 
       the discounted expected continuation value
    5. Return an ExercisePolicy object that uses the fitted boundary to decide
       whether to exercise at each (state, time) pair
    
    Parameters:
    -----------
    training_paths : ndarray, shape (n_paths, n_steps+1)
        Training price paths from risk-neutral GBM simulation
    contract : dict or object
        Contract specification
    
    Returns:
    --------
    object with exercise(current_state, time) method returning boolean vector
    """
    start_time = time.time()
    deadline = 135.0  # wall-clock limit with safety margin (evaluator kills at 150s)
    
    # Extract contract parameters
    K = _get_param(contract, 'K', 100.0)
    r = _get_param(contract, 'r', 0.05)
    sigma = _get_param(contract, 'sigma', 0.2)
    T = _get_param(contract, 'T', 1.0)
    n_steps = _get_param(contract, 'n_steps', 12)
    
    # Exercise dates: 12 monthly, no time-0
    dt = T / n_steps
    exercise_times = np.arange(1, n_steps + 1) * dt  # t_1, t_2, ..., t_12
    
    # Build a fine grid for the underlying price
    # Use log-space grid for better resolution with GBM dynamics
    S_min = 5.0
    S_max = 130.0
    n_grid = 10000  # Increased resolution
    log_S_grid = np.linspace(np.log(S_min), np.log(S_max), n_grid)
    S_grid = np.exp(log_S_grid)
    
    # Discount factors
    df = np.exp(-r * dt)
    
    # Precompute Gauss-Hermite quadrature nodes/weights for E[f(Z)] where Z ~ N(0,1)
    # Use numpy's built-in hermgauss
    n_quad = 60
    x_gh, w_gh = np.polynomial.hermite.hermgauss(n_quad)
    # Transform: for Z ~ N(0,1), E[f(Z)] = (1/sqrt(pi)) * integral exp(-x^2) f(sqrt(2)*x) dx
    z_nodes = np.sqrt(2.0) * x_gh
    z_weights = w_gh / np.sqrt(np.pi)
    
    # Backward induction
    # V stores the value function at the current exercise date
    # At maturity (last exercise date), the put is simply (K - S)+
    V = np.maximum(K - S_grid, 0.0)
    
    # Store the stopping boundary at each exercise date
    # boundary[i] = the threshold S below which we exercise at date i
    boundaries = np.full(n_steps, np.nan)
    
    # Work backwards from the second-to-last date to the first date
    mu = r - 0.5 * sigma**2
    sd_log = sigma * math.sqrt(dt)
    
    for i in range(n_steps - 2, -1, -1):
        # Check deadline
        if time.time() > deadline - 15:
            break
        
        # Compute continuation value: discounted expected value at next step
        # S_next = S_j * exp(mu*dt + sd_log * Z)
        # log_S_next = log_S_grid[:, None] + mu*dt + sd_log * z_nodes[None, :]
        log_S_next = log_S_grid[:, None] + mu * dt + sd_log * z_nodes[None, :]
        
        # Clip log_S_next to grid range in log-space
        log_S_min = np.log(S_min)
        log_S_max = np.log(S_max)
        log_S_next_clipped = np.clip(log_S_next, log_S_min, log_S_max)
        
        # Vectorized interpolation of V at log_S_next_clipped
        # We have V defined on log_S_grid.
        # For each point in log_S_next_clipped (shape n_grid x n_quad), find interval in log_S_grid.
        
        # Search for indices
        # idx is the index such that log_S_grid[idx-1] <= log_S_next_clipped < log_S_grid[idx]
        # np.searchsorted returns the index where the value would be inserted.
        # Since log_S_grid is sorted, this works.
        idx = np.searchsorted(log_S_grid, log_S_next_clipped)
        
        # Clip indices to valid range [1, n_grid-1] to allow for left/right neighbor access
        idx = np.clip(idx, 1, n_grid - 1)
        
        # Get left and right grid values
        x_left = log_S_grid[idx - 1]
        x_right = log_S_grid[idx]
        
        # Get corresponding V values
        V_left = V[idx - 1]
        V_right = V[idx]
        
        # Linear interpolation weights
        # w = (x - x_left) / (x_right - x_left)
        # Handle potential division by zero (though grid is fixed, so x_right != x_left)
        denom = x_right - x_left
        w = (log_S_next_clipped - x_left) / denom
        
        # Interpolated V
        V_interp = V_left * (1 - w) + V_right * w
        
        # Weighted sum over quadrature nodes
        # cont[j] = sum_q w_q * V_interp[j, q]
        cont = np.dot(V_interp, z_weights)
        cont *= df
        
        # Decision: exercise if (K - S) > continuation value
        exercise_payoff = np.maximum(K - S_grid, 0.0)
        should_exercise = exercise_payoff >= cont
        V = np.where(should_exercise, exercise_payoff, cont)
        
        # Find the boundary: the highest S where we still exercise
        # (for a put, exercise region is S <= boundary)
        if np.any(should_exercise):
            boundary_idx = np.where(should_exercise)[0]
            boundaries[i] = S_grid[boundary_idx[-1]]
        else:
            boundaries[i] = S_min
    
    # Fill any remaining unfilled boundaries (if we hit the deadline)
    for i in range(n_steps - 1, -1, -1):
        if np.isnan(boundaries[i]):
            # Use a heuristic: for a Bermudan put, the optimal exercise boundary
            # is below K. A reasonable default is around 0.8*K for early dates,
            # approaching K for later dates.
            # Linear interpolation from 0.75*K (first date) to 0.95*K (last date)
            frac = i / max(n_steps - 1, 1)
            boundaries[i] = K * (0.75 + 0.20 * frac)
    
    # Create the exercise policy
    policy = _ExercisePolicy(boundaries, n_steps, dt, K)
    return policy


def _get_param(contract, key, default):
    """Extract parameter from contract (dict or object)."""
    if contract is None:
        return default
    if isinstance(contract, dict):
        return contract.get(key, default)
    else:
        # Try common attribute names
        for attr in [key, key.lower(), key.upper(), '_' + key]:
            if hasattr(contract, attr):
                return getattr(contract, attr)
        # Try common variations
        name_map = {
            'K': ['K', 'strike', 'strike_price', 'knockout', 'knockout_price'],
            'r': ['r', 'rate', 'risk_free_rate', 'riskfree_rate'],
            'sigma': ['sigma', 'vol', 'volatility', 'vola'],
            'T': ['T', 'maturity', 'expiry', 'tenor'],
            'n_steps': ['n_steps', 'n_exercise', 'n_exercises', 'n_dates', 'steps'],
        }
        if key in name_map:
            for alt in name_map[key]:
                if hasattr(contract, alt):
                    return getattr(contract, alt)
        return default


def _gauss_hermite(n):
    """Compute Gauss-Hermite nodes and weights for n points using numpy."""
    from numpy.polynomial.hermite import hermgauss
    x, w = hermgauss(n)
    return x, w


class _ExercisePolicy:
    """Exercise policy based on fitted boundaries."""
    
    def __init__(self, boundaries, n_steps, dt, K):
        self.boundaries = boundaries  # boundaries[i] for exercise date i+1
        self.n_steps = n_steps
        self.dt = dt
        self.K = K
    
    def exercise(self, current_state, time):
        """
        Determine whether to exercise at the given time and state.
        
        Parameters:
        -----------
        current_state : array-like
            Current spot prices
        time : float or int
            Current time (could be continuous or step index)
        
        Returns:
        --------
        numpy array of booleans
        """
        current_state = np.asarray(current_state, dtype=float)
        
        # Determine which exercise date this corresponds to
        # time could be a float (continuous) or int (step index)
        if isinstance(time, (int, np.integer)):
            step = int(time)
            if step < 1:
                return np.zeros(len(current_state), dtype=bool)
            elif step > self.n_steps:
                return np.zeros(len(current_state), dtype=bool)
            else:
                idx = step - 1  # 0-based index into boundaries
        else:
            t = float(time)
            step = round(t / self.dt)
            if step < 1:
                return np.zeros(len(current_state), dtype=bool)
            elif step > self.n_steps:
                return np.zeros(len(current_state), dtype=bool)
            else:
                idx = step - 1
        
        # Exercise if S <= boundary
        boundary = self.boundaries[idx]
        # Only exercise if intrinsic value is positive (S < K)
        return (current_state <= boundary) & (current_state < self.K)


# EVOLVE-BLOCK-END

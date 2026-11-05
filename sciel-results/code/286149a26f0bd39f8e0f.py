"""Bermudan exercise-policy research: high-precision backward-induction approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using high-precision backward induction.
    
    Improvements over previous version:
    - Higher grid resolution (20,000 points) to minimize discretization error.
    - Higher quadrature order (100 points) for more accurate expectation.
    - Wider price range [1, 200] to capture tail effects.
    """
    start_time = time.time()
    deadline = 145.0  # Wall-clock limit with safety margin (evaluator kills at 150s)
    
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
    # Use a hybrid grid: log-spaced near zero for better put payoff resolution,
    # linear in the middle, and linear to S_max for tail coverage.
    # The put is only active for S < K, so we need high resolution below K.
    S_min = 0.5
    S_max = 200.0
    n_grid = 20000
    
    # Hybrid grid: 70% of points in [S_min, K], 30% in [K, S_max]
    n_low = int(n_grid * 0.70)
    n_high = n_grid - n_low
    S_low = np.geomspace(S_min, K, n_low)  # log-spaced for better resolution near 0
    S_high = np.linspace(K, S_max, n_high)[1:]  # skip K to avoid duplicate
    S_grid = np.concatenate([S_low, S_high])
    # Ensure monotonicity
    S_grid = np.sort(S_grid)
    
    # Discount factors
    df = np.exp(-r * dt)
    
    # Backward induction
    # V[i] stores the value function at exercise date i (index 0 = last date)
    # At maturity (last exercise date), the put is simply (K - S)+
    V = np.maximum(K - S_grid, 0.0)
    
    # Store the stopping boundary at each exercise date
    # boundary[i] = the threshold S below which we exercise at date i
    boundaries = np.zeros(n_steps)
    
    # Precompute Gauss-Hermite quadrature nodes and weights
    n_quad = 100  # Increased from 60 for better accuracy
    x_gh, w_gh = _gauss_hermite(n_quad)
    z_nodes = np.sqrt(2.0) * x_gh
    z_weights = w_gh / np.sqrt(np.pi)
    
    mu = r - 0.5 * sigma**2
    sd_log = sigma * math.sqrt(dt)
    
    # Precompute log grid for efficiency
    log_S_grid = np.log(S_grid)
    
    # Work backwards from the second-to-last date to the first date
    for i in range(n_steps - 2, -1, -1):
        # Compute continuation value: discounted expected value at next step
        # Vectorized: for all S_grid points simultaneously
        # S_next[j, k] = S_grid[j] * exp(mu*dt + sd_log * z_nodes[k])
        log_S_next = log_S_grid[:, np.newaxis] + mu * dt + sd_log * z_nodes[np.newaxis, :]
        S_next = np.exp(log_S_next)
        
        # Clip to grid range to avoid interpolation errors
        S_next_clipped = np.clip(S_next, S_min, S_max)
        
        # Interpolate V at S_next for each grid point
        # V_interp[j, k] = V interpolated at S_next[j, k]
        V_interp = np.zeros_like(S_next_clipped)
        # Loop over quadrature points is acceptable for n_quad=100
        # np.interp is C-optimized
        for k in range(n_quad):
            V_interp[:, k] = np.interp(S_next_clipped[:, k], S_grid, V)
        
        # Weighted average over quadrature points
        cont = df * (V_interp * z_weights[np.newaxis, :]).sum(axis=1)
        
        # Decision: exercise if (K - S) > continuation value
        exercise_payoff = np.maximum(K - S_grid, 0.0)
        should_exercise = exercise_payoff >= cont
        V = np.where(should_exercise, exercise_payoff, cont)
        
        # Find the boundary: the highest S where we still exercise
        if np.any(should_exercise):
            boundary_idx = np.where(should_exercise)[0]
            # Refine the boundary with local bisection for higher precision
            b_idx = boundary_idx[-1]
            if b_idx < n_grid - 1:
                # Bisection between S_grid[b_idx] and S_grid[b_idx+1]
                lo = S_grid[b_idx]
                hi = S_grid[b_idx + 1]
                n_bisect = 30
                for _ in range(n_bisect):
                    if time.time() > deadline - 10:
                        break
                    mid = 0.5 * (lo + hi)
                    # Compute continuation value at mid
                    log_mid = np.log(mid)
                    log_next = log_mid + mu * dt + sd_log * z_nodes[np.newaxis, :]
                    S_next_mid = np.exp(log_next)
                    S_next_mid_clipped = np.clip(S_next_mid, S_min, S_max)
                    V_interp_mid = np.zeros(n_quad)
                    for k in range(n_quad):
                        V_interp_mid[k] = np.interp(S_next_mid_clipped[k], S_grid, V)
                    cont_mid = df * (V_interp_mid * z_weights).sum()
                    payoff_mid = max(K - mid, 0.0)
                    if payoff_mid >= cont_mid:
                        lo = mid
                    else:
                        hi = mid
                boundaries[i] = 0.5 * (lo + hi)
            else:
                boundaries[i] = S_grid[b_idx]
        else:
            boundaries[i] = S_min
        
        # Check deadline
        if time.time() > deadline - 5:
            break
    
    # If we didn't finish all iterations, fill remaining boundaries with a heuristic
    # This is unlikely given the speed, but a safety net
    for i in range(n_steps - 1, -1, -1):
        if boundaries[i] == 0.0:
            # Approximate boundary using intrinsic value heuristic
            # For a put, boundary is typically below K
            # A rough estimate is K * exp(-r*T/2) or similar, but K*0.85 is safe
            boundaries[i] = K * 0.85
    
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
    """Compute Gauss-Hermite nodes and weights for n points."""
    n_int = int(n)
    if n_int <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    if n_int == 1:
        return np.array([0.0]), np.array([math.sqrt(math.pi)])
    
    try:
        from numpy.polynomial.hermite import hermgauss
        x, w = hermgauss(n_int)
        return x, w
    except (ImportError, Exception):
        pass
    
    # Fallback: build Jacobi matrix for Hermite polynomials with weight exp(-x^2)
    b = np.sqrt(np.arange(1, n_int, dtype=float))
    A = np.diag(b, 1) + np.diag(b, -1)
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    
    idx = np.argsort(eigenvalues)
    x = eigenvalues[idx]
    v = eigenvectors[:, idx]
    
    w = math.sqrt(math.pi) * (v[0, :] ** 2)
    
    return x, w


class _ExercisePolicy:
    """Exercise policy based on fitted boundaries."""
    
    def __init__(self, boundaries, n_steps, dt, K):
        self.boundaries = boundaries
        self.n_steps = n_steps
        self.dt = dt
        self.K = K
    
    def exercise(self, current_state, time):
        """
        Determine whether to exercise at the given time and state.
        """
        current_state = np.asarray(current_state, dtype=float)
        
        if isinstance(time, (int, np.integer)):
            step = int(time)
            if step < 1:
                return np.zeros(len(current_state), dtype=bool)
            elif step > self.n_steps:
                return np.zeros(len(current_state), dtype=bool)
            else:
                idx = step - 1
        else:
            t = float(time)
            step = round(t / self.dt)
            if step < 1:
                return np.zeros(len(current_state), dtype=bool)
            elif step > self.n_steps:
                return np.zeros(len(current_state), dtype=bool)
            else:
                idx = step - 1
        
        boundary = self.boundaries[idx]
        return (current_state <= boundary) & (current_state < self.K)

# EVOLVE-BLOCK-END

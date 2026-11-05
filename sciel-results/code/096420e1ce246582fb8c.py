"""Bermudan exercise-policy research: grid-based backward induction."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using backward induction on a price grid.
    
    Parameters:
    -----------
    training_paths : ndarray
        Training price paths (not used for direct fitting in this deterministic approach, 
        but part of the interface).
    contract : dict or object
        Contract specification with keys/attrs: K, r, sigma, T, n_steps.
        
    Returns:
    --------
    object with exercise(current_state, time) method returning boolean vector.
    """
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s hard kill

    # Extract contract parameters robustly
    K = _get_param(contract, 'K', 100.0)
    r = _get_param(contract, 'r', 0.05)
    sigma = _get_param(contract, 'sigma', 0.2)
    T = _get_param(contract, 'T', 1.0)
    n_steps = _get_param(contract, 'n_steps', 12)
    
    # Ensure n_steps is an integer
    n_steps = int(n_steps)
    
    # Exercise dates: 12 monthly, no time-0
    dt = T / n_steps
    # exercise_times = np.arange(1, n_steps + 1) * dt

    # Build a fine grid for the underlying price
    # Focus on the relevant range: evaluation spots are 90/100/110, K=100
    # Put exercise is relevant for S < K.
    # Use a grid that covers the likely range of S at each step.
    # S_min = 5.0, S_max = 150.0
    S_min = 5.0
    S_max = 150.0
    n_grid = 2000  # Higher resolution for better boundary precision
    S_grid = np.linspace(S_min, S_max, n_grid)
    
    # Discount factors
    df = np.exp(-r * dt)
    
    # Backward induction
    # V stores the value function at the current exercise date being computed
    # Start at maturity (last exercise date), value is intrinsic
    V = np.maximum(K - S_grid, 0.0)
    
    # Store the stopping boundary at each exercise date
    # boundaries[i] corresponds to exercise date i+1 (1-indexed)
    # We will store the boundary for date 1 at boundaries[0], etc.
    # But backward induction computes from last to first.
    # So we store in a list and reverse or index carefully.
    boundaries = [0.0] * n_steps
    
    # Precompute Gauss-Hermite quadrature nodes and weights
    # Integral of exp(-x^2) f(x) dx approx sum w_k f(x_k)
    # For GBM: S_next = S * exp((r - 0.5*sigma^2)dt + sigma*sqrt(dt)*Z)
    # Z ~ N(0,1). 
    # We can transform the expectation to a quadrature over Z.
    # E[f(S_next)] = (1/sqrt(2*pi)) * integral exp(-z^2/2) f(S*exp(...+sigma*sqrt(dt)*z)) dz
    # Let x = z/sqrt(2). Then dz = sqrt(2) dx.
    # Integral becomes (1/sqrt(pi)) * integral exp(-x^2) f(S*exp(...+sigma*sqrt(2*dt)*x)) dx
    # So we use Gauss-Hermite nodes x_k and weights w_k for exp(-x^2).
    # The factor is 1/sqrt(pi).
    
    n_quad = 50  # 50 points is usually sufficient for smooth functions
    x_gh, w_gh = _gauss_hermite(n_quad)
    
    # Transform nodes to Z-space for direct use in exponent
    # z = sqrt(2) * x
    z_nodes = np.sqrt(2.0) * x_gh
    # Weights for standard normal pdf integration:
    # w_z = w_gh * sqrt(2) / sqrt(2*pi) ? 
    # Let's derive:
    # E[f(Z)] = (1/sqrt(2*pi)) int exp(-z^2/2) f(z) dz
    # Let z = sqrt(2)x, dz = sqrt(2)dx
    # = (1/sqrt(2*pi)) int exp(-x^2) f(sqrt(2)x) sqrt(2) dx
    # = (1/sqrt(pi)) int exp(-x^2) f(sqrt(2)x) dx
    # So weight for f(x) in GH quadrature is w_k.
    # The integral approx is sum w_k f(x_k).
    # So E[f(Z)] approx (1/sqrt(pi)) * sum w_k f(sqrt(2) x_k)
    
    mu = r - 0.5 * sigma**2
    sd_log = sigma * math.sqrt(dt)
    
    log_S_grid = np.log(S_grid)
    
    # Work backwards from the second-to-last date to the first date
    # i is the index of the date we are evaluating (0 to n_steps-1)
    # We start with V at date n_steps (maturity).
    # We compute V at date n_steps-1, ..., 1.
    # boundaries[i] will be the boundary for date i+1.
    # So when we are at loop step j (counting back from 0), we are computing date n_steps - 1 - j.
    
    for step in range(n_steps - 1):
        # Current date index (1-based): d = n_steps - step
        # We are computing the value at date d, which depends on date d+1 (already in V).
        
        # Compute continuation value: discounted expected value at next step
        # S_next[j, k] = S_grid[j] * exp(mu*dt + sd_log * z_nodes[k])
        
        # Vectorized computation
        # log_S_next = log_S_grid[:, np.newaxis] + mu * dt + sd_log * z_nodes[np.newaxis, :]
        # S_next = np.exp(log_S_next)
        
        # To save memory and time, we can compute this in chunks or just directly.
        # n_grid=2000, n_quad=50. 100k elements. Very fast.
        
        log_S_next = log_S_grid[:, np.newaxis] + mu * dt + sd_log * z_nodes[np.newaxis, :]
        S_next = np.exp(log_S_next)
        
        # Clip to grid range to avoid interpolation issues
        S_next_clipped = np.clip(S_next, S_min, S_max)
        
        # Interpolate V at S_next for each grid point
        # V_interp[j, k] = V interpolated at S_next[j, k]
        # np.interp is fast for 1D.
        V_interp = np.zeros_like(S_next_clipped)
        for k in range(n_quad):
            V_interp[:, k] = np.interp(S_next_clipped[:, k], S_grid, V)
        
        # Weighted average over quadrature points
        # cont = df * (1/sqrt(pi)) * sum_k (w_k * V_interp[:, k])
        cont = df * (V_interp * (w_gh / math.sqrt(math.pi))[np.newaxis, :]).sum(axis=1)
        
        # Decision: exercise if (K - S) > continuation value
        exercise_payoff = np.maximum(K - S_grid, 0.0)
        should_exercise = exercise_payoff >= cont
        V = np.where(should_exercise, exercise_payoff, cont)
        
        # Find the boundary: the highest S where we still exercise
        # For a put, exercise region is S <= boundary.
        if np.any(should_exercise):
            boundary_idx = np.where(should_exercise)[0]
            # The boundary is the max S in the exercise region
            # We can refine this by finding the crossing point, but grid resolution is usually fine.
            boundaries[n_steps - 1 - step] = S_grid[boundary_idx[-1]]
        else:
            boundaries[n_steps - 1 - step] = S_min
        
        # Check deadline
        if time.time() > deadline - 10:
            break
    
    # If we didn't finish all iterations, fill remaining boundaries with a heuristic
    # The loop runs from step 0 to n_steps-2, filling boundaries[n_steps-1] down to boundaries[1].
    # boundaries[0] (date 1) is the last one computed.
    # If we broke early, boundaries[0] might be 0.0.
    
    for i in range(n_steps):
        if boundaries[i] == 0.0:
            # Approximate boundary using intrinsic value heuristic
            # For a put, boundary is typically below K.
            # A safe guess is 0.8 * K or similar.
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
    """Compute Gauss-Hermite nodes and weights for n points.
    
    Returns nodes x_k and weights w_k such that:
    sum_k w_k * f(x_k) approx integral_{-inf}^{inf} exp(-x^2) * f(x) dx
    """
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
    
    # Sort by eigenvalue
    idx = np.argsort(eigenvalues)
    x = eigenvalues[idx]
    v = eigenvectors[:, idx]
    
    # Weights: w_k = sqrt(pi) * (v[0,k])^2
    w = math.sqrt(math.pi) * (v[0, :] ** 2)
    
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
        Determine whether to exercise the option at the given state and time.
        
        Parameters:
        -----------
        current_state : array-like
            Current asset prices.
        time : float
            Current time.
            
        Returns:
        --------
        np.ndarray of bool
            True if exercise is optimal, False otherwise.
        """
        # Determine which exercise date we are at
        # time is continuous. We need to map it to the nearest exercise date index.
        # Exercise dates are at t_1 = dt, t_2 = 2dt, ..., t_n = n*dt.
        # If time is close to t_i, we use boundary[i-1].
        
        # Calculate the index of the exercise date
        # t_i = i * dt
        # i = time / dt
        # We round to the nearest integer, but must be within [1, n_steps].
        
        # Handle edge cases: if time is slightly off due to floating point, round.
        i_float = time / self.dt
        i_int = int(np.round(i_float))
        
        # Clamp to valid range [1, n_steps]
        if i_int < 1:
            i_int = 1
        elif i_int > self.n_steps:
            i_int = self.n_steps
            
        # Get the boundary for this date
        # boundaries[0] is for date 1, boundaries[1] for date 2, etc.
        boundary = self.boundaries[i_int - 1]
        
        # Exercise if current_state < boundary
        # Note: For a put, we exercise if S < K and it's optimal.
        # The boundary already accounts for the optimality.
        # If S > boundary, we continue.
        
        current_state = np.asarray(current_state)
        return current_state < boundary

# EVOLVE-BLOCK-END

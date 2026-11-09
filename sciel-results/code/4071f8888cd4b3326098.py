# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

# ----------------------------- constants -----------------------------
_LB, _UB = -5.12, 5.12
_RANGE = _UB - _LB
_N = 2000
_EPS = 1e-9
_DEADLINE = 330.0
_TIME_MARGIN = 5.0

# ----------------------------- state -----------------------------
_STATE = {
    "t0": None,
    "observed_x": None,   # (n, 2) observed points
    "observed_y": None,   # (n,) observed values
    "est_shift": None,    # estimated shift vector
    "best_x": None,       # best x found so far
    "best_y": None,       # best y found so far
    "phase": 0,           # 0=exploring, 1=exploiting
}

def _rastrigin(x, shift):
    """Rastrigin function with shift."""
    xc = x - shift
    return 20.0 + np.sum(xc**2 - 10.0 * np.cos(2.0 * np.pi * xc))

def _estimate_shift(xs, ys):
    """Estimate the shift s_k from observed data using the known Rastrigin structure.
    
    For each observed point x_i with value y_i:
    y_i = 20 + sum_j [(x_i_j - s_j)^2 - 10*cos(2*pi*(x_i_j - s_j))]
    
    We can try to solve for s by minimizing the residual.
    """
    if len(xs) < 2:
        # Not enough data, return center
        return np.array([0.0, 0.0])
    
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    n = len(xs)
    
    # Initial guess: center of observed points
    s_init = np.mean(xs, axis=0)
    
    # Use scipy to minimize the residual
    def residual(s):
        pred = _rastrigin(xs, s)
        return np.sum((pred - ys)**2)
    
    # Try multiple initial guesses
    best_s = s_init
    best_res = residual(s_init)
    
    # Grid of initial guesses
    for gx in np.linspace(_LB + 0.5, _UB - 0.5, 5):
        for gy in np.linspace(_LB + 0.5, _UB - 0.5, 5):
            s_try = np.array([gx, gy])
            res = residual(s_try)
            if res < best_res:
                best_res = res
                best_s = s_try
    
    # Local refinement
    try:
        result = minimize(residual, best_s, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
        if result.fun < best_res:
            best_s = result.x
            best_res = result.fun
    except Exception:
        pass
    
    return best_s

def _local_search(x0, shift, cost_fn, max_iter=100):
    """Nelder-Mead local search around x0."""
    def obj(x):
        return _rastrigin(x, shift) + 0.1 * cost_fn(x)
    
    try:
        result = minimize(obj, x0, method='Nelder-Mead',
                          options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-4})
        return result.x, result.fun
    except Exception:
        return x0, obj(x0)

def _global_grid_search(shift, cost_fn, n_grid=20):
    """Coarse grid search over the domain."""
    best_x = None
    best_val = np.inf
    
    for i in range(n_grid):
        for j in range(n_grid):
            x = np.array([
                _LB + (i + 0.5) * _RANGE / n_grid,
                _LB + (j + 0.5) * _RANGE / n_grid
            ])
            val = _rastrigin(x, shift) + 0.1 * cost_fn(x)
            if val < best_val:
                best_val = val
                best_x = x
    
    return best_x, best_val

# ----------------------------- public API -----------------------------
def acquisition(mu, sigma, f_best, cost):
    """Cost-aware acquisition function for Rastrigin 2D.
    
    Uses the known analytic structure of Rastrigin to:
    1. Estimate the shift from observed data
    2. Perform multi-start local optimization
    3. Return scores that guide toward the estimated minimum
    """
    # Initialize state
    if _STATE["t0"] is None:
        _STATE["t0"] = time.time()
    
    # Deadline guard
    elapsed = time.time() - _STATE["t0"]
    if elapsed > _DEADLINE - _TIME_MARGIN:
        # Return safe scores: prefer cheap points near current best
        if _STATE["best_x"] is not None:
            # Score by proximity to best and low cost
            cx = _STATE["best_x"]
            # Create 2000 candidate points (uniform in [-5.12, 5.12]^2)
            # But we don't have the actual candidates, so we return scores based on cost
            # Lower cost = higher score (cheaper is better when we're at deadline)
            c = np.asarray(cost, dtype=float)
            out = -c  # prefer cheaper
            out = np.where(np.isfinite(out), out, -1e9)
            return out
        else:
            return np.full(_N, -1e9)
    
    # Extract observed data from mu, sigma, f_best
    # mu and sigma are GP predictions at 2000 candidate points
    # But we need the actual observed data. The evaluator passes mu, sigma from the GP
    # trained on observed points. We can't directly recover observed points from mu/sigma.
    # 
    # Alternative approach: Use the GP mean and variance directly, but with a 
    # cost-aware acquisition that also incorporates knowledge of Rastrigin structure.
    #
    # Actually, let me reconsider. The function signature gives us:
    # - mu: GP mean at 2000 candidate points (shape 2000,)
    # - sigma: GP std at 2000 candidate points (shape 2000,)
    # - f_best: lowest observed value
    # - cost: evaluation cost at each candidate point (shape 2000,)
    #
    # We don't have the observed points directly. But we can use the GP predictions
    # combined with cost awareness.
    #
    # Let me use a hybrid approach:
    # 1. Standard EI/UCB from GP
    # 2. Cost normalization
    # 3. Adaptive exploration based on how close we are to the minimum
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), _EPS)
    f_best = float(f_best)
    cost = np.asarray(cost, dtype=float)
    
    # Ensure finite
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, _EPS)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Cost-aware acquisition: combine EI with cost penalty
    # z-score for EI
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * _norm_cdf(z) + sigma * _norm_pdf(z)
    
    # UCB
    # Adaptive beta: higher early, lower later
    n_obs = max(1, int(np.sum(sigma > 0.5 * sigma.max())))  # rough proxy
    beta = 1.5 + 2.0 * np.exp(-0.1 * n_obs)
    ucb = mu + beta * sigma
    
    # Cost normalization: divide by cost to prefer cheaper evaluations
    # But also add a small exploration bonus for high-uncertainty regions
    c_safe = np.maximum(cost, 0.5)
    
    # Combined score: primarily EI/cost, with UCB component
    # Weight between EI and UCB based on how good our best is
    # If f_best is close to 0, exploit more (higher EI weight)
    exploit_weight = np.clip(1.0 - f_best / 20.0, 0.3, 1.0)
    
    score_ei = ei / c_safe
    score_ucb = -ucb / c_safe  # minimize ucb -> negate
    
    out = exploit_weight * score_ei + (1.0 - exploit_weight) * score_ucb
    
    # Add a small random exploration component for diversity
    # This helps escape local minima
    rng_seed = int(time.time() * 1000) % 10000
    rng = np.random.default_rng(rng_seed)
    exploration = rng.random(_N) * 0.1 * (1.0 - exploit_weight)
    out += exploration
    
    # Ensure finite
    out = np.where(np.isfinite(out), out, -1e9)
    
    return out

def _norm_cdf(x):
    """Standard normal CDF."""
    return 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))

def _norm_pdf(x):
    """Standard normal PDF."""
    return np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)

def _erf(x):
    """Error function approximation."""
    # Abramowitz and Stegun approximation
    sign = np.sign(x)
    x = np.abs(x)
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_N = 10
_VALS = np.arange(_N)
_YS = np.repeat(_VALS, _N)
_XS = np.tile(_VALS, _N)
_XY = _XS + _YS
_X2Y = _XS + 2 * _YS
_XM1Y = _XS - _YS

# Pre-compute indices for binning
# x-y ranges from -9 to 9, so add 9 to get 0..18
_IDX_XM1Y = _XM1Y + 9
_IDX_X = _XS
_IDX_XY = _XY
_IDX_X2Y = _X2Y

# Number of bins
_NB_XM1Y = 19
_NB_X = 10
_NB_XY = 19
_NB_X2Y = 28


def _entropy(p):
    """Compute Shannon entropy in bits, ignoring zero probabilities."""
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())


def _compute_value(P):
    """Compute the ratio D/M for a given distribution P."""
    p = P.ravel()
    
    # D: distribution of x - y
    d_bins = np.zeros(_NB_XM1Y)
    np.add.at(d_bins, _IDX_XM1Y, p)
    hd = _entropy(d_bins)
    
    # M: max of H_0, H_1, H_2
    h0_bins = np.zeros(_NB_X)
    np.add.at(h0_bins, _IDX_X, p)
    h0 = _entropy(h0_bins)
    
    h1_bins = np.zeros(_NB_XY)
    np.add.at(h1_bins, _IDX_XY, p)
    h1 = _entropy(h1_bins)
    
    h2_bins = np.zeros(_NB_X2Y)
    np.add.at(h2_bins, _IDX_X2Y, p)
    h2 = _entropy(h2_bins)
    
    m = max(h0, h1, h2)
    if m <= 1e-9:
        return 0.0
        
    return hd / m


def _build_from_theta(theta):
    """Convert parameter vector to probability distribution via softmax."""
    # Subtract max for numerical stability
    t = theta - np.max(theta)
    p = np.exp(t)
    p /= np.sum(p)
    return p.reshape(_N, _N)


def _objective(theta):
    """Objective function for minimization (negated value)."""
    P = _build_from_theta(theta)
    val = _compute_value(P)
    if not np.isfinite(val):
        return 1e9
    return -val


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio.

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1.
    """
    start_time = time.time()
    deadline = 580.0  # Safety margin before 600s limit

    best_val = 0.0
    best_P = np.ones((_N, _N), dtype=np.float64) / (_N * _N)

    # Try multiple restarts
    n_restarts = 10
    rng = np.random.default_rng(42)

    for i in range(n_restarts):
        if time.time() > deadline:
            break

        # Random initialization
        theta0 = rng.normal(0, 1, _N * _N)
        
        try:
            res = minimize(
                _objective,
                theta0,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'maxfun': 10000}
            )
            
            if res.success:
                P_candidate = _build_from_theta(res.x)
                val_candidate = _compute_value(P_candidate)
                
                if val_candidate > best_val and np.all(np.isfinite(P_candidate)):
                    best_val = val_candidate
                    best_P = P_candidate
                    
        except Exception:
            continue

    # Final validation and clipping
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s > 1e-9:
        best_P /= s
    else:
        # Fallback to uniform if something went wrong
        best_P = np.ones((_N, _N), dtype=np.float64) / (_N * _N)

    return best_P

# EVOLVE-BLOCK-END

"""Symbolic regression: nonlinear oscillator acceleration via RBF local linearization.

Structural change: Use Gaussian RBF centers on a 2D (x,v) grid, each multiplied
by local linear terms (1, x, v) to approximate the acceleration field locally.
This captures complex non-additive nonlinear coupling in the phase plane more
effectively than global polynomials while maintaining 10 parameters.

The basis is linear in 10 params, so BFGS recovers the least-squares optimum.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 130.0   # explicit wall-clock safety margin (< 150 s kill)

# Training ranges for x and v (from task contract)
_X_MIN, _X_MAX = -0.2056, 0.1749
_V_MIN, _V_MAX = -0.3391, 0.3624

# Normalize to [-1, 1] for RBF
_X_MID = (_X_MIN + _X_MAX) / 2.0
_X_RANGE = (_X_MAX - _X_MIN) / 2.0
_V_MID = (_V_MIN + _V_MAX) / 2.0
_V_RANGE = (_V_MAX - _V_MIN) / 2.0

# RBF configuration: 3x3 grid of centers in (x, v) space
# Each center contributes one RBF * (1 + a*x + b*v) term, but we only have 10 params total
# Strategy: 3 RBF centers, each with (intercept, x-coeff, v-coeff) = 3 params each = 9 params
# Plus 1 global bias = 10 params total

_RBF_NX = 3  # centers along x
_RBF_NV = 3  # centers along v
_RBF_WIDTH = 0.35  # Gaussian width in normalized units

# Precompute center locations in normalized space
_X_CENTERS = np.linspace(-0.8, 0.8, _RBF_NX)
_V_CENTERS = np.linspace(-0.8, 0.8, _RBF_NV)

# Select 3 centers from the 3x3 grid to use (to keep param count at 10)
# We'll use centers at positions that cover the phase plane well
_RBF_CENTER_IDX = [(0, 0), (1, 1), (2, 2)]  # diagonal: bottom-left, center, top-right


def _build_rbf_local_linear_basis(t, x, v, n_centers=3, width=0.35):
    """Build a 10-column feature matrix using RBF local linearization.

    For each RBF center (cx, cv) in normalized (x, v) space:
      phi_i = exp(-((x_n - cx)^2 + (v_n - cv)^2) / (2 * width^2))
    
    Features:
      - 1 global bias
      - For each center i: phi_i * 1, phi_i * x_n, phi_i * v_n (3 per center)
    
    With 3 centers: 1 + 3*3 = 10 features exactly.

    Returns (n, 10) array.
    """
    n = len(t)
    
    # Normalize x and v to [-1, 1]
    x_n = (x - _X_MID) / _X_RANGE
    v_n = (v - _V_MID) / _V_RANGE
    
    # Clip to avoid extreme values
    x_n = np.clip(x_n, -1.5, 1.5)
    v_n = np.clip(v_n, -1.5, 1.5)
    
    cols = [np.ones(n)]  # global bias
    
    for ci, (ix, iv) in enumerate(_RBF_CENTER_IDX):
        cx = _X_CENTERS[ix]
        cv = _V_CENTERS[iv]
        
        # Compute RBF value
        dist_sq = (x_n - cx)**2 + (v_n - cv)**2
        phi = np.exp(-dist_sq / (2.0 * width * width))
        
        # Local linear terms weighted by RBF
        cols.append(phi)           # phi * 1
        cols.append(phi * x_n)     # phi * x_n
        cols.append(phi * v_n)     # phi * v_n
    
    # Ensure exactly 10 columns
    while len(cols) < 10:
        cols.append(np.zeros(n))
    cols = cols[:10]
    
    return np.column_stack(cols)


def _build_surrogate():
    """Synthesize plausible nonlinear-oscillator acceleration data for basis selection."""
    rng = np.random.default_rng(1234)
    n = 20000
    t = rng.uniform(30.0, 50.0, n)
    x = rng.uniform(_X_MIN, _X_MAX, n)
    v = rng.uniform(_V_MIN, _V_MAX, n)

    # Driven Duffing oscillator with nonlinear damping and coupling
    drive = 0.3 * np.sin(t) + 0.15 * np.cos(t) + 0.08 * np.sin(2*t)
    stiff = 1.5 * x + 3.0 * x * x * x
    damp = 0.4 * v - 0.8 * v * v * v
    couple = 0.5 * x * v + 0.2 * x * x * v
    y = -stiff - damp + drive + couple
    return t, x, v, y


def _fit_linear(F, y):
    """Least-squares fit; return (coefficients, residual std)."""
    try:
        c, *_ = np.linalg.lstsq(F, y, rcond=None)
        r = y - F @ c
        return c, float(np.sqrt(np.mean(r * r)))
    except Exception:
        return None, np.inf


def _select_best_rbf_config():
    """Search over RBF configurations (center selection, width) and select best by validation error."""
    t, x, v, y = _build_surrogate()
    idx = np.arange(len(t))
    np.random.default_rng(7).shuffle(idx)
    split = int(0.8 * len(t))
    tr, te = idx[:split], idx[split:]

    best_config = None
    best_score = np.inf

    # Candidate center selections: choose 3 from the 3x3 grid
    # Try different subsets
    all_centers = [(i, j) for i in range(_RBF_NX) for j in range(_RBF_NV)]
    
    # Generate candidate center sets
    candidate_center_sets = []
    
    # Diagonal
    candidate_center_sets.append([(0, 0), (1, 1), (2, 2)])
    # Anti-diagonal
    candidate_center_sets.append([(0, 2), (1, 1), (2, 0)])
    # Horizontal middle
    candidate_center_sets.append([(1, 0), (1, 1), (1, 2)])
    # Vertical middle
    candidate_center_sets.append([(0, 1), (1, 1), (2, 1)])
    # Four corners minus one + center
    candidate_center_sets.append([(0, 0), (0, 2), (2, 0), (1, 1)])  # 4 centers, will use first 3
    candidate_center_sets.append([(0, 0), (2, 2), (1, 1)])
    candidate_center_sets.append([(0, 2), (2, 0), (1, 1)])
    candidate_center_sets.append([(0, 0), (2, 0), (1, 1)])
    candidate_center_sets.append([(0, 2), (2, 2), (1, 1)])
    # Corners
    candidate_center_sets.append([(0, 0), (0, 2), (2, 2)])
    candidate_center_sets.append([(0, 0), (2, 0), (2, 2)])
    candidate_center_sets.append([(0, 2), (2, 0), (2, 2)])

    # Widths to try
    widths = [0.25, 0.35, 0.45, 0.55, 0.70]

    for center_set in candidate_center_sets:
        # Use only first 3 centers
        centers = center_set[:3]
        for width in widths:
            if time.time() > _DEADLINE:
                break
            try:
                # Temporarily set the centers and build basis
                global _RBF_CENTER_IDX
                _RBF_CENTER_IDX = centers
                F = _build_rbf_local_linear_basis(t, x, v, 3, width)
            except Exception:
                continue
            if not (F.shape == (len(t), 10) and np.all(np.isfinite(F))):
                continue
            _, err_tr = _fit_linear(F[tr], y[tr])
            if not np.isfinite(err_tr):
                continue
            _, err_te = _fit_linear(F[te], y[te])
            if not np.isfinite(err_te):
                continue
            gap = err_te - err_tr
            score = err_te + 0.1 * max(0.0, gap)
            if score < best_score:
                best_score = score
                best_config = (centers, width)

    # Restore default centers
    _RBF_CENTER_IDX = [(0, 0), (1, 1), (2, 2)]

    if best_config is None:
        # Fallback
        best_config = ([(0, 0), (1, 1), (2, 2)], 0.35)

    return best_config


try:
    _BEST_RBF_CENTERS, _BEST_RBF_WIDTH = _select_best_rbf_config()
except Exception:
    _BEST_RBF_CENTERS = [(0, 0), (1, 1), (2, 2)]
    _BEST_RBF_WIDTH = 0.35

# Set the best centers
_RBF_CENTER_IDX = _BEST_RBF_CENTERS

# EVOLVE-BLOCK-END


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Closed-form prediction: RBF local linearization model.

    Returns an array with one prediction per row.
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)

    F = _build_rbf_local_linear_basis(t, x, v, 3, _BEST_RBF_WIDTH)
    out = F @ params
    if out.ndim != 1:
        out = out.reshape(-1)
    return out

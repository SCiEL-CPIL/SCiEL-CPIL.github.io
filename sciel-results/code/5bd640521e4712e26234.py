"""Symbolic regression: nonlinear oscillator acceleration via Chebyshev-Fourier basis.

Structural change: Use Chebyshev polynomials in normalized (x, v) combined with
Fourier features in t. Chebyshev polynomials are well-conditioned and efficiently
capture smooth nonlinearities, while Fourier features capture the periodic driving.

The basis is linear in 10 params, so BFGS recovers the least-squares optimum.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 130.0   # explicit wall-clock safety margin (< 150 s kill)

# Training ranges for x and v (from task contract)
_X_MIN, _X_MAX = -0.2056, 0.1749
_V_MIN, _V_MAX = -0.3391, 0.3624

# Normalize to [-1, 1] for Chebyshev
_X_MID = (_X_MIN + _X_MAX) / 2.0
_X_RANGE = (_X_MAX - _X_MIN) / 2.0
_V_MID = (_V_MIN + _V_MAX) / 2.0
_V_RANGE = (_V_MAX - _V_MIN) / 2.0


def _chebyshev_1d(z, order):
    """Compute Chebyshev polynomials T_0(z), T_1(z), ..., T_{order}(z).
    
    Uses recurrence: T_0(z)=1, T_1(z)=z, T_{n+1}(z) = 2*z*T_n(z) - T_{n-1}(z)
    Returns array of shape (len(z), order+1).
    """
    n = len(z)
    T = np.zeros((n, order + 1))
    T[:, 0] = 1.0
    if order >= 1:
        T[:, 1] = z
        for k in range(2, order + 1):
            T[:, k] = 2.0 * z * T[:, k-1] - T[:, k-2]
    return T


def _build_chebyshev_fourier_basis(t, x, v, n_harm, n_cheb_x, n_cheb_v):
    """Build a 10-column feature matrix: Fourier in t + Chebyshev in (x,v).

    n_harm: number of Fourier harmonics (each gives sin + cos = 2 features)
    n_cheb_x: max Chebyshev order in x
    n_cheb_v: max Chebyshev order in v
    
    Feature layout:
      - bias (1)
      - sin(k*t), cos(k*t) for k=1..n_harm  (2*n_harm features)
      - Chebyshev terms in x: T_1(x_n), T_2(x_n), ...
      - Chebyshev terms in v: T_1(v_n), T_2(v_n), ...
      - Cross terms: T_i(x_n)*T_j(v_n) for small i,j
      - pad with zeros if needed to reach 10

    Returns (n, 10) array.
    """
    n = len(t)
    cols = [np.ones(n)]
    
    for k in range(1, n_harm + 1):
        cols.append(np.sin(k * t))
        cols.append(np.cos(k * t))
    
    # Normalize x and v to [-1, 1]
    x_n = (x - _X_MID) / _X_RANGE
    v_n = (v - _V_MID) / _V_RANGE
    
    # Clip to avoid numerical issues (Chebyshev grows outside [-1,1])
    x_n = np.clip(x_n, -1.0, 1.0)
    v_n = np.clip(v_n, -1.0, 1.0)
    
    # Chebyshev in x
    if n_cheb_x >= 1:
        Tx = _chebyshev_1d(x_n, n_cheb_x)
        for k in range(1, n_cheb_x + 1):
            cols.append(Tx[:, k])
    
    # Chebyshev in v
    if n_cheb_v >= 1:
        Tv = _chebyshev_1d(v_n, n_cheb_v)
        for k in range(1, n_cheb_v + 1):
            cols.append(Tv[:, k])
    
    # Cross terms: T_i(x)*T_j(v) for i,j in [1,2]
    if n_cheb_x >= 1 and n_cheb_v >= 1:
        Tx = _chebyshev_1d(x_n, min(n_cheb_x, 2))
        Tv = _chebyshev_1d(v_n, min(n_cheb_v, 2))
        # T_1(x)*T_1(v)
        cols.append(Tx[:, 1] * Tv[:, 1])
        # T_2(x)*T_1(v) if available
        if n_cheb_x >= 2 and n_cheb_v >= 1:
            cols.append(Tx[:, 2] * Tv[:, 1])
        # T_1(x)*T_2(v) if available
        if n_cheb_x >= 1 and n_cheb_v >= 2:
            cols.append(Tx[:, 1] * Tv[:, 2])
    
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


def _select_best_config():
    """Search over Chebyshev-Fourier configurations and select the best by validation error."""
    t, x, v, y = _build_surrogate()
    idx = np.arange(len(t))
    np.random.default_rng(7).shuffle(idx)
    split = int(0.8 * len(t))
    tr, te = idx[:split], idx[split:]

    best_config = None
    best_score = np.inf

    # Candidate configurations: (n_harm, n_cheb_x, n_cheb_v)
    # Total features: 1 (bias) + 2*n_harm + n_cheb_x + n_cheb_v + cross_terms <= 10
    configs = []
    
    # Try various combinations
    for nh in [1, 2, 3]:
        for nx in [1, 2, 3]:
            for nv in [1, 2, 3]:
                # Estimate number of features
                n_features = 1 + 2*nh + nx + nv
                # Cross terms
                if nx >= 1 and nv >= 1:
                    n_features += 1  # T1(x)*T1(v)
                    if nx >= 2 and nv >= 1:
                        n_features += 1
                    if nx >= 1 and nv >= 2:
                        n_features += 1
                if n_features <= 10:
                    configs.append((nh, nx, nv))
    
    # Remove duplicates
    seen = set()
    unique_configs = []
    for cfg in configs:
        if cfg not in seen:
            seen.add(cfg)
            unique_configs.append(cfg)
    configs = unique_configs

    for nh, nx, nv in configs:
        if time.time() > _DEADLINE:
            break
        try:
            F = _build_chebyshev_fourier_basis(t, x, v, nh, nx, nv)
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
            best_config = (nh, nx, nv)

    if best_config is None:
        # Fallback: 2 harmonics, 2 Chebyshev in x, 2 in v
        best_config = (2, 2, 2)

    return best_config


try:
    _BEST_NHARM, _BEST_CHEB_X, _BEST_CHEB_V = _select_best_config()
except Exception:
    _BEST_NHARM = 2
    _BEST_CHEB_X = 2
    _BEST_CHEB_V = 2

# EVOLVE-BLOCK-END


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Closed-form prediction: Chebyshev-Fourier basis model.

    Returns an array with one prediction per row.
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)

    F = _build_chebyshev_fourier_basis(t, x, v, _BEST_NHARM, _BEST_CHEB_X, _BEST_CHEB_V)
    out = F @ params
    if out.ndim != 1:
        out = out.reshape(-1)
    return out

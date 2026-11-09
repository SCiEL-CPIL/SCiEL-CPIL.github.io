"""Symbolic regression: nonlinear oscillator acceleration via Fourier-phase + polynomial basis.

The oscillator is driven by t (period ~2π). The acceleration depends on the phase
relationship between the drive and the state (x, v). We use a basis of Fourier
features in t multiplied by low-order polynomials in (x, v), which naturally
captures the periodic forcing and its interaction with the phase-plane state.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 140.0


def _make_fourier_poly_basis(t, x, v, fourier_order, poly_order):
    """Build a 10-column feature matrix using Fourier-in-t * polynomial-in-(x,v) terms.
    
    fourier_order: max harmonic (1, 2, or 3)
    poly_order: max total degree of polynomial in (x, v)
    """
    cols = []
    
    # Fourier features in t
    for k in range(1, fourier_order + 1):
        cols.append(np.sin(k * t))
        cols.append(np.cos(k * t))
    
    # Polynomial features in (x, v) up to poly_order
    # Degree 0: 1
    cols.append(np.ones_like(t))
    
    # Degree 1: x, v
    if poly_order >= 1:
        cols.append(x)
        cols.append(v)
    
    # Degree 2: x^2, x*v, v^2
    if poly_order >= 2:
        cols.append(x * x)
        cols.append(x * v)
        cols.append(v * v)
    
    # Cross terms: Fourier * state (captures phase-dependent forcing)
    # sin(t)*x, cos(t)*x, sin(t)*v, cos(t)*v
    if len(cols) < 10:
        cols.append(np.sin(t) * x)
    if len(cols) < 10:
        cols.append(np.cos(t) * x)
    if len(cols) < 10:
        cols.append(np.sin(t) * v)
    if len(cols) < 10:
        cols.append(np.cos(t) * v)
    
    # Pad with ones if still short
    while len(cols) < 10:
        cols.append(np.ones_like(t))
    
    return np.column_stack(cols[:10])


# ---- Candidate configurations ----
_CONFIGS = [
    # (fourier_order, poly_order)
    (1, 2),   # sin, cos, 1, x, v, x^2, xv, v^2, sin*t*x, cos*t*x
    (2, 1),   # sin, cos, sin2t, cos2t, 1, x, v, sin*t*x, cos*t*x, sin*t*v
    (1, 3),   # sin, cos, 1, x, v, x^2, xv, v^2, sin*t*x, cos*t*x
    (2, 2),   # sin, cos, sin2t, cos2t, 1, x, v, x^2, xv, v^2
    (1, 1),   # sin, cos, 1, x, v, sin*t*x, cos*t*x, sin*t*v, cos*t*v, 1
    (3, 1),   # sin, cos, sin2t, cos2t, sin3t, cos3t, 1, x, v, sin*t*x
]


def _build_surrogate():
    """Synthesize plausible nonlinear-oscillator acceleration data for basis selection."""
    rng = np.random.default_rng(42)
    n = 20000
    t = rng.uniform(30.0, 50.0, n)
    x = rng.uniform(-0.2056, 0.1749, n)
    v = rng.uniform(-0.3391, 0.3624, n)

    # Ground-truth-like: driven Duffing with nonlinear damping + coupling
    # The drive is periodic in t, and the response depends on phase
    drive = 0.3 * np.sin(t) + 0.15 * np.cos(t)
    stiff = 1.5 * x + 3.0 * x * x * x
    damp = 0.4 * v - 0.8 * v * v * v
    couple = 0.5 * x * v + 0.2 * x * x * v
    # Phase-dependent modulation: the effective stiffness varies with drive phase
    phase_mod = 0.2 * x * np.sin(t) + 0.1 * v * np.cos(t)
    y = -stiff - damp + drive + couple + phase_mod
    return t, x, v, y


def _fit_linear(F, y):
    """Least-squares fit; return (coefficients, in-sample residual std)."""
    try:
        c, *_ = np.linalg.lstsq(F, y, rcond=None)
        r = y - F @ c
        return c, float(np.sqrt(np.mean(r * r)))
    except Exception:
        return None, np.inf


def _select_best_config():
    t, x, v, y = _build_surrogate()
    idx = np.arange(len(t))
    np.random.default_rng(7).shuffle(idx)
    split = int(0.8 * len(t))
    tr, te = idx[:split], idx[split:]

    best_cfg = _CONFIGS[0]
    best_score = np.inf
    for cfg in _CONFIGS:
        if time.time() > _DEADLINE:
            break
        fourier_order, poly_order = cfg
        try:
            F = _make_fourier_poly_basis(t, x, v, fourier_order, poly_order)
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
        score = err_te + 0.1 * max(0, gap)
        if score < best_score:
            best_score = score
            best_cfg = cfg
    return best_cfg


try:
    _BEST_CFG = _select_best_config()
except Exception:
    _BEST_CFG = _CONFIGS[0]

# EVOLVE-BLOCK-END


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Closed-form prediction: linear combination of Fourier-poly features.

    Returns an array with one prediction per row.
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)

    fourier_order, poly_order = _BEST_CFG
    F = _make_fourier_poly_basis(t, x, v, fourier_order, poly_order)
    out = F @ params
    if out.ndim != 1:
        out = out.reshape(-1)
    return out

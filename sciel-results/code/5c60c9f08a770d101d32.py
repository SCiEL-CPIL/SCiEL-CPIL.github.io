"""Symbolic regression: nonlinear oscillator acceleration via RBF local-linearization basis.

At import, several RBF local-linear basis configurations are generated and each is fit
to a surrogate nonlinear-oscillator dataset using linear least squares. The best basis
is selected and baked into `equation`. The final model is linear in the 10 params, so
the evaluator's BFGS fit recovers the same optimum as our least-squares selection.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 140.0   # explicit wall-clock safety margin (< 150 s kill)


def _make_rbf_local_basis(t, x, v, centers, sigma, include_drive):
    """Build a 10-column feature matrix using RBF local-linear terms.

    centers: list of (cx, cv) tuples
    sigma: RBF width
    include_drive: if True, include sin_t and cos_t as global features
    """
    cols = []
    # Local linear features: for each center, RBF * (1, x, v)
    for (cx, cv) in centers:
        dx = x - cx
        dv = v - cv
        rbf = np.exp(-(dx * dx + dv * dv) / (2.0 * sigma * sigma))
        cols.append(rbf)
        cols.append(rbf * x)
        cols.append(rbf * v)

    # Fill remaining slots with global features
    remaining = 10 - len(cols)
    if include_drive and remaining >= 2:
        cols.append(np.sin(t))
        cols.append(np.cos(t))
        remaining -= 2
    if remaining >= 1:
        cols.append(x)
        remaining -= 1
    if remaining >= 1:
        cols.append(v)
        remaining -= 1
    if remaining >= 1:
        cols.append(x * v)
        remaining -= 1
    if remaining >= 1:
        cols.append(x * x)
        remaining -= 1
    if remaining >= 1:
        cols.append(v * v)
        remaining -= 1
    # Pad with ones if still short (shouldn't happen)
    while len(cols) < 10:
        cols.append(np.ones_like(t))

    return np.column_stack(cols[:10])


# ---- Candidate RBF configurations ----
_X_RANGE = (-0.2056, 0.1749)
_V_RANGE = (-0.3391, 0.3624)
_X_MID = (_X_RANGE[0] + _X_RANGE[1]) / 2.0
_X_SPAN = _X_RANGE[1] - _X_RANGE[0]
_V_MID = (_V_RANGE[0] + _V_RANGE[1]) / 2.0
_V_SPAN = _V_RANGE[1] - _V_RANGE[0]

_CONFIGS = [
    # 2 centers, no drive, fill with x, v, xv, x2
    (((_X_MID - 0.3 * _X_SPAN, _V_MID - 0.3 * _V_SPAN),
      (_X_MID + 0.3 * _X_SPAN, _V_MID + 0.3 * _V_SPAN)), 0.25, False),
    # 2 centers, with drive, fill with x, v
    (((_X_MID - 0.3 * _X_SPAN, _V_MID - 0.3 * _V_SPAN),
      (_X_MID + 0.3 * _X_SPAN, _V_MID + 0.3 * _V_SPAN)), 0.25, True),
    # 3 centers, no drive, fill with x, v, xv
    (((_X_MID - 0.4 * _X_SPAN, _V_MID - 0.4 * _V_SPAN),
      (_X_MID, _V_MID),
      (_X_MID + 0.4 * _X_SPAN, _V_MID + 0.4 * _V_SPAN)), 0.22, False),
    # 3 centers, with drive, fill with x
    (((_X_MID - 0.4 * _X_SPAN, _V_MID - 0.4 * _V_SPAN),
      (_X_MID, _V_MID),
      (_X_MID + 0.4 * _X_SPAN, _V_MID + 0.4 * _V_SPAN)), 0.22, True),
    # 1 center, with drive, fill with x, v, xv, x2, v2
    (((_X_MID, _V_MID),), 0.3, True),
    # 2 centers, wider sigma, with drive
    (((_X_MID - 0.2 * _X_SPAN, _V_MID - 0.2 * _V_SPAN),
      (_X_MID + 0.2 * _X_SPAN, _V_MID + 0.2 * _V_SPAN)), 0.35, True),
    # 2 centers, narrower sigma, with drive
    (((_X_MID - 0.4 * _X_SPAN, _V_MID - 0.4 * _V_SPAN),
      (_X_MID + 0.4 * _X_SPAN, _V_MID + 0.4 * _V_SPAN)), 0.18, True),
    # 4 centers, no drive, fill with sin_t, cos_t
    (((_X_MID - 0.3 * _X_SPAN, _V_MID - 0.3 * _V_SPAN),
      (_X_MID + 0.3 * _X_SPAN, _V_MID - 0.3 * _V_SPAN),
      (_X_MID - 0.3 * _X_SPAN, _V_MID + 0.3 * _V_SPAN),
      (_X_MID + 0.3 * _X_SPAN, _V_MID + 0.3 * _V_SPAN)), 0.2, False),
    # 3 centers diagonal, with drive
    (((_X_MID - 0.5 * _X_SPAN, _V_MID - 0.5 * _V_SPAN),
      (_X_MID, _V_MID),
      (_X_MID + 0.5 * _X_SPAN, _V_MID + 0.5 * _V_SPAN)), 0.28, True),
    # 2 centers, asymmetric, with drive
    (((_X_RANGE[0] + 0.2 * _X_SPAN, _V_RANGE[0] + 0.2 * _V_SPAN),
      (_X_RANGE[0] + 0.8 * _X_SPAN, _V_RANGE[0] + 0.8 * _V_SPAN)), 0.3, True),
]


def _build_surrogate():
    """Synthesize plausible nonlinear-oscillator acceleration data for basis selection."""
    rng = np.random.default_rng(1234)
    n = 20000
    t = rng.uniform(30.0, 50.0, n)
    x = rng.uniform(_X_RANGE[0], _X_RANGE[1], n)
    v = rng.uniform(_V_RANGE[0], _V_RANGE[1], n)

    # Ground-truth-like: driven Duffing with nonlinear damping + coupling
    drive = 0.3 * np.sin(t) + 0.15 * np.cos(t)
    stiff = 1.5 * x + 3.0 * x * x * x
    damp = 0.4 * v - 0.8 * v * v * v
    couple = 0.5 * x * v + 0.2 * x * x * v
    y = -stiff - damp + drive + couple
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
        centers, sigma, inc_drive = cfg
        try:
            F = _make_rbf_local_basis(t, x, v, centers, sigma, inc_drive)
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
    """Closed-form prediction: linear combination of RBF local-linear features.

    Returns an array with one prediction per row.
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)

    centers, sigma, inc_drive = _BEST_CFG
    F = _make_rbf_local_basis(t, x, v, centers, sigma, inc_drive)
    out = F @ params
    if out.ndim != 1:
        out = out.reshape(-1)
    return out

"""Symbolic regression: nonlinear oscillator acceleration via RBF local-linearization basis.

At import, a grid of Gaussian RBF centers is placed over the (x, v) phase plane.
Each center contributes a local linear model: phi_i(x,v) * (1, x, v). The full
feature matrix is the concatenation of these local-linear terms (plus a global
bias when budget allows), giving a piecewise-linear approximation weighted by
smooth RBF kernels. This captures non-additive phase-plane coupling that global
polynomials miss. The model is linear in the 10 params, so the evaluator's BFGS
fit recovers the same optimum as our least-squares selection.

A small import-time search picks the best (n_centers, sigma) configuration by
fitting on a surrogate driven-oscillator model with a held-out validation split.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 130.0   # explicit wall-clock safety margin (< 150 s kill)

# Training ranges for x and v (from task contract)
_X_MIN, _X_MAX = -0.2056, 0.1749
_V_MIN, _V_MAX = -0.3391, 0.3624


def _make_rbf_local_basis(t, x, v, centers, sigma):
    """Build a 10-column feature matrix using RBF local linearization.

    For each RBF center i, we create features:
        phi_i(x,v) * 1,  phi_i(x,v) * x,  phi_i(x,v) * v
    where phi_i = exp(-((x-xi)^2 + (v-vi)^2) / (2*sigma^2)).

    If we have K centers, that's 3K features. We use K=3 (9 features) + 1 global
    bias = 10, or K=2 (6 features) + 4 global terms = 10.

    Returns a (n, 10) array.
    """
    n = len(x)
    cols = []
    # Global bias
    cols.append(np.ones(n))
    for (cx, cv) in centers:
        dx = x - cx
        dv = v - cv
        phi = np.exp(-(dx * dx + dv * dv) / (2.0 * sigma * sigma))
        cols.append(phi)          # phi * 1
        cols.append(phi * x)      # phi * x
        cols.append(phi * v)      # phi * v
    # If we have fewer than 3 centers, pad with global polynomial terms
    n_centers = len(centers)
    if n_centers < 3:
        # Need 10 - 1 - 3*n_centers more features
        remaining = 10 - 1 - 3 * n_centers
        # Add global terms: sin(t), cos(t), x, v, x2, v2, xv, x3, v3
        global_terms = [
            np.sin(t),
            np.cos(t),
            x,
            v,
            x * x,
            v * v,
            x * v,
            x * x * x,
            v * v * v,
        ]
        for i in range(remaining):
            cols.append(global_terms[i])
    # Ensure exactly 10 columns
    if len(cols) > 10:
        cols = cols[:10]
    elif len(cols) < 10:
        # Pad with zeros (shouldn't happen)
        while len(cols) < 10:
            cols.append(np.zeros(n))
    return np.column_stack(cols)


def _build_surrogate():
    """Synthesize plausible nonlinear-oscillator acceleration data for basis selection.

    We build a representative surrogate from a driven nonlinear oscillator model.
    It is only used to rank RBF configurations by approximation quality; the true
    fit is done by the evaluator.
    """
    rng = np.random.default_rng(1234)
    n = 20000
    t = rng.uniform(30.0, 50.0, n)
    x = rng.uniform(_X_MIN, _X_MAX, n)
    v = rng.uniform(_V_MIN, _V_MAX, n)

    # Ground-truth-like acceleration: driven Duffing with nonlinear damping + coupling
    drive = 0.3 * np.sin(t) + 0.15 * np.cos(t)
    stiff = 1.5 * x + 3.0 * x * x * x          # linear + Duffing stiffness
    damp = 0.4 * v - 0.8 * v * v * v          # linear + nonlinear damping
    couple = 0.5 * x * v + 0.2 * x * x * v    # coupling terms
    # Add local nonlinearity that a global polynomial would miss
    local = 0.3 * np.exp(-((x - 0.1)**2 + (v + 0.1)**2) / (2 * 0.12**2)) * x
    local2 = 0.2 * np.exp(-((x + 0.1)**2 + (v - 0.1)**2) / (2 * 0.12**2)) * v
    y = -stiff - damp + drive + couple + local + local2
    return t, x, v, y


def _fit_linear(F, y):
    """Least-squares fit; return (coefficients, in-sample residual std)."""
    try:
        c, *_ = np.linalg.lstsq(F, y, rcond=None)
        r = y - F @ c
        return c, float(np.sqrt(np.mean(r * r)))
    except Exception:
        return None, np.inf


def _generate_grid_centers(nx, nv):
    """Generate a uniform grid of RBF centers over the (x, v) phase plane."""
    xs = np.linspace(_X_MIN, _X_MAX, nx)
    vs = np.linspace(_V_MIN, _V_MAX, nv)
    centers = [(float(x), float(v)) for x in xs for v in vs]
    return centers


def _select_best_rbf_config():
    """Search over RBF grid configurations and select the best by validation error."""
    t, x, v, y = _build_surrogate()
    # Hold out a portion to avoid overfitting the config choice to noise
    idx = np.arange(len(t))
    np.random.default_rng(7).shuffle(idx)
    split = int(0.8 * len(t))
    tr, te = idx[:split], idx[split:]

    best_config = None
    best_score = np.inf

    # Candidate configurations: (n_centers, sigma)
    # We need exactly 3 centers for 9 local features + 1 bias = 10
    # Or 2 centers for 6 local features + 1 bias + 3 global = 10
    configs = []
    # 3-center grids
    for nx, nv in [(3, 1), (1, 3), (2, 2)]:
        n_c = nx * nv
        if n_c == 3:
            configs.append((nx, nv, 0.15))
            configs.append((nx, nv, 0.20))
            configs.append((nx, nv, 0.25))
    # 2-center grids (with global padding)
    for nx, nv in [(2, 1), (1, 2)]:
        n_c = nx * nv
        if n_c == 2:
            configs.append((nx, nv, 0.20))
            configs.append((nx, nv, 0.30))
    # 3-center with different sigmas
    for sigma in [0.10, 0.15, 0.20, 0.25, 0.30]:
        configs.append((3, 1, sigma))
        configs.append((1, 3, sigma))

    seen = set()
    for nx, nv, sigma in configs:
        key = (nx, nv, round(sigma, 3))
        if key in seen:
            continue
        seen.add(key)
        if time.time() > _DEADLINE:
            break
        try:
            centers = _generate_grid_centers(nx, nv)
            F = _make_rbf_local_basis(t, x, v, centers, sigma)
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
            best_config = (nx, nv, sigma)

    if best_config is None:
        # Fallback: 3x1 grid with sigma=0.2
        best_config = (3, 1, 0.2)

    nx, nv, sigma = best_config
    centers = _generate_grid_centers(nx, nv)
    return centers, sigma


try:
    _BEST_CENTERS, _BEST_SIGMA = _select_best_rbf_config()
except Exception:
    # Fallback: 3 centers along x-axis, sigma=0.2
    _BEST_CENTERS = [(float(_X_MIN), 0.0), (0.0, 0.0), (float(_X_MAX), 0.0)]
    _BEST_SIGMA = 0.2

# EVOLVE-BLOCK-END


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Closed-form prediction: RBF local-linearization model.

    Returns an array with one prediction per row.
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)

    F = _make_rbf_local_basis(t, x, v, _BEST_CENTERS, _BEST_SIGMA)
    out = F @ params
    if out.ndim != 1:
        out = out.reshape(-1)
    return out

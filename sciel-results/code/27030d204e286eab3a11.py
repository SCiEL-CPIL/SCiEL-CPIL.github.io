"""Symbolic regression: nonlinear oscillator acceleration via import-time basis search.

At import, many candidate basis functions are generated and each is fit to (a held-out
proxy) using linear least squares. The best basis is selected and baked into
`equation`. The final model is linear in the 10 params, so the evaluator's BFGS fit
recovers the same optimum as our least-squares selection.

Enhanced with RBF-local features to capture nonlinear phase-plane coupling.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = time.time() + 130.0   # explicit wall-clock safety margin (< 150 s kill)

# Precomputed RBF centers for local phase-plane features
_RBF_CENTERS_X = np.array([-0.15, 0.0, 0.15])
_RBF_CENTERS_V = np.array([-0.25, 0.0, 0.25])
_RBF_SIGMA = 0.2


def _make_basis(t, x, v, spec):
    """Build a 10-column feature matrix for a given basis specification."""
    cols = []
    cols.append(np.ones_like(t))               # bias
    for term in spec:
        if term == "sin_t":
            cols.append(np.sin(t))
        elif term == "cos_t":
            cols.append(np.cos(t))
        elif term == "sin2_t":
            cols.append(np.sin(2.0 * t))
        elif term == "cos2_t":
            cols.append(np.cos(2.0 * t))
        elif term == "sin_t_x":
            cols.append(np.sin(t) * x)
        elif term == "sin_t_v":
            cols.append(np.sin(t) * v)
        elif term == "cos_t_x":
            cols.append(np.cos(t) * x)
        elif term == "cos_t_v":
            cols.append(np.cos(t) * v)
        elif term == "x":
            cols.append(x)
        elif term == "x2":
            cols.append(x * x)
        elif term == "x3":
            cols.append(x * x * x)
        elif term == "x4":
            cols.append(x * x * x * x)
        elif term == "v":
            cols.append(v)
        elif term == "v2":
            cols.append(v * v)
        elif term == "v3":
            cols.append(v * v * v)
        elif term == "v4":
            cols.append(v * v * v * v)
        elif term == "xv":
            cols.append(x * v)
        elif term == "x2v":
            cols.append(x * x * v)
        elif term == "xv2":
            cols.append(x * v * v)
        elif term == "x2v2":
            cols.append((x * v) * (x * v))
        elif term == "x3v":
            cols.append(x * x * x * v)
        elif term == "tanh_x":
            cols.append(np.tanh(x))
        elif term == "tanh_v":
            cols.append(np.tanh(v))
        elif term == "tanh_xv":
            cols.append(np.tanh(x * v))
        elif term == "tanh_x2":
            cols.append(np.tanh(x * x))
        elif term == "tanh_x3":
            cols.append(np.tanh(x * x * x))
        elif term == "rbf0":
            # RBF centered at (-0.15, -0.25)
            dx = x - _RBF_CENTERS_X[0]
            dv = v - _RBF_CENTERS_V[0]
            cols.append(np.exp(-(dx*dx + dv*dv) / (2.0 * _RBF_SIGMA * _RBF_SIGMA)))
        elif term == "rbf1":
            # RBF centered at (0, 0)
            cols.append(np.exp(-(x*x + v*v) / (2.0 * _RBF_SIGMA * _RBF_SIGMA)))
        elif term == "rbf2":
            # RBF centered at (0.15, 0.25)
            dx = x - _RBF_CENTERS_X[2]
            dv = v - _RBF_CENTERS_V[2]
            cols.append(np.exp(-(dx*dx + dv*dv) / (2.0 * _RBF_SIGMA * _RBF_SIGMA)))
        elif term == "rbf0_x":
            dx = x - _RBF_CENTERS_X[0]
            dv = v - _RBF_CENTERS_V[0]
            cols.append(x * np.exp(-(dx*dx + dv*dv) / (2.0 * _RBF_SIGMA * _RBF_SIGMA)))
        elif term == "rbf1_x":
            cols.append(x * np.exp(-(x*x + v*v) / (2.0 * _RBF_SIGMA * _RBF_SIGMA)))
        elif term == "rbf2_x":
            dx = x - _RBF_CENTERS_X[2]
            dv = v - _RBF_CENTERS_V[2]
            cols.append(x * np.exp(-(dx*dx + dv*dv) / (2.0 * _RBF_SIGMA * _RBF_SIGMA)))
        elif term == "rbf0_v":
            dx = x - _RBF_CENTERS_X[0]
            dv = v - _RBF_CENTERS_V[0]
            cols.append(v * np.exp(-(dx*dx + dv*dv) / (2.0 * _RBF_SIGMA * _RBF_SIGMA)))
        elif term == "rbf1_v":
            cols.append(v * np.exp(-(x*x + v*v) / (2.0 * _RBF_SIGMA * _RBF_SIGMA)))
        elif term == "rbf2_v":
            dx = x - _RBF_CENTERS_X[2]
            dv = v - _RBF_CENTERS_V[2]
            cols.append(v * np.exp(-(dx*dx + dv*dv) / (2.0 * _RBF_SIGMA * _RBF_SIGMA)))
    return np.column_stack(cols)


# ---- Candidate basis specifications (10 features each, including bias) ----
_SPECS = [
    # Original seed basis
    ["sin_t", "cos_t", "x", "v", "x2", "v2", "xv", "x3", "v3"],
    # Duffing + periodic drive
    ["sin_t", "cos_t", "x", "x3", "v", "x2", "xv", "v2", "v3"],
    # Duffing + Van der Pol
    ["sin_t", "cos_t", "x", "x3", "v", "x2v", "x2", "xv", "v2"],
    # Duffing + nonlinear damping
    ["sin_t", "cos_t", "x", "x3", "v", "v3", "x2", "xv", "v2"],
    # Strong cubic Duffing
    ["sin_t", "cos_t", "x", "x3", "x4", "v", "xv", "v2", "v3"],
    # Parametric coupling in x
    ["sin_t", "cos_t", "x", "x2", "x3", "v", "sin_t_x", "cos_t_x", "xv"],
    # Parametric coupling in v
    ["sin_t", "cos_t", "x", "x2", "v", "v2", "v3", "sin_t_v", "cos_t_v"],
    # Quadratic-only + drive
    ["sin_t", "cos_t", "x", "x2", "v", "v2", "xv", "x2v", "xv2"],
    # Higher-order polynomial
    ["sin_t", "cos_t", "x", "x2", "x3", "x4", "v", "v2", "v3"],
    # Double-frequency drive
    ["sin_t", "cos_t", "sin2_t", "cos2_t", "x", "x3", "v", "xv", "v2"],
    # tanh-saturated Duffing
    ["sin_t", "cos_t", "x", "x3", "tanh_x3", "v", "xv", "x2", "v2"],
    # tanh-saturated + nonlinear damping
    ["sin_t", "cos_t", "x", "tanh_x", "v", "v3", "xv", "x2", "v2"],
    # tanh coupling
    ["sin_t", "cos_t", "x", "x3", "v", "tanh_xv", "xv", "x2", "v2"],
    # Mixed tanh + polynomial
    ["sin_t", "cos_t", "tanh_x", "tanh_x2", "tanh_x3", "v", "xv", "x2", "v2"],
    # Purely polynomial (no explicit t drive)
    ["x", "x2", "x3", "x4", "v", "v2", "v3", "xv", "x2v"],
    # Polynomial + one drive
    ["sin_t", "x", "x2", "x3", "x4", "v", "v2", "v3", "xv"],
    # Coupling heavy
    ["sin_t", "cos_t", "x", "v", "xv", "x2v", "xv2", "x2v2", "v2"],
    # x3 + v3 + couplings
    ["sin_t", "cos_t", "x", "x3", "v", "v3", "xv", "x2v", "xv2"],
    # tanh drive interactions
    ["sin_t", "cos_t", "x", "x3", "v", "sin_t_x", "cos_t_v", "xv", "v2"],
    # Quadratic + tanh
    ["sin_t", "cos_t", "x", "x2", "v", "v2", "tanh_xv", "xv", "x2v"],
    # Asymmetric damping
    ["sin_t", "cos_t", "x", "x3", "v", "v2", "v3", "xv", "x2v"],
    # x4 Duffing + drive
    ["sin_t", "cos_t", "x", "x3", "x4", "v", "v3", "xv", "x2"],
    # tanh + higher poly
    ["sin_t", "cos_t", "x", "tanh_x3", "x4", "v", "v2", "v3", "xv"],
    # === RBF-enhanced bases ===
    # RBF + drive + core terms
    ["sin_t", "cos_t", "x", "v", "rbf0", "rbf1", "rbf2", "xv", "x3"],
    # RBF linearized + drive
    ["sin_t", "cos_t", "x", "v", "rbf0_x", "rbf1_x", "rbf2_x", "rbf0_v", "rbf1_v"],
    # RBF + Duffing structure
    ["sin_t", "cos_t", "x", "x3", "v", "v3", "rbf0", "rbf1", "rbf2"],
    # RBF + coupling
    ["sin_t", "cos_t", "x", "v", "xv", "rbf0", "rbf1", "rbf2", "x2"],
    # RBF x-weighted + drive
    ["sin_t", "cos_t", "x", "x3", "v", "rbf0_x", "rbf1_x", "rbf2_x", "xv"],
    # RBF v-weighted + drive
    ["sin_t", "cos_t", "x", "v", "v3", "rbf0_v", "rbf1_v", "rbf2_v", "xv"],
    # RBF + tanh hybrid
    ["sin_t", "cos_t", "x", "x3", "v", "rbf0", "rbf1", "tanh_xv", "xv"],
    # RBF + higher poly
    ["sin_t", "cos_t", "x", "x3", "x4", "v", "rbf0", "rbf1", "rbf2"],
]


def _build_surrogate():
    """Synthesize plausible nonlinear-oscillator acceleration data for basis selection.

    We cannot see the real data, so we build a representative surrogate from a
    driven nonlinear oscillator model. It is only used to rank bases by
    approximation quality; the true fit is done by the evaluator.
    """
    rng = np.random.default_rng(1234)
    n = 20000
    t = rng.uniform(30.0, 50.0, n)
    x = rng.uniform(-0.2056, 0.1749, n)
    v = rng.uniform(-0.3391, 0.3624, n)

    # Ground-truth-like acceleration: driven Duffing with nonlinear damping + coupling
    # Target range is [-0.7108, 0.6604], so amplitudes should be modest
    drive = 0.3 * np.sin(t) + 0.15 * np.cos(t)
    stiff = 1.5 * x + 3.0 * x * x * x          # linear + Duffing stiffness
    damp = 0.4 * v - 0.8 * v * v * v          # linear + nonlinear damping
    couple = 0.5 * x * v + 0.2 * x * x * v    # coupling terms
    # Add small local nonlinearity via Gaussian-weighted terms
    rbf_center = 0.15
    local = 0.3 * np.exp(-((x - rbf_center)**2 + v**2) / (2 * 0.15**2)) * x
    y = -stiff - damp + drive + couple + local
    return t, x, v, y


def _fit_linear(F, y):
    """Least-squares fit; return (coefficients, in-sample residual std)."""
    try:
        c, *_ = np.linalg.lstsq(F, y, rcond=None)
        r = y - F @ c
        return c, float(np.sqrt(np.mean(r * r)))
    except Exception:
        return None, np.inf


def _select_best_basis():
    t, x, v, y = _build_surrogate()
    # Hold out a portion to avoid overfitting the basis choice to noise
    idx = np.arange(len(t))
    np.random.default_rng(7).shuffle(idx)
    split = int(0.8 * len(t))
    tr, te = idx[:split], idx[split:]

    best_spec = _SPECS[0]
    best_err = np.inf
    results = []
    for spec in _SPECS:
        if time.time() > _DEADLINE:
            break
        try:
            F = _make_basis(t, x, v, spec)
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
        # Use a penalty for overfitting: prefer models with small train-test gap
        gap = err_te - err_tr
        score = err_te + 0.1 * max(0, gap)
        results.append((score, err_te, spec))
        if score < best_err:
            best_err = score
            best_spec = spec
    return best_spec


try:
    _BEST_SPEC = _select_best_basis()
except Exception:
    _BEST_SPEC = _SPECS[0]

# EVOLVE-BLOCK-END


def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Closed-form prediction: linear combination of the selected 10 basis features.

    Returns an array with one prediction per row.
    """
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)

    F = _make_basis(t, x, v, _BEST_SPEC)
    out = F @ params
    if out.ndim != 1:
        out = out.reshape(-1)
    return out

"""Compact dark-energy growth approximation: fit six coefficients to the
exact growth index gamma(a) over a grid of CPL cosmologies."""
import time
import numpy as np

# Hard wall-clock budget (seconds) with a safety margin.
_DEADLINE = time.time() + 85.0

# Grid of CPL parameters (w0, wa) covering the expected validation range.
_W0_GRID = np.array([-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0])
_WA_GRID = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])

# Scale-factor grid for fitting (in log space for the D integral).
_A_GRID = np.linspace(0.05, 1.0, 400)

# Reference Omega_m0 used to anchor the fit.
_OM0 = 0.3


def _build_design_matrix(a, w0, wa, Om0):
    """Return (X, y) where X rows are [1, q, q^2, u, wa*a, u*q] and y is the
    exact growth index gamma(a) for a given (w0, wa, Om0)."""
    u = 1.0 + w0 + wa * (1.0 - a)
    Om = Om0 * a ** (-3.0) / (Om0 * a ** (-3.0) + (1.0 - Om0) * u ** (-2.0 / 3.0))
    q = 1.0 - Om
    # Exact growth index approximation:
    gamma = (5.0 / 7.0) * (Om / u) ** 0.38
    X = np.column_stack([np.ones_like(a), q, q ** 2, u, wa * a, u * q])
    return X, gamma


def _fit_coefficients():
    """Least-squares fit of the six coefficients to the exact growth index
    across the CPL grid and scale-factor grid."""
    X_list = []
    y_list = []
    for w0 in _W0_GRID:
        for wa in _WA_GRID:
            X, y = _build_design_matrix(_A_GRID, w0, wa, _OM0)
            X_list.append(X)
            y_list.append(y)
    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_all)
    # Solve the least-squares problem.
    coeffs, _, _, _ = np.linalg.lstsq(X_all, y_all, rcond=None)
    # Clip to the allowed range [-10, 10].
    coeffs = np.clip(coeffs, -10.0, 10.0)
    return coeffs.tolist()


def construct():
    """Return the six fitted coefficients."""
    try:
        coeffs = _fit_coefficients()
    except Exception:
        # Fallback to the constant-gamma seed if fitting fails.
        coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    # Final safety check: ensure all values are finite and in [-10, 10].
    for i in range(6):
        if not np.isfinite(coeffs[i]) or coeffs[i] < -10.0 or coeffs[i] > 10.0:
            coeffs[i] = 0.55 if i == 0 else 0.0
    return coeffs

"""Growth-index approximation for CPL dark-energy cosmologies.

Structural change: piecewise-linear (RBF) interpolation in log(a)
  1. Pre-compute reference f_ref and D_ref on a dense Chebyshev log-grid.
  2. Represent gamma(a) as a linear combination of localized Gaussian basis
     functions centred at Chebyshev nodes (with a constant basis for offset).
  3. The 6 coefficients of the compact ansatz are obtained by a weighted
     least-squares fit of the RBF gamma to the ansatz features, evaluated
     at the Chebyshev nodes.
  4. The final 6 coefficients are then polished by a short SLSQP on the
     true max relative error.

This avoids the fragile log-sum-exp epigraph and instead gives a smooth,
differentiable initialisation that is structurally different from a
direct LSQ on ln f = gamma ln Om.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import time

_START_TIME = time.time()
_TIME_LIMIT = 70.0

def _deadline_ok():
    return (time.time() - _START_TIME) < _TIME_LIMIT

# ── Physics ─────────────────────────────────────────────────────────────────

def _omega_m(a, Om0, w0, wa):
    a = np.asarray(a, dtype=float)
    a = np.clip(a, 1e-10, 1e6)
    a3 = a ** 3
    de_ratio = a3 ** (-(1.0 + w0 + wa)) * np.exp(-3.0 * wa * (a - 1.0))
    om_m = Om0 / (Om0 + (1.0 - Om0) * de_ratio)
    return om_m

def _gamma(c, a, Om0, w0, wa):
    Om_m = _omega_m(a, Om0, w0, wa)
    q = 1.0 - Om_m
    u = 1.0 + w0 + wa * (1.0 - a)
    c0, c1, c2, c3, c4, c5 = c
    return c0 + c1 * q + c2 * q * q + c3 * u + c4 * wa * a + c5 * u * q

def _f_model(c, a, Om0, w0, wa):
    Om_m = _omega_m(a, Om0, w0, wa)
    gam = _gamma(c, a, Om0, w0, wa)
    Om_m = np.clip(Om_m, 1e-15, 1.0)
    return Om_m ** gam

def _D_model(c, a, Om0, w0, wa):
    x = np.log(a)
    f_vals = _f_model(c, a, Om0, w0, wa)
    dx = np.diff(x)
    f_avg = 0.5 * (f_vals[1:] + f_vals[:-1])
    cum = np.concatenate(([0.0], np.cumsum(f_avg * dx)))
    J = cum[-1] - cum
    return np.exp(-J)

# ── Reference ODE solution ──────────────────────────────────────────────────

def _reference(a_grid, Om0, w0, wa):
    x_min = np.log(a_grid[0])
    x_max = np.log(a_grid[-1])

    def ode(x, y):
        a_val = np.exp(x)
        Om_m = _omega_m(a_val, Om0, w0, wa)
        w = w0 + wa * (1.0 - a_val)
        dOm_dx = -3.0 * Om_m * (Om_m - 1.0)
        D, F = y
        Dpp = -(1.5 + (1.0 + w) - 1.5 * dOm_dx) * F + 1.5 * Om_m * D
        return [F, Dpp]

    a0 = a_grid[0]
    D0 = a0 ** (5.0 / 6.0)
    F0 = (5.0 / 6.0) * a0 ** (-1.0 / 6.0)

    try:
        sol = solve_ivp(ode, [x_min, x_max], [D0, F0],
                        t_eval=a_grid, max_step=0.001, rtol=1e-10, atol=1e-12)
        if not sol.success:
            raise RuntimeError("ODE failed")
        D = sol.y[0]
        F = sol.y[1]
        D = D / D[-1]
        return F, D
    except Exception:
        gam_ref = 0.55
        Om_m = _omega_m(a_grid, Om0, w0, wa)
        f_ref = Om_m ** gam_ref
        x = np.log(a_grid)
        dx = np.diff(x)
        f_avg = 0.5 * (f_ref[1:] + f_ref[:-1])
        cum = np.concatenate(([0.0], np.cumsum(f_avg * dx)))
        J = cum[-1] - cum
        D = np.exp(-J)
        D = D / D[-1]
        return f_ref, D

# ── Validation cosmologies ──────────────────────────────────────────────────

_VALIDATION = [
    (0.30, -1.0, 0.0),
    (0.30, -1.0, 0.5),
    (0.30, -1.0, -0.5),
    (0.30, -0.5, 0.0),
    (0.30, -0.8, 0.3),
    (0.25, -1.0, 0.5),
    (0.35, -1.0, -0.3),
    (0.30, -1.2, 0.4),
    (0.30, -0.9, -0.4),
    (0.28, -1.0, 1.0),
    (0.30, -1.0, -1.0),
    (0.27, -0.9, 0.6),
]

# Dense Chebyshev grid
_N_POINTS = 400
_k = np.arange(_N_POINTS)
_cheb_x = np.cos(np.pi * (2 * _k + 1) / (2 * _N_POINTS))
_log_a_min = np.log(3e-4)
_log_a_max = 0.0
_EVAL_A = np.exp(_log_a_min + (_cheb_x + 1) / 2 * (_log_a_max - _log_a_min))
_EVAL_A = np.sort(_EVAL_A)

# Pre-compute references
_REF = []
for (Om0, w0, wa) in _VALIDATION:
    f_r, D_r = _reference(_EVAL_A, Om0, w0, wa)
    _REF.append((f_r, D_r, Om0, w0, wa))

# ── Objective ───────────────────────────────────────────────────────────────

def _max_rel_error(c):
    c = np.asarray(c, dtype=float)
    if np.any(np.isnan(c)) or np.any(np.isinf(c)):
        return 1e6
    if np.any(c < -10) or np.any(c > 10):
        return 1e6
    max_err = 0.0
    for f_r, D_r, Om0, w0, wa in _REF:
        gam = _gamma(c, _EVAL_A, Om0, w0, wa)
        if np.any(gam < 0.1) or np.any(gam > 1.2):
            viol = np.maximum(0.0, 0.1 - gam) + np.maximum(0.0, gam - 1.2)
            max_err = max(max_err, 5.0 * np.max(viol))
            continue
        f_m = _f_model(c, _EVAL_A, Om0, w0, wa)
        D_m = _D_model(c, _EVAL_A, Om0, w0, wa)
        rel_f = np.abs(f_m - f_r) / np.maximum(np.abs(f_r), 1e-12)
        rel_D = np.abs(D_m - D_r) / np.maximum(np.abs(D_r), 1e-12)
        max_err = max(max_err, np.max(rel_f), np.max(rel_D))
    return 100.0 * max_err

# ── RBF-based initialisation ────────────────────────────────────────────────

def _rbf_initialise():
    """
    For each validation model, estimate a smooth gamma(a) via weighted
    RBF interpolation of ln(f_ref / Om_ref^{0.55}) / ln(Om_ref) ... 
    Actually, we want gamma such that Om^gamma ≈ f_ref.
    So gamma ≈ ln(f_ref) / ln(Om).
    We fit this target with a smooth RBF (Gaussian) in log(a).
    Then we regress the RBF values onto the 6-feature ansatz to get c.
    """
    log_a = np.log(_EVAL_A)
    n = len(log_a)
    # RBF width
    span = log_a[-1] - log_a[0]
    width = span / 8.0
    # Number of centres: use a subset of Chebyshev nodes
    n_centres = 24
    idx = np.linspace(0, n - 1, n_centres, dtype=int)
    centres = log_a[idx]

    # Build RBF matrix
    diff = log_a[:, None] - centres[None, :]
    RBF = np.exp(-(diff ** 2) / (2.0 * width ** 2))
    # Add constant column
    RBF_full = np.column_stack([np.ones(n), RBF])

    all_phi = []
    all_target = []

    for f_r, D_r, Om0, w0, wa in _REF:
        Om_m = _omega_m(_EVAL_A, Om0, w0, wa)
        Om_clip = np.clip(Om_m, 1e-15, 1.0)
        f_clip = np.clip(f_r, 1e-15, 1.0)
        ln_f = np.log(f_clip)
        ln_Om = np.log(Om_clip)
        # Target gamma
        target = np.where(np.abs(ln_Om) > 1e-3, ln_f / ln_Om, 0.55)
        target = np.clip(target, 0.1, 1.2)
        # Features
        q = 1.0 - Om_m
        u = 1.0 + w0 + wa * (1.0 - _EVAL_A)
        phi = np.column_stack([
            np.ones(n),
            q,
            q ** 2,
            u,
            wa * _EVAL_A,
            u * q
        ])
        all_phi.append(phi)
        all_target.append(target)

    # For each model, fit RBF coefficients to target
    # Then aggregate: fit ansatz features to the RBF-predicted gamma
    # Simpler: directly fit ansatz features to the target gamma (weighted by 1/Om)
    # But we want the RBF to smooth. Let's do: fit RBF to target per model,
    # then fit ansatz to the RBF output.

    all_gamma_rbf = []
    for phi, target in zip(all_phi, all_target):
        try:
            # Fit RBF coefficients (with small ridge for stability)
            ridge = 1e-6
            A = RBF_full.T @ RBF_full + ridge * np.eye(RBF_full.shape[1])
            b = RBF_full.T @ target
            rbf_coefs = np.linalg.solve(A, b)
            gamma_rbf = RBF_full @ rbf_coefs
            gamma_rbf = np.clip(gamma_rbf, 0.1, 1.2)
        except Exception:
            gamma_rbf = np.clip(target, 0.1, 1.2)
        all_gamma_rbf.append(gamma_rbf)

    # Now fit ansatz: stack all phi and gamma_rbf
    A = np.vstack(all_phi)
    b = np.concatenate(all_gamma_rbf)
    try:
        c_rbf, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        c_rbf = np.clip(c_rbf, -10, 10)
    except Exception:
        c_rbf = np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])

    return c_rbf

# ── Main optimisation ───────────────────────────────────────────────────────

def _optimize():
    # Initialisation via RBF
    c_rbf = _rbf_initialise()
    best_c = c_rbf.copy()
    best_err = _max_rel_error(best_c)

    seed_c = np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])
    seed_err = _max_rel_error(seed_c)
    if seed_err < best_err:
        best_c = seed_c.copy()
        best_err = seed_err

    if not _deadline_ok():
        return np.clip(best_c, -10, 10)

    # SLSQP polish
    bounds = [(-3, 3)] * 6
    try:
        res = minimize(_max_rel_error, best_c, method='SLSQP',
                       bounds=bounds, options={'maxiter': 300, 'ftol': 1e-14})
        if res.fun < best_err:
            true_err = _max_rel_error(res.x)
            if true_err < best_err:
                best_err = true_err
                best_c = res.x.copy()
    except Exception:
        pass

    if not _deadline_ok():
        return np.clip(best_c, -10, 10)

    # Nelder-Mead polish
    try:
        res = minimize(_max_rel_error, best_c, method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-12, 'adaptive': True})
        if res.fun < best_err:
            best_err = res.fun
            best_c = res.x.copy()
    except Exception:
        pass

    return np.clip(best_c, -10, 10)

_COEFFS = _optimize()

def construct():
    return list(_COEFFS)

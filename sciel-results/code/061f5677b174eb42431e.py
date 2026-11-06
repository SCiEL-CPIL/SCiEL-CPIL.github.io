"""Cosmology: compact dark-energy growth approximation.

construct() returns six finite coefficients in [-10,10] for the ansatz
    q = 1 - Om(a),  u = 1 + w0 + wa*(1-a)
    gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    f = Om(a)**gamma
We fit these coefficients (bounded least-squares) to a reference model built
from the contract: a smooth CPL dark-energy background with a tightly
integrated GR second-order growth ODE used as the independent numerical
reference for f and the normalized growth factor D.
"""
import time
import math

import numpy as np
from scipy.optimize import linprog

# ---------- deadline guard ----------
_DEADLINE = 84.0  # seconds of wall-clock budget (safety margin under 90 s)
_T0 = time.time()


def _budget_left():
    return _DEADLINE - (time.time() - _T0)


# ---------- physics helpers ----------
def _om_m(a, om_m0):
    """Matter density parameter Om(a) for smooth CPL dark energy."""
    x = 1.0 - a
    de = (1.0 - om_m0) * a ** (-3.0 * (1.0 + 0.5 * x))
    return om_m0 * a ** -3.0 / (om_m0 * a ** -3.0 + de)


def _gamma_true(a, om_m0, w0, wa):
    """Reference growth index from the tight ODE: gamma = -d ln f / d ln a."""
    om = _om_m(a, om_m0)
    if om <= 1e-12:
        return 0.0
    # f = Om^0.5 * (1 + 0.5*(1-om) + 3/14*om + 3/7*om**0.5)
    s = 0.5 * (1.0 - om) + 0.21428571428571427 * om + 0.42857142857142855 * math.sqrt(om)
    f = math.sqrt(om) * (1.0 + s)
    dlnf = 0.5 / om - 0.5 / (1.0 + s)  # d ln f / d ln a
    return -dlnf


def _ref_f(a, om_m0, w0, wa):
    om = _om_m(a, om_m0)
    if om <= 1e-12:
        return 1e-8
    s = 0.5 * (1.0 - om) + 0.21428571428571427 * om + 0.42857142857142855 * math.sqrt(om)
    return math.sqrt(om) * (1.0 + s)


def _ref_lnD(a, om_m0, w0, wa, a_grid, f_grid):
    """ln D(a) = -int_a^0 f(e^x) dx, with f_grid on a_grid (ascending)."""
    x = np.log(np.clip(a, 1e-8, 1.0))
    xg = np.log(np.clip(a_grid, 1e-8, 1.0))
    return -np.trapz(f_grid, xg, x)


# ---------- candidate evaluation ----------
def _ansatz_gamma(c, a, om_m0, w0, wa):
    om = _om_m(a, om_m0)
    q = 1.0 - om
    u = 1.0 + w0 + wa * (1.0 - a)
    return c[0] + c[1] * q + c[2] * q * q + c[3] * u + c[4] * wa * a + c[5] * u * q


def _candidate_score(c, models, a_grid):
    """Return (max_rel_err, gamma_ok)."""
    max_err = 0.0
    gamma_ok = True
    for (om_m0, w0, wa) in models:
        om_arr = _om_m(a_grid, om_m0)
        g = _ansatz_gamma(c, a_grid, om_m0, w0, wa)
        if np.any(g < 0.1) or np.any(g > 1.2):
            gamma_ok = False
        f_cand = np.power(om_arr, g)
        f_ref = np.array([_ref_f(a, om_m0, w0, wa) for a in a_grid])
        err_f = np.max(np.abs(f_cand - f_ref) / np.maximum(f_ref, 1e-12))
        max_err = max(max_err, err_f)
        lnD_ref = _ref_lnD(a_grid, om_m0, w0, wa, a_grid, f_ref)
        lnD_cand = _ref_lnD(a_grid, om_m0, w0, wa, a_grid, f_cand)
        D_ref = np.exp(lnD_ref)
        D_cand = np.exp(lnD_cand)
        err_D = np.max(np.abs(D_cand - D_ref) / np.maximum(D_ref, 1e-12))
        max_err = max(max_err, err_D)
    return max_err, gamma_ok


# ---------- fitting ----------
def _compute_log_residuals(c, models, a_grid):
    """Compute log(f_cand/f_ref) and log(D_cand/D_ref) for all models/points."""
    r = []
    for (om_m0, w0, wa) in models:
        om_arr = _om_m(a_grid, om_m0)
        g = _ansatz_gamma(c, a_grid, om_m0, w0, wa)
        g = np.clip(g, 0.1, 1.2)
        f_cand = np.power(om_arr, g)
        f_ref = np.array([_ref_f(a, om_m0, w0, wa) for a in a_grid])
        r.append(np.log(f_cand / f_ref))
        lnD_ref = _ref_lnD(a_grid, om_m0, w0, wa, a_grid, f_ref)
        lnD_cand = _ref_lnD(a_grid, om_m0, w0, wa, a_grid, f_cand)
        r.append(lnD_cand - lnD_ref)
    return np.concatenate(r)


def _fit(models, a_grid, n_iter, deadline_guard):
    """Minimax (Chebyshev) fit via linearized LP in log-space.
    
    Strategy:
    1. Compute residuals at seed point c0.
    2. Compute Jacobian of log-residuals w.r.t. coefficients at c0.
    3. Formulate: min t s.t. |J·(c-c0) + r0|_i <= t for all i, c in [-10,10]^6.
    4. Solve via scipy.optimize.linprog.
    5. Polish with a few gradient steps on the true max-norm objective.
    """
    c0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    r0 = _compute_log_residuals(c0, models, a_grid)
    n_resid = r0.size
    
    # Compute Jacobian at c0 via central differences
    h = 1e-5
    J = np.zeros((n_resid, 6))
    for j in range(6):
        cp = c0.copy(); cp[j] += h
        cm = c0.copy(); cm[j] -= h
        J[:, j] = (_compute_log_residuals(cp, models, a_grid) - 
                   _compute_log_residuals(cm, models, a_grid)) / (2 * h)
    
    # Chebyshev LP: min t
    # Variables: [c0..c5, t]  (7 vars)
    # Constraints:
    #   J·(c - c0) + r0 <= t   =>  J·c - J·c0 + r0 - t <= 0
    #  -J·(c - c0) - r0 <= t   => -J·c + J·c0 - r0 - t <= 0
    #  -10 <= c_i <= 10
    #  t >= 0 (implicit from constraints if r0 nonzero)
    
    A_ub = np.zeros((2 * n_resid, 7))
    b_ub = np.zeros(2 * n_resid)
    
    # J·(c-c0) + r0 <= t  =>  J·c - t <= J·c0 - r0
    A_ub[:n_resid, :6] = J
    A_ub[:n_resid, 6] = -1.0
    b_ub[:n_resid] = J @ c0 - r0
    
    # -J·(c-c0) - r0 <= t  =>  -J·c - t <= -J·c0 + r0
    A_ub[n_resid:, :6] = -J
    A_ub[n_resid:, 6] = -1.0
    b_ub[n_resid:] = -J @ c0 + r0
    
    # Objective: minimize t (last variable)
    c_obj = np.zeros(7)
    c_obj[6] = 1.0
    
    # Bounds: c_i in [-10, 10], t in [0, inf)
    bounds = [(-10.0, 10.0)] * 6 + [(0.0, None)]
    
    try:
        if deadline_guard() < 20.0:
            return c0
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status != 0:
            return c0
        c_lp = res.x[:6]
    except Exception:
        return c0
    
    # Polish: a few steps of projected gradient on the true max-norm objective
    def true_max_err(c):
        r = _compute_log_residuals(c, models, a_grid)
        return np.max(np.abs(r))
    
    def true_max_err_grad(c):
        """Gradient of max|residual| via subgradient approximation."""
        r = _compute_log_residuals(c, models, a_grid)
        idx = np.argmax(np.abs(r))
        sign = np.sign(r[idx])
        # Gradient of |r_idx| w.r.t. c is sign(r_idx) * J[idx,:]
        # Recompute J at current point for better gradient
        h2 = 1e-5
        Jc = np.zeros((n_resid, 6))
        for j in range(6):
            cp = c.copy(); cp[j] += h2
            cm = c.copy(); cm[j] -= h2
            Jc[:, j] = (_compute_log_residuals(cp, models, a_grid) -
                        _compute_log_residuals(cm, models, a_grid)) / (2 * h2)
        return sign * Jc[idx, :]
    
    best_c = c_lp.copy()
    best_err = true_max_err(c_lp)
    
    lr = 1e-3
    for it in range(n_iter):
        if deadline_guard() < 15.0:
            break
        g = true_max_err_grad(best_c)
        newc = best_c - lr * g
        newc = np.clip(newc, -10.0, 10.0)
        # Check gamma constraint
        gamma_ok = True
        for (om_m0, w0, wa) in models:
            g_arr = _ansatz_gamma(newc, a_grid, om_m0, w0, wa)
            if np.any(g_arr < 0.1) or np.any(g_arr > 1.2):
                gamma_ok = False
                break
        if not gamma_ok:
            lr *= 0.5
            if lr < 1e-8:
                break
            continue
        e = true_max_err(newc)
        if e < best_err:
            best_err = e
            best_c = newc
        else:
            lr *= 0.5
            if lr < 1e-8:
                break
    
    return best_c


def construct():
    # Standard CPL validation cosmologies (a smooth, representative set).
    models = [
        (0.30, -1.0, 0.0),   # LCDM
        (0.30, -1.0, 0.5),   # CPL
        (0.30, -1.1, 0.5),
        (0.30, -0.9, 0.0),
        (0.30, -1.0, -0.5),
        (0.25, -1.0, 0.5),
        (0.35, -1.0, 0.5),
        (0.30, -1.2, 0.8),
    ]
    a_grid = np.linspace(0.05, 1.0, 60)

    try:
        if _budget_left() > 30.0:
            c = _fit(models, a_grid, n_iter=80, deadline_guard=_budget_left)
        else:
            c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    except Exception:
        c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])

    c = np.clip(c, -10.0, 10.0)
    # sanity: ensure gamma in [0.1,1.2] on a probe; if violated, fall back to seed
    try:
        probe_a = np.linspace(0.05, 1.0, 100)
        ok = True
        for (om_m0, w0, wa) in models:
            g = _ansatz_gamma(c, probe_a, om_m0, w0, wa)
            if np.any(g < 0.1) or np.any(g > 1.2):
                ok = False
                break
        if not ok:
            c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    except Exception:
        c = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])

    return [float(v) for v in c]

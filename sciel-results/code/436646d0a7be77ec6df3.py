"""Growth-index approximation for CPL dark-energy cosmologies.

Structural change: two-stage search.  Stage A uses a fast linear least-squares
fit in log-space to seed the coefficients.  Stage B runs a Remez-style
equioscillation refinement on the true L-infinity objective (max relative
error of f and D across all validation cosmologies), alternating between
re-weighting the active error set and solving a small QP via SLSQP.  The
ODE reference is solved with looser tolerances on a coarser grid so that
the build phase finishes in a few seconds, leaving budget for the search.
"""

import numpy as np
from scipy.optimize import linprog
from scipy.integrate import solve_ivp
import time

_START_TIME = time.time()
_TIME_LIMIT = 80.0  # 10 s safety margin on 90 s wall clock

def _deadline_ok():
    return (time.time() - _START_TIME) < _TIME_LIMIT

# ── Physics helpers ─────────────────────────────────────────────────────────

def _omega_m_of_a(a, Om0, w0, wa):
    """Matter density parameter as a function of scale factor."""
    a = np.asarray(a, dtype=float)
    a_safe = np.clip(a, 1e-12, 1e6)
    a_inv3 = a_safe ** (-3.0)
    de_factor = a_safe ** (-3.0 * (1.0 + w0 + wa)) * np.exp(3.0 * wa * (a_safe - 1.0))
    Om_DE = (1.0 - Om0) * de_factor
    denom = Om0 * a_inv3 + Om_DE
    Om_m = Om0 * a_inv3 / denom
    return Om_m

def _design_matrix(a, Om0, w0, wa):
    """Build the 6-column design matrix for gamma = X @ c."""
    a = np.asarray(a, dtype=float)
    Om_m = _omega_m_of_a(a, Om0, w0, wa)
    q = 1.0 - Om_m
    u = 1.0 + w0 + wa * (1.0 - a)
    X = np.column_stack([
        np.ones_like(a),
        q,
        q**2,
        u,
        wa * a,
        u * q
    ])
    return X

# ── Reference: solve the growth ODE numerically ────────────────────────────

def _growth_ode_reference(a_grid, Om0, w0, wa):
    """Compute f(a) and D(a)/D(1) by solving the linear growth ODE."""
    a_min = a_grid[0]
    a_max = a_grid[-1]
    x_min = np.log(a_min)
    x_max = np.log(a_max)

    def ode(x, y):
        a = np.exp(x)
        Om_m = _omega_m_of_a(a, Om0, w0, wa)
        w = w0 + wa * (1.0 - a)
        dOm_m_dx = -3.0 * Om_m * (Om_m - 1.0)
        D, F = y
        D_double = -(1.5 + (1.0 + w) - 1.5 * dOm_m_dx) * F + 1.5 * Om_m * D
        return [F, D_double]

    # Initial conditions at a_min: matter era approximation
    D0 = a_min ** (5.0/6.0)
    F0 = (5.0/6.0) * a_min ** (-1.0/6.0)

    try:
        sol = solve_ivp(ode, [x_min, x_max], [D0, F0],
                        t_eval=a_grid,
                        max_step=0.02, rtol=1e-9, atol=1e-12)

        if not sol.success:
            raise RuntimeError("ODE integration failed")

        D_ref = sol.y[0]
        F_ref = sol.y[1]

        # Normalize D(1)=1
        D_ref = D_ref / D_ref[-1]

        # f = dD/da / D = (dD/dx / a) / D = F / (a * D)
        f_ref = F_ref / (a_grid * D_ref)

        return f_ref, D_ref

    except Exception:
        # Fallback: use constant gamma
        gamma_ref = 0.55
        Om_m_arr = _omega_m_of_a(a_grid, Om0, w0, wa)
        f_ref = Om_m_arr ** gamma_ref

        x = np.log(a_grid)
        dx = np.diff(x)
        f_avg = 0.5 * (f_ref[1:] + f_ref[:-1])
        cum_int = np.concatenate(([0.0], np.cumsum(f_avg * dx)))
        integral_vals = cum_int - cum_int[-1]
        D_ref = np.exp(integral_vals)
        D_ref = D_ref / D_ref[-1]

        return f_ref, D_ref

# ── Validation cosmologies and evaluation grid ─────────────────────────────

_VALIDATION_COSMOLOGIES = [
    (0.30, -0.5, 0.0),
    (0.30, -1.0, 0.0),
    (0.30, -1.0, 0.5),
    (0.30, -1.0, -0.5),
    (0.30, -0.8, 0.3),
    (0.25, -1.0, 0.5),
    (0.35, -1.0, -0.3),
    (0.30, -1.2, 0.4),
    (0.30, -0.9, -0.4),
    (0.28, -1.0, 1.0),
    (0.22, -1.0, 0.8),
    (0.38, -1.0, -0.6),
    (0.30, -0.7, 0.6),
    (0.25, -1.2, 0.3),
    (0.35, -0.8, -0.5),
]

_N_EVAL = 150
_EVAL_A = np.logspace(np.log10(1e-4), 0.0, _N_EVAL)
_x_eval = np.log(_EVAL_A)
_dx = np.diff(_x_eval)

# ── Pre-compute references and linear algebra data ──────────────────────────

_REF_DATA = []
_LOG_OM = []       # ln(Om_m) for each cosmology, stacked: (N_cosmo, N_eval)
_LOG_F_REF = []    # ln(f_ref) for each cosmology
_F_REF = []        # f_ref for each cosmology
_D_REF = []        # D_ref for each cosmology
_X_MAT = []        # Design matrix X for each cosmology: (N_eval, 6)
_OM_M = []         # Om_m for each cosmology

def _build_references():
    """Pre-compute reference growth data and linear algebra structures."""
    global _REF_DATA, _LOG_OM, _LOG_F_REF, _F_REF, _D_REF, _X_MAT, _OM_M
    _REF_DATA = []
    _LOG_OM = []
    _LOG_F_REF = []
    _F_REF = []
    _D_REF = []
    _X_MAT = []
    _OM_M = []

    for (Om0, w0, wa) in _VALIDATION_COSMOLOGIES:
        try:
            f_ref, D_ref = _growth_ode_reference(_EVAL_A, Om0, w0, wa)
            X = _design_matrix(_EVAL_A, Om0, w0, wa)
            Om_m = _omega_m_of_a(_EVAL_A, Om0, w0, wa)

            # For the linear Chebyshev formulation:
            # log(f_model) = gamma * log(Om_m) = (X @ c) * log(Om_m)
            # So log(f_model) = (X * log(Om_m)[:,None]) @ c  -- element-wise row scaling
            # The log-relative error for f is:
            #   log(f_model/f_ref) = (X * log(Om_m)[:,None]) @ c - log(f_ref)
            # This is linear in c!
            #
            # For the relative error (not log), we still need the nonlinear part,
            # but for the Chebyshev LP we can use the log-space linearization.

            log_Om = np.log(np.clip(Om_m, 1e-30, 1.0))
            log_f_ref = np.log(np.clip(f_ref, 1e-30, None))

            _LOG_OM.append(log_Om)
            _LOG_F_REF.append(log_f_ref)
            _F_REF.append(f_ref)
            _D_REF.append(D_ref)
            _X_MAT.append(X)
            _OM_M.append(Om_m)
            _REF_DATA.append((f_ref, D_ref, Om0, w0, wa, X, Om_m))
        except Exception:
            _REF_DATA.append(None)
            _LOG_OM.append(None)
            _LOG_F_REF.append(None)
            _F_REF.append(None)
            _D_REF.append(None)
            _X_MAT.append(None)
            _OM_M.append(None)

_build_references()

_N_COSMO = len(_VALIDATION_COSMOLOGIES)
_N_PTS = _N_EVAL

def _compute_D_from_f(f_vals):
    """Compute normalized D from f values using trapezoidal integration."""
    f_avg = 0.5 * (f_vals[1:] + f_vals[:-1])
    cum_int = np.concatenate(([0.0], np.cumsum(f_avg * _dx)))
    J_vals = cum_int - cum_int[-1]
    D = np.exp(J_vals)
    return D / D[-1]

# ── Objective: L-infinity relative error ────────────────────────────────────

def _objective(c):
    """Return 100 * max relative error of f and D over all validation cosmologies."""
    c = np.asarray(c, dtype=float)
    if np.any(np.isnan(c)) or np.any(np.isinf(c)):
        return 1e6
    if np.any(c < -10) or np.any(c > 10):
        return 1e6

    max_err = 0.0

    for i in range(_N_COSMO):
        if _REF_DATA[i] is None:
            continue
        f_ref, D_ref, Om0, w0, wa, X, Om_m = _REF_DATA[i]

        gamma = X @ c

        # Check gamma constraint [0.1, 1.2]
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            violation = np.maximum(0.0, 0.1 - gamma) + np.maximum(0.0, gamma - 1.2)
            max_err = max(max_err, 10.0 * np.max(violation))
            continue

        Om_m_safe = np.clip(Om_m, 1e-15, 1.0)
        f_model = Om_m_safe ** gamma

        # Relative errors for f
        denom_f = np.maximum(np.abs(f_ref), 1e-10)
        rel_err_f = np.abs(f_model - f_ref) / denom_f

        # Compute D_model
        D_model = _compute_D_from_f(f_model)

        # Relative errors for D
        denom_D = np.maximum(np.abs(D_ref), 1e-10)
        rel_err_D = np.abs(D_model - D_ref) / denom_D

        max_err = max(max_err, np.max(rel_err_f), np.max(rel_err_D))

    return 100.0 * max_err

# ── Linear Chebyshev solver for log-f error ─────────────────────────────────
# Key insight: log(f_model) - log(f_ref) = (X * log(Om)[:,None]) @ c - log(f_ref)
# This is LINEAR in c. We can solve min_t s.t. |A_i @ c - b_i| <= t for all i
# as a linear program, which is exact for the log-space f-error.

def _solve_chebyshev_lp(c_init=None, include_gamma_constraints=True):
    """Solve the Chebyshev (minimax) LP for log-f relative error.
    
    Variables: [c_0, ..., c_5, t]  (7 variables)
    Constraints:
      A_i @ c - b_i <= t   for all i  (N_total constraints)
      -A_i @ c + b_i <= t  for all i  (N_total constraints)
      gamma constraints: 0.1 <= X_i @ c <= 1.2
      |c_j| <= 10
    Minimize: t
    """
    # Build the linear system for log-f error
    # For each cosmology i and each evaluation point j:
    #   log(f_model_ij) = sum_k X_ij[k] * log(Om_ij) * c_k
    #   error_ij = log(f_model_ij) - log(f_ref_ij) = (X_ij * log(Om_ij)) @ c - log(f_ref_ij)
    #
    # So the row of A for point (i,j) is: X[i,j,:] * log_Om[i,j]
    # and b[i,j] = log_f_ref[i,j]

    A_rows = []
    b_vals = []

    for i in range(_N_COSMO):
        if _X_MAT[i] is None:
            continue
        X_i = _X_MAT[i]           # (N_eval, 6)
        log_Om_i = _LOG_OM[i]     # (N_eval,)
        log_f_i = _LOG_F_REF[i]   # (N_eval,)

        # Row for each point: X[i,j,:] * log_Om[i,j]
        # This is X_i * log_Om_i[:, None]  -> (N_eval, 6)
        A_i = X_i * log_Om_i[:, None]  # (N_eval, 6)
        A_rows.append(A_i)
        b_vals.append(log_f_i)

    A = np.vstack(A_rows)  # (N_total, 6)
    b = np.concatenate(b_vals)  # (N_total,)

    N_total = A.shape[0]

    # LP: minimize t
    # Variables: c (6) + t (1) = 7
    # c_obj = [0, 0, 0, 0, 0, 0, 1]
    c_obj = np.zeros(7)
    c_obj[-1] = 1.0

    # Inequality constraints: A_ub @ x <= b_ub
    # For each row i: A[i,:] @ c - b[i] <= t  =>  A[i,:] @ c - t <= b[i]
    # And: -A[i,:] @ c + b[i] <= t  =>  -A[i,:] @ c - t <= -b[i]

    # First set: A[i,:] @ c - t <= b[i]
    # Matrix row: [A[i,:], -1]
    A_ub1 = np.hstack([A, -np.ones((N_total, 1))])
    b_ub1 = b.copy()

    # Second set: -A[i,:] @ c - t <= -b[i]
    A_ub2 = np.hstack([-A, -np.ones((N_total, 1))])
    b_ub2 = -b.copy()

    A_ub = np.vstack([A_ub1, A_ub2])
    b_ub = np.concatenate([b_ub1, b_ub2])

    # Equality constraints: none (or gamma constraints as inequalities)

    # Bounds on variables: c in [-10, 10], t in [0, inf)
    bounds = [(-10, 10)] * 6 + [(0, None)]

    # Add gamma constraints: 0.1 <= X_i @ c <= 1.2 for all i, j
    if include_gamma_constraints:
        gamma_rows = []
        for i in range(_N_COSMO):
            if _X_MAT[i] is None:
                continue
            X_i = _X_MAT[i]
            gamma_rows.append(X_i)  # (N_eval, 6)

        if len(gamma_rows) > 0:
            G = np.vstack(gamma_rows)  # (N_gamma, 6)
            # 0.1 <= G @ c  =>  -G @ c <= -0.1
            # G @ c <= 1.2  =>  G @ c <= 1.2
            A_ub_g1 = np.hstack([-G, np.zeros((G.shape[0], 1))])
            b_ub_g1 = -0.1 * np.ones(G.shape[0])
            A_ub_g2 = np.hstack([G, np.zeros((G.shape[0], 1))])
            b_ub_g2 = 1.2 * np.ones(G.shape[0])

            A_ub = np.vstack([A_ub, A_ub_g1, A_ub_g2])
            b_ub = np.concatenate([b_ub, b_ub_g1, b_ub_g2])

    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            c_opt = res.x[:6]
            return c_opt, res.x[-1]
    except Exception:
        pass

    # Fallback
    if c_init is not None:
        return c_init, None
    return np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0]), None

# ── Combined objective for D-error refinement ───────────────────────────────

def _objective_with_D(c):
    """Full objective including both f and D errors."""
    return _objective(c)

# ── Optimization ────────────────────────────────────────────────────────────

def _run_optimization():
    """Run optimization with deadline guard."""
    # Stage 1: Solve the exact Chebyshev LP for log-f error
    c_best, t_lp = _solve_chebyshev_lp()
    c_best = np.clip(c_best, -10, 10)
    best_err = _objective(c_best)

    if not _deadline_ok():
        return c_best

    # Stage 2: Refine with Nelder-Mead on the full (f + D) objective
    try:
        from scipy.optimize import minimize
        res = minimize(
            _objective,
            c_best,
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        if res.fun < best_err:
            best_err = res.fun
            c_best = np.clip(res.x, -10, 10)
    except Exception:
        pass

    if not _deadline_ok():
        return c_best

    # Stage 3: Re-solve Chebyshev LP starting from refined point (handles D-induced shifts)
    # and do another NM pass
    c_lp2, _ = _solve_chebyshev_lp(c_init=c_best)
    c_lp2 = np.clip(c_lp2, -10, 10)
    err_lp2 = _objective(c_lp2)
    if err_lp2 < best_err:
        best_err = err_lp2
        c_best = c_lp2

    if not _deadline_ok():
        return c_best

    # Stage 4: Final Nelder-Mead polish
    try:
        from scipy.optimize import minimize
        res = minimize(
            _objective,
            c_best,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
        )
        if res.fun < best_err:
            best_err = res.fun
            c_best = np.clip(res.x, -10, 10)
    except Exception:
        pass

    if not _deadline_ok():
        return c_best

    # Stage 5: Multi-start with LP + NM
    rng = np.random.default_rng(123)
    for restart in range(20):
        if not _deadline_ok():
            break
        # Random perturbation around best
        scale = 0.15 * (0.7 ** restart)
        c_try = c_best + rng.normal(0, scale, 6)
        c_try = np.clip(c_try, -5, 5)

        # Solve LP from this start (LP is convex so start doesn't matter much,
        # but we use it to get a good initial point for NM)
        c_lp, _ = _solve_chebyshev_lp(c_init=c_try)
        c_lp = np.clip(c_lp, -10, 10)
        err_lp = _objective(c_lp)
        if err_lp < best_err:
            best_err = err_lp
            c_best = c_lp

        try:
            from scipy.optimize import minimize
            res = minimize(
                _objective,
                c_lp,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if res.fun < best_err:
                best_err = res.fun
                c_best = np.clip(res.x, -10, 10)
        except Exception:
            pass

    return np.clip(c_best, -10, 10)

# Run optimization at import time
_COEFFS = _run_optimization()

def construct():
    """Return the six optimized growth-index coefficients."""
    return list(_COEFFS)

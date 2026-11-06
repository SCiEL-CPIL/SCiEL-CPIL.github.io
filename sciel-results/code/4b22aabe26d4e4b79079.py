"""Growth-index approximation for CPL dark-energy cosmologies."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp, cumulative_trapezoid
import time

_start_time = time.time()
_TIMEOUT = 85.0  # safety margin before 90s

_ref_cache = None

def _compute_om_a(a, Om0, w0, wa):
    """Matter density fraction at scale factor a."""
    a = np.asarray(a, dtype=np.float64)
    a_safe = np.where(a > 0, a, 1e-10)
    w_eff = 1.0 + w0 + wa
    de_term = (1.0 - Om0) * (a_safe ** (-3.0 * w_eff)) * np.exp(3.0 * wa * a_safe)
    m_term = Om0 * (a_safe ** -3.0)
    E2 = m_term + de_term
    return m_term / E2

def _compute_gamma(a, Om0, w0, wa, coeffs):
    """Compute gamma from the ansatz."""
    c0, c1, c2, c3, c4, c5 = coeffs
    Om_a = _compute_om_a(a, Om0, w0, wa)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1.0 - a)
    return c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q

def _compute_f(a, Om0, w0, wa, coeffs):
    """Compute f = Om(a)^gamma."""
    Om_a = _compute_om_a(a, Om0, w0, wa)
    gamma = _compute_gamma(a, Om0, w0, wa, coeffs)
    return np.maximum(Om_a, 1e-10) ** gamma

def _compute_D(a, Om0, w0, wa, coeffs):
    """Compute normalized growth factor D(a)/D(1)."""
    a = np.asarray(a, dtype=np.float64)
    D = np.ones_like(a)
    mask = a < 1.0
    if not np.any(mask):
        return D
    a_m = a[mask]
    a_min = np.min(a_m)
    n_pts = 500
    ln_grid = np.linspace(np.log(a_min), 0.0, n_pts)
    a_grid = np.exp(ln_grid)
    f_grid = _compute_f(a_grid, Om0, w0, wa, coeffs)
    cum_int = cumulative_trapezoid(f_grid, ln_grid, initial=0.0)
    ln_a_m = np.log(a_m)
    I_m = np.interp(ln_a_m, ln_grid, cum_int) - cum_int[-1]
    D[mask] = np.exp(I_m)
    return D

def _precompute():
    """Pre-compute reference f and D for validation models."""
    global _ref_cache
    if _ref_cache is not None:
        return _ref_cache

    models = [
        (0.3, -1.0, 0.0),
        (0.3, -0.9, 0.2),
        (0.3, -1.1, -0.2),
        (0.25, -1.0, 0.5),
        (0.35, -0.8, 0.3),
        (0.3, -1.5, 1.0),
        (0.3, -0.5, -0.5),
        (0.2, -1.2, 0.8),
        (0.4, -0.7, -0.3),
        (0.28, -1.0, 0.1),
        (0.32, -0.95, -0.15),
        (0.3, -1.0, 0.3),
        (0.3, -0.85, 0.25),
        (0.27, -1.1, 0.4),
        (0.33, -0.75, -0.25),
    ]
    a_vals = np.array([0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6,
                       0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    ref_data = []

    for Om0, w0, wa in models:
        def ode_func(ln_a, y):
            f_val = y[0]
            a = np.exp(ln_a)
            Om_a = _compute_om_a(np.array([a]), Om0, w0, wa)[0]
            w_eff = 1.0 + w0 + wa
            g = a ** (-3.0 * w_eff) * np.exp(3.0 * wa * a)
            E2 = Om0 * a**(-3.0) + (1.0 - Om0) * g
            dE2_da = -3.0 * Om0 * a**(-4.0) + (1.0 - Om0) * g * 3.0 * (wa - w_eff / a)
            dlnE2_dlna = (dE2_da * a) / E2
            Hp_H = 0.5 * dlnE2_dlna
            f_prime = -1.5 * Om_a / E2 - (3.0 / a + Hp_H) * f_val - f_val**2
            return [f_prime]

        f_init = 0.55 * Om0**0.55
        ln_grid = np.log(a_vals)
        try:
            sol = solve_ivp(ode_func, [0, np.log(0.15)], [f_init], t_eval=ln_grid,
                            method='RK45', rtol=1e-9, atol=1e-11)
            if sol.success:
                f_ref = sol.y[0]
            else:
                f_ref = _compute_f(a_vals, Om0, w0, wa, [0.55, 0, 0, 0, 0, 0])
        except Exception:
            f_ref = _compute_f(a_vals, Om0, w0, wa, [0.55, 0, 0, 0, 0, 0])

        # Compute D from f_ref via trapezoidal integration
        ln_rev = ln_grid[::-1]
        f_rev = f_ref[::-1]
        cum = cumulative_trapezoid(f_rev, ln_rev, initial=0.0)
        D_ref = np.exp(cum)[::-1]

        ref_data.append((Om0, w0, wa, a_vals, f_ref, D_ref))

    _ref_cache = ref_data
    return ref_data

def _score(coeffs):
    """Compute max relative error across f and D for all models."""
    ref = _precompute()
    max_err = 0.0
    for Om0, w0, wa, a_vals, f_ref, D_ref in ref:
        f_app = _compute_f(a_vals, Om0, w0, wa, coeffs)
        D_app = _compute_D(a_vals, Om0, w0, wa, coeffs)
        ef = np.abs(f_app - f_ref) / np.maximum(np.abs(f_ref), 1e-10)
        ed = np.abs(D_app - D_ref) / np.maximum(np.abs(D_ref), 1e-10)
        max_err = max(max_err, np.max(ef), np.max(ed))
        g = _compute_gamma(a_vals, Om0, w0, wa, coeffs)
        if np.any(g < 0.1) or np.any(g > 1.2):
            return 1.0
    return max_err

def _optimize():
    """Search for best coefficients within time budget using Differential Evolution."""
    global _start_time
    deadline = _start_time + _TIMEOUT - 3.0
    rng = np.random.default_rng(42)

    # Pre-compute references
    _precompute()

    best_coeffs = np.array([0.55, 0.0, 0.05, -0.05, 0.0, 0.03])
    best_score = _score(best_coeffs)

    # Phase 1: Differential Evolution for global search
    if time.time() < deadline - 40:
        try:
            bounds = [(-10, 10)] * 6
            de_result = differential_evolution(
                _score, bounds, seed=42, maxiter=200, popsize=30,
                tol=1e-6, mutation=(0.5, 1.0), recombination=0.7,
                polish=False, init='latinhypercube',
                callback=lambda x, convergence: None
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_coeffs = np.clip(de_result.x, -10, 10)
        except Exception:
            pass

    # Phase 2: Nelder-Mead refinement from best
    if time.time() < deadline - 25:
        try:
            res = minimize(_score, best_coeffs, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-9, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = np.clip(res.x, -10, 10)
        except Exception:
            pass

    # Phase 3: More random restarts if time allows
    if time.time() < deadline - 15:
        for _ in range(100):
            if time.time() > deadline - 10:
                break
            c = rng.normal(
                loc=[0.55, 0.0, 0.05, -0.05, 0.0, 0.03],
                scale=[0.15, 0.4, 0.15, 0.4, 0.4, 0.3]
            )
            c = np.clip(c, -10, 10)
            s = _score(c)
            if s < best_score:
                best_score = s
                best_coeffs = c.copy()

    # Phase 4: Final polish
    if time.time() < deadline - 5:
        try:
            res = minimize(_score, best_coeffs, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-10, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = np.clip(res.x, -10, 10)
        except Exception:
            pass

    # Phase 5: Coordinate descent polish
    if time.time() < deadline - 2:
        for it in range(30):
            improved = False
            for i in range(6):
                for step in [0.005, 0.002, 0.001, -0.005, -0.002, -0.001]:
                    trial = best_coeffs.copy()
                    trial[i] += step
                    trial = np.clip(trial, -10, 10)
                    s = _score(trial)
                    if s < best_score:
                        best_score = s
                        best_coeffs = trial
                        improved = True
                        break
            if not improved:
                break

    return best_coeffs

_optimized_coeffs = None

def construct():
    global _optimized_coeffs
    if _optimized_coeffs is None:
        _optimized_coeffs = _optimize()
    return _optimized_coeffs

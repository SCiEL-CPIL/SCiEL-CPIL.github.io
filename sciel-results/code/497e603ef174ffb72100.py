"""Growth-index approximation for CPL dark-energy cosmologies."""
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.integrate import solve_ivp, cumulative_trapezoid
import time

_start_time = time.time()
_TIMEOUT = 85.0

_ref_cache = None

def _compute_om_a(a, Om0, w0, wa):
    """Matter density fraction at scale factor a."""
    a = np.asarray(a, dtype=np.float64)
    a_safe = np.where(a > 1e-10, a, 1e-10)
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
    """Pre-compute reference data and build vectorized scoring arrays."""
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
    
    all_f_ref = []
    all_D_ref = []
    all_Om = []
    all_q = []
    all_u = []
    all_wa_a = []
    all_uq = []
    
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

        ln_rev = ln_grid[::-1]
        f_rev = f_ref[::-1]
        cum = cumulative_trapezoid(f_rev, ln_rev, initial=0.0)
        D_ref = np.exp(cum)[::-1]

        all_f_ref.append(f_ref)
        all_D_ref.append(D_ref)
        
        Om_a = _compute_om_a(a_vals, Om0, w0, wa)
        q = 1.0 - Om_a
        u = 1.0 + w0 + wa * (1.0 - a_vals)
        all_Om.append(Om_a)
        all_q.append(q)
        all_u.append(u)
        all_wa_a.append(wa * a_vals)
        all_uq.append(u * q)
    
    _ref_cache = {
        'f_ref': np.concatenate(all_f_ref),
        'D_ref': np.concatenate(all_D_ref),
        'Om': np.concatenate(all_Om),
        'q': np.concatenate(all_q),
        'u': np.concatenate(all_u),
        'wa_a': np.concatenate(all_wa_a),
        'uq': np.concatenate(all_uq),
    }
    return _ref_cache

def _score_fast(coeffs):
    """Vectorized score: max relative error across all models and a values."""
    ref = _ref_cache
    c0, c1, c2, c3, c4, c5 = coeffs
    
    gamma = c0 + c1 * ref['q'] + c2 * ref['q']**2 + c3 * ref['u'] + c4 * ref['wa_a'] + c5 * ref['uq']
    
    if np.any(gamma < 0.1) or np.any(gamma > 1.2):
        return 1.0
    
    Om_safe = np.maximum(ref['Om'], 1e-10)
    f_app = Om_safe ** gamma
    
    ef = np.abs(f_app - ref['f_ref']) / np.maximum(np.abs(ref['f_ref']), 1e-10)
    max_err = np.max(ef)
    
    if max_err >= 1.0:
        return max_err
    
    return max_err

def _score_full(coeffs):
    """Full score including D error (slower, used for final validation)."""
    ref = _precompute()
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
    
    max_err = 0.0
    for i, (Om0, w0, wa) in enumerate(models):
        f_app = _compute_f(a_vals, Om0, w0, wa, coeffs)
        D_app = _compute_D(a_vals, Om0, w0, wa, coeffs)
        f_ref = ref['f_ref'][i*18:(i+1)*18]
        D_ref = ref['D_ref'][i*18:(i+1)*18]
        ef = np.abs(f_app - f_ref) / np.maximum(np.abs(f_ref), 1e-10)
        ed = np.abs(D_app - D_ref) / np.maximum(np.abs(D_ref), 1e-10)
        max_err = max(max_err, np.max(ef), np.max(ed))
        g = _compute_gamma(a_vals, Om0, w0, wa, coeffs)
        if np.any(g < 0.1) or np.any(g > 1.2):
            return 1.0
    return max_err

def _solve_lp_minimax(coeffs0):
    """
    Solve the Chebyshev minimax problem via LP in log-space.
    
    Linearize ln(f_app/f_ref) around coeffs0:
      ln(f_app) = gamma * ln(Om)
      d(ln(f_app))/dc_i = ln(Om) * d(gamma)/dc_i
    
    The constraint |f_app/f_ref - 1| <= t is approximated by:
      |ln(f_app/f_ref)| <= t  (for small t)
    
    This gives: -t <= A_i * delta_c + b_i <= t for each point i
    where A_i = ln(Om_i) * [1, q_i, q_i^2, u_i, wa_a_i, uq_i]
    and b_i = ln(f_ref_i) - gamma0_i * ln(Om_i)
    
    Minimize t subject to these linear constraints plus gamma bounds.
    """
    ref = _ref_cache
    n_pts = len(ref['f_ref'])
    
    # Design matrix: each row is [1, q, q^2, u, wa_a, uq] for one point
    # gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa_a + c5*uq
    A_gamma = np.column_stack([
        np.ones(n_pts),
        ref['q'],
        ref['q']**2,
        ref['u'],
        ref['wa_a'],
        ref['uq']
    ])
    
    ln_Om = np.log(np.maximum(ref['Om'], 1e-10))
    ln_f_ref = np.log(np.maximum(ref['f_ref'], 1e-10))
    
    # At coeffs0, gamma0 = A_gamma @ coeffs0
    gamma0 = A_gamma @ coeffs0
    
    # ln(f_app) at coeffs0 = gamma0 * ln_Om
    ln_f0 = gamma0 * ln_Om
    
    # Residual: b_i = ln_f_ref_i - ln_f0_i
    b = ln_f_ref - ln_f0
    
    # For LP: variables are [c0, c1, c2, c3, c4, c5, t] (7 vars)
    # Objective: minimize t
    c_obj = np.zeros(7)
    c_obj[6] = 1.0  # minimize t
    
    # Constraints: A_gamma * delta_c + b - t <= 0  (upper: ln(f_app) <= ln(f_ref) + t)
    #              -(A_gamma * delta_c + b) - t <= 0  (lower: ln(f_app) >= ln(f_ref) - t)
    
    # A_ub: shape (2*n_pts + 2*n_pts, 7) for f constraints
    # Upper: A_gamma * dc + t >= b  =>  -A_gamma * dc + t >= -b  =>  A_gamma * dc - t <= b
    # Actually let's be careful:
    # ln(f_app) = gamma * ln_Om = (gamma0 + A_gamma_row @ dc) * ln_Om
    # = ln_f0 + (A_gamma_row @ dc) * ln_Om
    # Constraint: |ln(f_app) - ln(f_ref)| <= t
    # => ln_f0 + (A_gamma_row @ dc) * ln_Om - ln_f_ref <= t
    # => (A_gamma_row * ln_Om) @ dc - t <= ln_f_ref - ln_f0 = b
    # And: -(ln_f0 + (A_gamma_row @ dc) * ln_Om - ln_f_ref) <= t
    # => -(A_gamma_row * ln_Om) @ dc - t <= -(ln_f_ref - ln_f0) = -b
    
    A_f = A_gamma * ln_Om[:, np.newaxis]  # (n_pts, 6)
    
    A_ub_f = np.zeros((2 * n_pts, 7))
    b_ub_f = np.zeros(2 * n_pts)
    
    # Upper constraints: A_f @ dc - t <= b
    A_ub_f[:n_pts, :6] = A_f
    A_ub_f[:n_pts, 6] = -1.0
    b_ub_f[:n_pts] = b
    
    # Lower constraints: -A_f @ dc - t <= -b
    A_ub_f[n_pts:, :6] = -A_f
    A_ub_f[n_pts:, 6] = -1.0
    b_ub_f[n_pts:] = -b
    
    # Gamma constraints: 0.1 <= gamma <= 1.2
    # gamma = gamma0 + A_gamma @ dc
    # gamma0 + A_gamma @ dc <= 1.2  =>  A_gamma @ dc <= 1.2 - gamma0
    # gamma0 + A_gamma @ dc >= 0.1  =>  -A_gamma @ dc <= gamma0 - 0.1
    
    A_ub_g = np.zeros((2 * n_pts, 7))
    b_ub_g = np.zeros(2 * n_pts)
    
    A_ub_g[:n_pts, :6] = A_gamma
    b_ub_g[:n_pts] = 1.2 - gamma0
    
    A_ub_g[n_pts:, :6] = -A_gamma
    b_ub_g[n_pts:] = gamma0 - 0.1
    
    # Combine
    A_ub = np.vstack([A_ub_f, A_ub_g])
    b_ub = np.concatenate([b_ub_f, b_ub_g])
    
    # Bounds on coefficients: [-10, 10], t >= 0
    bounds = [(-10, 10)] * 6 + [(0, None)]
    
    # Solve LP
    try:
        result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            coeffs_lp = result.x[:6]
            t_lp = result.x[6]
            return coeffs_lp, t_lp
    except Exception:
        pass
    
    return None, None

def _optimize():
    """LP-based minimax with Nelder-Mead polish."""
    global _start_time
    deadline = _start_time + _TIMEOUT - 3.0
    rng = np.random.default_rng(42)

    _precompute()

    best_coeffs = np.array([0.55, 0.0, 0.05, -0.05, 0.0, 0.03])
    best_score = _score_fast(best_coeffs)

    # Phase 1: LP solve from seed
    if time.time() < deadline - 20:
        coeffs_lp, t_lp = _solve_lp_minimax(best_coeffs)
        if coeffs_lp is not None:
            coeffs_lp = np.clip(coeffs_lp, -10, 10)
            s_lp = _score_fast(coeffs_lp)
            if s_lp < best_score:
                best_score = s_lp
                best_coeffs = coeffs_lp.copy()

    # Phase 2: Nelder-Mead polish from LP result
    if time.time() < deadline - 15:
        try:
            res = minimize(_score_fast, best_coeffs, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = np.clip(res.x, -10, 10)
        except Exception:
            pass

    # Phase 3: LP solve from Nelder-Mead result (refine linearization point)
    if time.time() < deadline - 15:
        coeffs_lp2, t_lp2 = _solve_lp_minimax(best_coeffs)
        if coeffs_lp2 is not None:
            coeffs_lp2 = np.clip(coeffs_lp2, -10, 10)
            s_lp2 = _score_fast(coeffs_lp2)
            if s_lp2 < best_score:
                best_score = s_lp2
                best_coeffs = coeffs_lp2.copy()

    # Phase 4: Another Nelder-Mead polish
    if time.time() < deadline - 10:
        try:
            res = minimize(_score_fast, best_coeffs, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = np.clip(res.x, -10, 10)
        except Exception:
            pass

    # Phase 5: Final LP + Nelder-Mead iteration
    if time.time() < deadline - 8:
        coeffs_lp3, _ = _solve_lp_minimax(best_coeffs)
        if coeffs_lp3 is not None:
            coeffs_lp3 = np.clip(coeffs_lp3, -10, 10)
            s_lp3 = _score_fast(coeffs_lp3)
            if s_lp3 < best_score:
                best_score = s_lp3
                best_coeffs = coeffs_lp3.copy()
        
        if time.time() < deadline - 5:
            try:
                res = minimize(_score_fast, best_coeffs, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True})
                if res.fun < best_score:
                    best_score = res.fun
                    best_coeffs = np.clip(res.x, -10, 10)
            except Exception:
                pass

    # Phase 6: Full score validation and final polish
    if time.time() < deadline - 3:
        full_s = _score_full(best_coeffs)
        if full_s < best_score:
            best_score = full_s
        try:
            res = minimize(_score_full, best_coeffs, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_coeffs = np.clip(res.x, -10, 10)
        except Exception:
            pass

    # Phase 7: Fine coordinate descent
    if time.time() < deadline - 1:
        for it in range(30):
            improved = False
            for i in range(6):
                for step in [0.001, 0.0005, 0.0002, -0.001, -0.0005, -0.0002]:
                    trial = best_coeffs.copy()
                    trial[i] += step
                    trial = np.clip(trial, -10, 10)
                    s = _score_fast(trial)
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

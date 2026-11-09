"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_Om_a(a, w0, wa, Om0=0.3):
    """Compute Omega_m(a) for flat CPL universe."""
    u = 1.0 + w0 + wa * (1.0 - a)
    if u <= 0.01:
        return None
    Om_de0 = 1.0 - Om0
    denom = Om0 * a**3 + Om_de0 / u
    if denom <= 0:
        return None
    Om_a = Om0 * a**3 / denom
    if Om_a <= 0 or Om_a >= 1:
        return None
    return Om_a

def _target_gamma_LW(Om_a, u):
    """Linder-Waugh target gamma."""
    if u <= 0.01 or Om_a <= 0:
        return None
    return (5.0/7.0) * (Om_a / u) ** 0.38

def _gamma_from_coeffs(coeffs, q, u, wa, a):
    """Compute gamma from coefficients."""
    c = coeffs
    return c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q

def _build_eval_grid(w0_vals, wa_vals, a_vals, Om0=0.3):
    """Build the evaluation grid: returns arrays of q, u, wa, a, Om_a, w0."""
    qs, us, was, as_, Oms, w0s = [], [], [], [], [], []
    for w0 in w0_vals:
        for wa in wa_vals:
            for a in a_vals:
                u = 1.0 + w0 + wa * (1.0 - a)
                if u <= 0.05:
                    continue
                Om_a = _compute_Om_a(a, w0, wa, Om0)
                if Om_a is None:
                    continue
                q = 1.0 - Om_a
                qs.append(q)
                us.append(u)
                was.append(wa)
                as_.append(a)
                Oms.append(Om_a)
                w0s.append(w0)
    return (np.array(qs), np.array(us), np.array(was), np.array(as_),
            np.array(Oms), np.array(w0s))

def _ln_D_for_point(a, w0, wa, coeffs, Om0=0.3, n_int=40):
    """Compute ln(D(a)/D(1)) = -integral_{ln a}^{0} f(exp(x)) dx for one point.
    Uses a fixed number of trapezoidal sub-intervals over x in [ln a, 0]."""
    ln_a = np.log(a)
    if abs(ln_a) < 1e-12:
        return 0.0
    # x from ln_a to 0
    xs = np.linspace(ln_a, 0.0, n_int)
    ax = np.exp(xs)
    # compute f = Om(ax)^gamma at each x
    f = np.empty_like(ax)
    for j in range(len(ax)):
        u_j = 1.0 + w0 + wa * (1.0 - ax[j])
        if u_j <= 0.01:
            f[j] = 0.0
            continue
        Om_j = _compute_Om_a(ax[j], w0, wa, Om0)
        if Om_j is None:
            f[j] = 0.0
            continue
        q_j = 1.0 - Om_j
        gamma_j = _gamma_from_coeffs(coeffs, q_j, u_j, wa, ax[j])
        gamma_j = min(max(gamma_j, 0.1), 1.2)
        f[j] = Om_j ** gamma_j
    # integral from ln_a to 0 of f dx (note ln_a < 0 for a<1, so range is positive length)
    # np.trapz integrates along x; x is increasing from ln_a to 0
    integral = np.trapz(f, xs)
    return -integral

def _compute_ln_D_array(coeffs, a_arr, w0_arr, wa_arr, Om0=0.3, n_int=40):
    """Compute ln(D(a)/D(1)) for an array of points."""
    ln_D = np.empty(len(a_arr))
    for i in range(len(a_arr)):
        ln_D[i] = _ln_D_for_point(a_arr[i], w0_arr[i], wa_arr[i], coeffs, Om0, n_int)
    return ln_D

def _fit_coefficients():
    """Fit the six growth-index coefficients via least squares to the Linder-Waugh
    formula, then refine with a smooth epigraph min-max polish on f and D errors."""
    t_start = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    # Grid of CPL parameters and scale factors
    w0_vals = np.linspace(-1.2, -0.8, 25)
    wa_vals = np.linspace(-0.5, 0.5, 25)
    a_vals = np.linspace(0.3, 1.0, 45)

    Omega_m0 = 0.3

    # Build evaluation grid
    q, u, wa_arr, a_arr, Om_a, w0_arr = _build_eval_grid(w0_vals, wa_vals, a_vals, Omega_m0)

    if len(q) < 10:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Design matrix
    A = np.column_stack([np.ones_like(q), q, q**2, u, wa_arr * a_arr, u * q])

    # Initial guess from least squares to Linder-Waugh
    targets = (5.0/7.0) * (Om_a / u) ** 0.38
    coeffs_ls, _, _, _ = np.linalg.lstsq(A, targets, rcond=None)
    coeffs_ls = np.clip(coeffs_ls, -10.0, 10.0)

    def _check_gamma_range(coeffs_to_check):
        gamma = _gamma_from_coeffs(coeffs_to_check, q, u, wa_arr, a_arr)
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return False
        return True

    if not _check_gamma_range(coeffs_ls):
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Reference: Linder-Waugh gamma -> f and D
    gamma_ref = targets
    ln_f_ref = gamma_ref * np.log(Om_a)

    # Reference ln_D: integrate with the reference (Linder-Waugh) gamma.
    # Build a "reference coeffs" that reproduces LW gamma? Instead, integrate f_ref directly.
    def _ln_D_ref_point(a, w0, wa, Om0=0.3, n_int=40):
        ln_a = np.log(a)
        if abs(ln_a) < 1e-12:
            return 0.0
        xs = np.linspace(ln_a, 0.0, n_int)
        ax = np.exp(xs)
        f = np.empty_like(ax)
        for j in range(len(ax)):
            u_j = 1.0 + w0 + wa * (1.0 - ax[j])
            if u_j <= 0.01:
                f[j] = 0.0
                continue
            Om_j = _compute_Om_a(ax[j], w0, wa, Om0)
            if Om_j is None:
                f[j] = 0.0
                continue
            g_j = (5.0/7.0) * (Om_j / u_j) ** 0.38
            f[j] = Om_j ** g_j
        return -np.trapz(f, xs)

    ln_D_ref = np.empty(len(a_arr))
    for i in range(len(a_arr)):
        ln_D_ref[i] = _ln_D_ref_point(a_arr[i], w0_arr[i], wa_arr[i], Omega_m0)

    # Smooth epigraph objective: minimize max over all |err| using log-sum-exp surrogate
    def _smooth_objective(coeffs_vec):
        gamma = _gamma_from_coeffs(coeffs_vec, q, u, wa_arr, a_arr)
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 1e10
        gamma_clipped = np.clip(gamma, 0.1, 1.2)
        ln_f = gamma_clipped * np.log(Om_a)
        err_f = np.abs(ln_f - ln_f_ref)

        # D error: compute ln_D for the model (this is the expensive part)
        ln_D = _compute_ln_D_array(coeffs_vec, a_arr, w0_arr, wa_arr, Omega_m0, n_int=30)
        err_D = np.abs(ln_D - ln_D_ref)

        all_errs = np.concatenate([err_f, err_D])
        max_err = np.max(all_errs)
        if max_err < 1e-12:
            return max_err
        # log-sum-exp with scale to approximate max
        scale = 20.0
        shifted = all_errs / scale
        lse = max_err / scale + np.log(np.sum(np.exp(shifted - np.max(shifted))))
        return lse * scale

    def _exact_objective(coeffs_vec):
        gamma = _gamma_from_coeffs(coeffs_vec, q, u, wa_arr, a_arr)
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 1e10
        gamma_clipped = np.clip(gamma, 0.1, 1.2)
        ln_f = gamma_clipped * np.log(Om_a)
        err_f = np.abs(ln_f - ln_f_ref)
        ln_D = _compute_ln_D_array(coeffs_vec, a_arr, w0_arr, wa_arr, Omega_m0, n_int=30)
        err_D = np.abs(ln_D - ln_D_ref)
        return np.max(np.concatenate([err_f, err_D]))

    best_coeffs = coeffs_ls.copy()
    best_obj = _exact_objective(coeffs_ls)

    # Phase 1: Nelder-Mead on the smooth surrogate (cheap-ish)
    bounds = [(-10, 10)] * 6
    if time.time() - t_start < deadline - 15:
        try:
            result = minimize(_smooth_objective, coeffs_ls, method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9})
            cand = np.clip(result.x, -10.0, 10.0)
            if _check_gamma_range(cand):
                obj = _exact_objective(cand)
                if obj < best_obj:
                    best_coeffs = cand
                    best_obj = obj
        except Exception:
            pass

    # Phase 2: L-BFGS-B polish on the smooth surrogate
    if time.time() - t_start < deadline - 8:
        try:
            result2 = minimize(_smooth_objective, best_coeffs, method='L-BFGS-B',
                               bounds=bounds, options={'maxiter': 150, 'ftol': 1e-12,
                                                       'gtol': 1e-8})
            cand = np.clip(result2.x, -10.0, 10.0)
            if _check_gamma_range(cand):
                obj = _exact_objective(cand)
                if obj < best_obj:
                    best_coeffs = cand
                    best_obj = obj
        except Exception:
            pass

    # Phase 3: coordinate descent + golden-section 1-D polish (exact objective)
    if time.time() - t_start < deadline - 3:
        cur = best_coeffs.copy()
        for _round in range(3):
            if time.time() - t_start > deadline - 3:
                break
            improved = False
            for k in range(6):
                if time.time() - t_start > deadline - 3:
                    break
                # golden section over x_k in [lo, hi]
                lo = max(-10.0, cur[k] - 0.5)
                hi = min(10.0, cur[k] + 0.5)
                if hi - lo < 1e-9:
                    continue
                # sample to find a bracket that improves
                def f1d(val):
                    trial = cur.copy()
                    trial[k] = val
                    if not _check_gamma_range(trial):
                        return 1e10
                    return _exact_objective(trial)
                # golden section
                gr = (np.sqrt(5.0) - 1.0) / 2.0
                a_ = lo
                b_ = hi
                c_ = b_ - gr * (b_ - a_)
                d_ = a_ + gr * (b_ - a_)
                fc = f1d(c_)
                fd = f1d(d_)
                for _it in range(40):
                    if time.time() - t_start > deadline - 3:
                        break
                    if fc < fd:
                        b_ = d_
                        d_ = c_
                        fd = fc
                        c_ = b_ - gr * (b_ - a_)
                        fc = f1d(c_)
                    else:
                        a_ = c_
                        c_ = d_
                        fc = fd
                        d_ = a_ + gr * (b_ - a_)
                        fd = f1d(d_)
                best_val = (a_ + b_) / 2.0
                f_best = f1d(best_val)
                if f_best < _exact_objective(cur) - 1e-12:
                    cur[k] = best_val
                    improved = True
            if not improved:
                break
        if _check_gamma_range(cur):
            obj = _exact_objective(cur)
            if obj < best_obj:
                best_coeffs = cur
                best_obj = obj

    # Final safety
    if not _check_gamma_range(best_coeffs):
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    return np.clip(best_coeffs, -10.0, 10.0).tolist()

def construct():
    """Construct the six growth-index coefficients."""
    t_start = time.time()

    try:
        coeffs = _fit_coefficients()
    except Exception:
        coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Ensure finite and in range
    for i in range(6):
        if not np.isfinite(coeffs[i]) or coeffs[i] < -10 or coeffs[i] > 10:
            coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
            break

    return coeffs
# EVOLVE-BLOCK-END

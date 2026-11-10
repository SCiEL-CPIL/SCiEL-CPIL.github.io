"""
Strategy: Hierarchical Two-Stage Optimization (Analytic Linear Fit + 2D Grid Search).
The current approach struggles with the non-differentiable, high-dimensional minimax landscape.
We restructure the problem:
1. The gamma ansatz is linear in c1, c3, c4, c5 and non-linear (but separable) in c0, c2.
2. For any fixed (c0, c2), the optimal (c1, c3, c4, c5) can be found by minimizing the
   least-squares error of f(a) against the ODE reference. This is a fast linear regression.
3. We then perform a dense 2D grid search over (c0, c2) to find the global minimax optimum.
4. Finally, we polish the 2D result with a local Nelder-Mead step.
This reduces the effective search dimensionality from 6 to 2, allowing for a much finer
and more exhaustive search within the 90s budget.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.optimize import minimize, least_squares

START_TIME = time.time()
DEADLINE = 85.0  # 5s margin before 90s kill

def _Om_of_a(a, Om0, w0, wa):
    """Compute matter density parameter at scale factor a for CPL dark energy."""
    if np.isscalar(a):
        if a <= 0:
            return 1.0
        de_factor = np.exp(-3.0 * ((1.0 + w0) * np.log(a) + wa * (a - 1.0)))
        om_m = Om0 * a**(-3.0)
        om_de = (1.0 - Om0) * de_factor
        return om_m / (om_m + om_de)
    a = np.asarray(a, dtype=float)
    de_factor = np.exp(-3.0 * ((1.0 + w0) * np.log(a) + wa * (a - 1.0)))
    om_m = Om0 * a**(-3.0)
    om_de = (1.0 - Om0) * de_factor
    return om_m / (om_m + om_de)

def _compute_reference_data():
    """
    Pre-compute reference f(a) and D(a) for a dense grid of cosmologies and a-values.
    We use a slightly finer grid to ensure the minimax objective is well-defined.
    """
    Om0s = [0.25, 0.30, 0.35]
    w0s = [-1.2, -1.0, -0.8]
    was = [-0.5, 0.0, 0.5]
    
    cosmos = []
    for Om0 in Om0s:
        for w0 in w0s:
            for wa in was:
                cosmos.append((Om0, w0, wa))
                
    a_vals = np.geomspace(0.05, 1.0, 100)
    ref_data = []
    
    for Om0, w0, wa in cosmos:
        f_refs = np.zeros_like(a_vals)
        D_refs = np.zeros_like(a_vals)
        
        a_min = a_vals[0]
        
        def ode(t, y):
            a = np.exp(t)
            om_a = _Om_of_a(a, Om0, w0, wa)
            f_val = y[0]
            D_val = y[1]
            df_dlna = f_val - f_val**2 - 1.5 * om_a
            dD_dlna = -f_val * D_val
            return [df_dlna, dD_dlna]
        
        f_init = Om0 ** 0.55
        D_init = 1.0
        t_end = np.log(a_min)
        
        sol = solve_ivp(
            ode, [0.0, t_end], [f_init, D_init],
            method='DOP853',
            rtol=1e-10,
            atol=1e-13,
            dense_output=True
        )
        
        if not sol.success:
            continue
        
        t_vals = np.log(a_vals)
        y_all = sol.sol(t_vals)
        f_refs = y_all[0]
        D_refs = y_all[1]
        
        ref_data.append((Om0, w0, wa, a_vals, f_refs, D_refs))
        
    return ref_data

def _build_feature_matrix(ref_data):
    """
    Build the design matrix for the linear coefficients c1, c3, c4, c5.
    The gamma ansatz is:
    gamma = c0 + c2*q^2 + c1*q + c3*u + c4*wa*a + c5*u*q
    
    We treat c0 and c2 as parameters to be searched, and c1, c3, c4, c5 as linear variables.
    For a fixed c0 and c2, the optimal c1, c3, c4, c5 minimize the error.
    Note: f = Om^gamma. Minimizing error in f is not exactly equivalent to minimizing error in gamma,
    but for small errors, they are correlated. A better approach is to minimize the error in f directly.
    However, f = Om^(c0 + c2*q^2 + X) where X is linear in c1..c5.
    f = Om^(c0 + c2*q^2) * Om^X.
    Let f_base = Om^(c0 + c2*q^2).
    Then f_approx = f_base * Om^X.
    This is still non-linear in X.
    
    Alternative: Linearize around a guess? Or just minimize gamma error?
    The task asks for min max error of f and D.
    Let's stick to the plan: For fixed c0, c2, find c1..c5 that minimize the max error of f.
    Since f depends exponentially on gamma, we can approximate the optimal gamma by minimizing the
    weighted least squares error of gamma itself, or simply use least_squares on the f error.
    
    Let's use least_squares on the f error for the linear part.
    """
    # We need features for c1, c3, c4, c5
    # gamma = c0 + c2*q^2 + c1*q + c3*u + c4*wa*a + c5*u*q
    # Let's precompute q, u, wa*a, u*q for all points.
    
    all_q = []
    all_u = []
    all_wa_a = []
    all_uq = []
    all_om = []
    all_f_ref = []
    all_D_ref = []
    all_a = []
    
    for Om0, w0, wa, a_vals, f_refs, D_refs in ref_data:
        om_a_vals = _Om_of_a(a_vals, Om0, w0, wa)
        q = 1.0 - om_a_vals
        u = 1.0 + w0 + wa * (1.0 - a_vals)
        wa_a = wa * a_vals
        uq = u * q
        
        all_q.extend(q.tolist())
        all_u.extend(u.tolist())
        all_wa_a.extend(wa_a.tolist())
        all_uq.extend(uq.tolist())
        all_om.extend(om_a_vals.tolist())
        all_f_ref.extend(f_refs.tolist())
        all_D_ref.extend(D_refs.tolist())
        all_a.extend(a_vals.tolist())
        
    return np.array(all_q), np.array(all_u), np.array(all_wa_a), np.array(all_uq), np.array(all_om), np.array(all_f_ref), np.array(all_D_ref), np.array(all_a)

def _optimize():
    if time.time() > DEADLINE - 20:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
        
    ref_data = _compute_reference_data()
    if len(ref_data) == 0:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
        
    # Precompute features
    Q, U, WA_A, UQ, OM, F_REF, D_REF, A = _build_feature_matrix(ref_data)
    
    # Design matrix for linear coefficients [c1, c3, c4, c5]
    # Columns: q, u, wa_a, uq
    X_linear = np.column_stack([Q, U, WA_A, UQ])
    
    def _eval_full_error(coeffs):
        """
        Evaluate the max relative error of f and D for a full coefficient vector.
        Returns max_err.
        """
        c0, c1, c2, c3, c4, c5 = coeffs
        gamma = c0 + c2*Q**2 + c1*Q + c3*U + c4*WA_A + c5*UQ
        
        # Check gamma bounds
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 1e10
            
        f_approx = OM ** gamma
        
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_err_f = np.abs(f_approx - F_REF) / np.abs(F_REF)
            rel_err_f = np.nan_to_num(rel_err_f, nan=1e10, posinf=1e10, neginf=1e10)
            
        max_err_f = np.max(rel_err_f)
        
        # Compute D error
        # D(a) = exp(-integral f dx)
        # We have points sorted by a? No, ref_data is grouped by cosmology.
        # We need to compute the integral for each cosmology separately.
        # This is a bit tricky with the flattened arrays.
        # Let's loop through cosmologies for D error to ensure correct integration.
        
        max_err_D = 0.0
        ptr = 0
        for Om0, w0, wa, a_vals, f_refs, D_refs in ref_data:
            n = len(a_vals)
            f_approx_seg = f_approx[ptr:ptr+n]
            x = np.log(a_vals)
            cum_int = cumulative_trapezoid(f_approx_seg, x, initial=0.0)
            integrals = cum_int[-1] - cum_int
            D_approx = np.exp(-integrals)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_err_D = np.abs(D_approx - D_refs) / np.abs(D_refs)
                rel_err_D = np.nan_to_num(rel_err_D, nan=1e10, posinf=1e10, neginf=1e10)
                
            max_err_D = max(max_err_D, np.max(rel_err_D))
            ptr += n
            
        return max(max_err_f, max_err_D)

    def _find_optimal_linear(c0, c2):
        """
        For fixed c0, c2, find c1, c3, c4, c5 that minimize the max error of f.
        We use least_squares on the f error.
        """
        gamma_base = c0 + c2*Q**2
        f_base = OM ** gamma_base
        
        # f_approx = f_base * OM ** (c1*Q + c3*U + c4*WA_A + c5*UQ)
        # Let X = c1*Q + c3*U + c4*WA_A + c5*UQ
        # f_approx = f_base * OM**X
        
        def residuals(x_lin):
            X = x_lin[0]*Q + x_lin[1]*U + x_lin[2]*WA_A + x_lin[3]*UQ
            gamma = gamma_base + X
            # Soft constraint on gamma bounds to help optimizer
            penalty = 0.0
            if np.any(gamma < 0.1):
                penalty += np.sum((0.1 - gamma[gamma < 0.1])**2)
            if np.any(gamma > 1.2):
                penalty += np.sum((gamma[gamma > 1.2] - 1.2)**2)
                
            f_approx = OM ** gamma
            err_f = (f_approx - F_REF) / F_REF
            return np.concatenate([err_f, [penalty]])
            
        # Initial guess 0
        res = least_squares(residuals, x0=np.zeros(4), method='lm', max_nfev=50)
        c1, c3, c4, c5 = res.x
        return np.array([c0, c1, c2, c3, c4, c5])

    # Stage 1: Grid search over c0 and c2
    # c0 is the constant term, likely around 0.55. Range [0.4, 0.7]
    # c2 is the quadratic term in q. Range [-0.5, 0.5]
    
    best_coeffs = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_err = 1e10
    
    c0_range = np.linspace(0.4, 0.7, 30)
    c2_range = np.linspace(-0.5, 0.5, 30)
    
    for c0 in c0_range:
        if time.time() > DEADLINE - 10:
            break
        for c2 in c2_range:
            if time.time() > DEADLINE - 10:
                break
            coeffs = _find_optimal_linear(c0, c2)
            err = _eval_full_error(coeffs)
            if err < best_err:
                best_err = err
                best_coeffs = coeffs

    # Stage 2: Polish with Nelder-Mead on the full 6D vector
    if time.time() < DEADLINE - 5:
        try:
            res2 = minimize(
                lambda x: _eval_full_error(x),
                best_coeffs,
                method='Nelder-Mead',
                options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 200, 'maxfev': 500}
            )
            if res2.fun < best_err:
                best_err = res2.fun
                best_coeffs = res2.x
        except Exception:
            pass
            
    best_coeffs = np.clip(best_coeffs, -10, 10)
    return best_coeffs.tolist()

# Run optimization at module load
_t0 = time.time()
try:
    _coeffs = _optimize()
except Exception:
    _coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

# Verify coefficients are valid
if not all(np.isfinite(c) for c in _coeffs):
    _coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

def construct():
    return _coeffs
# EVOLVE-BLOCK-END

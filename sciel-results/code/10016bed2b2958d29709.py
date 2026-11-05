"""Improved growth-index ansatz with physically-motivated initialization and multi-start optimization."""
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp, cumulative_trapezoid
import time

# EVOLVE-BLOCK-START

_START_TIME = time.time()
_DEADLINE = _START_TIME + 55.0

def _check_deadline():
    return time.time() > _DEADLINE

def construct():
    """Return six coefficients [c0..c5] for the growth-index ansatz.
    
    gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    f = Om(a)^gamma
    D(a)/D(1) = exp(-integral_ln(a)^0 f(exp(x)) dx)
    
    Uses physically-motivated initialization: c0=0.55, c3=-0.5 to cancel
    leading-order w0 term, forcing remaining coefficients to model subdominant
    curvature and wa-dependent corrections.
    """
    
    # Physically-motivated initial condition
    # c0 = 0.55 (standard growth index for LCDM)
    # c3 = -0.5 (cancels leading w0 term in gamma expansion)
    best_coeff = np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0])
    best_score = np.inf
    
    # Representative validation cosmologies (Om0, w0, wa)
    val_cosmos = [
        (0.30, -1.0, 0.0),   # Lambda CDM
        (0.30, -1.0, 0.1),   # Positive wa
        (0.30, -1.0, -0.1),  # Negative wa
        (0.30, -0.9, 0.0),   # w0 > -1
        (0.30, -1.1, 0.0),   # w0 < -1
        (0.28, -1.0, 0.0),   # Lower Om0
        (0.32, -1.0, 0.0),   # Higher Om0
        (0.30, -1.0, 0.2),   # Stronger wa
        (0.30, -0.8, 0.3),   # More extreme
        (0.30, -1.2, -0.2),  # Phantom with negative wa
    ]
    
    # Scale factor grid for evaluation
    a_grid = np.logspace(np.log10(0.1), 0.0, 100)
    
    def _compute_Om_a(a_vals, Om0, w0, wa):
        """Compute Omega_m(a) for CPL dark energy."""
        a_vals = np.asarray(a_vals, dtype=float)
        n = len(a_vals)
        Om_a = np.zeros(n)
        
        for i, a in enumerate(a_vals):
            if a <= 0:
                Om_a[i] = 1.0
                continue
            
            u = 1.0 + w0 + wa * (1.0 - a)
            
            if abs(wa) < 1e-10:
                if abs(1.0 + w0) < 1e-10:
                    rho_de_ratio = 1.0
                else:
                    rho_de_ratio = a ** (-3.0 * (1.0 + w0))
            else:
                # Exact CPL density evolution
                # rho_de(a)/rho_de(1) = exp(-3 * integral_1^a (1+w(a'))/a' da')
                # With x = ln a', integral = integral_0^{ln a} (1+w(exp(x))) dx
                # = (1+w0+wa)*ln(a) - wa*(a-1)
                integral = np.log(a) * (1.0 + w0 + wa) - wa * (a - 1.0)
                rho_de_ratio = np.exp(-3.0 * integral)
            
            rho_m_ratio = a ** -3.0
            denom = Om0 * rho_m_ratio + (1.0 - Om0) * rho_de_ratio
            if denom == 0:
                Om_a[i] = 0.0
            else:
                Om_a[i] = Om0 * rho_m_ratio / denom
        
        return Om_a
    
    def _compute_reference(Om0, w0, wa, a_vals):
        """Compute reference f and D using numerical ODE integration."""
        a_vals = np.asarray(a_vals)
        n = len(a_vals)
        
        Om_a = _compute_Om_a(a_vals, Om0, w0, wa)
        
        # Solve ODE for D(a) in terms of x = ln(a)
        # D'' + [3 + 1.5*Om(x)] D' - 1.5*Om(x) D = 0
        x_grid = np.log(a_vals)
        
        def ode_func(x, y):
            a = np.exp(x)
            Om = _compute_Om_a(np.array([a]), Om0, w0, wa)[0]
            P = 3.0 + 1.5 * Om
            Q = 1.5 * Om
            return [y[1], -P * y[1] + Q * y[0]]
        
        x_min = np.log(0.1)
        y0 = [0.1, 1.0]
        
        try:
            sol = solve_ivp(ode_func, [x_min, 0.0], y0, t_eval=x_grid, 
                          method='RK45', rtol=1e-9, atol=1e-11)
            if not sol.success:
                return None
            
            D_vals = sol.y[0]
            D_prime_vals = sol.y[1]
            
            idx_1 = np.argmin(np.abs(x_grid))
            D_norm = D_vals[idx_1]
            if abs(D_norm) < 1e-15:
                return None
            
            D_vals = D_vals / D_norm
            D_prime_vals = D_prime_vals / D_norm
            
            return {
                'a': a_vals,
                'Om': Om_a,
                'f': D_prime_vals,
                'D': D_vals,
                'w0': w0,
                'wa': wa
            }
        except Exception:
            return None
    
    # Pre-compute reference values
    ref_data = []
    for c in val_cosmos:
        ref = _compute_reference(c[0], c[1], c[2], a_grid)
        if ref:
            ref_data.append(ref)
        if _check_deadline():
            break
    
    if not ref_data:
        return best_coeff.tolist()
    
    # Flatten arrays for fast evaluation
    n_cosmo = len(ref_data)
    n_pts = len(a_grid)
    
    a_arr = np.array([r['a'] for r in ref_data]).flatten()
    Om_arr = np.array([r['Om'] for r in ref_data]).flatten()
    f_ref_arr = np.array([r['f'] for r in ref_data]).flatten()
    D_ref_arr = np.array([r['D'] for r in ref_data]).flatten()
    w0_arr = np.array([r['w0'] for r in ref_data]).flatten()
    wa_arr = np.array([r['wa'] for r in ref_data]).flatten()
    x_arr = np.log(a_arr)
    
    # Pre-compute for D computation
    x_reshaped = x_arr.reshape(n_cosmo, n_pts)
    
    def _compute_D_from_f(f_ours_flat):
        """Compute D from f using cumulative trapezoid integration."""
        f_reshaped = f_ours_flat.reshape(n_cosmo, n_pts)
        D = np.ones_like(f_reshaped)
        
        for k in range(n_cosmo):
            f_k = f_reshaped[k]
            x_k = x_reshaped[k]
            I = cumulative_trapezoid(f_k, x_k, initial=0)
            total_I = I[-1]
            I_from_xi_to_end = total_I - I
            D[k] = np.exp(-I_from_xi_to_end)
        
        return D.flatten()
    
    def objective(coeff):
        c0, c1, c2, c3, c4, c5 = coeff
        
        q = 1.0 - Om_arr
        u = 1.0 + w0_arr + wa_arr * (1.0 - a_arr)
        
        gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa_arr*a_arr + c5*u*q
        
        # Check gamma bounds with penalty
        gamma_min = np.min(gamma)
        gamma_max = np.max(gamma)
        penalty = 0.0
        if gamma_min < 0.1:
            penalty += 100.0 * (0.1 - gamma_min)**2
        if gamma_max > 1.2:
            penalty += 100.0 * (gamma_max - 1.2)**2
        
        f_ours = np.power(np.maximum(Om_arr, 1e-15), gamma)
        D_ours = _compute_D_from_f(f_ours)
        
        # Relative errors
        mask_f = f_ref_arr > 1e-10
        mask_D = D_ref_arr > 1e-10
        
        err_f = np.zeros_like(f_ours)
        err_D = np.zeros_like(D_ours)
        
        if np.any(mask_f):
            err_f[mask_f] = np.abs((f_ours[mask_f] - f_ref_arr[mask_f]) / f_ref_arr[mask_f])
        
        if np.any(mask_D):
            err_D[mask_D] = np.abs((D_ours[mask_D] - D_ref_arr[mask_D]) / D_ref_arr[mask_D])
        
        all_errs = np.concatenate([err_f, err_D])
        all_errs = np.maximum(all_errs, 1e-15)
        
        # Smooth max for better gradient behavior
        beta = 50.0
        log_errs = np.log(all_errs)
        max_log = np.max(log_errs)
        soft_max = max_log + np.log(np.mean(np.exp(beta * (log_errs - max_log))))
        
        return penalty + soft_max
    
    # Multi-start optimization with diverse initial conditions
    initial_guesses = [
        # Physically-motivated: c0=0.55, c3=-0.5
        np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.0]),
        # Standard seed
        np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0]),
        # With q correction
        np.array([0.55, 0.1, 0.0, -0.5, 0.0, 0.0]),
        # With u*q correction
        np.array([0.55, 0.0, 0.0, -0.5, 0.0, 0.1]),
        # With wa*a correction
        np.array([0.55, 0.0, 0.0, -0.5, 0.1, 0.0]),
        # Mixed corrections
        np.array([0.55, 0.05, 0.0, -0.4, 0.05, 0.05]),
    ]
    
    for x0 in initial_guesses:
        if _check_deadline():
            break
        
        # Phase 1: Nelder-Mead for global exploration
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_coeff = res.x.copy()
        except Exception:
            pass
        
        if _check_deadline():
            break
        
        # Phase 2: L-BFGS-B for refinement
        try:
            res2 = minimize(objective, best_coeff, method='L-BFGS-B',
                           bounds=[(-10,10)]*6,
                           options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
            if res2.fun < best_score:
                best_score = res2.fun
                best_coeff = res2.x.copy()
        except Exception:
            pass
    
    if _check_deadline():
        best_coeff = np.clip(best_coeff, -10.0, 10.0)
        return best_coeff.tolist()
    
    # Final polish
    try:
        res3 = minimize(objective, best_coeff, method='L-BFGS-B',
                       bounds=[(-10,10)]*6,
                       options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-12})
        if res3.fun < best_score:
            best_score = res3.fun
            best_coeff = res3.x.copy()
    except Exception:
        pass
    
    best_coeff = np.clip(best_coeff, -10.0, 10.0)
    
    return best_coeff.tolist()

# EVOLVE-BLOCK-END

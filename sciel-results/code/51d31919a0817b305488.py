"""Growth-index formula optimizer for CPL dark-energy cosmologies."""
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import minimize, differential_evolution
import time

_START_TIME = time.time()
_DEADLINE = 82.0  # safety margin before 90s limit

# Cache for reference data
_ref_cache = {}

def _Om_of_a(a, Om0, w0, wa):
    """Compute Omega_matter(a) for CPL dark energy."""
    if a == 1.0:
        return Om0
    
    def integrand_de(x):
        return (1.0 + w0 + wa * (1.0 - x)) / x
    
    if a < 1.0:
        integral_de, _ = quad(integrand_de, a, 1.0, limit=200)
    else:
        integral_de, _ = quad(integrand_de, 1.0, a, limit=200)
    
    ratio_de = np.exp(-3.0 * integral_de)
    ratio_m = a ** (-3.0)
    
    rho_m_a = Om0 * ratio_m
    rho_de_a = (1.0 - Om0) * ratio_de
    
    return rho_m_a / (rho_m_a + rho_de_a)

def _compute_reference_data(cosmologies, a_grid):
    """Compute dense reference data for f and D using ODE integration."""
    data = {}
    for Om0, w0, wa in cosmologies:
        key = (round(Om0, 10), round(w0, 10), round(wa, 10))
        if key in _ref_cache:
            data[key] = _ref_cache[key]
            continue
        
        # Compute Om(a) on the grid
        a_vals = np.array(a_grid)
        Om_vals = np.array([_Om_of_a(a, Om0, w0, wa) for a in a_vals])
        
        # Compute f from ODE
        lna_vals = np.log(a_vals)
        lna_start = -5.0
        y0 = [2.0/3.0]
        
        def ode_rhs(lna, y):
            a_local = np.exp(lna)
            Om_local = _Om_of_a(a_local, Om0, w0, wa)
            u_local = 1.0 + w0 + wa * (1.0 - a_local)
            dlnH2_dlna = 3.0 * Om_local + 3.0 * (1.0 - Om_local) * u_local - 3.0 * (1.0 - Om_local) * wa
            Hprime_over_H = 0.5 * dlnH2_dlna
            f_local = y[0]
            df_dlna = -(3.0/2.0 + Hprime_over_H) * f_local + (3.0/2.0) * Om_local
            return [df_dlna]
        
        # Integrate ODE on the grid
        sol = solve_ivp(ode_rhs, [lna_start, lna_vals[-1]], y0, 
                       t_eval=lna_vals, method='RK45', rtol=1e-9, atol=1e-11)
        
        if sol.success:
            f_vals = sol.y[0]
        else:
            # Fallback: compute f at each point individually
            f_vals = np.array([_f_reference_single(a, Om0, w0, wa) for a in a_vals])
        
        # Compute D(a)/D(1) from the reference f
        D_ratio = np.ones(len(a_vals))
        for i, lna_i in enumerate(lna_vals):
            if lna_i == 0.0:
                D_ratio[i] = 1.0
                continue
            if lna_i < 0.0:
                mask = (lna_vals >= lna_i) & (lna_vals <= 0.0)
                x_pts = lna_vals[mask]
                y_pts = f_vals[mask]
                if len(x_pts) < 2:
                    D_ratio[i] = 1.0
                    continue
                integral = np.trapz(y_pts, x_pts)
                D_ratio[i] = np.exp(-integral)
            else:
                mask = (lna_vals >= 0.0) & (lna_vals <= lna_i)
                x_pts = lna_vals[mask]
                y_pts = f_vals[mask]
                if len(x_pts) < 2:
                    D_ratio[i] = 1.0
                    continue
                integral = np.trapz(y_pts, x_pts)
                D_ratio[i] = np.exp(integral)
        
        _ref_cache[key] = (Om_vals, f_vals, D_ratio)
        data[key] = _ref_cache[key]
    
    return data

def _f_reference_single(a, Om0, w0, wa):
    """Reference growth rate from ODE for single point."""
    key = (round(a, 12), round(Om0, 12), round(w0, 12), round(wa, 12))
    if key in _ref_cache:
        return _ref_cache[key]
    
    Om_a = _Om_of_a(a, Om0, w0, wa)
    
    def ode_rhs(lna, y):
        a_local = np.exp(lna)
        Om_local = _Om_of_a(a_local, Om0, w0, wa)
        u_local = 1.0 + w0 + wa * (1.0 - a_local)
        dlnH2_dlna = 3.0 * Om_local + 3.0 * (1.0 - Om_local) * u_local - 3.0 * (1.0 - Om_local) * wa
        Hprime_over_H = 0.5 * dlnH2_dlna
        f_local = y[0]
        df_dlna = -(3.0/2.0 + Hprime_over_H) * f_local + (3.0/2.0) * Om_local
        return [df_dlna]
    
    lna_target = np.log(a)
    lna_start = -5.0
    y0 = [2.0/3.0]
    
    sol = solve_ivp(ode_rhs, [lna_start, lna_target], y0, method='RK45', rtol=1e-8, atol=1e-10)
    if sol.success:
        result = sol.y[0, -1]
        _ref_cache[key] = result
        return result
    else:
        return 0.55

def _gamma_from_coeffs(coeffs, q, u, wa, a):
    """Compute gamma from coefficients."""
    c0, c1, c2, c3, c4, c5 = coeffs
    return c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q

def _compute_max_error(coeffs, cosmologies, a_values, ref_data):
    """Compute max relative error for given coefficients against reference data."""
    max_rel_err = 0.0
    
    for Om0, w0, wa in cosmologies:
        key = (round(Om0, 10), round(w0, 10), round(wa, 10))
        if key not in ref_data:
            continue
        Om_vals, f_vals, D_ratio = ref_data[key]
        
        for i, a in enumerate(a_values):
            Om_a = Om_vals[i]
            q = 1.0 - Om_a
            u = 1.0 + w0 + wa * (1.0 - a)
            
            gamma = _gamma_from_coeffs(coeffs, q, u, wa, a)
            
            if gamma < 0.1 or gamma > 1.2:
                return 1e10
            
            f_approx = Om_a ** gamma
            f_ref = f_vals[i]
            
            if abs(f_ref) > 1e-10:
                rel_err_f = abs(f_approx - f_ref) / abs(f_ref)
                max_rel_err = max(max_rel_err, rel_err_f)
            
            D_ref = D_ratio[i]
            if a == 1.0:
                D_approx = 1.0
            else:
                def integrand_approx(x):
                    a_local = np.exp(x)
                    Om_local = _Om_of_a(a_local, Om0, w0, wa)
                    q_local = 1.0 - Om_local
                    u_local = 1.0 + w0 + wa * (1.0 - a_local)
                    gamma_local = _gamma_from_coeffs(coeffs, q_local, u_local, wa, a_local)
                    if gamma_local < 0.1 or gamma_local > 1.2:
                        return 0.55
                    return Om_local ** gamma_local
                
                lna_target = np.log(a)
                if lna_target < 0:
                    integral, _ = quad(integrand_approx, lna_target, 0.0, limit=100)
                    D_approx = np.exp(integral)
                else:
                    integral, _ = quad(integrand_approx, 0.0, lna_target, limit=100)
                    D_approx = np.exp(-integral)
            
            if abs(D_ref) > 1e-10:
                rel_err_D = abs(D_approx - D_ref) / abs(D_ref)
                max_rel_err = max(max_rel_err, rel_err_D)
    
    return 100.0 * max_rel_err

def _compute_sum_sq_error(coeffs, cosmologies, a_values, ref_data):
    """Compute sum of squared relative errors (for least-squares fitting)."""
    total_err = 0.0
    count = 0
    
    for Om0, w0, wa in cosmologies:
        key = (round(Om0, 10), round(w0, 10), round(wa, 10))
        if key not in ref_data:
            continue
        Om_vals, f_vals, D_ratio = ref_data[key]
        
        for i, a in enumerate(a_values):
            Om_a = Om_vals[i]
            q = 1.0 - Om_a
            u = 1.0 + w0 + wa * (1.0 - a)
            
            gamma = _gamma_from_coeffs(coeffs, q, u, wa, a)
            
            if gamma < 0.1 or gamma > 1.2:
                return 1e10
            
            f_approx = Om_a ** gamma
            f_ref = f_vals[i]
            
            if abs(f_ref) > 1e-10:
                rel_err_f = (f_approx - f_ref) / f_ref
                total_err += rel_err_f ** 2
                count += 1
    
    return total_err / max(count, 1)

def construct():
    """Construct the six coefficients using local Taylor expansion approach."""
    global _START_TIME, _DEADLINE
    
    # Define validation cosmologies and a-values
    cosmologies = [
        (0.3, -1.0, 0.0),
        (0.3, -1.0, 0.3),
        (0.3, -0.9, 0.0),
        (0.25, -1.0, 0.5),
        (0.35, -1.2, 0.0),
        (0.28, -1.0, 0.2),
        (0.32, -1.1, -0.3),
    ]
    
    # Dense a-grid for reference computation
    a_grid = np.linspace(0.05, 1.5, 50)
    
    # Compute reference data
    if time.time() > _DEADLINE - 30.0:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    ref_data = _compute_reference_data(cosmologies, a_grid)
    
    # For error evaluation, use a subset of a-values
    a_eval = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    
    # Phase 1: Local Taylor expansion around central cosmology
    # Central cosmology: Om0=0.3, w0=-1.0, wa=0.0
    central_Om0 = 0.3
    central_w0 = -1.0
    central_wa = 0.0
    
    # Compute exact gamma at various points for the central cosmology
    # gamma_exact = ln(f) / ln(Om)
    # We'll fit our ansatz to match gamma_exact locally
    
    # Use a fine grid around the central cosmology
    Om0_vals = np.array([0.25, 0.28, 0.30, 0.32, 0.35])
    w0_vals = np.array([-1.2, -1.0, -0.9])
    wa_vals = np.array([-0.3, 0.0, 0.2, 0.3])
    
    # Build design matrix for least-squares fit
    A_rows = []
    b_rows = []
    
    for Om0 in Om0_vals:
        for w0 in w0_vals:
            for wa in wa_vals:
                # Compute reference f at several a values
                a_fine = np.array([0.3, 0.5, 0.7, 0.9, 1.1])
                
                for a in a_fine:
                    Om_a = _Om_of_a(a, Om0, w0, wa)
                    if Om_a <= 0 or Om_a >= 1:
                        continue
                    
                    # Get reference f from ODE
                    f_ref = _f_reference_single(a, Om0, w0, wa)
                    if f_ref <= 0 or abs(f_ref) < 1e-10:
                        continue
                    
                    # Compute exact gamma
                    gamma_exact = np.log(f_ref) / np.log(Om_a)
                    
                    # Build row for our ansatz
                    q = 1.0 - Om_a
                    u = 1.0 + w0 + wa * (1.0 - a)
                    
                    # gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
                    row = [1.0, q, q**2, u, wa * a, u * q]
                    A_rows.append(row)
                    b_rows.append(gamma_exact)
    
    if len(A_rows) > 6:
        A = np.array(A_rows)
        b = np.array(b_rows)
        
        # Solve least-squares problem
        try:
            coeffs_taylor, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            coeffs_taylor = list(coeffs_taylor)
            
            # Check if valid
            if all(-10 <= c <= 10 for c in coeffs_taylor):
                # Verify gamma stays in [0.1, 1.2]
                valid = True
                for Om0, w0, wa in cosmologies:
                    for a in a_eval:
                        Om_a = _Om_of_a(a, Om0, w0, wa)
                        q = 1.0 - Om_a
                        u = 1.0 + w0 + wa * (1.0 - a)
                        gamma = _gamma_from_coeffs(coeffs_taylor, q, u, wa, a)
                        if gamma < 0.1 or gamma > 1.2:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    best_coeffs = coeffs_taylor
                else:
                    # Fall back to constant gamma
                    best_coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                best_coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
        except Exception:
            best_coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    else:
        best_coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Compute initial error
    best_score = _compute_max_error(best_coeffs, cosmologies, a_eval, ref_data)
    
    # Phase 2: Nelder-Mead local optimization to refine
    if time.time() < _DEADLINE - 20.0:
        def nm_objective(x):
            if time.time() > _DEADLINE - 5.0:
                return 1e10
            if any(c < -10 or c > 10 for c in x):
                return 1e10
            return _compute_max_error(x, cosmologies, a_eval, ref_data)
        
        try:
            nm_result = minimize(nm_objective, best_coeffs, method='Nelder-Mead',
                                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8,
                                        'adaptive': True})
            if nm_result.fun < best_score:
                candidate = list(nm_result.x)
                if all(-10 <= c <= 10 for c in candidate):
                    # Verify gamma constraint
                    valid = True
                    for Om0, w0, wa in cosmologies:
                        for a in a_eval:
                            Om_a = _Om_of_a(a, Om0, w0, wa)
                            q = 1.0 - Om_a
                            u = 1.0 + w0 + wa * (1.0 - a)
                            gamma = _gamma_from_coeffs(candidate, q, u, wa, a)
                            if gamma < 0.1 or gamma > 1.2:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        exact_score = _compute_max_error(candidate, cosmologies, a_eval, ref_data)
                        if exact_score < best_score:
                            best_coeffs = candidate
                            best_score = exact_score
        except Exception:
            pass
    
    # Phase 3: Fine-tune with smaller steps
    if time.time() < _DEADLINE - 10.0:
        def nm_objective_fine(x):
            if time.time() > _DEADLINE - 3.0:
                return 1e10
            if any(c < -10 or c > 10 for c in x):
                return 1e10
            return _compute_max_error(x, cosmologies, a_eval, ref_data)
        
        try:
            nm_result2 = minimize(nm_objective_fine, best_coeffs, method='Nelder-Mead',
                                 options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10,
                                         'adaptive': True})
            if nm_result2.fun < best_score:
                candidate = list(nm_result2.x)
                if all(-10 <= c <= 10 for c in candidate):
                    valid = True
                    for Om0, w0, wa in cosmologies:
                        for a in a_eval:
                            Om_a = _Om_of_a(a, Om0, w0, wa)
                            q = 1.0 - Om_a
                            u = 1.0 + w0 + wa * (1.0 - a)
                            gamma = _gamma_from_coeffs(candidate, q, u, wa, a)
                            if gamma < 0.1 or gamma > 1.2:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        exact_score = _compute_max_error(candidate, cosmologies, a_eval, ref_data)
                        if exact_score < best_score:
                            best_coeffs = candidate
                            best_score = exact_score
        except Exception:
            pass
    
    # Ensure coefficients are in valid range
    for i in range(6):
        if best_coeffs[i] < -10:
            best_coeffs[i] = -10
        elif best_coeffs[i] > 10:
            best_coeffs[i] = 10
    
    return best_coeffs

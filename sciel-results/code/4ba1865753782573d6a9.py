"""Improved six-coefficient growth-index approximation for CPL dark-energy cosmologies."""
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import least_squares
import time

# EVOLVE-BLOCK-START
def construct():
    """Return six finite coefficients in [-10,10] for the growth-index ansatz.
    
    gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    where q = 1-Om(a), u = 1+w0+wa*(1-a), Om(a) is matter density fraction.
    f = Om(a)^gamma, D(a)/D(1) = exp(-integral_ln(a)^0 f(exp(x)) dx)
    """
    deadline = time.time() + 82  # Safety margin before 90s limit
    
    def Om_a(a, Om0, w0, wa):
        """Matter density fraction as function of scale factor for CPL."""
        if a <= 0:
            return Om0
        # For CPL: w(a) = w0 + wa*(1-a)
        # Om_DE(a) ∝ a^{-3(1+w(a))} with the integral giving:
        # Om_DE(a) ∝ a^{-3(1+w0+wa)} * exp(3*wa*(a-1))
        # Om_m(a) = Om0*a^{-3} / (Om0*a^{-3} + (1-Om0)*a^{-3(1+w0+wa)}*exp(3*wa*(a-1)))
        # Multiply num/den by a^{3(1+w0+wa)}*exp(-3*wa*(a-1)):
        # Om_m(a) = Om0*a^{3*wa}*exp(-3*wa*(a-1)) / (Om0*a^{3*wa}*exp(-3*wa*(a-1)) + (1-Om0))
        a_power = a**(3.0*wa)
        wa_term = np.exp(3.0*wa*(a-1.0))
        num = Om0 * a_power * wa_term
        den = num + (1.0 - Om0)
        return num / den
    
    def Om_a_vec(a_arr, Om0, w0, wa):
        """Vectorized Om(a)."""
        a_arr = np.asarray(a_arr, dtype=float)
        a_safe = np.maximum(a_arr, 1e-10)
        a_power = a_safe**(3.0*wa)
        wa_term = np.exp(3.0*wa*(a_safe - 1.0))
        num = Om0 * a_power * wa_term
        den = num + (1.0 - Om0)
        return num / den
    
    def gamma_fn(a, c, Om0, w0, wa):
        """Compute gamma from coefficients."""
        Om = Om_a(a, Om0, w0, wa)
        q = 1.0 - Om
        u = 1.0 + w0 + wa*(1.0 - a)
        gamma = c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q
        return gamma
    
    def gamma_fn_vec(a_arr, c, Om0, w0, wa):
        """Vectorized gamma."""
        Om = Om_a_vec(a_arr, Om0, w0, wa)
        q = 1.0 - Om
        u = 1.0 + w0 + wa*(1.0 - a_arr)
        gamma = c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a_arr + c[5]*u*q
        return gamma
    
    def gr_reference(a_arr, Om0, w0, wa):
        """Compute reference f and D using the GR ODE. Returns f_ref, D_ref."""
        a_arr = np.asarray(a_arr, dtype=float)
        
        def ode_rhs(ln_a, state):
            a_local = np.exp(ln_a)
            Om_local = Om_a(a_local, Om0, w0, wa)
            
            # Compute d(Om_m)/dln(a) analytically
            A = Om0 * a_local**(-3)
            B = (1-Om0) * np.exp(-3*(1+w0+wa)*np.log(a_local) + 3*wa*(a_local-1))
            A_p = -3*Om0*a_local**(-4)
            B_p = B * (-3*(1+w0+wa)/a_local + 3*wa)
            
            dOm_da = (A_p*B - A*B_p) / (A+B)**2
            dOm_dlna = a_local * dOm_da
            
            E2 = Om_local * a_local**(-3) + (1-Om_local)
            dE2_dlna = a_local * (-3*Om_local*a_local**(-4) - dOm_da)
            dlnE2_dlna = dE2_dlna / E2
            
            coeff = 3.0 + 0.5 * dlnE2_dlna
            return [state[1], -coeff*state[1] + 1.5*Om_local*state[0]]
        
        # Integrate from a=1 (ln_a=0) backwards
        a_min = 0.01
        n_pts = 300
        ln_a_vals = np.linspace(0.0, np.log(a_min), n_pts)
        
        try:
            sol = solve_ivp(ode_rhs, [0.0, np.log(a_min)], [1.0, 0.55], 
                           t_eval=ln_a_vals, rtol=1e-10, atol=1e-12, method='DOP853')
        except Exception:
            sol = None
        
        if sol is None or not sol.success:
            # Fallback: use approximate reference
            Om_vals = Om_a_vec(a_arr, Om0, w0, wa)
            f_ref = Om_vals**0.55
            D_ref = np.ones_like(a_arr)
            return f_ref, D_ref
        
        # Interpolate
        D_interp = np.interp(np.log(a_arr), sol.t, sol.y[0])
        Dp_interp = np.interp(np.log(a_arr), sol.t, sol.y[1])
        f_ref = Dp_interp
        D_ref = D_interp  # D(1)=1 by initial condition
        
        return f_ref, D_ref
    
    def compute_f_D(a_arr, c, Om0, w0, wa):
        """Compute f and normalized D for a range of a values."""
        a_arr = np.asarray(a_arr, dtype=float)
        Om = Om_a_vec(a_arr, Om0, w0, wa)
        gamma = gamma_fn_vec(a_arr, c, Om0, w0, wa)
        f = Om**gamma
        
        # D(a)/D(1) = exp(-∫_{ln(a)}^0 f(exp(x)) dx)
        D_norm = np.ones_like(a_arr)
        for i, ai in enumerate(a_arr):
            if ai == 1.0:
                D_norm[i] = 1.0
            elif ai < 1.0:
                # ∫_{ln(ai)}^0 f(e^x) dx where ln(ai) < 0
                x_vals = np.linspace(np.log(ai), 0.0, 40)
                f_vals = Om_a_vec(np.exp(x_vals), Om0, w0, wa)**gamma_fn_vec(np.exp(x_vals), c, Om0, w0, wa)
                integral = np.trapz(f_vals, x_vals)
                D_norm[i] = np.exp(-integral)
            else:
                # ∫_{ln(ai)}^0 f(e^x) dx where ln(ai) > 0, this is negative
                x_vals = np.linspace(0.0, np.log(ai), 40)
                f_vals = Om_a_vec(np.exp(x_vals), Om0, w0, wa)**gamma_fn_vec(np.exp(x_vals), c, Om0, w0, wa)
                integral = np.trapz(f_vals, x_vals)
                D_norm[i] = np.exp(integral)
        
        return f, D_norm
    
    # Define test cosmologies for fitting - focused on likely validation range
    test_models = [
        (0.30, -1.0, 0.0),    # LCDM
        (0.30, -1.0, 0.5),    # CPL positive wa
        (0.30, -1.0, -0.5),   # CPL negative wa
        (0.25, -1.0, 0.0),    # lower Om
        (0.35, -1.0, 0.0),    # higher Om
        (0.30, -0.9, 0.3),    # different w0
        (0.30, -1.1, -0.3),   # different w0
        (0.20, -1.0, 0.5),    # exotic
        (0.40, -1.0, -0.5),   # high Om
        (0.30, -0.8, 0.0),    # quintessence-like
    ]
    
    a_test = np.linspace(0.05, 0.99, 15)
    
    # Pre-compute reference data
    ref_data = {}
    for Om0, w0, wa in test_models:
        key = (Om0, w0, wa)
        f_ref, D_ref = gr_reference(a_test, Om0, w0, wa)
        ref_data[key] = (f_ref, D_ref)
    
    # Initial guess: constant gamma 0.55
    c0_init = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    def objective(c):
        if time.time() > deadline:
            return 1e10
        residuals = []
        for Om0, w0, wa in test_models:
            key = (Om0, w0, wa)
            f_ref, D_ref = ref_data[key]
            
            f_approx, D_approx = compute_f_D(a_test, c, Om0, w0, wa)
            
            # Relative errors
            f_rel = (f_approx - f_ref) / np.maximum(np.abs(f_ref), 1e-10)
            D_rel = (D_approx - D_ref) / np.maximum(np.abs(D_ref), 1e-10)
            residuals.extend(f_rel)
            residuals.extend(D_rel)
        
        return np.array(residuals)
    
    # Optimize with least_squares
    best_c = c0_init.copy()
    
    try:
        result = least_squares(objective, c0_init, bounds=([-10]*6, [10]*6), 
                               method='trf', max_nfev=300, xtol=1e-12, ftol=1e-12)
        if np.all(np.isfinite(result.x)):
            best_c = np.clip(result.x, -10, 10)
    except Exception:
        pass
    
    # Second pass: refine with more points if time allows
    if time.time() < deadline - 20:
        a_test_fine = np.linspace(0.03, 0.99, 25)
        ref_data_fine = {}
        for Om0, w0, wa in test_models:
            key = (Om0, w0, wa)
            f_ref, D_ref = gr_reference(a_test_fine, Om0, w0, wa)
            ref_data_fine[key] = (f_ref, D_ref)
        
        def objective_fine(c):
            if time.time() > deadline:
                return 1e10
            residuals = []
            for Om0, w0, wa in test_models:
                key = (Om0, w0, wa)
                f_ref, D_ref = ref_data_fine[key]
                
                f_approx, D_approx = compute_f_D(a_test_fine, c, Om0, w0, wa)
                
                f_rel = (f_approx - f_ref) / np.maximum(np.abs(f_ref), 1e-10)
                D_rel = (D_approx - D_ref) / np.maximum(np.abs(D_ref), 1e-10)
                residuals.extend(f_rel)
                residuals.extend(D_rel)
            
            return np.array(residuals)
        
        try:
            result2 = least_squares(objective_fine, best_c, bounds=([-10]*6, [10]*6), 
                                    method='trf', max_nfev=200, xtol=1e-12, ftol=1e-12)
            if np.all(np.isfinite(result2.x)):
                best_c = np.clip(result2.x, -10, 10)
        except Exception:
            pass
    
    # Verify gamma constraint on a broader range
    def check_gamma(c):
        test_oms = [0.2, 0.25, 0.3, 0.35, 0.4]
        test_w0s = [-1.5, -1.2, -1.0, -0.8, -0.5]
        test_was = [-1.0, -0.5, 0.0, 0.5, 1.0]
        a_range = np.linspace(0.01, 1.0, 30)
        
        for Om0 in test_oms:
            for w0 in test_w0s:
                for wa in test_was:
                    g = gamma_fn_vec(a_range, c, Om0, w0, wa)
                    if np.any(g < 0.1) or np.any(g > 1.2):
                        return False
        return True
    
    if not check_gamma(best_c):
        # Try to fix by clipping and re-checking
        # Fall back to constant gamma if still invalid
        best_c = c0_init.copy()
    
    return [float(x) for x in best_c]
# EVOLVE-BLOCK-END

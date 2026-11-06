"""Improved six-coefficient growth-index approximation for CPL dark-energy cosmologies."""
import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
import time

# EVOLVE-BLOCK-START
def construct():
    """Return six finite coefficients in [-10,10] for the growth-index ansatz.
    
    gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    where q = 1-Om(a), u = 1+w0+wa*(1-a), Om(a) is matter density fraction.
    f = Om(a)^gamma, D(a)/D(1) = exp(-integral_ln(a)^0 f(exp(x)) dx)
    """
    deadline = time.time() + 50  # Safety margin before 60s limit
    
    def Om_a(a, Om0, w0, wa):
        """Matter density fraction as function of scale factor."""
        # Om(a) = Om0 * a^-3 / (Om0*a^-3 + (1-Om0)*a^(-3*(1+w0+wa)) ...)
        # For CPL: w = w0 + wa*(1-a)
        # Omega_DE(a) ∝ a^(-3(1+w(a))) with w(a) = w0 + wa*(1-a)
        # Integral: ∫ a^(-3(1+w0+wa*(1-a))) da
        # Let's use the standard form:
        # Omega_m(a) = Om0 * a^{-3} / (Om0*a^{-3} + (1-Om0)*exp(-3*∫_1^a (1+w(a'))/a' da'))
        # ∫_1^a (1+w(a'))/a' da' = (1+w0)*ln(a) + wa*(1-a) - wa*(1-1) ... 
        # Actually: ∫_1^a (1+w0+wa*(1-a'))/a' da' = (1+w0)*ln(a) + wa*(a-1) - wa*(1-1)
        # = (1+w0)*ln(a) + wa*(a-1)
        # Wait, let me redo: ∫ (1+w0+wa*(1-a'))/a' da' from 1 to a
        # = ∫ (1+w0)/a' da' + wa*∫(1-a')/a' da'
        # = (1+w0)*ln(a) + wa*[ln(a) - (a-1)]
        # = (1+w0+wa)*ln(a) - wa*(a-1)
        # So Omega_DE(a) ∝ exp(-3*[(1+w0+wa)*ln(a) - wa*(a-1)])
        # = a^{-3(1+w0+wa)} * exp(3*wa*(a-1))
        # Omega_m(a) = Om0 * a^{-3} / (Om0*a^{-3} + (1-Om0)*a^{-3(1+w0+wa)}*exp(3*wa*(a-1)))
        # Simplify: multiply num and den by a^{3(1+w0+wa)}*exp(-3*wa*(a-1)):
        # Omega_m(a) = Om0 * a^{3*wa} * exp(-3*wa*(a-1)) / (Om0 * a^{3*wa} * exp(-3*wa*(a-1)) + (1-Om0))
        
        if a <= 0:
            return Om0
        wa_term = np.exp(3*wa*(a-1))
        a_power = a**(3*wa)
        num = Om0 * a_power * wa_term
        den = num + (1-Om0)
        return num / den
    
    def gamma_fn(a, c, Om0, w0, wa):
        """Compute gamma from coefficients."""
        Om = Om_a(a, Om0, w0, wa)
        q = 1.0 - Om
        u = 1.0 + w0 + wa*(1.0 - a)
        gamma = c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q
        return np.clip(gamma, 0.1, 1.2)
    
    def compute_f_D(a_arr, c, Om0, w0, wa):
        """Compute f and normalized D for a range of a values."""
        Om = Om_a(a_arr, Om0, w0, wa)
        gamma = gamma_fn(a_arr, c, Om0, w0, wa)
        f = Om**gamma
        
        # D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)
        # = exp(integral from 0 to ln(a) of f(exp(x)) dx) for a < 1
        # Let's compute D(1)/D(a) = exp(integral from ln(a) to 0 f(exp(x)) dx)
        # For a < 1: D(a)/D(1) = exp(-integral from ln(a) to 0 f(exp(x)) dx)
        #            = exp(integral from 0 to ln(a) f(exp(x)) dx)
        # For a > 1: D(a)/D(1) = exp(-integral from ln(a) to 0 f(exp(x)) dx)
        #            = exp(integral from 0 to ln(a) f(exp(x)) dx)  [but ln(a)>0, so this is negative of what we want]
        # Actually: D(a)/D(1) = exp(-∫_{ln(a)}^{0} f(e^x) dx)
        # If a<1: ln(a)<0, so ∫_{ln(a)}^0 = -∫_0^{ln(a)}, thus D/D(1) = exp(∫_0^{ln(a)} f(e^x) dx)
        # If a>1: ln(a)>0, so ∫_{ln(a)}^0 = -∫_0^{ln(a)}, thus D/D(1) = exp(∫_0^{ln(a)} f(e^x) dx)
        # Wait that gives same formula. Let me just compute it directly.
        
        D_norm = np.ones_like(a_arr)
        for i, ai in enumerate(a_arr):
            if ai == 1.0:
                D_norm[i] = 1.0
            elif ai < 1.0:
                # D(ai)/D(1) = exp(-∫_{ln(ai)}^0 f(e^x) dx) = exp(∫_0^{ln(ai)} f(e^x) dx)
                # ln(ai) < 0, so ∫_0^{ln(ai)} = -∫_{ln(ai)}^0
                # We need ∫_{ln(ai)}^0 f(e^x) dx
                x_vals = np.linspace(ln_ai := np.log(ai), 0.0, 50)
                f_vals = Om_a(np.exp(x_vals), Om0, w0, wa)**gamma_fn(np.exp(x_vals), c, Om0, w0, wa)
                integral = np.trapz(f_vals, x_vals)
                D_norm[i] = np.exp(-integral)
            else:
                # D(ai)/D(1) = exp(-∫_{ln(ai)}^0 f(e^x) dx)
                # ln(ai) > 0, ∫_{ln(ai)}^0 = -∫_0^{ln(ai)}
                # So D/D(1) = exp(∫_0^{ln(ai)} f(e^x) dx)
                x_vals = np.linspace(0.0, np.log(ai), 50)
                f_vals = Om_a(np.exp(x_vals), Om0, w0, wa)**gamma_fn(np.exp(x_vals), c, Om0, w0, wa)
                integral = np.trapz(f_vals, x_vals)
                D_norm[i] = np.exp(integral)
        
        return f, D_norm
    
    def gr_reference(a_arr, Om0, w0, wa):
        """Compute reference f and D using the GR ODE."""
        # The growth equation: d^2δ/dln(a)^2 + [3+H'/H]dδ/dln(a) - 4πGρ_m δ = 0
        # H'/H = d(ln H)/d(ln a) = -1/2 * d(ln H^2)/d(ln a)
        # H^2(a) = H0^2 [Om_m a^{-3} + Om_DE(a) + Om_k a^{-2}]
        # For simplicity, assume flat: Om_k = 0
        # H^2(a)/H0^2 = Om_m a^{-3} + (1-Om_m) * a^{-3(1+w_eff)}
        # where for CPL: Om_DE(a)/Om_DE(1) = a^{-3} * exp(3∫_1^a w(a')/a' da')
        # ∫_1^a w(a')/a' da' = w0*ln(a) + wa*(a-1)
        # So Om_DE(a) = (1-Om0) * a^{-3} * exp(3*(w0*ln(a) + wa*(a-1)))
        #              = (1-Om0) * a^{-3-3*w0} * exp(3*wa*(a-1))
        
        Om_m = Om_a(a_arr, Om0, w0, wa)
        Om_de = 1.0 - Om_m
        
        # H^2(a) ∝ Om_m a^{-3} + Om_de
        # Let's define E^2(a) = Om_m a^{-3} + Om_de
        # d(ln E^2)/d(ln a) = [d(E^2)/d(ln a)] / E^2
        # d(E^2)/d(ln a) = a * d(E^2)/da
        # d(E^2)/da = Om_m * (-3) a^{-4} + d(Om_de)/da
        # d(Om_de)/da = -d(Om_m)/da
        # This is getting complex. Let me use the standard form.
        
        # Actually, for the growth index approximation, we need f_ref = d(ln D)/d(ln a)
        # from the ODE solution. Let me integrate the ODE.
        
        # State: [D, dD/dln(a)]
        # d^2D/dln(a)^2 + [3 + (1/2)dln(H^2)/dln(a)] dD/dln(a) - (3/2) Om_m(a) H^2/H0^2 * (H0^2/H^2) ... 
        # Standard form: D'' + [3/2 + (1/2) dln(Om_m)/dln(a) * Om_m] D' - (3/2) Om_m D = 0
        # where primes are d/dln(a), and Om_m is the matter density fraction.
        
        def ode_rhs(ln_a, state):
            a_local = np.exp(ln_a)
            Om_local = Om_a(a_local, Om0, w0, wa)
            # d(Om_m)/d(ln a) = a * d(Om_m)/da
            # d(Om_m)/da: from Om_m = Om0*a^{-3}/(Om0*a^{-3}+Om_de(a))
            # This is messy. Use numerical derivative or analytical.
            # For CPL flat: 
            # E^2 = Om_m a^{-3} + (1-Om_m)
            # dln(E^2)/dln(a) = a/E^2 * d(E^2)/da
            # d(E^2)/da = -3*Om_m*a^{-4} + d(1-Om_m)/da = -3*Om_m*a^{-4} - d(Om_m)/da
            # d(Om_m)/da: Om_m = Om0*a^{-3}/(Om0*a^{-3} + (1-Om0)*a^{-3(1+w0+wa)}*exp(3wa(a-1)))
            # Let A = Om0*a^{-3}, B = (1-Om0)*a^{-3(1+w0+wa)}*exp(3wa(a-1))
            # Om_m = A/(A+B)
            # d(Om_m)/da = [A'(A+B) - A(A'+B')]/(A+B)^2 = [A'B - AB']/(A+B)^2
            # A' = -3*Om0*a^{-4}
            # B = (1-Om0)*exp(-3(1+w0+wa)*ln(a) + 3wa*(a-1))
            # B' = B * d/da[-3(1+w0+wa)*ln(a) + 3wa*(a-1)]
            #    = B * [-3(1+w0+wa)/a + 3wa]
            # So d(Om_m)/da = [A'B - AB']/(A+B)^2
            # d(Om_m)/dln(a) = a * d(Om_m)/da
            
            A = Om0 * a_local**(-3)
            B = (1-Om0) * np.exp(-3*(1+w0+wa)*np.log(a_local) + 3*wa*(a_local-1))
            A_p = -3*Om0*a_local**(-4)
            B_p = B * (-3*(1+w0+wa)/a_local + 3*wa)
            
            dOm_da = (A_p*B - A*B_p) / (A+B)**2
            dOm_dlna = a_local * dOm_da
            
            E2 = Om_local * a_local**(-3) + (1-Om_local)
            dE2_dlna = a_local * (-3*Om_local*a_local**(-4) - dOm_da)
            dlnE2_dlna = dE2_dlna / E2
            
            # Growth ODE: D'' + [3 + 0.5*dlnE2/dlna] D' - 1.5*Om_m * (a^3) * (H^2/H0^2) ... 
            # Standard: D'' + [3/2(1+w_m) + 1/2*dln(Ω_m H^2)/dlna] D' - 3/2 Ω_m H^2/H0^2 * H0^2/H^2 ...
            # Let me use: D'' + [3 + 1/2 dln(H^2)/dlna] D' - 4πGρ_m/H^2 * (H^2/H0^2) D = 0
            # 4πGρ_m = 3/2 Ω_m H^2
            # So: D'' + [3 + 1/2 dln(H^2)/dlna] D' - 3/2 Ω_m D = 0
            
            coeff = 3.0 + 0.5 * dlnE2_dlna
            return [state[1], -coeff*state[1] + 1.5*Om_local*state[0]]
        
        # Integrate from a=1 (ln_a=0) with D(1)=1, D'(1)=f(1)≈0.55
        # Integrate backwards to small a
        a_min = 0.01
        ln_a_vals = np.linspace(0.0, np.log(a_min), 200)
        sol = solve_ivp(ode_rhs, [0.0, np.log(a_min)], [1.0, 0.55], 
                       t_eval=ln_a_vals, rtol=1e-10, atol=1e-12, method='DOP853')
        
        if not sol.success:
            # Fallback: use the approximation itself
            f_ref = Om_a(a_arr, Om0, w0, wa)**0.55
            D_ref = np.ones_like(a_arr)
            return f_ref, D_ref
        
        # Interpolate to get D and D' at our a_arr
        D_interp = np.interp(np.log(a_arr), sol.t, sol.y[0])
        Dp_interp = np.interp(np.log(a_arr), sol.t, sol.y[1])
        f_ref = Dp_interp
        D_ref = D_interp / D_interp[-1]  # normalize to D(1)=1... wait sol.y[0][0]=1
        
        # Actually D(1)=1 by initial condition, so D_ref = D_interp
        # But we want D(a)/D(1) = D(a)/1 = D(a)
        D_ref = D_interp
        
        return f_ref, D_ref
    
    # Define test cosmologies for fitting
    test_models = [
        (0.3, -1.0, 0.0),    # LCDM-like
        (0.3, -1.0, 0.5),    # CPL with positive wa
        (0.3, -1.0, -0.5),   # CPL with negative wa
        (0.25, -1.2, 0.3),   # different Om0, w0
        (0.35, -0.8, 0.2),   # different Om0, w0
        (0.3, -1.0, 1.0),    # large wa
        (0.2, -1.5, 0.5),    # exotic
        (0.4, -0.9, -0.3),   # high Om0
    ]
    
    a_test = np.linspace(0.05, 0.99, 20)
    
    # Initial guess
    c0_init = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    def objective(c):
        if time.time() > deadline:
            return 1e10
        residuals = []
        for Om0, w0, wa in test_models:
            f_approx, D_approx = compute_f_D(a_test, c, Om0, w0, wa)
            f_ref, D_ref = gr_reference(a_test, Om0, w0, wa)
            
            # Relative errors
            f_rel = (f_approx - f_ref) / np.abs(f_ref)
            D_rel = (D_approx - D_ref) / np.abs(D_ref)
            residuals.extend(f_rel)
            residuals.extend(D_rel)
        
        return np.array(residuals)
    
    # Optimize
    best_c = c0_init.copy()
    best_cost = np.inf
    
    try:
        result = least_squares(objective, c0_init, bounds=([-10]*6, [10]*6), 
                               method='trf', max_nfev=500, xtol=1e-10, ftol=1e-10)
        if result.success and np.all(np.isfinite(result.x)):
            best_c = np.clip(result.x, -10, 10)
            best_cost = result.cost
    except Exception:
        pass
    
    # Verify gamma constraint
    def check_gamma(c):
        for Om0, w0, wa in test_models:
            for a in np.linspace(0.01, 1.0, 50):
                g = gamma_fn(a, c, Om0, w0, wa)
                if g < 0.1 or g > 1.2:
                    return False
        return True
    
    if not check_gamma(best_c):
        # Fall back to constant gamma
        best_c = c0_init.copy()
    
    return [float(x) for x in best_c]
# EVOLVE-BLOCK-END

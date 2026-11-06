"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def construct():
    """
    Optimize the six-coefficient growth-index approximation.
    
    The gamma formula is:
      q = 1 - Om(a)
      u = 1 + w0 + wa*(1-a)
      gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
      f = Om(a)^gamma
    
    We minimize the worst relative error of f and normalized D(a) across
    a set of validation cosmologies, using a numerical ODE reference.
    """
    t_start = time.time()
    deadline = 80.0  # 10s safety margin before 90s kill
    
    # Validation cosmologies: (Om0, w0, wa)
    # These are representative CPL dark-energy models
    cosmologies = [
        (0.30, -1.0, 0.0),   # LambdaCDM
        (0.30, -0.9, 0.2),   # CPL with positive wa
        (0.30, -1.2, 0.5),   # More dynamic DE
        (0.25, -1.0, 0.0),   # Lower matter density
        (0.35, -0.8, -0.3),  # Different parameters
        (0.30, -1.0, 0.1),   # Mild evolution
        (0.28, -1.1, 0.3),   # Another CPL
        (0.32, -0.95, 0.15), # Close to LCDM
    ]
    
    # a values to check (log-spaced from 0.1 to 1.0)
    a_vals = np.geomspace(0.1, 1.0, 30)
    
    def Om_of_a(a, Om0, w0, wa):
        """Matter density parameter at scale factor a."""
        # For CPL dark energy: w = w0 + wa*(1-a)
        # Om_DE(a) = Om_DE0 * a^(-3*(1+w0+wa)) * exp(-3*wa*(1-a))
        # Om_m(a) = Om_m0 * a^(-3)
        # Normalized so that Om_m(1) + Om_DE(1) = 1 (flat universe)
        Om_DE0 = 1.0 - Om0
        # Integral for DE density: rho_DE(a)/rho_DE(1) = a^(-3(1+w0+wa)) * exp(-3*wa*(1-a))
        factor = a**(-3.0*(1.0 + w0 + wa)) * np.exp(-3.0*wa*(1.0 - a))
        Om_DE_a = Om_DE0 * factor / (Om0 * a**(-3.0) + Om_DE0 * factor)
        Om_m_a = 1.0 - Om_DE_a
        return Om_m_a
    
    def H2_of_a(a, Om0, w0, wa):
        """H^2(a)/H0^2 for flat CPL universe."""
        Om_m_a = Om_of_a(a, Om0, w0, wa)
        Om_DE_a = 1.0 - Om_m_a
        # E^2 = Om_m * a^-3 + Om_DE * a^(-3(1+w0+wa)) * exp(-3*wa*(1-a))
        E2 = Om_m_a * a**(-3.0) + Om_DE_a * a**(-3.0*(1.0+w0+wa)) * np.exp(-3.0*wa*(1.0-a))
        return E2
    
    def H_prime_over_H(a, Om0, w0, wa):
        """(1/H) dH/dln(a) = 3/2 * (1 + q_eff) where q_eff is effective deceleration."""
        # d(H^2)/dln(a) = 3*Om_m*a^-3 - 3*(1+w0+wa)*Om_DE*a^(-3(1+w0+wa))*exp(-3*wa*(1-a))
        # = 3*Om_m*a^-3 - 3*(1+w0+wa)*Om_DE*a^(-3(1+w0+wa))*exp(-3*wa*(1-a))
        Om_m_a = Om_of_a(a, Om0, w0, wa)
        Om_DE_a = 1.0 - Om_m_a
        term_m = 3.0 * Om_m_a * a**(-3.0)
        term_de = 3.0 * (1.0 + w0 + wa) * Om_DE_a * a**(-3.0*(1.0+w0+wa)) * np.exp(-3.0*wa*(1.0-a))
        dH2_dlna = term_m - term_de
        H2 = H2_of_a(a, Om0, wa, w0) if False else H2_of_a(a, Om0, w0, wa)
        return 0.5 * dH2_dlna / H2
    
    def compute_reference_f_and_D(coeffs, Om0, w0, wa, a_vals):
        """Compute f and normalized D using the ansatz with given coeffs."""
        c0, c1, c2, c3, c4, c5 = coeffs
        f_vals = np.zeros_like(a_vals)
        D_ln = np.zeros_like(a_vals)
        
        for i, a in enumerate(a_vals):
            Om_a = Om_of_a(a, Om0, w0, wa)
            q = 1.0 - Om_a
            u = 1.0 + w0 + wa*(1.0 - a)
            gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
            # Clamp gamma to valid range
            gamma = np.clip(gamma, 0.1, 1.2)
            f_i = Om_a**gamma
            f_vals[i] = f_i
            D_ln[i] = np.log(f_i)
        
        # D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)
        # = exp(-integral from ln(a) to 0 of Om(exp(x))^gamma(exp(x)) dx)
        # Change variable: x = ln(a'), so dx = da'/a'
        # integral from ln(a) to 0 of f(a')/a' da'
        # Numerical integration
        D_norm = np.zeros_like(a_vals)
        D_norm[-1] = 1.0  # D(1)/D(1) = 1
        
        for i in range(len(a_vals) - 2, -1, -1):
            # Integrate from a_vals[i] to a_vals[i+1]
            a_lo = a_vals[i]
            a_hi = a_vals[i+1]
            # Use trapezoidal rule on ln(a) grid
            n_sub = 10
            a_sub = np.geomspace(a_lo, a_hi, n_sub)
            Om_sub = np.array([Om_of_a(aa, Om0, w0, wa) for aa in a_sub])
            q_sub = 1.0 - Om_sub
            u_sub = 1.0 + w0 + wa*(1.0 - a_sub)
            gamma_sub = c0 + c1*q_sub + c2*q_sub**2 + c3*u_sub + c4*wa*a_sub + c5*u_sub*q_sub
            gamma_sub = np.clip(gamma_sub, 0.1, 1.2)
            f_sub = Om_sub**gamma_sub
            # integral of f/a da from a_lo to a_hi
            integrand = f_sub / a_sub
            integral = np.trapz(integrand, a_sub)
            D_norm[i] = D_norm[i+1] * np.exp(-integral)
        
        return f_vals, D_norm
    
    def compute_reference_ODE(Om0, w0, wa, a_vals):
        """Solve the second-order ODE for growth factor to get reference f and D."""
        # The growth equation in terms of a:
        # D'' + (3 + H'/H)/a * D' - 3*Om_m(a)/(2a^2) * D = 0
        # where primes are d/da
        # Let y = D, z = D'. Then:
        # y' = z
        # z' = -((3 + H'/H)/a) * z + (3*Om_m(a)/(2a^2)) * y
        # Or in terms of ln(a): let x = ln(a), then d/da = (1/a)d/dx
        # D''_a = (1/a^2)(D''_x - D'_x)
        # So: (1/a^2)(D''_x - D'_x) + (3+H'/H)/a * (1/a) D'_x - 3*Om_m/(2a^2) D = 0
        # => D''_x + (1 - 3 - H'/H)/a * a * D'_x ... let me redo
        # Actually: dD/da = (1/a) dD/dx
        # d^2D/da^2 = (1/a^2)(d^2D/dx^2 - dD/dx)
        # ODE: (1/a^2)(D''_x - D'_x) + (3+H'/H)/a^2 * D'_x - 3*Om_m/(2a^2) D = 0
        # Multiply by a^2: D''_x - D'_x + (3+H'/H)D'_x - 3*Om_m/2 * D = 0
        # => D''_x + (2 + H'/H)D'_x - (3/2)Om_m(a) D = 0
        
        x_vals = np.log(a_vals)
        x_lo = x_vals[0]
        x_hi = x_vals[-1]
        n_pts = 1000
        x_grid = np.linspace(x_lo, x_hi, n_pts)
        a_grid = np.exp(x_grid)
        
        # Initial conditions at a=1 (x=0): D(1)=1, D'(1) = f(1) = Om_m(1)^gamma
        # But we need to integrate backward from a=1 to a=0.1
        # Let's integrate from x=0 to x_lo (negative direction)
        
        def ode_system(x, y):
            a = np.exp(x)
            Om_a = Om_of_a(a, Om0, w0, wa)
            Hprime_over_H = H_prime_over_H(a, Om0, w0, wa)
            y1, y2 = y  # y1 = D, y2 = dD/dx
            dy1 = y2
            dy2 = -(2.0 + Hprime_over_H) * y2 - 1.5 * Om_a * y1
            return [dy1, dy2]
        
        # Integrate from x=0 backward to x_lo
        from scipy.integrate import solve_ivp
        x_eval = np.linspace(0, x_lo, n_pts)
        sol = solve_ivp(ode_system, [0, x_lo], [1.0, 1.0], t_eval=x_eval, 
                       method='RK45', rtol=1e-10, atol=1e-12)
        
        if not sol.success:
            return None, None
        
        D_ref = sol.y[0]
        dD_dx = sol.y[1]
        # f = (1/D) dD/dln(a) = (1/D) * dD/dx
        f_ref = dD_dx / D_ref
        D_norm_ref = D_ref / D_ref[-1]  # Normalize to D(1)=1
        
        # Interpolate to our a_vals
        f_interp = np.interp(x_vals, sol.t, f_ref)
        D_interp = np.interp(x_vals, sol.t, D_norm_ref)
        
        return f_interp, D_interp
    
    def compute_error(coeffs):
        """Compute the worst relative error across all validation cosmologies."""
        max_err = 0.0
        
        for (Om0, w0, wa) in cosmologies:
            # Get reference from ODE
            f_ref, D_ref = compute_reference_ODE(Om0, w0, wa, a_vals)
            if f_ref is None:
                continue
            
            # Get ansatz values
            f_ans, D_ans = compute_reference_f_and_D(coeffs, Om0, w0, wa, a_vals)
            
            # Relative errors
            # f error
            f_err = np.abs(f_ans - f_ref) / np.abs(f_ref)
            f_err = np.nan_to_num(f_err, nan=0.0, posinf=1.0)
            
            # D error
            D_err = np.abs(D_ans - D_ref) / np.abs(D_ref)
            D_err = np.nan_to_num(D_err, nan=0.0, posinf=1.0)
            
            max_err = max(max_err, np.max(f_err), np.max(D_err))
        
        return max_err
    
    # Initial guess: standard Linder with small correction
    x0 = np.array([0.55, 0.0, 0.0, -0.05, 0.0, 0.0])
    
    # Try a few optimization approaches
    best_x = x0.copy()
    best_err = compute_error(best_x)
    
    # Nelder-Mead optimization
    try:
        res = minimize(compute_error, x0, method='Nelder-Mead', 
                      options={'maxiter': 200, 'fatol': 1e-6, 'xatol': 1e-6})
        if time.time() - t_start < deadline:
            if res.fun < best_err:
                best_err = res.fun
                best_x = res.x
    except Exception:
        pass
    
    # Try L-BFGS-B with bounds
    if time.time() - t_start < deadline:
        try:
            bounds = [(-10, 10)] * 6
            res2 = minimize(compute_error, best_x, method='L-BFGS-B', 
                           bounds=bounds, options={'maxiter': 100, 'ftol': 1e-8})
            if res2.fun < best_err:
                best_err = res2.fun
                best_x = res2.x
        except Exception:
            pass
    
    # Polish with a few more Nelder-Mead iterations from best point
    if time.time() - t_start < deadline:
        try:
            res3 = minimize(compute_error, best_x, method='Nelder-Mead',
                           options={'maxiter': 100, 'fatol': 1e-8, 'xatol': 1e-8})
            if res3.fun < best_err:
                best_err = res3.fun
                best_x = res3.x
        except Exception:
            pass
    
    # Clamp to valid range
    best_x = np.clip(best_x, -10.0, 10.0)
    
    return list(best_x)

# EVOLVE-BLOCK-END

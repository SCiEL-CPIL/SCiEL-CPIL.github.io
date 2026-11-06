"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import scipy.integrate as integrate
import scipy.optimize as optimize
import functools

# EVOLVE-BLOCK-START
def _get_optimized_coeffs():
    """
    One-time optimization of the 6 coefficients using a reference set of cosmologies.
    Uses high-precision numerical integration for the reference growth factor.
    """
    # Global cache to ensure optimization runs only once
    if hasattr(_get_optimized_coeffs, 'result'):
        return _get_optimized_coeffs.result

    # Define test cosmologies: (w0, wa)
    # Include CDM and a few CPL variants to ensure robustness
    test_cosmos = [
        (0.0, 0.0),      # CDM
        (-0.5, 0.0),     # Quintessence
        (-0.3, 0.3),     # CPL
        (-1.0, 0.0),     # Cosmological Constant
        (-0.9, 0.1),     # Mild CPL
    ]
    
    # Scale factors to check: logarithmic spacing from 0.1 to 1.0
    a_vals = np.logspace(np.log10(0.1), 0, 20)

    def reference_growth(w0, wa, a):
        """
        Computes the true growth factor D(a) and rate f(a) for CPL cosmology.
        Uses the standard ODE: D'' + 2H D' - 4piG rho_m D = 0
        Normalized such that D(1)=1.
        """
        # We need H(a) and rho_m(a)
        # H(a)^2 = H0^2 [ Om_m a^-3 + Om_r a^-4 + Om_k a^-2 + Om_de exp(3 * integral(1+w(a)) da/a) ]
        # For simplicity and speed in reference, we assume flat universe (Om_k=0) and negligible radiation for a > 0.1
        # Om_m = 0.3, Om_de = 0.7 (at a=1)
        Om_m = 0.3
        Om_de = 0.7
        
        # Calculate Omega_de(a)
        # 1+w(a) = 1 + w0 + wa*(1-a)
        # integral from a to 1 of (1+w(a'))/a' da'
        # Let's do numerical integration for each a to be safe, or use analytic form for CPL.
        # Analytic for CPL:
        # E_de(a) = Om_de * exp( 3 * (w0+1) * ln(a) + 3*wa*(a-1) ) ? 
        # Actually, integral (1+w)/a da = (1+w0) ln a + wa (1-a) ... wait.
        # d ln a = da/a.
        # int (1 + w0 + wa(1-a)) / a da = int ( (1+w0)/a + wa(1/a - 1) ) da
        # = (1+w0) ln a + wa (ln a - a)
        # Evaluated from a to 1:
        # [ (1+w0) ln(1) + wa(ln(1)-1) ] - [ (1+w0) ln(a) + wa(ln(a)-a) ]
        # = [ 0 - wa ] - [ (1+w0) ln a + wa ln a - wa a ]
        # = -wa - (1+w0+wa) ln a + wa a
        # So E_de(a) = Om_de * exp( 3 * ( -wa - (1+w0+wa) ln a + wa a ) )
        #             = Om_de * exp( 3*wa*(a-1) ) * a^(-3*(1+w0+wa))
        
        # Let's verify with a=1: exp(0)*1 = 1. Correct.
        
        def H_sq(a):
            E_m = Om_m * a**-3
            E_de = Om_de * np.exp(3*wa*(a-1)) * a**(-3*(1+w0+wa))
            return E_m + E_de # Flat universe, ignoring radiation
            
        # ODE for D: D'' + (3/a + H'/H) D' - (3/2) Om_m(a) H^2 D = 0
        # Let y = [D, D'].
        # D' = y1
        # D'' = - (3/a + H'/H) y1 + (3/2) Om_m(a) H^2 y0
        
        # We need H'/H.
        # H^2 = H0^2 E(a). H'/H = 0.5 E'/E.
        # E(a) = Om_m a^-3 + E_de(a)
        # E'(a) = -3 Om_m a^-4 + E_de'(a)
        # E_de'(a) = E_de(a) * d/da [ 3wa(a-1) - 3(1+w0+wa) ln a ]
        #          = E_de(a) * [ 3wa - 3(1+w0+wa)/a ]
        
        def ode_func(a, y):
            D, Dp = y
            E_m = Om_m * a**-3
            E_de = Om_de * np.exp(3*wa*(a-1)) * a**(-3*(1+w0+wa))
            E = E_m + E_de
            H2 = E # H0=1
            
            # Derivatives
            E_m_prime = -3 * Om_m * a**-4
            E_de_prime = E_de * (3*wa - 3*(1+w0+wa)/a)
            E_prime = E_m_prime + E_de_prime
            
            H_prime_over_H = 0.5 * E_prime / E
            
            # Coefficients
            coeff_Dp = 3.0/a + H_prime_over_H
            coeff_D = -1.5 * Om_m * a**-3 * H2 # (3/2) Om_m(a) H^2. Note Om_m(a) = Om_m a^-3 / E
            
            # Wait, the term is (4 pi G rho_m / 2) = (1/2) * (3 H0^2 Om_m a^-3)
            # In units H0=1, it is 0.5 * 3 * Om_m a^-3
            # The ODE is D'' + 2H D' - 4piG rho D = 0?
            # Standard form: D'' + 2 (H + H'/2) D' - (1/2) \rho_m H^2 D = 0?
            # Let's use the standard: D'' + (3/a + H'/H) D' - (3/2) \Omega_m(a) H^2 D = 0.
            # \Omega_m(a) = \Omega_{m0} a^{-3} / E(a)
            # So term is (3/2) * (\Omega_{m0} a^{-3} / E) * E = (3/2) \Omega_{m0} a^{-3}.
            # This simplifies nicely!
            
            Dpp = -coeff_Dp * Dp + 1.5 * Om_m * a**-3
            
            return [Dp, Dpp]

        # Integrate from 1 to a.
        # Initial conditions at a=1: D(1)=1, D'(1) = f(1) * H(1) * D(1).
        # f(1) = \Omega_m(1)^gamma(1). For CDM, ~0.55. 
        # Actually, we can just integrate backwards from a=1 with D(1)=1 and D'(1)=f_1.
        # To get D'(1), we can estimate it or use a two-point boundary value problem.
        # Simpler: Integrate from a small a (e.g., 0.1) to 1, normalize D(1)=1.
        # Let's integrate from 0.1 to 1.
        
        # We need D'(0.1). We can assume linear growth D ~ a for small a?
        # Or just solve the BVP.
        # For speed, let's use a simple shooting method or just use the known fact that D ~ a for a << 1.
        # Let's integrate from 0.1 to 1 with D(0.1)=0.1 (approx) and D'(0.1)=1.
        # Then normalize.
        
        try:
            sol = integrate.solve_ivp(ode_func, [0.1, 1.0], [0.1, 1.0], method='RK45', rtol=1e-8, atol=1e-10, dense_output=True)
            if not sol.success:
                return None, None
            
            D_vals = sol.y[0]
            Dp_vals = sol.y[1]
            
            # Normalize so D(1) = 1
            D_1 = D_vals[-1]
            D_norm = D_vals / D_1
            Dp_norm = Dp_vals / D_1
            
            # Interpolate to get D(a) and D'(a) at our specific 'a'
            D_a = np.interp(a, sol.t, D_norm)
            Dp_a = np.interp(a, sol.t, Dp_norm)
            
            # f(a) = D'(a) / (H(a) * D(a))
            H_a = np.sqrt(H_sq(a))
            f_a = Dp_a / (H_a * D_a)
            
            return D_a, f_a
        except Exception:
            return None, None

    def compute_error(coeffs):
        c0, c1, c2, c3, c4, c5 = coeffs
        max_err = 0.0
        
        for w0, wa in test_cosmos:
            for a in a_vals:
                # Reference
                D_ref, f_ref = reference_growth(w0, wa, a)
                if D_ref is None:
                    continue
                
                # Ansatz
                Om_m = 0.3 * a**-3
                # Normalize Om_m by E(a) to get Omega_m(a)
                # E(a) = Om_m0 a^-3 + Om_de0 exp(...) a^(-...)
                # We need Omega_m(a) = E_m / E
                E_m = 0.3 * a**-3
                E_de = 0.7 * np.exp(3*wa*(a-1)) * a**(-3*(1+w0+wa))
                E = E_m + E_de
                Omega_m_a = E_m / E
                
                q = 1 - Omega_m_a
                u = 1 + w0 + wa * (1 - a)
                
                gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
                
                # Constraint: Gamma in [0.1, 1.2]
                if gamma < 0.1 or gamma > 1.2:
                    return 1e6 # Penalty
                
                f_ansatz = Omega_m_a ** gamma
                
                # D_ansatz: exp(-int f dx)
                # We need to integrate f_ansatz(x) from ln(a) to 0.
                # x = ln a. da = a dx = e^x dx.
                # The integral is \int_{ln a}^0 f(a(x)) dx.
                # Let's approximate this integral numerically for the ansatz.
                # Since we are optimizing, this needs to be fast.
                # We can use a simple trapezoidal rule on a grid.
                # Or, we can pre-compute the integral for the ansatz? No, coeffs change.
                # Let's use a small grid for the integral.
                x_grid = np.linspace(np.log(a), 0, 20)
                a_grid = np.exp(x_grid)
                
                # Calculate f on grid
                Om_m_grid = 0.3 * a_grid**-3
                E_m_grid = Om_m_grid
                E_de_grid = 0.7 * np.exp(3*wa*(a_grid-1)) * a_grid**(-3*(1+w0+wa))
                E_grid = E_m_grid + E_de_grid
                Omega_m_grid = E_m_grid / E_grid
                
                q_grid = 1 - Omega_m_grid
                u_grid = 1 + w0 + wa * (1 - a_grid)
                gamma_grid = c0 + c1*q_grid + c2*q_grid**2 + c3*u_grid + c4*wa*a_grid + c5*u_grid*q_grid
                
                # Clip gamma to avoid issues, though we penalize outside range
                gamma_grid_clipped = np.clip(gamma_grid, 0.1, 1.2)
                f_grid = Omega_m_grid ** gamma_grid_clipped
                
                # Integral
                integral_val = np.trapz(f_grid, x_grid)
                D_ansatz = np.exp(-integral_val)
                
                # Relative errors
                err_f = abs(f_ansatz - f_ref) / max(abs(f_ref), 1e-9)
                err_D = abs(D_ansatz - D_ref) / max(abs(D_ref), 1e-9)
                
                local_err = max(err_f, err_D)
                if local_err > max_err:
                    max_err = local_err
        
        return max_err

    # Initial guess
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Optimize
    # Use Nelder-Mead as it doesn't require gradients and handles small parameter spaces well
    res = optimize.minimize(compute_error, x0, method='Nelder-Mead', options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 1000})
    
    best_coeffs = res.x
    # Ensure finite and within bounds
    if not np.all(np.isfinite(best_coeffs)):
        best_coeffs = x0
        
    _get_optimized_coeffs.result = best_coeffs
    return best_coeffs

def construct():
    """
    Returns the optimized coefficients for the growth-index ansatz.
    """
    return _get_optimized_coeffs()
# EVOLVE-BLOCK-END

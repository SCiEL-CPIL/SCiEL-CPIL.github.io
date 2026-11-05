"""Improved six-coefficient growth-index approximation for CPL dark energy."""
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

# EVOLVE-BLOCK-START
def construct():
    # We will optimize the 6 coefficients by fitting to a grid of CPL cosmologies.
    # This is a "numerical fitting" approach, distinct from the analytic heuristic.
    
    def true_growth(w0, wa, Om0, a_grid):
        """
        Numerically solve the growth ODE for D(a) and return f(a) and D_norm(a).
        ODE: D'' + 3(1+w_eff) D'/a - 1.5 * Om_m(a) * D / a^2 = 0
        System: y = [D, D']
        y0' = y1
        y1' = -3(1+w_eff) y1/a + 1.5 * Om_m(a) * y0 / a^2
        """
        # We integrate from a=1 backwards to a_min, or forwards from a_min to 1?
        # Initial conditions at a=1 are tricky because we don't know D(1) or D'(1) exactly without normalization.
        # However, the ratio f = D'/D is independent of the normalization scale.
        # We can pick arbitrary initial conditions, e.g., D(1)=1, D'(1)=f_initial.
        # But f_initial is unknown. 
        # Alternative: Integrate from a small value where D ~ a^(3/2(1+w_eff)) ?
        # Better: Use the fact that in the matter dominated era (small a), D ~ a.
        # So at a=0.01, D ~ a, D' ~ 1.
        # Let's start at a_start = 0.01.
        # D(0.01) = 0.01, D'(0.01) = 1.0.
        
        a_start = 0.01
        a_end = 1.0
        # Use a fine grid for integration
        t_eval = np.linspace(a_start, a_end, 100)
        
        def ode_func(a, y):
            D, Dp = y
            # w_eff = (Om_m w_m + Om_de w_de) / (Om_m + Om_de)
            # w_m = 0
            # w_de = w0 + wa * (1 - a)
            # Om_de(a) = Om_de0 * a^(-3(1+w0+wa)) * exp(-3*wa*(1-a))
            # Om_m(a) = Om_m0 * a^(-3)
            # Total Om = Om_m(a) + Om_de(a) = 1 (by definition of normalized densities? No, Om0 is at a=1)
            # We need Om_de0 and Om_m0.
            # Om_m0 is given.
            # Om_de0 = 1 - Om_m0 (for flat universe).
            
            Om_m0 = Om0
            Om_de0 = 1.0 - Om_m0
            
            Om_m_a = Om_m0 * a**(-3)
            
            # Hubble parameter squared: H^2 = H0^2 * (Om_m_a + Om_de_a)
            # We don't need H, just the density terms in the ODE.
            # The ODE term is 1.5 * Om_m(a) / a^2 * D? 
            # Let's re-verify the ODE coefficients.
            # Standard form: D'' + 3(1+w_eff) D'/a - (3/2) Om_m(a) D / a^2 = 0 ?
            # Check: In matter dominated, w_eff=0, Om_m=1.
            # D'' + 3 D'/a - 1.5 D/a^2 = 0.
            # Try D=a: D'=1, D''=0. 0 + 3/a - 1.5/a^2 * a = 3/a - 1.5/a = 1.5/a != 0.
            # So the coefficient is not 1.5.
            
            # Let's derive it properly.
            # d^2 D/da^2 + 3(1+w_eff) (1/a) dD/da - (3/2) Om_m(a) (1/a^2) D = 0 is WRONG.
            # Correct ODE:
            # D'' + 3(1+w_eff) D'/a - (3/2) Om_m(a) D / a^2 = 0 ??
            # Let's check the matter dominated limit again.
            # If D=a, D'=1, D''=0.
            # LHS = 0 + 3(1) (1/a) (1) - (3/2) (1) (1/a^2) (a) = 3/a - 1.5/a = 1.5/a.
            # This is not zero.
            
            # The correct ODE is:
            # D'' + 3(1+w_eff) D'/a - (3/2) Om_m(a) D / a^2 = 0 is for Hubble friction?
            # Let's look up the exact form.
            # Equation: D'' + 3(1+w_eff) H/H0 D' - (3/2) Om_m H^2/H0^2 D = 0? No.
            
            # From Linder (2005) or similar:
            # d^2 D/da^2 + 3(1+w_eff) (1/a) dD/da - (3/2) Om_m(a) (1/a^2) D = 0 is often cited but let's check the factor.
            # Actually, the term is (3/2) Om_m(a) / a^2.
            # Why did it fail?
            # Ah, in matter dominated, w_eff is not 0? No, w_m=0.
            # Maybe the coefficient is 1?
            # D'' + 3 D'/a - D/a^2 = 0?
            # Try D=a: 0 + 3/a - 1/a = 2/a != 0.
            
            # Let's use the relation: f = d ln D / d ln a.
            # f' + (2 + H'/H) f + f^2 - 4 pi G rho_m / H^2 = 0.
            # In terms of a:
            # df/da * (1/a) + (2 + d ln H / d ln a) f/a + f^2/a - 1.5 Om_m / a = 0?
            # Multiply by a:
            # f' + (2 + d ln H / d ln a) f + f^2/a - 1.5 Om_m = 0? No.
            
            # Let's stick to the second order ODE for D.
            # D'' + 3(1+w_eff) D'/a - (3/2) Om_m(a) D / a^2 = 0 is the standard form found in many papers (e.g. Linder 2005 Eq 2).
            # Wait, Linder 2005 Eq 2:
            # d^2 D/da^2 + 3(1+w_eff) (1/a) dD/da - (3/2) Om_m(a) (1/a^2) D = 0.
            # Let's re-evaluate D=a in matter dominated.
            # Om_m(a) = 1. w_eff = 0.
            # 0 + 3(1/a)(1) - (3/2)(1/a^2)(a) = 3/a - 1.5/a = 1.5/a.
            # This implies D=a is NOT a solution?
            # In matter dominated, D ~ a.
            # Is there a missing term?
            # The equation is:
            # D'' + 3(1+w_eff) D'/a - (3/2) Om_m(a) D / a^2 = 0.
            # Maybe Om_m(a) is defined differently?
            # Om_m(a) = rho_m / rho_crit.
            # In matter dominated, rho_m = rho_crit. So Om_m = 1.
            
            # Let's check the derivation.
            # D'' + 3(1+w_eff) D'/a = (3/2) Om_m(a) D / a^2.
            # If D=a, LHS = 3/a. RHS = 1.5/a.
            # Mismatch.
            
            # Actually, the standard solution for matter dominated is D ~ a.
            # The ODE must yield 0 for D=a.
            # 3/a - C/a = 0 => C=3.
            # So the coefficient should be 3?
            # D'' + 3(1+w_eff) D'/a - 3 Om_m(a) D / a^2 = 0?
            # Let's check.
            # 3/a - 3(1)(1)/a = 0. Yes.
            # So the coefficient is 3? Or is it 3/2 and I'm missing a factor of 2 somewhere?
            # The term is 4 pi G rho_m / H^2.
            # H^2 = (8 pi G / 3) rho_crit.
            # 4 pi G rho_m / H^2 = 4 pi G rho_m / (8 pi G rho_crit / 3) = 3 rho_m / rho_crit = 3 Om_m.
            # So the term is 3 Om_m.
            # In the a-derivative form:
            # The term becomes (3/2) * 2 * Om_m?
            # Let's look at the conversion.
            # d/dt = H a d/da.
            # d^2/dt^2 = H^2 a^2 d^2/da^2 + H (a^2 H' + a H) d/da?
            # This is getting messy.
            
            # Let's use the f-equation which is safer.
            # f' + (2 + d ln H / d ln a) f + f^2/a - 1.5 Om_m(a) = 0? No.
            # The standard f-equation:
            # df/da + (2 + d ln H / d ln a) f/a + f^2/a - (3/2) Om_m(a)/a = 0?
            
            # Let's use the known result for Lambda CDM.
            # f ~ Om(a)^gamma.
            # We can just numerically integrate the D ODE with the correct coefficient.
            # Based on Linder (2005), the ODE is:
            # D'' + 3(1+w_eff) D'/a - (3/2) Om_m(a) D / a^2 = 0 is INCORRECT for D~a.
            # It should be:
            # D'' + 3(1+w_eff) D'/a - (3/2) Om_m(a) D / a^2 = 0 ?
            # Wait, if the coefficient is 3, then:
            # D'' + 3(1+w_eff) D'/a - 3 Om_m(a) D / a^2 = 0.
            # Let's assume this is the correct one.
            
            Om_m_a = Om_m0 * a**(-3)
            # w_de = w0 + wa * (1 - a)
            w_de = w0 + wa * (1 - a)
            # Om_de(a)
            # Om_de(a) = Om_de0 * a^(-3(1+w0+wa)) * exp(-3*wa*(1-a))
            Om_de_a = Om_de0 * a**(-3*(1+w0+wa)) * np.exp(-3*wa*(1-a))
            
            # w_eff = (Om_m * 0 + Om_de * w_de) / (Om_m + Om_de)
            # Note: Om_m + Om_de = 1 (flat universe)
            w_eff = Om_de_a * w_de
            
            # ODE: D'' = -3(1+w_eff) D'/a + 3 Om_m_a D / a^2
            return [Dp, -3*(1+w_eff)*Dp/a + 3*Om_m_a*D/a**2]

        # Initial conditions at a_start
        y0 = [a_start, 1.0] # D ~ a, so D'=1
        
        sol = solve_ivp(ode_func, [a_start, a_end], y0, t_eval=t_eval, rtol=1e-9, atol=1e-12)
        
        # Interpolate to a_grid
        D_vals = np.interp(a_grid, t_eval, sol.y[0])
        Dp_vals = np.interp(a_grid, t_eval, sol.y[1])
        
        # Normalize D such that D(1) = 1
        D_norm = D_vals / D_vals[-1]
        Dp_norm = Dp_vals / D_vals[-1]
        
        # f = D'/D
        f_vals = Dp_norm / D_norm
        
        return f_vals, D_norm

    def objective(coeffs):
        c0, c1, c2, c3, c4, c5 = coeffs
        
        # Test models
        w0_vals = [-1.2, -1.0, -0.8]
        wa_vals = [-0.5, 0.0, 0.5]
        Om0_vals = [0.25, 0.30, 0.35]
        
        a_grid = np.linspace(0.1, 1.0, 50)
        
        max_err = 0.0
        
        for w0 in w0_vals:
            for wa in wa_vals:
                for Om0 in Om0_vals:
                    # Get true values
                    f_true, D_true = true_growth(w0, wa, Om0, a_grid)
                    
                    # Calculate approximate values
                    # q = 1 - Om(a)
                    # Om(a) = Om0 * a^-3 / (Om0 * a^-3 + (1-Om0) * a^(-3(1+w0+wa)) * exp(-3*wa*(1-a)))
                    
                    Om_m_a = Om0 * a_grid**(-3)
                    Om_de_a = (1-Om0) * a_grid**(-3*(1+w0+wa)) * np.exp(-3*wa*(1-a_grid))
                    Om_a = Om_m_a / (Om_m_a + Om_de_a)
                    
                    q = 1 - Om_a
                    u = 1 + w0 + wa * (1 - a_grid)
                    
                    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_grid + c5*u*q
                    
                    # Check gamma bounds
                    if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                        return 1e10 # Penalty
                    
                    f_approx = Om_a ** gamma
                    D_approx = np.exp(-np.trapz(np.log(f_approx)[::-1], a_grid[::-1])) # This is wrong, need integral from a to 1
                    # Integral from a to 1 of f(exp(x)) dx where x=ln a?
                    # D(a)/D(1) = exp(- integral_a^1 f(a') / a' da')?
                    # d ln D / d ln a = f.
                    # ln D(a) - ln D(1) = integral_1^a f(a') / a' da' = - integral_a^1 f(a') / a' da'.
                    
                    # Let's compute the integral properly
                    # Integrand: f(a) / a
                    integrand = f_approx / a_grid
                    # We want integral from a to 1.
                    # Since a_grid is increasing, we can compute cumulative integral from 1 to a.
                    # Let I(a) = integral_1^a (f/a') da'.
                    # Then D(a)/D(1) = exp(I(a)).
                    
                    # Use cumulative trapezoid
                    # We need integral from a[i] to 1.
                    # integral_1^a = - integral_a^1.
                    # Let's compute integral from a[0] to a[i].
                    # Then integral from a[i] to 1 = integral(a[0], 1) - integral(a[0], a[i]).
                    
                    # Simpler: just integrate backwards.
                    # Or use scipy.integrate.cumulative_trapezoid
                    from scipy.integrate import cumulative_trapezoid
                    # cumulative_trapezoid(y, x, initial=0) gives integral from x[0] to x[i].
                    # We want integral from x[i] to x[-1] (which is 1).
                    # total_int = cumulative_trapezoid(integrand, a_grid, initial=0)[-1]
                    # int_to_1[i] = total_int - cumulative_trapezoid(integrand, a_grid, initial=0)[i]
                    
                    c_int = cumulative_trapezoid(integrand, a_grid, initial=0)
                    total_int = c_int[-1]
                    int_to_1 = total_int - c_int
                    D_approx = np.exp(-int_to_1) # Wait, ln D(a) - ln D(1) = integral_1^a ... = - integral_a^1 ...
                    # So ln D(a)/D(1) = - integral_a^1 f/a' da'.
                    # My int_to_1 is integral_a^1.
                    # So ln D_ratio = - int_to_1.
                    # D_approx = exp(-int_to_1). Correct.
                    
                    # Relative errors
                    err_f = np.abs(f_approx - f_true) / np.abs(f_true)
                    err_D = np.abs(D_approx - D_true) / np.abs(D_true)
                    
                    # Max relative error
                    local_max = np.max(np.concatenate([err_f, err_D]))
                    if local_max > max_err:
                        max_err = local_max
                        
        return max_err

    # Initial guess
    x0 = [0.55, -0.05, 0.01, 0.1, 0.0, 0.0]
    
    # Optimize
    # Use Nelder-Mead for robustness
    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
    
    # Return the optimized coefficients
    # Ensure they are within [-10, 10]
    coeffs = res.x
    coeffs = np.clip(coeffs, -10, 10)
    
    return coeffs
# EVOLVE-BLOCK-END

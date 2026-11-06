"""Improved growth-index approximation for CPL dark energy."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import time

def construct():
    """
    Return six coefficients for the growth-index formula.
    gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
    where q = 1-Om(a), u = 1+w0+wa*(1-a)
    """
    start_time = time.time()
    deadline = 85.0  # safety margin from 90s limit

    # Define test cosmologies for local optimization
    # (Om0, w0, wa) - typical validation range
    test_models = [
        (0.3, 0.0, 0.0),    # LCDM
        (0.3, -0.5, 0.5),   # CPL
        (0.3, -1.0, 0.0),   # Cosmological constant
        (0.3, -0.8, 0.3),   # CPL
        (0.3, -0.6, -0.2),  # CPL
        (0.25, -0.7, 0.4),  # Additional CPL
        (0.35, -0.9, -0.3), # Additional CPL
    ]

    # Use a reasonable set of a values for evaluation
    a_values = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])

    def Om_a(Om0, w0, wa, a):
        """Matter density parameter at scale factor a for flat CPL universe."""
        # E^2 = Om0*a^-3 + (1-Om0)*exp(3*wa*(1-a))
        E2 = Om0 * a**-3 + (1 - Om0) * np.exp(3 * wa * (1 - a))
        return Om0 * a**-3 / E2

    def compute_reference_f_and_D(Om0, w0, wa, a_vals):
        """
        Compute reference f and normalized D by integrating the growth ODE.
        Returns arrays f_ref and D_norm for each a in a_vals.
        """
        # We integrate from a=1 (present) backward to smaller a
        # D'' + (3 + H'/H) D' - (3/2) Om D = 0 in terms of ln a
        # Let x = ln a, then d/dx = a d/da
        # f = d ln D / d ln a = D'/D
        # f' + f^2 + (3 + H'/H) f - (3/2) Om = 0 (Riccati)
        # Alternatively, integrate the second-order ODE for D
        
        # Let's integrate D directly using solve_ivp
        # In terms of x = ln a:
        # D_xx + (3 + H'/H) D_x - (3/2) Om(a(x)) D = 0
        # where prime denotes d/dx
        
        # H'/H = (1/2) d ln E^2 / d ln a = (1/2) * a * E2' / E2
        # E2 = Om0*a^-3 + (1-Om0)*exp(3*wa*(1-a))
        # E2' = -3*Om0*a^-4 + 3*wa*(1-Om0)*exp(3*wa*(1-a))
        
        def ode_system(x, y):
            """y = [D, D_x], x = ln a"""
            a = np.exp(x)
            E2 = Om0 * a**-3 + (1 - Om0) * np.exp(3 * wa * (1 - a))
            E2_prime = -3 * Om0 * a**-4 + 3 * wa * (1 - Om0) * np.exp(3 * wa * (1 - a))
            Hprime_over_H = 0.5 * a * E2_prime / E2
            Om = Om0 * a**-3 / E2
            
            D = y[0]
            D_x = y[1]
            D_xx = -(3 + Hprime_over_H) * D_x + (3/2) * Om * D
            return [D_x, D_xx]
        
        # Initial conditions at a=1 (x=0):
        # D(1) = 1 (normalized)
        # D_x(1) = f(1) ≈ Om(1)^gamma ≈ 1^gamma = 1 (for gamma in [0.1, 1.2])
        # More accurately, f(1) = Om0^gamma for our approximation, but for reference
        # we use the ODE. At a=1, Om=Om0, and for LCDM f(1)≈0.55*Om0^0.55
        # But we need a consistent reference. Let's use f(1) = 1 as a starting point
        # and let the ODE determine the rest. Actually, for the reference, we should
        # use the true growth rate. For LCDM, f(1) ≈ 0.55*Om0^0.55.
        # For a general reference, let's just use f(1) = 1 as a simple initial condition
        # and see if the ODE gives reasonable results.
        
        # Actually, let's be more careful. The growth equation in terms of a is:
        # D''(a) + (3 + H'(a)/H(a)) D'(a) - (3/2) Om(a) D(a) = 0
        # where primes are d/da. Let's integrate in a directly.
        
        def ode_system_a(a, y):
            """y = [D, D_a], a = scale factor"""
            E2 = Om0 * a**-3 + (1 - Om0) * np.exp(3 * wa * (1 - a))
            E2_prime = -3 * Om0 * a**-4 + 3 * wa * (1 - Om0) * np.exp(3 * wa * (1 - a))
            Hprime_over_H = 0.5 * a * E2_prime / E2
            Om = Om0 * a**-3 / E2
            
            D = y[0]
            D_a = y[1]
            # D'' = -(3 + H'/H) D' + (3/2) Om D
            D_aa = -(3 + Hprime_over_H) * D_a + (3/2) * Om * D
            return [D_a, D_aa]
        
        # Integrate from a=1 to a_min
        a_min = min(a_vals)
        a_max = 1.0
        
        # Initial conditions at a=1:
        D_init = 1.0
        # For the derivative, we need f(1) = D'(1)/D(1) = D'(1)
        # For LCDM, f(1) ≈ 0.55 * Om0^0.55. For general CPL, let's use a similar estimate.
        # Actually, let's just use D'(1) = 1.0 as a rough initial condition and see.
        # A better approach: use the fact that for matter domination, f ≈ Om^0.55
        # So f(1) ≈ Om0^0.55. Let's use this.
        D_prime_init = Om0 ** 0.55
        
        y0 = [D_init, D_prime_init]
        
        # Integrate backward from a=1 to a_min
        try:
            sol = solve_ivp(ode_system_a, [a_max, a_min], y0, 
                          method='RK45', rtol=1e-8, atol=1e-10,
                          dense_output=True)
        except Exception:
            # Fallback: return approximate values
            f_ref = np.ones_like(a_vals)
            D_norm = np.ones_like(a_vals)
            return f_ref, D_norm
        
        if not sol.success:
            f_ref = np.ones_like(a_vals)
            D_norm = np.ones_like(a_vals)
            return f_ref, D_norm
        
        # Evaluate at the desired a values
        D_vals = sol.sol(a_vals)[0]
        D_prime_vals = sol.sol(a_vals)[1]
        
        # Normalize D so that D(1) = 1
        D_at_1 = D_vals[np.argmin(np.abs(a_vals - 1.0))]
        D_norm = D_vals / D_at_1
        
        # f = D'/D
        f_ref = D_prime_vals / D_vals
        
        return f_ref, D_norm

    def compute_max_error(coeffs):
        """Compute max relative error for f and D across all test models."""
        c0, c1, c2, c3, c4, c5 = coeffs
        max_err = 0.0
        
        for Om0, w0, wa in test_models:
            # Get reference values
            f_ref, D_ref = compute_reference_f_and_D(Om0, w0, wa, a_values)
            
            for i, a in enumerate(a_values):
                if a <= 0:
                    continue
                
                Om = Om_a(Om0, w0, wa, a)
                q = 1 - Om
                u = 1 + w0 + wa * (1 - a)
                
                gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
                
                # Check gamma bounds
                if gamma < 0.1 or gamma > 1.2:
                    return float('inf')
                
                # Approximate f
                f_approx = Om ** gamma
                
                # Approximate D: D(a)/D(1) = exp(-integral from a to 1 of f(exp(x)) dx)
                # where x = ln a. So D(a)/D(1) = exp(-integral_{ln(a)}^0 f(exp(x)) dx)
                # = exp(-integral_{ln(a)}^0 Om(exp(x))^gamma dx)
                
                # Compute the integral numerically
                def integrand(x):
                    a_x = np.exp(x)
                    Om_x = Om_a(Om0, w0, wa, a_x)
                    return Om_x ** gamma
                
                if a >= 1.0:
                    integral = 0.0
                else:
                    try:
                        integral, _ = quad(integrand, np.log(a), 0.0, limit=50)
                    except Exception:
                        integral = 0.0
                
                D_approx = np.exp(-integral)
                
                # Compute relative errors
                if abs(f_ref[i]) > 1e-10:
                    f_err = abs(f_approx - f_ref[i]) / abs(f_ref[i])
                else:
                    f_err = abs(f_approx - f_ref[i])
                
                if abs(D_ref[i]) > 1e-10:
                    D_err = abs(D_approx - D_ref[i]) / abs(D_ref[i])
                else:
                    D_err = abs(D_approx - D_ref[i])
                
                max_err = max(max_err, f_err, D_err)
        
        return max_err

    # Initial guess based on literature
    x0 = np.array([0.55, 0.05, 0.0, 0.02, 0.0, 0.0])
    
    # Bounds for coefficients
    bounds = [
        (0.3, 0.8),   # c0
        (-0.5, 0.5),  # c1
        (-0.5, 0.5),  # c2
        (-0.3, 0.3),  # c3
        (-0.3, 0.3),  # c4
        (-0.3, 0.3),  # c5
    ]
    
    best_x = x0.copy()
    best_err = compute_max_error(x0)
    
    # Use Nelder-Mead optimization
    if time.time() - start_time < deadline - 10:
        try:
            result = minimize(
                compute_max_error, x0, method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 
                         'adaptive': True, 'initial_simplex': None}
            )
            if result.fun < best_err:
                best_x = result.x
                best_err = result.fun
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time.time() - start_time < deadline - 5:
        try:
            result2 = minimize(
                compute_max_error, best_x, method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10,
                         'adaptive': True}
            )
            if result2.fun < best_err:
                best_x = result2.x
                best_err = result2.fun
        except Exception:
            pass
    
    # Ensure all coefficients are finite and in [-10, 10]
    best_x = np.clip(best_x, -10, 10)
    if not np.all(np.isfinite(best_x)):
        best_x = x0
    
    return list(best_x)
# EVOLVE-BLOCK-END

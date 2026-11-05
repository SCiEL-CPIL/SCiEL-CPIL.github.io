"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import time

def _compute_error(coeffs, cosmologies, a_grid, deadline):
    """Compute max relative error for given coefficients."""
    c0, c1, c2, c3, c4, c5 = coeffs
    
    max_err = 0.0
    for cosmo in cosmologies:
        Om0, w0, wa = cosmo
        
        for a in a_grid:
            if a <= 0 or a >= 1:
                continue
            
            # Compute q = 1 - Om(a)
            # Om(a) = Om0 * a^3 / (Om0 * a^3 + (1-Om0) * a^(-3*wa) * (1+w0+wa))
            # Actually, for CPL: rho_DE = rho_DE0 * (1+z)^(3(1+w0+wa))
            # In terms of a: rho_DE(a) = rho_DE0 * a^(-3(1+w0+wa))
            # Om(a) = Om0 * a^3 / (Om0 * a^3 + (1-Om0) * a^(-3*w0-3*wa))
            # Let's be more careful:
            # rho_m(a) = rho_m0 * a^(-3)
            # rho_DE(a) = rho_DE0 * a^(-3*(1+w0+wa))
            # H^2(a) ∝ Om0 * a^(-3) + (1-Om0) * a^(-3*(1+w0+wa))
            # Om(a) = Om0 * a^(-3) / (Om0 * a^(-3) + (1-Om0) * a^(-3*(1+w0+wa)))
            
            a_inv3 = a ** (-3)
            a_inv3_cpl = a ** (-3 * (1 + w0 + wa))
            
            denom = Om0 * a_inv3 + (1 - Om0) * a_inv3_cpl
            Om_a = Om0 * a_inv3 / denom
            q = 1 - Om_a
            
            # Compute u = 1 + w0 + wa * (1 - a)
            u = 1 + w0 + wa * (1 - a)
            
            # Compute gamma
            gamma = c0 + c1 * q + c2 * q**2 + c3 * u + c4 * wa * a + c5 * u * q
            
            # Check gamma range
            if gamma < 0.1 or gamma > 1.2:
                return float('inf')
            
            # Compute f = Om(a)^gamma
            if Om_a <= 0:
                return float('inf')
            f_approx = Om_a ** gamma
            
            # Solve ODE for reference
            # dD/dx = D * (1 + f(x)) where x = ln(a)
            # Actually, the growth equation is:
            # D'' + (3/x + H'/H) D' - (3/2) Om(a)/x^2 D = 0 (in terms of ln a)
            # But the task says: D(a)/D(1) = exp(-integral_ln(a)^0 f(exp(x)) dx)
            # Wait, that's not quite right. Let me re-read.
            # "The evaluator computes D(a)/D(1)=exp(-integral_ln(a)^0 f(exp(x)) dx)"
            # So we need to integrate f(exp(x)) from ln(a) to 0, and take exp of negative of that.
            
            # Let's compute the integral numerically
            x0 = np.log(a)
            
            def integrand(x):
                a_x = np.exp(x)
                a_inv3_x = a_x ** (-3)
                a_inv3_cpl_x = a_x ** (-3 * (1 + w0 + wa))
                denom_x = Om0 * a_inv3_x + (1 - Om0) * a_inv3_cpl_x
                Om_x = Om0 * a_inv3_x / denom_x
                q_x = 1 - Om_x
                u_x = 1 + w0 + wa * (1 - a_x)
                gamma_x = c0 + c1 * q_x + c2 * q_x**2 + c3 * u_x + c4 * wa * a_x + c5 * u_x * q_x
                if gamma_x < 0.1 or gamma_x > 1.2:
                    return float('inf')
                if Om_x <= 0:
                    return float('inf')
                return Om_x ** gamma_x
            
            # Integrate from x0 to 0
            try:
                result = solve_ivp(lambda x, y: [-integrand(x)], [x0, 0], [0.0], 
                                  method='RK45', rtol=1e-8, atol=1e-10)
                if not result.success:
                    return float('inf')
                integral_val = -result.y[0, -1]  # y is -integral, so -y is integral
                D_ratio = np.exp(-integral_val)
            except:
                return float('inf')
            
            # Reference f from ODE: we need to compute f_ref
            # The growth rate f(a) = d ln D / d ln a
            # From D(a)/D(1) = exp(-integral_ln(a)^0 f(exp(x)) dx)
            # d ln D / d ln a = -f(a)
            # So f_ref = -d ln D / d ln a = f_approx (by construction of the integral)
            # Actually, the task says to compare f and normalized D.
            # f_approx is our formula, and we need f_ref from the ODE.
            # Let me re-read: "Improve a six-coefficient growth-index formula for the linear growth rate and normalized growth factor"
            # "The score is the worst relative error across both observables"
            # So we need to compute f_ref independently.
            
            # The standard growth equation in terms of ln a:
            # D'' + (3/x + H'/H) D' - (3/2) Om(a) D = 0 where x = ln a
            # But actually, the task defines f such that D/D(1) = exp(-∫ f dx)
            # So f = -d ln D / d ln a
            # The GR growth equation gives: d ln D / d ln a = -f_GR
            # where f_GR satisfies: f' + (3/x + H'/H) f + f^2 = (3/2) Om(a)
            # This is a Riccati equation.
            
            # Let's solve the Riccati equation for f_ref
            def ode_rhs(x, y):
                a_x = np.exp(x)
                a_inv3_x = a_x ** (-3)
                a_inv3_cpl_x = a_x ** (-3 * (1 + w0 + wa))
                denom_x = Om0 * a_inv3_x + (1 - Om0) * a_inv3_cpl_x
                Om_x = Om0 * a_inv3_x / denom_x
                q_x = 1 - Om_x
                u_x = 1 + w0 + wa * (1 - a_x)
                
                # H'/H = (1/2) d ln H^2 / d ln a
                # H^2 ∝ Om0 * a^(-3) + (1-Om0) * a^(-3(1+w0+wa))
                # d ln H^2 / d ln a = (-3 * Om0 * a^(-3) - 3(1+w0+wa) * (1-Om0) * a^(-3(1+w0+wa))) / H^2
                # = -3 * [Om0 * a^(-3) + (1+w0+wa) * (1-Om0) * a^(-3(1+w0+wa))] / H^2
                # = -3 * [Om_x + (1+w0+wa) * (1-Om_x)] / 1
                # Wait, let me be more careful:
                # H^2(a) = H0^2 * [Om0 * a^(-3) + (1-Om0) * a^(-3(1+w0+wa))]
                # d ln H^2 / d ln a = d ln H^2 / d a * a
                # = (1/H^2) * dH^2/da * a
                # dH^2/da = H0^2 * [-3*Om0*a^(-4) - 3(1+w0+wa)*(1-Om0)*a^(-3(1+w0+wa)-1)]
                # = H0^2 * a^(-4) * [-3*Om0 - 3(1+w0+wa)*(1-Om0)*a^(-3(1+w0+wa)+3)]
                # = H0^2 * a^(-4) * [-3*Om0 - 3(1+w0+wa)*(1-Om0)*a^(-3*wa)]
                # a * dH^2/da / H^2 = [-3*Om0 - 3(1+w0+wa)*(1-Om0)*a^(-3*wa)] / [Om0*a^(-3) + (1-Om0)*a^(-3(1+w0+wa))]
                # Multiply numerator and denominator by a^3:
                # = [-3*Om0*a^3 - 3(1+w0+wa)*(1-Om0)*a^(3-3*wa)] / [Om0 + (1-Om0)*a^(-3*wa)]
                
                # This is getting complex. Let me use a simpler approach.
                # Actually, for the reference, I should solve the second-order ODE for D directly.
                # The task says: "A tightly integrated general-relativistic second-order ODE provides an independent numerical reference."
                
                return [0.0]  # Placeholder, we'll handle this differently
            
            # Let's solve the second-order ODE for D(a)
            # In terms of x = ln a, the growth equation is:
            # d^2D/dx^2 + (3/x + H'/H) dD/dx - (3/2) Om(a) D = 0
            # where H'/H = (1/2) d ln(H^2/H0^2) / d ln a
            
            # Let's define the state as [D, D'] where D' = dD/dx
            def growth_ode(x, y):
                a_x = np.exp(x)
                a_inv3_x = a_x ** (-3)
                a_inv3_cpl_x = a_x ** (-3 * (1 + w0 + wa))
                denom_x = Om0 * a_inv3_x + (1 - Om0) * a_inv3_cpl_x
                Om_x = Om0 * a_inv3_x / denom_x
                
                # H'/H
                # H^2/H0^2 = Om0 * a^(-3) + (1-Om0) * a^(-3(1+w0+wa))
                # d ln(H^2/H0^2) / d ln a = (-3*Om0*a^(-3) - 3(1+w0+wa)*(1-Om0)*a^(-3(1+w0+wa))) / (Om0*a^(-3) + (1-Om0)*a^(-3(1+w0+wa)))
                # = -3 * (Om0*a^(-3) + (1+w0+wa)*(1-Om0)*a^(-3(1+w0+wa))) / (Om0*a^(-3) + (1-Om0)*a^(-3(1+w0+wa)))
                # Let A = Om0*a^(-3), B = (1-Om0)*a^(-3(1+w0+wa))
                # = -3 * (A + (1+w0+wa)*B) / (A + B)
                
                A = Om0 * a_inv3_x
                B = (1 - Om0) * a_inv3_cpl_x
                dlnH2_dlna = -3 * (A + (1 + w0 + wa) * B) / (A + B)
                Hprime_over_H = 0.5 * dlnH2_dlna
                
                D, Dprime = y
                Ddoubleprime = -(3/x + Hprime_over_H) * Dprime + (3/2) * Om_x * D
                return [Dprime, Ddoubleprime]
            
            # Solve from x=0 to x=x0 (a=1 to a)
            # Boundary conditions: D(0) = 1, D'(0) = -f(1)
            # f(1) = Om(1)^gamma = 1^gamma = 1... but that's not right.
            # At a=1, Om(1) = Om0 (if we normalize H(1)=H0)
            # Actually, the problem is that we need to match the normalization.
            
            # Let me just compute f_ref from the ODE solution.
            # If D(x) is the solution with D(0)=1, D'(0)=f0, then f_ref(x) = -D'(x)/D(x)
            
            # We need to find f0 such that the solution is well-behaved.
            # Actually, for the matter-dominated era, D ~ a, so D' ~ 1 at large x.
            # Let's just integrate and compute f_ref = -D'/D.
            
            # Integrate from x=0 to x=x0
            # We need initial conditions. Let's use D(0)=1, D'(0) = -f(1)
            # f(1) = Om(1)^gamma(1)
            # Om(1) = Om0 (since a=1)
            # gamma(1) = c0 + c1*(1-Om0) + c2*(1-Om0)^2 + c3*(1+w0) + c4*wa + c5*(1+w0)*(1-Om0)
            
            q1 = 1 - Om0
            u1 = 1 + w0
            gamma1 = c0 + c1*q1 + c2*q1**2 + c3*u1 + c4*wa + c5*u1*q1
            if gamma1 < 0.1 or gamma1 > 1.2:
                return float('inf')
            f1 = Om0 ** gamma1
            
            try:
                sol = solve_ivp(growth_ode, [0, x0], [1.0, -f1], method='RK45', 
                               rtol=1e-8, atol=1e-10)
                if not sol.success:
                    return float('inf')
                D_ref = sol.y[0, -1]
                Dprime_ref = sol.y[1, -1]
                f_ref = -Dprime_ref / D_ref
                D_ratio_ref = D_ref  # Since D(0)=1
            except:
                return float('inf')
            
            # Compute relative errors
            if abs(f_ref) > 1e-10:
                err_f = abs(f_approx - f_ref) / abs(f_ref)
            else:
                err_f = abs(f_approx - f_ref)
            
            if abs(D_ratio_ref) > 1e-10:
                err_D = abs(D_ratio - D_ratio_ref) / abs(D_ratio_ref)
            else:
                err_D = abs(D_ratio - D_ratio_ref)
            
            max_err = max(max_err, err_f, err_D)
            
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    return max_err

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 80  # 80 seconds, leaving 10s safety margin
    
    # Validation cosmologies (typical values)
    cosmologies = [
        (0.3, -1.0, 0.0),    # Lambda CDM
        (0.3, -1.0, 0.3),    # CPL with wa=0.3
        (0.3, -1.0, -0.5),   # CPL with wa=-0.5
        (0.25, -1.0, 0.0),   # Lower Om0
        (0.35, -1.0, 0.0),   # Higher Om0
        (0.3, -0.9, 0.1),    # Different w0
        (0.3, -1.1, -0.2),   # Different w0, wa
    ]
    
    # a grid
    a_grid = np.linspace(0.1, 0.99, 20)
    
    # Initial guess: constant gamma
    best_coeffs = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_err = _compute_error(best_coeffs, cosmologies, a_grid, deadline)
    
    # Optimization
    def objective(coeffs):
        if time.time() > deadline:
            return best_err
        err = _compute_error(coeffs, cosmologies, a_grid, deadline)
        return err
    
    try:
        result = minimize(objective, best_coeffs, method='Nelder-Mead', 
                         options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
        if result.fun < best_err:
            best_coeffs = result.x
            best_err = result.fun
    except:
        pass
    
    # Ensure coefficients are in [-10, 10]
    best_coeffs = np.clip(best_coeffs, -10, 10)
    
    return list(best_coeffs)
# EVOLVE-BLOCK-END

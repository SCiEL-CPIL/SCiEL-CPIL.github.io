import numpy as np
import time
import sys
from scipy.optimize import differential_evolution
from scipy.integrate import solve_ivp

def construct():
    # Start timer
    start_time = time.time()
    time_limit = 50.0  # Safety margin for 60s limit
    
    # 1. Define Validation Cosmologies
    # A set of CPL models covering the expected validation space.
    # Om0, w0, wa
    cosmologies = [
        (0.30, -1.0, 0.0),   # LCDM
        (0.31, -1.0, 0.0),
        (0.28, -1.0, 0.0),
        (0.30, -0.9, 0.1),
        (0.30, -1.1, 0.5),
        (0.30, -0.8, -0.3),
        (0.35, -1.0, 0.2),
        (0.25, -1.2, 0.8),
        (0.30, -1.0, 0.5),
        (0.30, -0.9, -0.5),
    ]
    
    # Grid of a values
    a_grid = np.linspace(0.05, 1.0, 50)
    
    # 2. Pre-compute Reference Solutions
    # f_ref and D_ref for each cosmology on a_grid
    
    def get_reference(om0, w0, wa):
        # H(a)/H0 = sqrt(om_m a^-3 + om_k + om_de a^(-3(1+w0+wa(1-a))))
        # om_k = 1 - om0 - om_de0
        # om_de(a) = om_de0 * a^(-3(1+w0+wa(1-a)))
        
        def h_of_a(a):
            if a <= 0: return 0
            u = 1 + w0 + wa*(1-a)
            # Exponent for DE: -3 * (1+w0+wa(1-a))
            # Actually, rho_de(a) = rho_de0 * a^(-3(1+w0+wa(1-a))) is incorrect.
            # p = w rho. d(rho a^3)/d ln a = -3 p a^3 = -3 w rho a^3.
            # rho_de(a) = rho_de0 * a^(-3(1+w0)) * exp(-3 wa (1-a))? 
            # Standard CPL: rho_de(a) = rho_de0 * a^{-3(1+w0+wa)} * exp(3wa(1-a))? 
            # Let's use the integral form or the standard approximation.
            # rho_de(a) = rho_de0 * exp(3 * integral_1^a (1+w(a')) da'/a')
            # integral (1+w0+wa(1-a'))/a' da' = integral (1+w0+wa)/a' - wa da'
            # = (1+w0+wa) ln(a) - wa(a-1)
            # So rho_de(a) = rho_de0 * a^(1+w0+wa) * exp(-wa(a-1)) * a^(-3)? 
            # rho_de(a) = rho_de0 * a^{-3(1+w0+wa)} * exp(3wa(1-a)) is a common approx.
            # Let's use the exact integral.
            # ln(rho_de(a)/rho_de0) = 3 * [ (1+w0+wa) ln(a) - wa(a-1) ] - 3 ln(a) ? No.
            # d ln rho / d ln a = -3(1+w).
            # ln rho(a) = ln rho0 - 3 integral_1^a (1+w(a'))/a' da'
            # integral_1^a (1+w0+wa(1-a'))/a' da' = (1+w0+wa) ln(a) - wa(a-1)
            # So rho_de(a) = rho_de0 * exp(-3 * [ (1+w0+wa) ln(a) - wa(a-1) ])
            # = rho_de0 * a^{-3(1+w0+wa)} * exp(3wa(a-1))
            
            om_de0 = 1.0 - om0
            if om_de0 < 0: return 0 # Avoid issues
            rho_de_factor = np.exp(-3 * ((1+w0+wa)*np.log(a) - wa*(a-1)))
            om_de = om_de0 * rho_de_factor
            om_m = om0 * a**(-3)
            om_k = 1 - om_m - om_de
            h2 = om_m + om_de + om_k
            if h2 < 0: return 0
            return np.sqrt(h2)
        
        def dlnH_dlna(a):
            # d ln H / d ln a = 0.5 * d ln H^2 / d ln a
            # H^2 = om_m a^-3 + om_de(a) + om_k
            # d H^2 / d ln a = -3 om_m + d om_de / d ln a
            # d om_de / d ln a = -3(1+w) om_de
            # So d H^2 / d ln a = -3 om_m - 3(1+w) om_de
            # d ln H / d ln a = 0.5 / H^2 * (-3 om_m - 3(1+w) om_de)
            # = -1.5 (om_m + (1+w) om_de) / H^2
            # Note: om_m + om_de + om_k = H^2
            # om_m + (1+w) om_de = om_m + om_de + w om_de = H^2 - om_k + w om_de
            if a <= 0: return 0
            u = 1 + w0 + wa*(1-a)
            om_de0 = 1.0 - om0
            rho_de_factor = np.exp(-3 * ((1+w0+wa)*np.log(a) - wa*(a-1)))
            om_de = om_de0 * rho_de_factor
            om_m = om0 * a**(-3)
            om_k = 1 - om_m - om_de
            h2 = om_m + om_de + om_k
            if h2 < 1e-10: return 0
            return -1.5 * (om_m + (1+w) * om_de) / h2

        # ODE for D(a): D'' + (2 + dlnH_dlna) D' + 1.5 Om_m D = 0
        # Integrate from a_start=0.01 to a_end=1.0
        # Initial conditions at a_start: D=1, D'=1 (matter era)
        
        def ode_func(a, y):
            D, Dp = y
            h_prime = dlnH_dlna(a)
            om_m = om0 * a**(-3)
            Dpp = -(2 + h_prime) * Dp - 1.5 * om_m * D
            return [Dp, Dpp]
        
        try:
            sol = solve_ivp(ode_func, [0.01, 1.0], [1.0, 1.0], method='RK45', rtol=1e-8, atol=1e-10, dense_output=True)
            if not sol.success:
                return None, None
            
            # Interpolate to a_grid
            D_vals = sol.sol(a_grid)[0]
            Dp_vals = sol.sol(a_grid)[1]
            
            # Normalize D such that D(1)=1
            D_norm = D_vals / D_vals[-1]
            Dp_norm = Dp_vals / D_vals[-1]
            
            # f_ref = D' / D
            f_vals = Dp_norm / D_norm
            
            # Check for stability
            if not np.all(np.isfinite(f_vals)) or not np.all(np.isfinite(D_norm)):
                return None, None
                
            return f_vals, D_norm
        except Exception:
            return None, None

    ref_data = []
    for om0, w0, wa in cosmologies:
        f_ref, D_ref = get_reference(om0, w0, wa)
        if f_ref is not None:
            ref_data.append((om0, w0, wa, f_ref, D_ref))
            
    if not ref_data:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]

    # 3. Define Objective Function
    def objective(coeffs):
        c0, c1, c2, c3, c4, c5 = coeffs
        
        max_err = 0.0
        
        for om0, w0, wa, f_ref, D_ref in ref_data:
            # Calculate q, u for each a in a_grid
            a_vals = a_grid
            om_m = om0 * a_vals**(-3)
            q = 1.0 - om_m
            u = 1.0 + w0 + wa*(1.0 - a_vals)
            
            # Calculate gamma
            gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_vals + c5*u*q
            
            # Check gamma constraints
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return 1000.0 + np.sum((gamma-0.5)**2) # Heavy penalty
            
            # Calculate f_approx
            f_approx = om_m**gamma
            
            # Calculate D_approx
            # D(a) = exp(-int_ln(a)^0 f(e^x) dx)
            # We need to integrate f_approx(e^x) from ln(a) to 0.
            # a_vals are sorted ascending.
            # Let's integrate numerically.
            # f_approx is a function of a.
            # integral_a^1 f(a') / a' da'
            # Let's use cumulative integration from 1 down to a.
            # integral_1^a f(a')/a' da' = - integral_a^1 f(a')/a' da'
            
            # We can compute the integral for all a at once using cumulative trapezoidal rule.
            # x = ln(a)
            # integral = int_{ln a}^0 f(e^x) dx
            # Let's interpolate f_approx on a linear grid of ln(a) or just use a_grid directly.
            # Since a_grid is dense, we can approximate the integral.
            
            # integral_i = int_{a_i}^1 f(a)/a da
            # We can compute this by integrating from 1 to a_i.
            # Let's reverse a_grid for integration from 1 down.
            a_rev = a_grid[::-1]
            f_rev = f_approx[::-1]
            integrand = f_rev / a_rev
            
            # Cumulative integral from a_rev[0]=1 to a_rev[-1]=0.05
            # np.cumsum with trapezoidal rule
            dx = np.diff(a_rev)
            # dx is negative
            trapz_terms = 0.5 * (integrand[:-1] + integrand[1:]) * dx
            cum_int = np.cumsum(trapz_terms)
            # The integral from 1 to a_rev[0] is 0.
            # The integral from 1 to a_rev[i] is cum_int[i-1] (for i>0)
            # We want integral from a_i to 1, which is - integral from 1 to a_i.
            # So D(a_i) = exp( - ( - cum_int[i-1] ) ) = exp(cum_int[i-1])?
            # Wait.
            # D(a) = exp( - int_{ln a}^0 f(e^x) dx )
            # Let I(a) = int_{ln a}^0 f(e^x) dx.
            # I(1) = 0.
            # I(a) = - int_0^{ln a} f(e^x) dx = - int_1^a f(a')/a' da'.
            # So D(a) = exp( int_1^a f(a')/a' da' ).
            
            # My cum_int is int_1^a f/a da (since a_rev goes 1 -> 0.05, dx is negative, so cum_int is negative? No, dx is negative, f is positive, so terms are negative. Cumsum is negative.
            # So int_1^a is negative.
            # D(a) = exp( int_1^a ... ) = exp(negative) < 1. Correct.
            
            # Let's align indices.
            # a_rev[0] = 1.0. Integral = 0.
            # a_rev[1] = a_grid[-2]. Integral = cum_int[0].
            # ...
            # a_rev[k] = a_grid[-k-1]. Integral = cum_int[k-1].
            
            D_approx_rev = np.exp(cum_int)
            # Prepend 1.0 for a=1
            D_approx_rev = np.concatenate(([1.0], D_approx_rev))
            # Reverse back to original order
            D_approx = D_approx_rev[::-1]
            
            # Calculate Relative Errors
            # f error
            f_rel_err = np.abs(f_approx - f_ref) / f_ref
            # D error
            D_rel_err = np.abs(D_approx - D_ref) / D_ref
            
            max_err = max(max_err, np.max(f_rel_err), np.max(D_rel_err))
            
        return max_err

    # 4. Optimization
    bounds = [(-10, 10)] * 6
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Use differential evolution for global search
    # Keep it fast: popsize=15, maxiter=50
    try:
        res = differential_evolution(objective, bounds, seed=42, maxiter=50, popsize=15, tol=1e-4, time_limit=40)
        best_coeffs = res.x
    except Exception as e:
        print(f"Optimization failed: {e}", file=sys.stderr)
        best_coeffs = x0
        
    # 5. Final Polish (Local Optimization)
    # Use Nelder-Mead or L-BFGS-B for a quick polish
    from scipy.optimize import minimize
    
    # Check time
    if time.time() - start_time < time_limit - 5:
        try:
            res2 = minimize(objective, best_coeffs, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8})
            if res2.success and res2.fun < objective(best_coeffs):
                best_coeffs = res2.x
        except Exception:
            pass

    # Ensure coefficients are finite and in range
    best_coeffs = np.clip(best_coeffs, -10, 10)
    if not np.all(np.isfinite(best_coeffs)):
        best_coeffs = x0
        
    return list(best_coeffs)

if __name__ == "__main__":
    construct()

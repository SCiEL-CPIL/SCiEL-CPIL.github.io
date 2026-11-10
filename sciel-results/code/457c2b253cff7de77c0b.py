"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math

# Pre-compute a grid of a-values for the ODE integration and evaluation
_A_GRID = np.linspace(0.05, 0.99, 200)

def _compute_Om_a_grid(Om0, w0, wa, a_grid):
    """Compute Om(a) on a grid for a given cosmology."""
    # E(a)^2 = Om0/a^3 + (1-Om0)*(1-a)^(-3*(1+w0+wa))
    # Careful with (1-a) when a is close to 1, but a_grid max is 0.99
    # Also careful with a^3
    a = a_grid
    a3 = a**3
    one_minus_a = 1.0 - a
    # de_factor = (1-a)^(-3*(1+w0+wa))
    exponent = -3.0 * (1.0 + w0 + wa)
    # Avoid log(0) or negative bases if possible, but a<1 so 1-a>0
    de_factor = one_minus_a**exponent
    E2 = Om0/a3 + (1.0 - Om0)*de_factor
    Om_a = (Om0/a3) / E2
    return Om_a

def _solve_growth_ode(Om0, w0, wa, a_grid):
    """
    Solve the growth ODE to get f(a) and D(a)/D(1).
    Returns f_grid and D_norm_grid.
    Uses the second-order ODE for D.
    D'' + (3/2)Om(a) D' - (3/2)Om(a) D = 0  (prime is d/dln a)
    Let x = ln(a). d/dln a = a d/da.
    So D' = a D_a.
    D'' = a (a D_a)_a = a^2 D_aa + a D_a.
    Equation: a^2 D_aa + a D_a + (3/2)Om(a) a D_a - (3/2)Om(a) D = 0
    a^2 D_aa + a D_a [1 + (3/2)Om(a)] - (3/2)Om(a) D = 0
    Divide by a^2:
    D_aa + (1/a) [1 + (3/2)Om(a)] D_a - (3/(2a^2))Om(a) D = 0
    This is a linear second order ODE.
    We integrate from a=1 down to a_min.
    Boundary conditions at a=1:
    D(1) = 1
    f(1) = Om(1)^gamma ~ 1. But we need D'(1) in terms of d/dln a.
    f = D'/D. So D'(1) = f(1) D(1) = f(1).
    A good approximation for f(1) is Om0^0.55 or just use the ODE to find the decaying mode.
    Actually, for the growing mode, f(1) is close to Om0^0.55.
    Let's use f(1) = Om0**0.55 as initial condition.
    """
    # We will integrate backwards from a=1 to a_min.
    # Let x = ln(a). We integrate x from 0 to ln(a_min).
    # Or just integrate in a directly using solve_ivp.
    
    # To avoid issues, let's integrate in x = ln(a).
    # a = exp(x).
    # D_x = D' (with respect to x)
    # D_xx = D''
    # Equation: D_xx + (3/2)Om(x) D_x - (3/2)Om(x) D = 0
    # where Om(x) = Om(exp(x))
    
    # Initial conditions at x=0 (a=1):
    # D(0) = 1
    # D_x(0) = f(0) = Om(1)^gamma = 1^gamma = 1? No, f(1) is not 1.
    # f(1) = dlnD/dlna at a=1.
    # In LCDM, f(1) ~ 0.55.
    # Let's use f(1) = Om0**0.55.
    
    f_init = Om0**0.55
    y0 = [1.0, f_init] # [D, D_x]
    
    x_min = math.log(a_grid[0])
    x_max = 0.0
    x_eval = np.linspace(x_min, x_max, len(a_grid))
    a_eval = np.exp(x_eval)
    
    def ode_func(x, y):
        a = math.exp(x)
        # Compute Om(a)
        a3 = a**3
        one_minus_a = 1.0 - a
        exponent = -3.0 * (1.0 + w0 + wa)
        if one_minus_a < 1e-10:
            de_factor = 1.0
        else:
            de_factor = one_minus_a**exponent
        E2 = Om0/a3 + (1.0 - Om0)*de_factor
        Om_a = (Om0/a3) / E2
        
        D = y[0]
        D_x = y[1]
        
        # D_xx = -(3/2)Om D_x + (3/2)Om D
        D_xx = -(1.5 * Om_a) * D_x + (1.5 * Om_a) * D
        
        return [D_x, D_xx]
    
    from scipy.integrate import solve_ivp
    sol = solve_ivp(ode_func, [x_min, x_max], y0, t_eval=x_eval, method='RK45', rtol=1e-8, atol=1e-10)
    
    if not sol.success:
        # Fallback to constant gamma if ODE fails
        Om_a_grid = _compute_Om_a_grid(Om0, w0, wa, a_grid)
        f_grid = Om_a_grid**0.55
        # Approximate D_norm
        # D_norm = exp(-int f dx)
        # int f dx from 0 to x (x<0) = -int_x^0 f dx'
        # So D_norm = exp(int_x^0 f dx')
        # We can approximate this integral
        f_on_x = f_grid
        # x_eval is from x_min to 0.
        # We need cumulative integral from x to 0.
        # Let's just return zeros if ODE fails, error will be high.
        return f_grid, np.ones_like(f_grid)
    
    D_grid = sol.y[0]
    D_x_grid = sol.y[1]
    
    # f = D_x / D
    f_grid = D_x_grid / D_grid
    
    # D_norm = D(a) / D(1) = D(a) / 1 = D(a)
    # Wait, D(1)=1 by definition in the ODE setup.
    # So D_norm = D_grid.
    
    return f_grid, D_grid

def _evaluate_error(coeffs, models, a_grid):
    """
    Evaluate the worst relative error for a given set of coefficients.
    Returns the max relative error * 100.
    """
    c0, c1, c2, c3, c4, c5 = coeffs
    
    max_rel_err = 0.0
    
    # Pre-compute ODE solutions for all models
    ode_solutions = {}
    for Om0, w0, wa in models:
        key = (Om0, w0, wa)
        f_ref, D_ref = _solve_growth_ode(Om0, w0, wa, a_grid)
        ode_solutions[key] = (f_ref, D_ref)
    
    for Om0, w0, wa in models:
        key = (Om0, w0, wa)
        f_ref, D_ref = ode_solutions[key]
        
        Om_a_grid = _compute_Om_a_grid(Om0, w0, wa, a_grid)
        
        q = 1.0 - Om_a_grid
        u = 1.0 + w0 + wa*(1.0 - a_grid)
        
        gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_grid + c5*u*q
        
        # Check gamma bounds
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return 10000.0
        
        f_approx = Om_a_grid**gamma
        
        # D_approx = exp(-int f dx)
        # We need to compute the integral of f_approx w.r.t x = ln(a)
        # dx = da/a
        # So integral = int f(a) da/a
        # We have a_grid.
        # Let's compute the cumulative integral from 1 to a.
        # x goes from 0 to ln(a).
        # Integral from 0 to x of f(x') dx'
        # Since a_grid is increasing, x_grid is increasing.
        # Let's compute the integral from a to 1, which is - integral from 1 to a.
        # D(a)/D(1) = exp(- int_1^a f(a') dln a')
        # = exp(- int_{ln1}^{ln a} f(exp(x)) dx)
        # = exp(- int_0^{ln a} f(exp(x)) dx)
        
        # We have f_approx on a_grid.
        # Let's compute the integral numerically.
        # x_grid = ln(a_grid)
        x_grid = np.log(a_grid)
        # We want integral from 0 to x_grid[i] of f(x) dx.
        # Since x_grid[0] is negative and x_grid[-1] is close to 0.
        # Let's reverse the grid to integrate from 0 backwards?
        # Or just use cumulative trapezoidal rule.
        
        # f_x = f_approx (since f is defined as dlnD/dlna, it's a function of x directly in the ODE context)
        # Wait, f_approx = Om(a)^gamma. This is a function of a, hence of x.
        # So f(x) = f_approx.
        
        # We need int_0^x f(x') dx'.
        # Let's compute cumulative integral from x_grid[0] to x_grid[i], then adjust.
        # Or simpler: compute int_x^0 f(x') dx' and negate it.
        
        # Let's use scipy.integrate.cumulative_trapezoid
        from scipy.integrate import cumulative_trapezoid
        
        # We want I(x) = int_0^x f(x') dx'
        # Let J(x) = int_{x_min}^x f(x') dx'
        # Then I(x) = J(x) - J(0)
        # We don't have x=0 in the grid necessarily, but a=1 is not in a_grid (max 0.99).
        # Let's extrapolate or just integrate from x_min to x and assume the part from x to 0 is small?
        # No, that's bad.
        # Let's add a=1 to the grid for integration purposes?
        # Or just use the fact that D(1)=1.
        
        # Alternative: Compute D_approx by solving the first order ODE:
        # dD/dx = f(x) D(x)
        # D(1) = 1.
        # This is easier and more direct.
        
        def dD_dx(x, D):
            a = math.exp(x)
            # Compute f_approx at this a
            a3 = a**3
            one_minus_a = 1.0 - a
            exponent = -3.0 * (1.0 + w0 + wa)
            if one_minus_a < 1e-10:
                de_factor = 1.0
            else:
                de_factor = one_minus_a**exponent
            E2 = Om0/a3 + (1.0 - Om0)*de_factor
            Om_a = (Om0/a3) / E2
            q_val = 1.0 - Om_a
            u_val = 1.0 + w0 + wa*(1.0 - a)
            gamma_val = c0 + c1*q_val + c2*q_val**2 + c3*u_val + c4*wa*a + c5*u_val*q_val
            f_val = Om_a**gamma_val
            return f_val * D
        
        from scipy.integrate import solve_ivp
        x_eval = x_grid
        # Integrate from x=0 to x_min (backwards)
        # solve_ivp can integrate backwards if t_eval is decreasing?
        # Or just swap limits.
        # Let's integrate from x_min to 0 and then flip?
        # No, we want D(x) for x in x_grid.
        # Let's integrate from 0 to x_min.
        # t_span = (0, x_min)
        # y0 = [1.0] at x=0.
        
        sol = solve_ivp(dD_dx, [0.0, x_min], [1.0], t_eval=x_grid, method='RK45', rtol=1e-8, atol=1e-10)
        
        if not sol.success:
            D_approx = np.ones_like(D_ref)
        else:
            D_approx = sol.y[0]
        
        # Calculate relative errors
        # f error
        rel_err_f = np.abs(f_approx - f_ref) / np.maximum(np.abs(f_ref), 1e-10)
        # D error
        rel_err_D = np.abs(D_approx - D_ref) / np.maximum(np.abs(D_ref), 1e-10)
        
        current_max = np.max(np.concatenate([rel_err_f, rel_err_D]))
        if current_max > max_rel_err:
            max_rel_err = current_max
            
    return max_rel_err * 100.0

def construct():
    # Test cosmologies: (Om0, w0, wa)
    test_models = [
        (0.3, -1.0, 0.0),    # LambdaCDM
        (0.3, -0.9, -0.3),   # Quintessence
        (0.3, -1.2, 0.5),    # Phantom
        (0.3, -0.8, 0.0),    # w0wa with w0 > -1
        (0.2, -1.0, 0.0),    # Low Omega_m
        (0.4, -1.0, 0.0),    # High Omega_m
        (0.3, -1.0, -1.0),   # Strong wa
        (0.3, -1.0, 1.0),    # Strong positive wa
        (0.3, -0.5, 0.0),    # High w0
        (0.3, -1.5, 0.5),    # Very phantom
    ]
    
    a_grid = _A_GRID
    
    # Start with the seed
    x0 = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    start_time = time.time()
    deadline = start_time + 80  # 80 seconds deadline
    
    best_x = x0
    best_err = _evaluate_error(x0, test_models, a_grid)
    
    # Simple local optimization
    # Use Nelder-Mead
    for i in range(50):
        if time.time() > deadline:
            break
        
        result = minimize(_evaluate_error, best_x, args=(test_models, a_grid), method='Nelder-Mead',
                         options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if result.fun < best_err:
            best_err = result.fun
            best_x = result.x
    
    # Clamp to [-10, 10]
    best_x = np.clip(best_x, -10, 10)
    
    return list(best_x)
# EVOLVE-BLOCK-END

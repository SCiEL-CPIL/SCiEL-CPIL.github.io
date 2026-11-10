"""Improved growth-index approximation with optimized coefficients."""
import numpy as np
from scipy.integrate import solve_ivp, quad
import time

def _Om_of_a(a, Om0, w0, wa):
    """Compute Omega_m(a) for CPL dark energy."""
    a = np.asarray(a, dtype=float)
    u = 1.0 + w0 + wa * (1.0 - a)
    # DE density: rho_de = rho_de0 * a^(-3(1+w0+wa)) = a^(-3-3w0-3wa)
    # rho_m = a^(-3)
    # Omega_m(a) = Om0 * a^(-3) / (Om0*a^(-3) + (1-Om0)*a^(-3-3w0-3wa))
    a3 = a**3
    a3_de = a3 * a**(3*(w0 + wa*(1.0 - a)))  # a^(3+3w0+3wa(1-a)) = a^(3+3w0+3wa)*a^(-3wa*a)
    # More carefully: rho_de/rho_de0 = a^(-3(1+w0+wa(1-a))) = a^(-3-3w0-3wa+3wa*a)
    # So in denominator: Om0*a^(-3) + (1-Om0)*a^(-3-3w0-3wa+3wa*a)
    # Divide by a^(-3): Om0 + (1-Om0)*a^(-3w0-3wa+3wa*a)
    # a^(-3w0-3wa+3wa*a) = a^(-3w0-3wa) * a^(3wa*a)
    exponent = -3.0*w0 - 3.0*wa + 3.0*wa*a
    de_factor = np.exp(exponent * np.log(a))
    denom = Om0 + (1.0 - Om0) * de_factor
    return Om0 / denom

def _gamma_of_a(a, Om0, w0, wa, c):
    """Compute gamma(a) from the six-coefficient ansatz."""
    c0, c1, c2, c3, c4, c5 = c
    q = 1.0 - Om0 / _Om_of_a(a, Om0, w0, wa)  # q = 1 - Om(a)
    u = 1.0 + w0 + wa * (1.0 - a)
    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
    return gamma

def _f_of_a(a, Om0, w0, wa, c):
    """Compute f(a) = Om(a)^gamma(a)."""
    Om = _Om_of_a(a, Om0, w0, wa)
    gamma = _gamma_of_a(a, Om0, w0, wa, c)
    return Om**gamma

def _reference_f_and_D(a_grid, Om0, w0, wa):
    """
    Compute reference f(a) and normalized D(a) by solving the GR ODE.
    Returns arrays of f and D/D(1) on a_grid.
    """
    # Solve for D(a) using the second-order ODE for linear growth.
    # In terms of x = ln(a), d2D/dx2 + (3/2 + H'/H) dD/dx - (3/2) Om(a) D = 0
    # where H'/H = (d/dx) ln H = (1/2) d/dx ln(1 + ... )
    # For CPL: H^2(a) = H0^2 [Om0 a^-3 + (1-Om0) a^(-3(1+w0+wa(1-a)))]
    # Let's work in x = ln a, so a = e^x.
    # H^2(x) = H0^2 [Om0 e^{-3x} + (1-Om0) e^{-3x(1+w0+wa(1-e^{-x}))}]
    # dH/dx / H = (1/2) d/dx ln H^2
    
    def h2_of_x(x):
        a = np.exp(x)
        Om_a = _Om_of_a(a, Om0, w0, wa)
        # H^2/H0^2 = Om0*a^-3 + (1-Om0)*a^(-3(1+w0+wa(1-a)))
        # = a^-3 * [Om0 + (1-Om0)*a^(-3w0-3wa(1-a))]
        a3 = a**3
        de_exp = -3.0*w0 - 3.0*wa*(1.0 - a)
        return a3**(-1) * (Om0 + (1.0 - Om0) * np.exp(de_exp))
    
    def dlnH2_dx(x):
        a = np.exp(x)
        Om_a = _Om_of_a(a, Om0, w0, wa)
        # d/dx ln H^2 = d/dx [ -3x + ln(Om0 + (1-Om0)*exp(-3w0-3wa(1-e^{-x}))) ]
        # = -3 + [ (1-Om0)*exp(-3w0-3wa(1-e^{-x})) * d/dx(-3wa(1-e^{-x})) ] / [Om0 + (1-Om0)*exp(-3w0-3wa(1-e^{-x}))]
        # d/dx(-3wa(1-e^{-x})) = -3wa*e^{-x} = -3wa*a
        de_exp = -3.0*w0 - 3.0*wa*(1.0 - a)
        de_term = (1.0 - Om0) * np.exp(de_exp)
        denom = Om0 + de_term
        if denom == 0:
            return -3.0
        return -3.0 + de_term * (-3.0*wa*a) / denom
    
    def ode_rhs(x, y):
        D, dD = y
        Om_a = _Om_of_a(np.exp(x), Om0, w0, wa)
        dlnH2 = dlnH2_dx(x)
        # d2D/dx2 + (3/2 + dlnH2/2) dD/dx - (3/2) Om_a D = 0
        d2D = -(1.5 + 0.5*dlnH2) * dD + 1.5 * Om_a * D
        return [dD, d2D]
    
    # Integrate from x=0 (a=1) backwards to x_min
    x_min = np.log(a_grid[0])
    x_max = 0.0
    # Initial conditions at x=0: D(0)=1, dD/dx(0) = f(1) = Om0^gamma(1)
    # We need a reference gamma. Use a reasonable value.
    gamma_1 = 0.55  # approximate
    dD0 = Om0**gamma_1
    
    # Use a dense grid for integration
    n_pts = max(200, len(a_grid)*5)
    x_eval = np.linspace(x_min, x_max, n_pts)
    
    sol = solve_ivp(ode_rhs, [x_min, x_max], [1.0, dD0], 
                    method='RK45', rtol=1e-10, atol=1e-12,
                    dense_output=True, max_step=0.01)
    
    if not sol.success:
        # Fallback: use constant gamma
        D = np.ones_like(a_grid)
        f = _f_of_a(a_grid, Om0, w0, wa, [0.55,0,0,0,0,0])
        return f, D
    
    D_vals = sol.sol(np.log(a_grid))[0]
    f_vals = sol.sol(np.log(a_grid))[1] / D_vals
    
    return f_vals, D_vals

def _objective(c, models):
    """
    Compute max relative error over all models and a values.
    Returns the objective (lower is better).
    """
    max_err = 0.0
    for (Om0, w0, wa) in models:
        a_grid = np.geomspace(0.1, 1.0, 50)
        try:
            f_ref, D_ref = _reference_f_and_D(a_grid, Om0, w0, wa)
        except:
            continue
        
        f_app = _f_of_a(a_grid, Om0, w0, wa, c)
        D_app = _normalized_D(a_grid, Om0, w0, wa, c)
        
        # Check gamma bounds
        gamma_vals = _gamma_of_a(a_grid, Om0, w0, wa, c)
        if np.any(gamma_vals < 0.1) or np.any(gamma_vals > 1.2):
            return 100.0  # penalty
        
        # Relative errors
        f_err = np.abs(f_app - f_ref) / np.abs(f_ref)
        D_err = np.abs(D_app - D_ref) / np.abs(D_ref)
        
        max_err = max(max_err, np.max(f_err), np.max(D_err))
    
    return 100.0 * max_err

def _normalized_D(a_grid, Om0, w0, wa, c):
    """Compute D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)."""
    def integrand(x):
        a = np.exp(x)
        return _f_of_a(a, Om0, w0, wa, c)
    
    D_vals = np.ones_like(a_grid)
    for i, a in enumerate(a_grid):
        if a >= 1.0:
            D_vals[i] = 1.0
        else:
            val, _ = quad(integrand, np.log(a), 0.0, limit=100)
            D_vals[i] = np.exp(-val)
    return D_vals

def construct():
    """
    Find optimal coefficients c = [c0, c1, c2, c3, c4, c5].
    Uses physics-constrained affine strategy: fix c1=0, c2=0,
    optimize c0, c3, c4, c5.
    """
    start_time = time.time()
    deadline = 80.0  # safety margin before 90s limit
    
    # Representative CPL models for optimization
    models = [
        (0.30, -1.0, 0.0),   # LCDM
        (0.30, -1.0, 0.5),   # CPL positive wa
        (0.30, -1.0, -0.5),  # CPL negative wa
        (0.25, -1.0, 0.0),   # lower Om0
        (0.35, -1.0, 0.0),   # higher Om0
        (0.30, -0.9, 0.0),   # w0 != -1
        (0.30, -1.1, 0.0),   # w0 < -1
        (0.30, -1.0, 1.0),   # large wa
        (0.30, -1.0, -1.0),  # large negative wa
    ]
    
    # Physics-constrained: c1=0, c2=0
    # Optimize c0, c3, c4, c5
    from scipy.optimize import minimize
    
    def obj_wrapper(x):
        c = [x[0], 0.0, 0.0, x[1], x[2], x[3]]
        return _objective(c, models)
    
    # Initial guess
    x0 = [0.55, 0.0, 0.0, 0.0]
    
    best_x = x0
    best_obj = obj_wrapper(x0)
    
    # Try multiple starting points
    starts = [
        [0.55, 0.0, 0.0, 0.0],
        [0.5, 0.1, 0.0, 0.0],
        [0.6, -0.1, 0.0, 0.0],
        [0.55, 0.05, 0.05, 0.0],
        [0.55, -0.05, 0.05, 0.0],
        [0.5, 0.0, 0.1, 0.0],
        [0.6, 0.0, -0.1, 0.0],
    ]
    
    for s in starts:
        if time.time() - start_time > deadline:
            break
        try:
            res = minimize(obj_wrapper, s, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x
        except:
            pass
    
    # Final polish with tighter tolerance
    if time.time() - start_time < deadline - 5:
        try:
            res = minimize(obj_wrapper, best_x, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x
        except:
            pass
    
    c = [best_x[0], 0.0, 0.0, best_x[1], best_x[2], best_x[3]]
    
    # Clamp to [-10, 10]
    c = [max(-10.0, min(10.0, float(v))) for v in c]
    
    return c

# Module-level call to ensure construct is ready
construct()

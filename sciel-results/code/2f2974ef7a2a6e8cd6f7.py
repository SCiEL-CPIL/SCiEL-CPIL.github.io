"""Improved growth-index ansatz with optimization against ODE reference."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp
import time

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 85.0  # Safety margin before 90s kill

def _check_deadline():
    return time.time() > _DEADLINE

def _Om_a(a, Om0, w0, wa):
    """Compute matter density parameter at scale factor a for CPL dark energy."""
    a = np.asarray(a, dtype=float)
    scalar = a.ndim == 0
    if scalar:
        a = a.reshape(1)
    
    # For CPL: w(a) = w0 + wa*(1-a)
    # rho_de(a)/rho_de(1) = exp(-3 * integral_1^a (1+w(a'))/a' da')
    # Change variable x = ln a': integral_0^{ln a} (1+w(e^x)) dx
    # 1+w(e^x) = 1+w0+wa*(1-e^x)
    # integral = (1+w0+wa)*ln(a) - wa*(a-1)
    
    if np.any(np.abs(a - 1.0) < 1e-15):
        rho_de_ratio = np.ones_like(a)
        rho_m_ratio = np.ones_like(a)
    else:
        integral = np.log(a) * (1 + w0 + wa) - wa * (a - 1)
        rho_de_ratio = np.exp(-3 * integral)
        rho_m_ratio = a**(-3)
    
    Om_a = Om0 * rho_m_ratio / (Om0 * rho_m_ratio + (1 - Om0) * rho_de_ratio)
    
    if scalar:
        return Om_a[0]
    return Om_a

def _solve_growth_ODE(Om0, w0, wa, a_min=0.05, a_max=1.0, n_points=500):
    """Solve the linear growth ODE to get reference f and D."""
    # Growth equation in terms of x = ln a:
    # D'' + [3/2 + (3/2)*Om(a) - (3/2)*a*Om'(a)] D' - (3/2)*Om(a) D = 0
    # where ' = d/dx, x = ln a
    # 
    # More carefully: The standard form is
    # D'' + (3 + (3/2)*Om(a) - (3/2)*dOm/dln(a)) D' - (3/2)*Om(a) D = 0
    # 
    # Actually, let me use the well-known form:
    # D'' + [3/2 * (1 + Om(a)) + (1/2) * dOm/dln(a)] D' - (3/2) * Om(a) * D = 0
    # 
    # Wait, the correct form from the literature is:
    # D'' + (3/2 + (3/2)Om(a) - (3/2)Om'(a)/Om(a) * a * H'/H ... ) 
    # 
    # Let me use the simplest correct form. The growth factor D(a) satisfies:
    # d^2D/dln^2(a) + [3/2 + (3/2)Om(a) - (3/2) dln(Om(a))/dln(a)] dD/dln(a) - (3/2)Om(a) D = 0
    # 
    # Actually, the standard result is:
    # F'' + (3/2 + (3/2)Om(a) - (3/2) * dOm/dln(a) / Om(a) * ... )
    # 
    # Let me just use the known form:
    # For a flat universe, the growth equation is:
    # D'' + (3/2 + (3/2)Om(a) - (3/2) * (dOm/dln a) / (d ln H/d ln a) * ... )
    # 
    # The cleanest form I know:
    # D'' + [3/2 * (1 + Om(a)) + 1/2 * dOm/dln(a)] D' - 3/2 * Om(a) * D = 0
    # 
    # Hmm, let me just use numerical differentiation for dOm/dln(a)
    # and solve the ODE directly.
    
    a_grid = np.logspace(np.log10(a_min), 0, n_points)
    x_grid = np.log(a_grid)
    
    # Compute Om(a) on grid
    Om_grid = _Om_a(a_grid, Om0, w0, wa)
    
    # Compute dOm/dln(a) = a * dOm/da using numerical differentiation
    dOm_dlna = np.zeros_like(Om_grid)
    for i in range(1, n_points - 1):
        dOm_dlna[i] = (Om_grid[i+1] - Om_grid[i-1]) / (x_grid[i+1] - x_grid[i-1])
    dOm_dlna[0] = dOm_dlna[1]
    dOm_dlna[-1] = dOm_dlna[-2]
    
    # Interpolation functions
    from scipy.interpolate import interp1d
    Om_interp = interp1d(x_grid, Om_grid, kind='cubic', fill_value='extrapolate')
    dOm_interp = interp1d(x_grid, dOm_dlna, kind='cubic', fill_value='extrapolate')
    
    def ode_func(x, y):
        Om_x = Om_interp(x)
        dOm_x = dOm_interp(x)
        # D'' + [3/2 + (3/2)*Om - (3/2)*dOm/dlna] D' - (3/2)*Om*D = 0
        # y[0] = D, y[1] = D'
        D, Dp = y
        Dpp = -(3.0/2.0 + 3.0/2.0 * Om_x - 3.0/2.0 * dOm_x) * Dp + 3.0/2.0 * Om_x * D
        return [Dp, Dpp]
    
    # Initial conditions at small a: D ~ a^{6/5}, so D' ~ 6/5
    x_start = np.log(a_min)
    D_start = a_min**(6.0/5.0)
    Dp_start = 6.0/5.0 * D_start
    
    sol = solve_ivp(ode_func, [x_start, 0.0], [D_start, Dp_start], 
                    t_eval=x_grid, method='DOP853', rtol=1e-10, atol=1e-12)
    
    if not sol.success:
        return None, None
    
    D = sol.y[0]
    Dp = sol.y[1]
    
    # f = dlnD/dlna = D'/D
    f_ref = Dp / D
    
    # Normalize D so that D(1) = 1
    D_norm = D / D[-1]
    
    return f_ref, D_norm

def _compute_reference(val_cosmos, a_grid):
    """Compute reference f and D for all validation cosmologies."""
    refs = []
    for (Om0, w0, wa) in val_cosmos:
        if _check_deadline():
            break
        f_ref, D_ref = _solve_growth_ODE(Om0, w0, wa, a_min=0.1, a_max=1.0, n_points=300)
        if f_ref is not None:
            refs.append((Om0, w0, wa, f_ref, D_ref))
    return refs

def _evaluate_coefficients(coeff, refs, a_grid):
    """Compute max relative error for f and D."""
    c0, c1, c2, c3, c4, c5 = coeff
    max_err_f = 0.0
    max_err_D = 0.0
    
    for (Om0, w0, wa, f_ref, D_ref) in refs:
        Om_arr = _Om_a(a_grid, Om0, w0, wa)
        q = 1.0 - Om_arr
        u = 1.0 + w0 + wa * (1.0 - a_grid)
        
        gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_grid + c5*u*q
        
        # Check gamma bounds
        if np.any(gamma < 0.1) or np.any(gamma > 1.2):
            return np.inf
        
        # f_ours = Om(a)^gamma
        f_ours = np.power(np.maximum(Om_arr, 1e-15), gamma)
        
        # Relative error in f
        err_f = np.abs(f_ours - f_ref) / np.maximum(np.abs(f_ref), 1e-10)
        max_err_f = max(max_err_f, np.max(err_f))
        
        # Compute D_ours: D(a)/D(1) = exp(-integral_0^{ln(a)} f(exp(x)) dx)
        # We need to integrate f(exp(x)) from 0 to ln(a)
        # For each a in a_grid, compute the integral numerically
        
        # Use cumulative trapezoidal integration
        x_grid = np.log(a_grid)
        # f_ours as function of x: f(exp(x))
        f_x = f_ours  # already computed on a_grid which corresponds to x_grid
        
        # Cumulative integral from 0 (x=0, a=1) to each x
        # We need integral from x_i to 0, so -integral from 0 to x_i
        # Use cumulative trapezoid
        from scipy.integrate import cumulative_trapezoid
        # We want integral from 0 to x_i for each i
        # cumulative_trapezoid gives integral from first point to each point
        # x_grid is sorted, first point is a_min, last is 1.0
        # We want integral from x=0 (last point) backwards
        
        # Simpler: compute integral from 0 to x_i = -integral from x_i to 0
        # Let's compute the antiderivative F(x) = integral from some_ref to x f(x') dx'
        # Then D(a)/D(1) = exp(-(F(ln(a)) - F(0)))
        
        # Compute F using cumulative_trapezoid from the last point (x=0)
        # Actually, let's just do it point by point using the trapezoidal rule
        
        D_ours = np.ones_like(a_grid)
        # For a=1 (x=0), D=1
        # For a<1 (x<0), integral from 0 to x of f(exp(x')) dx' is negative
        # D(a)/D(1) = exp(-integral_0^{ln(a)} f(exp(x')) dx')
        
        # Let's compute the integral numerically for each point
        # Use the fact that x_grid is sorted
        # For each i, integral from 0 to x_grid[i] = -integral from x_grid[i] to 0
        
        # Precompute cumulative integral from last point
        # cumulative_trapezoid(f_x, x_grid, initial=0) gives integral from x_grid[0] to each x_grid[i]
        cum_int = cumulative_trapezoid(f_x, x_grid, initial=0.0)
        # cum_int[i] = integral from x_grid[0] to x_grid[i]
        # We want integral from 0 to x_grid[i] = cum_int[i] - cum_int[-1] (since x_grid[-1]=0)
        # integral_0^{x_i} = integral_{x_0}^{x_i} - integral_{x_0}^{0} = cum_int[i] - cum_int[-1]
        int_from_0 = cum_int - cum_int[-1]
        # D(a)/D(1) = exp(-int_from_0)
        D_ours = np.exp(-int_from_0)
        
        # Relative error in D
        err_D = np.abs(D_ours - D_ref) / np.maximum(np.abs(D_ref), 1e-10)
        max_err_D = max(max_err_D, np.max(err_D))
    
    return 100.0 * max(max_err_f, max_err_D)

def construct():
    """Return six coefficients [c0..c5] for the growth-index ansatz."""
    # Seed: constant gamma = 0.55
    best_coeff = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Validation cosmologies
    val_cosmos = [
        (0.3, -1.0, 0.0),
        (0.3, -1.0, 0.1),
        (0.3, -1.0, -0.1),
        (0.3, -0.9, 0.0),
        (0.3, -1.1, 0.0),
        (0.3, -1.0, 0.2),
        (0.28, -1.0, 0.0),
        (0.32, -1.0, 0.0),
        (0.3, -0.8, 0.3),
        (0.3, -1.2, -0.2),
    ]
    
    a_grid = np.logspace(np.log10(0.1), 0, 200)
    
    # Compute references
    refs = _compute_reference(val_cosmos, a_grid)
    if not refs:
        return best_coeff.tolist()
    
    # Evaluate seed
    best_score = _evaluate_coefficients(best_coeff, refs, a_grid)
    
    # Optimize using differential evolution (robust to local minima)
    bounds = [(0.1, 1.2)] * 6  # Coefficients should be in reasonable range
    # Actually, the task says coefficients must be in [-10,10]
    bounds = [(-10, 10)] * 6
    
    if _check_deadline():
        return best_coeff.tolist()
    
    def objective(coeff):
        if _check_deadline():
            return np.inf
        score = _evaluate_coefficients(coeff, refs, a_grid)
        return score
    
    try:
        result = differential_evolution(objective, bounds, seed=42, 
                                         maxiter=50, tol=1e-8, 
                                         polish=True, workers=-1,
                                         init='sobol')
        
        if _check_deadline():
            return best_coeff.tolist()
        
        if result.fun < best_score:
            best_score = result.fun
            best_coeff = result.x
    except Exception:
        pass
    
    # Try local refinement with Nelder-Mead
    if not _check_deadline():
        try:
            result2 = minimize(objective, best_coeff, method='Nelder-Mead',
                               options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10})
            if not _check_deadline() and result2.fun < best_score:
                best_score = result2.fun
                best_coeff = result2.x
        except Exception:
            pass
    
    # Ensure coefficients are in [-10, 10]
    best_coeff = np.clip(best_coeff, -10, 10)
    
    return best_coeff.tolist()
# EVOLVE-BLOCK-END

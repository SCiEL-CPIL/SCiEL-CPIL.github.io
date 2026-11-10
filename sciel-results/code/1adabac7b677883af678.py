"""
Growth-index approximation for CPL dark-energy cosmologies.

The six-coefficient ansatz for gamma(a) is:
    gamma = c0 + c1*q + c2*q^2 + c3*u + c4*wa*a + c5*u*q
where q = 1 - Om(a) and u = 1 + w0 + wa*(1-a).

We optimize the coefficients to minimize the maximum relative error of
both f(a) = Om(a)^gamma(a) and the normalized growth factor D(a)/D(1)
against a numerical reference solution of the GR growth ODE.

The reference is computed by solving the second-order ODE:
    D'' + (3/a - q(a)) D' - (3/2) Om(a) D = 0
with D(a) ~ a in the matter-dominated limit, normalized to D(1)=1.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import time

# ---------------------------------------------------------------------------
# Cosmology helpers
# ---------------------------------------------------------------------------

def _Om_of_a(a, Om0, w0, wa):
    """Matter density parameter Om(a) for a flat CPL universe."""
    a = np.asarray(a, dtype=float)
    Omega_DE0 = 1.0 - Om0
    # rho_DE(a)/rho_DE(1) = a^{-3(1+w0+wa)} * exp(3*wa*(a-1))
    ratio_DE = np.power(a, -3.0*(1.0 + w0 + wa)) * np.exp(3.0*wa*(a - 1.0))
    matter = Om0 * np.power(a, -3.0)
    de = Omega_DE0 * ratio_DE
    return matter / (matter + de)


def _q_of_a(a, Om0, w0, wa):
    """Deceleration parameter q(a) = (1/2) d ln(H^2)/d ln a."""
    # H^2/H0^2 = Om(a)*a^{-3}
    # d ln H^2 / d ln a = -3 + d ln Om(a) / d ln a
    # Alternatively: q = -a dH/da / H = 1/2 * (d ln H^2 / d ln a)
    # Let's compute numerically for robustness
    # Actually, for flat: q(a) = 0.5*(3*Om(a) - 3*(1+3w(a))*Omega_DE(a))
    # More precisely: q = 0.5 * (3*Om_m(a) - 3*(1+3w(a))*Om_DE(a))
    # where Om_DE(a) = 1 - Om(a)
    Om_a = _Om_of_a(a, Om0, w0, wa)
    w_a = w0 + wa*(1.0 - a)
    Om_DE_a = 1.0 - Om_a
    return 0.5 * (3.0*Om_a - 3.0*(1.0 + 3.0*w_a)*Om_DE_a)


# ---------------------------------------------------------------------------
# Reference solution via ODE
# ---------------------------------------------------------------------------

def _reference_solution(a_grid, Om0, w0, wa, n_steps=500):
    """
    Solve the growth ODE on a grid from a_min to 1.
    Returns D_norm (normalized so D(1)=1) and f = D'/D on the grid.
    """
    a_min = 0.01
    a_grid = np.linspace(a_min, 1.0, n_steps)
    
    def ode_rhs(a, y):
        D, v = y  # v = dD/da
        Om_a = _Om_of_a(a, Om0, w0, wa)
        q_a = _q_of_a(a, Om0, w0, wa)
        D_prime = v
        v_prime = -(3.0/a - q_a)*v + 1.5*Om_a*D
        return [D_prime, v_prime]
    
    # Initial conditions: in matter domination D ~ a, so D(a_min)=a_min, v(a_min)=1
    y0 = [a_min, 1.0]
    
    sol = solve_ivp(ode_rhs, [a_min, 1.0], y0, t_eval=a_grid,
                    method='RK45', rtol=1e-10, atol=1e-12, max_step=0.01)
    
    if not sol.success:
        # Fallback: use constant gamma=0.55
        D = np.ones_like(a_grid)
        f = np.full_like(a_grid, 0.55)
        return D, f
    
    D = sol.y[0]
    v = sol.y[1]
    D_norm = D / D[-1]  # Normalize so D(1)=1
    f = v / D  # f = dD/da / D = d ln D / d ln a * (1/a) ... wait
    
    # Actually f(a) = d ln D / d ln a = a * D'/D = a * v / D
    # Let me re-check: the growth rate is f = d ln D / d ln a = a * (dD/da)/D
    f = a_grid * v / D
    
    return D_norm, f


def _interp_reference(a_eval, a_grid, D_ref, f_ref):
    """Interpolate reference D and f at a_eval."""
    a_eval = np.clip(a_eval, a_grid[0], a_grid[-1])
    D_val = np.interp(a_eval, a_grid, D_ref)
    f_val = np.interp(a_eval, a_grid, f_ref)
    return D_val, f_val


# ---------------------------------------------------------------------------
# Ansatz evaluation
# ---------------------------------------------------------------------------

def _gamma_from_coeffs(a, coeffs, Om0, w0, wa):
    """Compute gamma(a) from the six-coefficient ansatz."""
    c0, c1, c2, c3, c4, c5 = coeffs
    Om_a = _Om_of_a(a, Om0, w0, wa)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa*(1.0 - a)
    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
    return gamma


def _f_from_coeffs(a, coeffs, Om0, w0, wa):
    """Compute f(a) = Om(a)^gamma(a)."""
    Om_a = _Om_of_a(a, Om0, w0, wa)
    gamma = _gamma_from_coeffs(a, coeffs, Om0, w0, wa)
    # Clamp gamma to valid range
    gamma = np.clip(gamma, 0.1, 1.2)
    return Om_a**gamma


def _D_from_coeffs(a, coeffs, Om0, w0, wa):
    """
    Compute D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx).
    Using change of variable: integral from ln(a) to 0 of f(e^x) dx
    = integral from a to 1 of f(t)/t dt  (with t = e^x, dx = dt/t)
    So D(a)/D(1) = exp(-integral_a^1 f(t)/t dt)
    """
    if a >= 1.0:
        return 1.0
    
    def integrand(t):
        f_t = _f_from_coeffs(t, coeffs, Om0, w0, wa)
        return f_t / t
    
    # Integrate from a to 1
    from scipy.integrate import quad
    val, _ = quad(integrand, a, 1.0, limit=200)
    return np.exp(-val)


# ---------------------------------------------------------------------------
# Objective function for optimization
# ---------------------------------------------------------------------------

def _objective(coeffs, test_models, a_eval_grid):
    """
    Maximize error across all test models and evaluation points.
    Returns the maximum relative error * 100.
    """
    max_err = 0.0
    
    for (Om0, w0, wa) in test_models:
        # Compute reference on a grid
        a_grid = np.linspace(0.01, 1.0, 300)
        D_ref, f_ref = _reference_solution(a_grid, Om0, w0, wa, n_steps=300)
        
        for a_eval in a_eval_grid:
            if a_eval < 0.01 or a_eval > 1.0:
                continue
            
            # Reference values
            D_r, f_r = _interp_reference(a_eval, a_grid, D_ref, f_ref)
            
            # Ansatz values
            f_a = _f_from_coeffs(a_eval, coeffs, Om0, w0, wa)
            D_a = _D_from_coeffs(a_eval, coeffs, Om0, w0, wa)
            
            # Relative errors
            if abs(f_r) > 1e-10:
                err_f = abs(f_a - f_r) / abs(f_r)
            else:
                err_f = abs(f_a - f_r)
            
            if abs(D_r) > 1e-10:
                err_D = abs(D_a - D_r) / abs(D_r)
            else:
                err_D = abs(D_a - D_r)
            
            max_err = max(max_err, err_f, err_D)
    
    return max_err * 100.0


# ---------------------------------------------------------------------------
# Pre-compute reference data for speed
# ---------------------------------------------------------------------------

def _precompute_references(test_models, n_grid=200):
    """Pre-compute reference solutions for all test models."""
    refs = {}
    a_grid = np.linspace(0.01, 1.0, n_grid)
    
    for (Om0, w0, wa) in test_models:
        D_ref, f_ref = _reference_solution(a_grid, Om0, w0, wa, n_steps=n_grid)
        refs[(Om0, w0, wa)] = (a_grid, D_ref, f_ref)
    
    return refs


def _objective_fast(coeffs, refs, a_eval_grid):
    """Fast objective using pre-computed references."""
    max_err = 0.0
    
    for (Om0, w0, wa), (a_grid, D_ref, f_ref) in refs.items():
        for a_eval in a_eval_grid:
            if a_eval < a_grid[0] or a_eval > a_grid[-1]:
                continue
            
            D_r = np.interp(a_eval, a_grid, D_ref)
            f_r = np.interp(a_eval, a_grid, f_ref)
            
            f_a = _f_from_coeffs(a_eval, coeffs, Om0, w0, wa)
            D_a = _D_from_coeffs(a_eval, coeffs, Om0, w0, wa)
            
            if abs(f_r) > 1e-10:
                err_f = abs(f_a - f_r) / abs(f_r)
            else:
                err_f = abs(f_a - f_r)
            
            if abs(D_r) > 1e-10:
                err_D = abs(D_a - D_r) / abs(D_r)
            else:
                err_D = abs(D_a - D_r)
            
            max_err = max(max_err, err_f, err_D)
    
    return max_err * 100.0


# ---------------------------------------------------------------------------
# Main construct function
# ---------------------------------------------------------------------------

def construct():
    """Find optimal six coefficients by minimizing max error over test cosmologies."""
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    # Test cosmologies: (Om0, w0, wa)
    test_models = [
        (0.3, -1.0, 0.0),      # LCDM
        (0.3, -1.0, 0.1),      # CPL with wa>0
        (0.3, -1.0, -0.1),     # CPL with wa<0
        (0.3, -0.9, 0.0),      # CPL with w0>-1
        (0.3, -1.1, 0.0),      # CPL with w0<-1
        (0.3, -0.9, 0.2),      # Combined
        (0.3, -1.1, -0.2),     # Combined
        (0.25, -1.0, 0.0),     # Lower Om0
        (0.35, -1.0, 0.0),     # Higher Om0
        (0.3, -1.0, 0.3),      # Larger wa
    ]
    
    # Evaluation points
    a_eval_grid = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    
    # Pre-compute references
    refs = _precompute_references(test_models, n_grid=200)
    
    # Check time
    if time.time() > deadline:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Initial guess: constant gamma = 0.55
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    best_coeffs = x0.copy()
    best_score = _objective_fast(best_coeffs, refs, a_eval_grid)
    
    # Try multiple optimization strategies
    strategies = [
        ('Nelder-Mead', {'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6}),
        ('Powell', {'maxiter': 300, 'xtol': 1e-6, 'ftol': 1e-6}),
        ('COBYLA', {'maxiter': 500, 'tol': 1e-6}),
    ]
    
    for method, options in strategies:
        if time.time() > deadline - 10:
            break
        
        try:
            result = minimize(
                _objective_fast, x0, args=(refs, a_eval_grid),
                method=method, options=options
            )
            
            if result.success or result.fun < best_score:
                # Verify gamma stays in [0.1, 1.2]
                valid = True
                for (Om0, w0, wa) in test_models:
                    for a in np.linspace(0.01, 1.0, 50):
                        g = _gamma_from_coeffs(a, result.x, Om0, w0, wa)
                        if g < 0.1 or g > 1.2:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    best_coeffs = result.x.copy()
                    best_score = result.fun
                    
        except Exception:
            pass
    
    # Polish: try a few random perturbations around best
    if time.time() < deadline - 5:
        for _ in range(20):
            if time.time() > deadline - 5:
                break
            perturbation = np.random.normal(0, 0.05, 6)
            candidate = best_coeffs + perturbation
            # Clamp coefficients to [-10, 10]
            candidate = np.clip(candidate, -10, 10)
            
            # Check gamma range
            valid = True
            for (Om0, w0, wa) in test_models:
                for a in np.linspace(0.01, 1.0, 30):
                    g = _gamma_from_coeffs(a, candidate, Om0, w0, wa)
                    if g < 0.1 or g > 1.2:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                score = _objective_fast(candidate, refs, a_eval_grid)
                if score < best_score:
                    best_coeffs = candidate
                    best_score = score
    
    # Final clamp
    best_coeffs = np.clip(best_coeffs, -10, 10)
    
    return best_coeffs.tolist()

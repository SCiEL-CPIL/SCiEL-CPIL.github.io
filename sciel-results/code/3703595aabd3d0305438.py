"""Improved growth-index approximation via on-the-fly optimization of 6 coefficients."""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp, quad

# EVOLVE-BLOCK-START

def construct():
    """Return the best 6 coefficients found via optimization within time budget."""
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill
    
    # Physically-motivated initial condition:
    # c0 = 0.55 (standard constant gamma), c3 = -0.5 to cancel leading w0 term
    # In the growth index expansion, gamma ~ 0.55 - 0.5*w0 + ... for CPL
    # This forces remaining coefficients to model subdominant corrections
    seed = [0.55, 0.0, 0.0, -0.5, 0.0, 0.0]
    best_coeffs = seed.copy()
    best_score = float('inf')
    
    # Test cosmologies (disjoint public validation set approximation)
    TEST_COSMOLOGIES = [
        (0.3, -1.0, 0.0),      # LambdaCDM
        (0.3, -1.0, 0.3),      # CPL
        (0.25, -0.9, 0.2),     # varying
        (0.35, -1.1, 0.1),     # another
        (0.3, -1.0, -0.5),     # phantom-ish
        (0.28, -1.0, 0.0),     # Planck-like
        (0.3, -0.5, 0.5),      # quintessence-like
        (0.32, -0.6, 0.1),     # mild variation
    ]
    
    # Pre-compute reference solutions for all test cosmologies
    # This is crucial for speed: solve ODE once per cosmology, cache results
    ref_cache = {}
    for Om0, w0, wa in TEST_COSMOLOGIES:
        key = (Om0, w0, wa)
        if time.time() > deadline - 30:
            break
        try:
            f_ref_grid, D_ref_grid, a_grid = _precompute_reference(Om0, w0, wa)
            ref_cache[key] = (f_ref_grid, D_ref_grid, a_grid)
        except Exception:
            pass
    
    # Fallback: if no references computed, use simpler approach
    if not ref_cache:
        for Om0, w0, wa in TEST_COSMOLOGIES[:4]:
            key = (Om0, w0, wa)
            try:
                f_ref_grid, D_ref_grid, a_grid = _precompute_reference(Om0, w0, wa)
                ref_cache[key] = (f_ref_grid, D_ref_grid, a_grid)
            except Exception:
                pass
    
    if not ref_cache:
        return seed
    
    # Define the objective function using cached references
    def objective(coeffs):
        c0, c1, c2, c3, c4, c5 = coeffs
        
        # Validate gamma range [0.1, 1.2] on a representative grid
        if not _gamma_in_range(c0, c1, c2, c3, c4, c5):
            return 1e10
        
        max_err = 0.0
        
        for (Om0, w0, wa), (f_ref_grid, D_ref_grid, a_grid) in ref_cache.items():
            if len(a_grid) == 0:
                continue
            
            # Compute approximate f and D on the same grid
            Om_a = Om0 * a_grid**3
            q = 1.0 - Om_a
            u = 1.0 + w0 + wa * (1.0 - a_grid)
            gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a_grid + c5*u*q
            
            # Check gamma bounds
            if np.any(gamma < 0.1) or np.any(gamma > 1.2):
                return 1e10
            
            f_approx = Om_a ** gamma
            
            # Relative error for f
            denom_f = np.maximum(np.abs(f_ref_grid), 1e-12)
            rel_err_f = np.abs(f_approx - f_ref_grid) / denom_f
            max_err = max(max_err, float(np.max(rel_err_f)))
            
            # Relative error for D
            D_approx = _normalized_growth_vec(c0, c1, c2, c3, c4, c5, Om0, w0, wa, a_grid)
            denom_D = np.maximum(np.abs(D_ref_grid), 1e-12)
            rel_err_D = np.abs(D_approx - D_ref_grid) / denom_D
            max_err = max(max_err, float(np.max(rel_err_D)))
            
            if max_err > 1.0:  # Early exit if error already large
                break
        
        return 100.0 * max_err
    
    # Global bounds for coefficients
    bounds = [(-10, 10)] * 6
    
    # Evaluate seed
    try:
        seed_score = objective(seed)
        if seed_score < best_score:
            best_score = seed_score
    except Exception:
        pass
    
    # Try differential evolution first (robust global optimizer)
    try:
        if time.time() < deadline - 25:
            result_de = differential_evolution(
                objective, bounds, seed=42, maxiter=30, tol=1e-5,
                polish=True, workers=-1, init='sobol',
                popsize=15
            )
            if result_de.fun < best_score:
                best_score = result_de.fun
                best_coeffs = result_de.x.tolist()
    except Exception:
        pass
    
    # Refine with L-BFGS-B
    try:
        if time.time() < deadline - 15:
            result_lbfgs = minimize(
                objective, best_coeffs, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result_lbfgs.fun < best_score:
                best_score = result_lbfgs.fun
                best_coeffs = result_lbfgs.x.tolist()
    except Exception:
        pass
    
    # Final high-precision polish with Nelder-Mead on tight box
    try:
        if time.time() < deadline - 5:
            # Tighten bounds around current best
            center = np.array(best_coeffs)
            tight_bounds = [(max(-10, c - 0.5), min(10, c + 0.5)) for c in center]
            result_nm = minimize(
                objective, best_coeffs, method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            # Clip to original bounds
            result_nm.x = np.clip(result_nm.x, -10, 10)
            if result_nm.fun < best_score:
                best_score = result_nm.fun
                best_coeffs = result_nm.x.tolist()
    except Exception:
        pass
    
    # Final check: ensure coefficients are finite and in [-10, 10]
    best_coeffs = np.array(best_coeffs)
    if not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.array(seed)
    best_coeffs = np.clip(best_coeffs, -10, 10)
    
    return best_coeffs.tolist()


def _gamma_in_range(c0, c1, c2, c3, c4, c5):
    """Check if gamma stays in [0.1, 1.2] on a representative grid."""
    # Check on a grid of (q, u, wa, a) covering plausible ranges
    for q in np.linspace(0, 1, 11):
        for u in np.linspace(-2, 3, 11):
            for wa in np.linspace(-1, 1, 5):
                for a in np.linspace(0.05, 0.99, 5):
                    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
                    if gamma < 0.1 or gamma > 1.2:
                        return False
    return True


def _precompute_reference(Om0, w0, wa, n_points=100):
    """Pre-compute reference f and D on a grid using the ODE."""
    a_grid = np.linspace(0.05, 0.99, n_points)
    
    def ode_rhs(x, y):
        a_x = math.exp(x)
        Om_m = Om0 * a_x**3
        Om_de = max(0.0, 1.0 - Om_m)
        
        # H^2/H0^2 for CPL dark energy
        h2 = Om_m + Om_de * (a_x ** (-3*(1+w0+wa))) * math.exp(-3*wa*(1-a_x))
        if h2 <= 0:
            return [0, 0]
        
        de_factor = math.exp(3*(1+w0+wa)*x + 3*wa*(1-a_x))
        T = Om_de * de_factor
        dT_dx = -3*Om_m * de_factor + Om_de * de_factor * (3*(1+w0+wa) - 3*wa*a_x)
        dh2_dx = 3*Om_m + dT_dx
        dlnH_dx = 0.5 * dh2_dx / h2
        coeff_f = 3.0 * Om_m / h2
        
        f = y[0]
        fp = y[1]
        fpp = -(3 + dlnH_dx) * fp + coeff_f * f
        return [fp, fpp]
    
    # Integrate from early times to a=1
    x_start = math.log(0.05)
    f_start = 0.05 ** 0.6
    fp_start = 0.6 * f_start
    
    sol = solve_ivp(ode_rhs, [x_start, 0.0], [f_start, fp_start],
                    method='DOP853', rtol=1e-10, atol=1e-12, dense_output=True)
    
    if not sol.success:
        # Fallback: use constant gamma approximation
        f_ref_grid = (Om0 * a_grid**3) ** 0.55
        D_ref_grid = np.ones_like(a_grid)
        for i, a in enumerate(a_grid):
            if a < 1.0:
                val, _ = quad(lambda x: (Om0 * math.exp(x)**3) ** 0.55, 0, -math.log(a), limit=50)
                D_ref_grid[i] = math.exp(-val)
        return f_ref_grid, D_ref_grid, a_grid
    
    # Extract f on the grid
    f_ref_grid = np.array([sol.sol(math.log(a))[0] for a in a_grid])
    
    # Compute D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)
    D_ref_grid = np.ones_like(a_grid)
    for i, a in enumerate(a_grid):
        if a >= 1.0:
            D_ref_grid[i] = 1.0
            continue
        x_lower = math.log(a)
        # Use the dense solution for integration
        n_int = 200
        xs = np.linspace(x_lower, 0.0, n_int)
        f_vals = np.array([sol.sol(x)[0] for x in xs])
        integral = np.trapz(f_vals, xs)
        D_ref_grid[i] = math.exp(-integral)
    
    return f_ref_grid, D_ref_grid, a_grid


def _normalized_growth_vec(c0, c1, c2, c3, c4, c5, Om0, w0, wa, a_grid):
    """Compute D(a)/D(1) using the approximate f on a vectorized grid."""
    D_ratio = np.ones_like(a_grid)
    
    for i, a in enumerate(a_grid):
        if a >= 1.0:
            D_ratio[i] = 1.0
            continue
        
        Om_a = Om0 * a**3
        q = 1.0 - Om_a
        u = 1.0 + w0 + wa * (1.0 - a)
        gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
        
        if gamma < 0.1:
            gamma = 0.1
        elif gamma > 1.2:
            gamma = 1.2
        
        # Integrate f(exp(x)) from ln(a) to 0
        def integrand(x):
            a_x = math.exp(x)
            Om_ax = Om0 * a_x**3
            if Om_ax <= 0:
                return 0.0
            q_x = 1.0 - Om_ax
            u_x = 1.0 + w0 + wa * (1.0 - a_x)
            g_x = c0 + c1*q_x + c2*q_x**2 + c3*u_x + c4*wa*a_x + c5*u_x*q_x
            if g_x < 0.1:
                g_x = 0.1
            elif g_x > 1.2:
                g_x = 1.2
            return Om_ax ** g_x
        
        x_lower = math.log(a)
        val, _ = quad(integrand, x_lower, 0.0, limit=100)
        D_ratio[i] = math.exp(-val)
    
    return D_ratio

# EVOLVE-BLOCK-END

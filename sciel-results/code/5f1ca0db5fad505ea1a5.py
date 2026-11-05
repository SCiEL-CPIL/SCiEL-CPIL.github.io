"""Improved growth-index approximation via on-the-fly optimization of 6 coefficients."""
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import solve_ivp, quad
import warnings
warnings.filterwarnings("ignore")

# EVOLVE-BLOCK-START

_REFERENCE_CACHE = {}

def _solve_ode(Om0, w0, wa, a_grid):
    """Solve the growth ODE and return f(a) and D(a)/D(1) on a_grid."""
    key = (round(Om0, 6), round(w0, 6), round(wa, 6), tuple(a_grid))
    if key in _REFERENCE_CACHE:
        return _REFERENCE_CACHE[key]
    
    def ode_rhs(x, y):
        a_val = math.exp(x)
        w_val = w0 + wa * (1.0 - a_val)
        f_val = y[0]
        return [-f_val*f_val - 1.5*(1.0 + w_val)*f_val]
    
    a_min = max(float(np.min(a_grid)), 0.02)
    x_start = math.log(a_min)
    f_start = a_min ** 0.6
    
    sol = solve_ivp(ode_rhs, [x_start, 0.0], [f_start],
                    method='DOP853', rtol=1e-10, atol=1e-12,
                    dense_output=True, max_step=0.02)
    
    if not sol.success:
        gamma_ref = 0.55
        f_ref = (Om0 * a_grid**3) ** gamma_ref
        D_ref = np.array([_integrate_f(Om0, w0, wa, a, gamma_ref) for a in a_grid])
    else:
        f_ref = np.array([sol.sol(math.log(a))[0] for a in a_grid])
        D_ref = np.ones_like(a_grid)
        # Use cumulative integration from a=1 backwards
        # D(a)/D(1) = exp(-∫_ln(a)^0 f(exp(x)) dx)
        for i, a in enumerate(a_grid):
            if a >= 1.0:
                D_ref[i] = 1.0
            else:
                x_lo = math.log(a)
                n_pts = 150
                xs = np.linspace(x_lo, 0.0, n_pts)
                f_vals = np.array([sol.sol(x)[0] for x in xs])
                integral = np.trapz(f_vals, xs)
                D_ref[i] = math.exp(-integral)
    
    _REFERENCE_CACHE[key] = (f_ref, D_ref)
    return f_ref, D_ref


def _integrate_f(Om0, w0, wa, a, gamma):
    """Integrate f = Om(a)^gamma from ln(a) to 0."""
    if a >= 1.0:
        return 1.0
    def integrand(x):
        a_x = math.exp(x)
        return (Om0 * a_x**3) ** gamma
    val, _ = quad(integrand, math.log(a), 0.0, limit=100, epsabs=1e-12, epsrel=1e-10)
    return math.exp(-val)


def _gamma_vec(c, a, Om0, w0, wa):
    """Vectorized gamma computation."""
    Om_a = Om0 * a**3
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1.0 - a)
    return c[0] + c[1]*q + c[2]*q**2 + c[3]*u + c[4]*wa*a + c[5]*u*q


def _objective(c, models, a_grid, sharpness=50.0):
    """Compute soft-max approximation to worst relative error."""
    c = np.clip(c, -10.0, 10.0)
    
    # Quick gamma range check on dense grid
    a_dense = np.linspace(0.02, 0.999, 50)
    for Om0, w0, wa in models:
        g = _gamma_vec(c, a_dense, Om0, w0, wa)
        if np.any(g < 0.1) or np.any(g > 1.2):
            # Soft penalty for gamma violation
            viol = np.maximum(0.1 - g, 0) + np.maximum(g - 1.2, 0)
            return 100.0 * (0.05 + np.max(viol))
    
    errors = []
    for Om0, w0, wa in models:
        f_ref, D_ref = _solve_ode(Om0, w0, wa, a_grid)
        
        g = _gamma_vec(c, a_grid, Om0, w0, wa)
        g = np.clip(g, 0.05, 1.25)
        f_approx = (Om0 * a_grid**3) ** g
        
        rel_f = np.abs(f_approx - f_ref) / np.maximum(np.abs(f_ref), 1e-12)
        errors.append(rel_f)
        
        # Compute D_approx via integration
        D_approx = np.ones_like(a_grid)
        for i, a in enumerate(a_grid):
            if a >= 1.0:
                continue
            x_lo = math.log(a)
            n_pts = 80
            xs = np.linspace(x_lo, 0.0, n_pts)
            a_x = np.exp(xs)
            Om_x = Om0 * a_x**3
            q_x = 1.0 - Om_x
            u_x = 1.0 + w0 + wa * (1.0 - a_x)
            g_x = c[0] + c[1]*q_x + c[2]*q_x**2 + c[3]*u_x + c[4]*wa*a_x + c[5]*u_x*q_x
            g_x = np.clip(g_x, 0.05, 1.25)
            f_x = Om_x ** g_x
            integral = np.trapz(f_x, xs)
            D_approx[i] = math.exp(-integral)
        
        rel_D = np.abs(D_approx - D_ref) / np.maximum(np.abs(D_ref), 1e-12)
        errors.append(rel_D)
    
    if not errors:
        return 1e10
    
    all_err = np.concatenate(errors)
    # Soft max
    s = sharpness
    m = np.max(all_err)
    soft_max = m + np.log(np.sum(np.exp(s * (all_err - m)))) / s
    return 100.0 * soft_max


def construct():
    start = time.time()
    deadline = start + 78.0
    
    seed = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_c = seed.copy()
    best_val = float('inf')
    
    # Diverse test cosmologies covering the expected validation range
    MODELS = [
        (0.30, -1.0, 0.0),
        (0.28, -1.0, 0.0),
        (0.30, -1.0, 0.3),
        (0.30, -1.0, -0.3),
        (0.25, -0.9, 0.2),
        (0.35, -1.1, 0.1),
        (0.30, -0.8, 0.4),
        (0.32, -1.2, -0.2),
    ]
    
    a_grid = np.linspace(0.05, 0.98, 30)
    
    # Precompute references
    for Om0, w0, wa in MODELS:
        try:
            _solve_ode(Om0, w0, wa, a_grid)
        except Exception:
            pass
    
    def obj(c):
        if time.time() > deadline:
            return 1e10
        return _objective(c, MODELS, a_grid, sharpness=50.0)
    
    try:
        best_val = obj(best_c)
    except Exception:
        best_val = 1e10
    
    bounds = [(-10, 10)] * 6
    
    # Phase 1: Differential evolution
    try:
        if time.time() < deadline - 35:
            res = differential_evolution(
                obj, bounds, seed=42, maxiter=25, tol=1e-6,
                polish=False, init='sobol', popsize=15
            )
            if res.fun < best_val:
                best_val = res.fun
                best_c = np.clip(res.x, -10, 10)
    except Exception:
        pass
    
    # Phase 2: L-BFGS-B local refinement
    try:
        if time.time() < deadline - 15:
            res = minimize(
                obj, best_c, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_c = np.clip(res.x, -10, 10)
    except Exception:
        pass
    
    # Phase 3: Sharp polish
    try:
        if time.time() < deadline - 5:
            def obj_sharp(c):
                if time.time() > deadline:
                    return 1e10
                return _objective(c, MODELS, a_grid, sharpness=200.0)
            
            res = minimize(
                obj_sharp, best_c, method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_c = np.clip(res.x, -10, 10)
    except Exception:
        pass
    
    best_c = np.array(best_c)
    if not np.all(np.isfinite(best_c)):
        best_c = seed
    best_c = np.clip(best_c, -10, 10)
    
    # Final gamma validation on dense grid
    a_dense = np.linspace(0.01, 0.999, 100)
    for Om0, w0, wa in MODELS:
        g = _gamma_vec(best_c, a_dense, Om0, w0, wa)
        if np.any(g < 0.1) or np.any(g > 1.2):
            best_c = seed
            break
    
    return best_c.tolist()
# EVOLVE-BLOCK-END

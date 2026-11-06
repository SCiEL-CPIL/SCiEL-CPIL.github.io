"""Improved growth-index formula optimizer for CPL dark-energy cosmologies."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
import time

def _Om_a(a, Om0, w0, wa):
    """Matter density parameter at scale factor a."""
    a = np.asarray(a, dtype=float)
    # For CPL: rho_m ~ a^-3, rho_de ~ a^(-3(1+w0+wa))
    # Om(a) = Om0 * a^-3 / (Om0 * a^-3 + (1-Om0) * a^(-3(1+w0+wa)))
    # = 1 / (1 + (1-Om0)/Om0 * a^(3*wa))
    exponent = 3.0 * wa
    ratio = (1.0 - Om0) / Om0 * np.power(a, exponent)
    return 1.0 / (1.0 + ratio)

def _solve_growth_ode(Om0, w0, wa, a_min=0.05, a_max=1.0, n_points=200):
    """Solve the GR ODE for growth factor and extract gamma."""
    a_grid = np.linspace(a_min, a_max, n_points)
    
    def ode_func(a, y):
        # y = [D, dD/da]
        D, dD = y
        Om = _Om_a(a, Om0, w0, wa)
        # d²D/da² + (3/a + H'/H) dD/da - 3 Om(a) H²/H dD/da = 0
        # Actually: d²D/da² + (3/a + d(ln H)/da) dD/da - (3/2) Om(a) (H/H)² dD/da = 0
        # Standard form: d²D/da² + (3/a + q) dD/da - (3/2) Om(a) dD/da = 0
        # where q = -a d(ln H)/da... let me use the standard form
        # The growth equation in terms of a:
        # D'' + (3/a + H'/H) D' = (3/2) Om(a) D'  ... no
        # Correct: D'' + (3/a + dlnH/da) D' - (3/2) Om(a) (H/H) D' = 0
        # Simplified: D'' + (3/a + q_dyn) D' = (3/2) Om(a) D'
        # where q_dyn = -a dlnH/da
        # For CPL: H²/H0² = Om0 a^-3 + (1-Om0) a^(-3(1+w0+wa))
        # dlnH/da = (1/2) dln(H²)/da
        H2 = Om0 * a**(-3) + (1.0 - Om0) * a**(-3*(1+w0+wa))
        dH2_da = -3*Om0*a**(-4) + (1-Om0)*(-3*(1+w0+wa))*a**(-3*(1+w0+wa)-1)
        dlnH_da = 0.5 * dH2_da / H2
        q_dyn = a * dlnH_da  # = a * dlnH/da
        # Growth ODE: D'' + (3/a + q_dyn) D' - (3/2) Om(a) D' = 0
        # Wait, the standard form is:
        # D'' + (3/a + dlnH/da) D' = (3/2) Om(a) (H/H) D'  ... this isn't right either
        # Let me use: f = dlnD/dlna, then f' + f² + (3/2)(1+Om) - (3/2)Om = 0
        # where prime = d/dlna
        # In terms of a: a f' + f² + (3/2)(1+Om) - (3/2)Om = 0
        # => a f' = -f² - (3/2)(1+Om) + (3/2)Om = -f² - 3/2 + Om
        # So: f' = (-f² - 1.5 + Om(a)) / a
        # And D' = f * D / a
        Om_a = _Om_a(a, Om0, w0, wa)
        f = y[1] / y[0] / a if y[0] != 0 else 0.55  # f = dlnD/dlna = a*D'/D
        # Actually let me track [D, f] where f = dlnD/dlna
        # D' = f*D/a
        # f' = (-f² - 1.5 + Om(a)) / a
        return [f * D / a, (-f**2 - 1.5 + Om_a) / a]
    
    # Initial conditions at a=1: D(1)=1, f(1) ≈ 0.55 (approximate)
    # Better: use f(1) = Om0**0.55 as initial guess, then refine
    y0 = [1.0, Om0**0.55]
    
    sol = solve_ivp(ode_func, [a_min, a_max], y0, t_eval=a_grid, 
                    method='DOP853', rtol=1e-10, atol=1e-12, max_step=0.01)
    
    if not sol.success:
        return None, None
    
    D = sol.y[0]
    f_num = sol.y[1] / sol.y[0] / a_grid  # f = dlnD/dlna = a*D'/D... 
    # Wait, I defined y[1] as f = dlnD/dlna, so f_num = sol.y[1]
    # And D' = f*D/a, so dD/da = f*D/a
    f_num = sol.y[1]  # this is already f = dlnD/dlna
    
    # Extract gamma from f = Om(a)^gamma => gamma = ln(f) / ln(Om(a))
    Om_a = _Om_a(a_grid, Om0, w0, wa)
    # Avoid log of values near 0 or 1
    mask = (Om_a > 0.01) & (Om_a < 0.99) & (f_num > 0.01)
    gamma_true = np.full_like(a_grid, 0.55)
    if np.any(mask):
        gamma_true[mask] = np.log(f_num[mask]) / np.log(Om_a[mask])
    
    return a_grid, gamma_true, D

def _compute_gamma_from_coeffs(coeffs, a, Om0, w0, wa):
    """Compute gamma from the 6-coefficient formula."""
    c0, c1, c2, c3, c4, c5 = coeffs
    Om = _Om_a(a, Om0, w0, wa)
    q = 1.0 - Om
    u = 1.0 + w0 + wa * (1.0 - a)
    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
    return gamma

def _objective(coeffs, test_cosmos, a_grid, gamma_refs, D_refs):
    """Compute max relative error in f and D across all test cosmologies."""
    c0, c1, c2, c3, c4, c5 = coeffs
    max_err = 0.0
    
    for i, (Om0, w0, wa) in enumerate(test_cosmos):
        a = a_grid[i]
        gamma_ref = gamma_refs[i]
        D_ref = D_refs[i]
        
        # Compute our gamma and f
        gamma_our = _compute_gamma_from_coeffs(coeffs, a, Om0, w0, wa)
        
        # Enforce gamma in [0.1, 1.2]
        gamma_our = np.clip(gamma_our, 0.1, 1.2)
        
        Om = _Om_a(a, Om0, w0, wa)
        f_our = np.power(Om, gamma_our)
        f_ref = np.power(Om, gamma_ref)
        
        # Relative error in f
        err_f = np.abs(f_our - f_ref) / np.maximum(np.abs(f_ref), 1e-10)
        max_err = max(max_err, np.max(err_f))
        
        # Compute D_our from f_our
        # D(a)/D(1) = exp(-∫_0^a f(a')/a' da')
        # Use trapezoidal integration
        x = np.log(a)
        f_over_a = f_our / a
        integral = np.zeros_like(a)
        for j in range(1, len(a)):
            # ∫ from x[0] to x[j] of f(exp(x))/exp(x) dx = ∫ f(a)/a da
            # Using trapezoidal on a-grid
            integral[j] = integral[j-1] + 0.5*(f_over_a[j] + f_over_a[j-1])*(a[j]-a[j-1])
        
        D_our = np.exp(-integral)
        
        # Normalize both to D(1)=1 (they already are since integral from a_min)
        # Actually D(a)/D(1) means D(a) normalized so D(1)=1
        # Our integral is from a_min to a, so D_our[a_min]=1, not D_our[1]=1
        # Need to normalize: D_norm(a) = D_our(a) / D_our(1)
        # But D_our(1) = exp(-integral at a=1)
        D_our_norm = D_our / D_our[-1]  # D_our[-1] is at a=1
        
        # Same for reference
        # D_ref is already the growth factor with D(1)=1 from ODE
        # Actually from ODE, D(1)=1 by initial condition
        D_ref_norm = D_ref / D_ref[-1]
        
        # Relative error in D
        err_D = np.abs(D_our_norm - D_ref_norm) / np.maximum(np.abs(D_ref_norm), 1e-10)
        max_err = max(max_err, np.max(err_D))
    
    return max_err

def construct():
    """Optimize 6 coefficients for the growth-index formula."""
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    # Test cosmologies to fit (representative range)
    test_cosmos = [
        (0.30, -1.0, 0.0),   # LCDM
        (0.30, -0.9, 0.2),   # CPL with positive wa
        (0.30, -1.1, 0.0),   # Phantom-ish
        (0.30, -0.8, 0.3),   # Stronger wa
        (0.30, -1.0, 0.5),   # Large wa
        (0.30, -1.2, 0.0),   # More negative w0
    ]
    
    # Generate reference data
    a_grid_all = []
    gamma_refs = []
    D_refs = []
    
    for Om0, w0, wa in test_cosmos:
        result = _solve_growth_ode(Om0, w0, wa, a_min=0.1, a_max=1.0, n_points=150)
        if result is None:
            continue
        a_grid, gamma_true, D = result
        a_grid_all.append(a_grid)
        gamma_refs.append(gamma_true)
        D_refs.append(D)
    
    if len(a_grid_all) == 0:
        return [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Initial guess: constant gamma
    best_coeffs = [0.55, 0.0, 0.0, 0.0, 0.0, 0.0]
    best_score = _objective(best_coeffs, test_cosmos[:len(a_grid_all)], a_grid_all, gamma_refs, D_refs)
    
    # Differential evolution
    bounds = [(-1.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
    
    # Run DE with time check
    def de_objective(x):
        if time.time() > deadline:
            return 100.0  # Penalty
        return _objective(x, test_cosmos[:len(a_grid_all)], a_grid_all, gamma_refs, D_refs)
    
    try:
        result = differential_evolution(
            de_objective, bounds,
            seed=42, maxiter=50, popsize=15,
            tol=1e-6, mutation=(0.5, 1.0), recombination=0.8,
            polish=True, updating='immediate'
        )
        if result.success and result.fun < best_score:
            best_coeffs = result.x
            best_score = result.fun
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 5.0:
        # Not enough time for local refinement
        return [float(c) for c in best_coeffs]
    
    # Local refinement with Nelder-Mead
    from scipy.optimize import minimize
    try:
        result2 = minimize(
            lambda x: _objective(x, test_cosmos[:len(a_grid_all)], a_grid_all, gamma_refs, D_refs),
            best_coeffs, method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        if result2.fun < best_score:
            best_coeffs = result2.x
            best_score = result2.fun
    except Exception:
        pass
    
    # Ensure coefficients are in [-10, 10]
    best_coeffs = [float(np.clip(c, -10.0, 10.0)) for c in best_coeffs]
    
    return best_coeffs
# EVOLVE-BLOCK-END

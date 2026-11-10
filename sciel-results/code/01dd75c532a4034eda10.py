"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.integrate import solve_ivp
# EVOLVE-BLOCK-START

def _ode_rhs(x, y, Om0, w0, wa):
    a = np.exp(x)
    Om = Om0 * a**(-3)
    Ode = 1.0 - Om
    w = w0 + wa * (1.0 - a)
    coeff = 1.5 * (Om + Ode * (1.0 + 3.0 * w))
    return [y[1], -coeff * y[1] + 1.5 * Om * y[0]]

def _numerical_f_and_D(x_grid, Om0, w0, wa):
    """Solve ODE and extract the physical growing mode."""
    x0 = np.log(0.01)
    x1 = 0.0
    # Solve with two basis solutions
    sol1 = solve_ivp(_ode_rhs, [x0, x1], [1.0, 0.0], args=(Om0, w0, wa),
                     method='DOP853', rtol=1e-10, atol=1e-12, dense_output=True)
    sol2 = solve_ivp(_ode_rhs, [x0, x1], [0.0, 1.0], args=(Om0, w0, wa),
                     method='DOP853', rtol=1e-10, atol=1e-12, dense_output=True)
    
    # Evaluate at grid points
    D1 = sol1.sol(x_grid)[0]
    D1p = sol1.sol(x_grid)[1]
    D2 = sol2.sol(x_grid)[0]
    D2p = sol2.sol(x_grid)[1]
    
    # The physical growing mode is the one that grows (D > 0 and increasing)
    # In CPL cosmologies, the growing mode dominates at late times
    # Use the solution with larger D at late times
    # Combine: D = A*D1 + B*D2, choose so that D is positive and growing
    # Simple approach: use D2 (initial slope 1) which is typically the growing mode
    # But let's be more careful: find the combination that is positive at a=1
    D_at_1 = D2[-1]
    if D_at_1 < 0:
        D_phys = -D2
        Dp_phys = -D2p
    else:
        D_phys = D2
        Dp_phys = D2p
    
    # Normalize D(1) = 1
    Dn = D_phys / D_phys[-1]
    fn = Dp_phys / D_phys
    return fn, Dn

def _gamma_of(a, Om0, w0, wa, coeffs):
    c0, c1, c2, c3, c4, c5 = coeffs
    Om = Om0 * a**(-3)
    q = 1.0 - Om
    u = 1.0 + w0 + wa * (1.0 - a)
    return c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q

def _compute_relative_errors(coeffs, params, a_grid):
    """Compute max relative error in f and normalized D."""
    x_grid = np.log(a_grid)
    max_err_f = 0.0
    max_err_D = 0.0
    
    for (Om0, w0, wa) in params:
        fn_num, Dn_num = _numerical_f_and_D(x_grid, Om0, w0, wa)
        
        Om = Om0 * a_grid**(-3)
        q = 1.0 - Om
        u = 1.0 + w0 + wa * (1.0 - a_grid)
        
        gamma = _gamma_of(a_grid, Om0, w0, wa, coeffs)
        
        # Compute f and D from gamma
        valid = Om > 1e-6
        f_approx = np.ones_like(Om)
        D_approx = np.ones_like(Om)
        
        if np.any(valid):
            f_approx[valid] = Om[valid] ** gamma[valid]
            # D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)
            # = exp(-integral from 0 to ln(a) of f(exp(x)) dx) with sign flip
            # Actually: D(a)/D(1) = exp(-∫_a^1 f(a')/a' da') = exp(-∫_ln(a)^0 f(e^x) dx)
            x_vals = np.linspace(np.log(a_grid[valid]), 0, 100)
            f_vals = np.exp(np.log(Om0 * np.exp(x_vals)**(-3)) * gamma)
            # Wait, need to be more careful
            # f(e^x) = Om(e^x)^gamma = (Om0 * e^{-3x})^gamma
            # But gamma depends on a, so we need to evaluate gamma at each point
            # This is complex; let's use numerical integration
            from scipy.integrate import quad
            D_ratios = []
            for ai in a_grid[valid]:
                xi = np.log(ai)
                def integrand(x):
                    a_x = np.exp(x)
                    Om_x = Om0 * a_x**(-3)
                    q_x = 1.0 - Om_x
                    u_x = 1.0 + w0 + wa * (1.0 - a_x)
                    gamma_x = c0 + c3*u_x + c4*wa*a_x + c5*u_x*q_x
                    return Om_x ** gamma_x
                val, _ = quad(integrand, xi, 0, limit=100)
                D_ratios.append(np.exp(-val))
            D_approx[valid] = D_ratios
        
        # Compute relative errors
        err_f = np.abs(f_approx - fn_num) / np.abs(fn_num)
        err_D = np.abs(D_approx - Dn_num) / np.abs(Dn_num)
        
        # Only consider valid regions
        err_f = err_f[valid]
        err_D = err_D[valid]
        
        if len(err_f) > 0:
            max_err_f = max(max_err_f, np.max(err_f))
            max_err_D = max(max_err_D, np.max(err_D))
    
    return max_err_f, max_err_D

def _fit_coefficients():
    """Fit c0, c3, c4, c5 using grid search + refinement."""
    params = [
        (0.30, -1.0, 0.0),
        (0.30, -0.9, 0.3),
        (0.30, -1.1, 0.5),
        (0.25, -1.0, 0.2),
        (0.35, -0.8, 0.4),
        (0.30, -1.0, -0.5),
        (0.28, -1.2, 0.6),
        (0.32, -0.7, 0.1),
    ]
    a_grid = np.logspace(-2, 0, 50)  # Fewer points for speed
    
    # Use a smarter approach: fit gamma directly but with proper weighting
    # gamma = ln(f) / ln(Om), fit linear model gamma ≈ c0 + c3*u + c4*wa*a + c5*u*q
    rows = []
    targets = []
    weights = []
    
    for (Om0, w0, wa) in params:
        x_grid = np.log(a_grid)
        fn_num, Dn_num = _numerical_f_and_D(x_grid, Om0, w0, wa)
        
        Om = Om0 * a_grid**(-3)
        q = 1.0 - Om
        u = 1.0 + w0 + wa * (1.0 - a_grid)
        
        valid = (Om > 0.01) & (Om < 0.99) & (fn_num > 0.01)
        
        for i in range(len(a_grid[valid])):
            ai = a_grid[valid][i]
            ui = u[valid][i]
            qi = q[valid][i]
            fi = fn_num[valid][i]
            Omi = Om[valid][i]
            
            gamma_target = np.log(fi) / np.log(Omi)
            rows.append([1.0, ui, wa * ai, ui * qi])
            targets.append(gamma_target)
            # Weight by how well-defined gamma is
            weights.append(abs(np.log(Omi)))
    
    if len(rows) == 0:
        return [0.55, 0.0, 0.0, 0.05, 0.1, 0.0]
    
    A = np.array(rows)
    b = np.array(targets)
    w = np.array(weights)
    
    # Weighted least squares
    Aw = A * w[:, None]
    bw = b * w
    sol, res, rank, sv = np.linalg.lstsq(Aw, bw, rcond=None)
    c0, c3, c4, c5 = sol
    
    # Refine with local search
    best_coeffs = [c0, 0.0, 0.0, c3, c4, c5]
    best_err = np.inf
    
    # Try small perturbations
    for dc0 in [-0.02, -0.01, 0.0, 0.01, 0.02]:
        for dc3 in [-0.02, -0.01, 0.0, 0.01, 0.02]:
            for dc4 in [-0.02, -0.01, 0.0, 0.01, 0.02]:
                for dc5 in [-0.02, -0.01, 0.0, 0.01, 0.02]:
                    trial = [c0+dc0, 0.0, 0.0, c3+dc3, c4+dc4, c5+dc5]
                    # Quick error estimate using a subset
                    err_f, err_D = _compute_relative_errors(trial, params[:4], a_grid[:20])
                    err = max(err_f, err_D)
                    if err < best_err:
                        best_err = err
                        best_coeffs = trial
    
    return best_coeffs

def construct():
    try:
        coeffs = _fit_coefficients()
        coeffs = [float(np.clip(c, -10.0, 10.0)) for c in coeffs]
        if not all(np.isfinite(c) for c in coeffs):
            raise ValueError
        return coeffs
    except Exception:
        return [0.55, 0.0, 0.0, 0.05, 0.1, 0.0]
# EVOLVE-BLOCK-END

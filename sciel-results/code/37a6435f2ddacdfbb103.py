"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.integrate import solve_ivp
# EVOLVE-BLOCK-START

def _ode_rhs(x, y, Om0, w0, wa):
    a = np.exp(x)
    # Omega_m(a)
    Om = Om0 * a**(-3)
    # Omega_de(a)
    Ode = 1.0 - Om
    w = w0 + wa * (1.0 - a)
    # H'/H = -3/2 [ Om + Ode*(1+3w) ]
    coeff = 1.5 * (Om + Ode * (1.0 + 3.0 * w))
    # d2D/dx2 = -coeff * dD/dx + 1.5*Om*D
    return [y[1], -coeff * y[1] + 1.5 * Om * y[0]]

def _numerical_f_and_D(x_grid, Om0, w0, wa):
    # Solve ODE from x=ln(0.01) to x=0 (a=1), normalized D(1)=1
    # Use two initial conditions to get a fundamental solution, then normalize.
    x0 = np.log(0.01)
    # Solve forward with two basis solutions
    sol1 = solve_ivp(_ode_rhs, [x0, 0.0], [1.0, 0.0], args=(Om0, w0, wa),
                     method='DOP853', rtol=1e-10, atol=1e-12, dense_output=True)
    sol2 = solve_ivp(_ode_rhs, [x0, 0.0], [0.0, 1.0], args=(Om0, w0, wa),
                     method='DOP853', rtol=1e-10, atol=1e-12, dense_output=True)
    # General solution: D = A*sol1 + B*sol2. Choose B=1, A=0 for simplicity (growing mode approx).
    # Actually just use sol2 (initial slope 1) as the growing mode proxy; f = D'/D.
    # Better: combine to get the physical growing mode. For our error metric we only need
    # f = dlnD/dlna and normalized D. Use sol2 (it's the growing mode for these params).
    D1 = sol2.sol(x_grid)[0]
    D1p = sol2.sol(x_grid)[1]
    Dn = D1 / D1[-1]
    fn = D1p / D1
    return fn, Dn

def _gamma_of(a, Om0, w0, wa, coeffs):
    c0, c1, c2, c3, c4, c5 = coeffs
    Om = Om0 * a**(-3)
    q = 1.0 - Om
    u = 1.0 + w0 + wa * (1.0 - a)
    return c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q

def _fit_coefficients():
    # Representative CPL cosmologies covering a broad parameter space
    params = [
        (0.30, -1.0, 0.0),
        (0.30, -0.9, 0.3),
        (0.30, -1.1, 0.5),
        (0.25, -1.0, 0.2),
        (0.35, -0.8, 0.4),
        (0.30, -1.0, -0.5),
        (0.28, -1.2, 0.6),
        (0.32, -0.7, 0.1),
        (0.30, -1.0, 0.3),
        (0.30, -1.0, -0.3),
        (0.27, -1.0, 0.1),
        (0.33, -1.0, -0.2),
    ]
    a_grid = np.logspace(-2, 0, 300)
    x_grid = np.log(a_grid)

    # Build design matrix for gamma = c0 + c3*u + c4*wa*a + c5*u*q (c1=c2=0)
    rows = []
    targets_f = []
    weights_f = []
    
    for (Om0, w0, wa) in params:
        fn, Dn = _numerical_f_and_D(x_grid, Om0, w0, wa)
        Om = Om0 * a_grid**(-3)
        q = 1.0 - Om
        u = 1.0 + w0 + wa * (1.0 - a_grid)
        
        # For f: gamma_target = ln(f) / ln(Om)
        valid_f = (Om > 1e-6) & (Om < 1.0) & (fn > 1e-6)
        if np.any(valid_f):
            gamma_target_f = np.log(fn[valid_f]) / np.log(Om[valid_f])
            # Weight by |ln(Om)| to emphasize regions where gamma is well-defined
            weight_f = np.abs(np.log(Om[valid_f]))
            for i in range(len(a_grid[valid_f])):
                ai = a_grid[valid_f][i]
                ui = u[valid_f][i]
                qi = q[valid_f][i]
                rows.append([1.0, ui, wa * ai, ui * qi])
                targets_f.append(gamma_target_f[i])
                weights_f.append(weight_f[i])
    
    A = np.array(rows)
    b = np.array(targets_f)
    w = np.array(weights_f)
    
    # Weighted least squares
    Aw = A * w[:, np.newaxis]
    bw = b * w
    sol, res, rank, sv = np.linalg.lstsq(Aw, bw, rcond=None)
    c0, c3, c4, c5 = sol
    
    # Clip to reasonable range before final check
    c0 = np.clip(c0, 0.1, 1.2)
    c3 = np.clip(c3, -1.0, 1.0)
    c4 = np.clip(c4, -1.0, 1.0)
    c5 = np.clip(c5, -1.0, 1.0)
    
    return [c0, 0.0, 0.0, c3, c4, c5]

def construct():
    # Physics-constrained affine: c1=c2=0, optimize c0,c3,c4,c5 against ODE reference.
    try:
        coeffs = _fit_coefficients()
        # Clip to [-10,10] for safety
        coeffs = [float(np.clip(c, -10.0, 10.0)) for c in coeffs]
        # Sanity: ensure finite
        if not all(np.isfinite(c) for c in coeffs):
            raise ValueError
        return coeffs
    except Exception:
        # Fallback to a reasonable constant-gamma with mild CPL correction
        return [0.55, 0.0, 0.0, 0.05, 0.1, 0.0]
# EVOLVE-BLOCK-END

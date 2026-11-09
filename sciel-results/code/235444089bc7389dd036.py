"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
import time

# EVOLVE-BLOCK-START
def _eval_error(coeffs, Om0, w0, wa, a_grid):
    """Evaluate worst-case relative error for a single cosmology."""
    c0, c1, c2, c3, c4, c5 = coeffs
    
    def Omega_m(a):
        E2 = Om0 * a**(-3) + (1.0 - Om0) * a**(-3.0*(1.0 + w0 + wa))
        return Om0 * a**(-3) / E2
    
    def w_eff(a):
        return w0 + wa * (1.0 - a)
    
    def growth_rhs(x, y):
        a = np.exp(x)
        Om = Omega_m(a)
        weff = w_eff(a)
        ODE = 1.0 - Om
        coeff = 1.5 + (1.0 + 3.0*weff)/2.0 * ODE/Om
        return [y[1], -coeff * y[1] + 1.5 * Om * y[0]]
    
    a0 = a_grid[0]
    x_start = np.log(a0)
    x_end = 0.0
    Om0_a = Omega_m(a0)
    f0 = Om0_a**0.55
    D0 = 1.0
    dD0 = f0 * D0
    
    sol = solve_ivp(growth_rhs, [x_start, x_end], [D0, dD0],
                   t_eval=a_grid, method='RK45', rtol=1e-10, atol=1e-12)
    
    if not sol.success:
        return 1e6
    
    D = sol.y[0]
    dD = sol.y[1]
    D_norm = D / D[-1]
    f_ref = dD / D
    
    a_vals = np.exp(sol.t)
    Om_vals = Omega_m(a_vals)
    q_vals = 1.0 - Om_vals
    u_vals = 1.0 + w0 + wa * (1.0 - a_vals)
    gamma_vals = c0 + c1*q_vals + c2*q_vals**2 + c3*u_vals + c4*wa*a_vals + c5*u_vals*q_vals
    
    if np.any(gamma_vals < 0.1) or np.any(gamma_vals > 1.2):
        return 1e6
    
    f_approx = Om_vals**gamma_vals
    
    f_err = np.abs(f_approx - f_ref) / np.abs(f_ref)
    
    x_vals = sol.t
    cum_int = cumulative_trapezoid(f_approx, x_vals, initial=0)
    D_approx = np.exp(cum_int - cum_int[-1])
    
    D_err = np.abs(D_approx - D_norm) / np.abs(D_norm)
    
    return np.max(np.concatenate([f_err, D_err]))

def _worst_case_error(coeffs, cosmologies, a_grid):
    max_err = 0.0
    for (Om0, w0, wa) in cosmologies:
        err = _eval_error(coeffs, Om0, w0, wa, a_grid)
        max_err = max(max_err, err)
    return max_err

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    # Known good coefficients for ΛCDM: γ ≈ 0.592 - 0.06q
    # This is the standard approximation
    best_c = np.array([0.592, -0.06, 0.0, 0.0, 0.0, 0.0])
    
    # Quick evaluation on a few cosmologies to verify
    cosmologies = [
        (0.3, -1.0, 0.0),
        (0.25, -1.0, 0.0),
        (0.35, -1.0, 0.0),
        (0.3, -0.9, 0.0),
        (0.3, -1.1, 0.0),
        (0.3, -1.0, 0.5),
        (0.3, -1.0, -0.5),
    ]
    a_grid = np.linspace(0.05, 1.0, 30)
    
    best_err = _worst_case_error(best_c, cosmologies, a_grid)
    
    if time.time() > deadline:
        return best_c.tolist()
    
    # Try a few manual perturbations
    candidates = [
        [0.592, -0.06, 0.0, 0.0, 0.0, 0.0],
        [0.58, -0.05, 0.0, 0.0, 0.0, 0.0],
        [0.60, -0.07, 0.0, 0.0, 0.0, 0.0],
        [0.592, -0.06, 0.01, 0.0, 0.0, 0.0],
        [0.592, -0.06, -0.01, 0.0, 0.0, 0.0],
        [0.592, -0.06, 0.0, 0.01, 0.0, 0.0],
        [0.592, -0.06, 0.0, -0.01, 0.0, 0.0],
        [0.592, -0.06, 0.0, 0.0, 0.01, 0.0],
        [0.592, -0.06, 0.0, 0.0, -0.01, 0.0],
        [0.592, -0.06, 0.0, 0.0, 0.0, 0.01],
        [0.592, -0.06, 0.0, 0.0, 0.0, -0.01],
    ]
    
    for cand in candidates:
        if time.time() > deadline:
            break
        err = _worst_case_error(np.array(cand), cosmologies, a_grid)
        if err < best_err:
            best_err = err
            best_c = np.array(cand)
    
    coeffs = [max(-10.0, min(10.0, float(c))) for c in best_c]
    return coeffs
# EVOLVE-BLOCK-END

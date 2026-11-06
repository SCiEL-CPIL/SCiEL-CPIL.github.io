"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import time

_REF_COSMOLOGIES = [
    (0.27, -1.0, 0.0),
    (0.30, -1.0, 0.0),
    (0.27, -1.2, 0.4),
    (0.30, -0.9, 0.2),
    (0.25, -1.0, 0.0),
    (0.32, -1.0, 0.0),
    (0.27, -1.1, 0.3),
    (0.30, -0.8, 0.1),
]

_REF_A_GRID = np.linspace(0.05, 1.0, 150)

def _Om_m(a, Om_m0):
    return Om_m0 * a**(-3)

def _Om_de(a, Om_m0, w0, wa):
    Om_de0 = 1.0 - Om_m0
    return Om_de0 * np.exp(-3.0 * (1.0 + w0 + wa) * (1.0 - a))

def _compute_reference_f_and_D(cosmo, a_grid):
    Om_m0, w0, wa = cosmo
    n = len(a_grid)
    
    def ode_f(x, f):
        a = np.exp(x)
        Om_m = _Om_m(a, Om_m0)
        Om_de = _Om_de(a, Om_m0, w0, wa)
        w = w0 + wa * (1.0 - a)
        return -1.5 * Om_m - 0.5 * Om_de * (1.0 + 3.0 * w)
    
    gamma_0 = 0.55
    f0 = Om_m0 ** gamma_0
    
    x_grid = np.log(a_grid)
    x_eval = np.sort(x_grid)
    sol = solve_ivp(ode_f, [0.0, x_eval[0]], [f0], t_eval=x_eval, 
                    method='RK45', rtol=1e-10, atol=1e-12)
    f_vals_sorted = sol.y[0]
    
    sort_idx = np.argsort(np.argsort(x_grid))
    f_ref = f_vals_sorted[sort_idx]
    
    # Compute D ratio
    x_sorted = x_eval
    f_sorted = f_vals_sorted
    dx = np.diff(x_sorted)
    avg_f = 0.5 * (f_sorted[:-1] + f_sorted[1:])
    dF = avg_f * dx
    cum_int = np.cumsum(dF[::-1])[::-1]
    
    sort_idx2 = np.argsort(np.argsort(x_grid))
    cum_int_orig = cum_int[sort_idx2]
    D_ratio = np.exp(-cum_int_orig)
    
    return f_ref, D_ratio

def _compute_model_f_and_D(coeffs, cosmo, a_grid):
    c0, c1, c2, c3, c4, c5 = coeffs
    Om_m0, w0, wa = cosmo
    a = a_grid
    Om_m = _Om_m(a, Om_m0)
    q = 1.0 - Om_m
    u = 1.0 + w0 + wa * (1.0 - a)
    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
    gamma = np.clip(gamma, 0.1, 1.2)
    f_model = Om_m ** gamma
    
    x_grid = np.log(a_grid)
    x_sorted = np.sort(x_grid)
    a_sorted = np.exp(x_sorted)
    Om_m_s = _Om_m(a_sorted, Om_m0)
    q_s = 1.0 - Om_m_s
    u_s = 1.0 + w0 + wa * (1.0 - a_sorted)
    gamma_s = c0 + c1*q_s + c2*q_s**2 + c3*u_s + c4*wa*a_sorted + c5*u_s*q_s
    gamma_s = np.clip(gamma_s, 0.1, 1.2)
    f_s = Om_m_s ** gamma_s
    
    dx = np.diff(x_sorted)
    avg_f = 0.5 * (f_s[:-1] + f_s[1:])
    dF = avg_f * dx
    cum_int = np.cumsum(dF[::-1])[::-1]
    
    sort_idx = np.argsort(np.argsort(x_grid))
    cum_int_orig = cum_int[sort_idx]
    D_ratio = np.exp(-cum_int_orig)
    
    return f_model, D_ratio

# Precompute reference data
_REF_DATA = {}
for _i, _cosmo in enumerate(_REF_COSMOLOGIES):
    _REF_DATA[_i] = _compute_reference_f_and_D(_cosmo, _REF_A_GRID)

def _objective(coeffs):
    max_err = 0.0
    for i, cosmo in enumerate(_REF_COSMOLOGIES):
        f_ref, D_ref = _REF_DATA[i]
        f_mod, D_mod = _compute_model_f_and_D(coeffs, cosmo, _REF_A_GRID)
        err_f = np.abs(f_mod - f_ref) / np.abs(f_ref)
        err_D = np.abs(D_mod - D_ref) / np.abs(D_ref)
        max_err = max(max_err, np.max(err_f), np.max(err_D))
    return 100.0 * max_err

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    best_x = x0.copy()
    best_val = _objective(x0)
    
    try:
        result = minimize(_objective, x0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8,
                                   'adaptive': True})
        if result.fun < best_val:
            best_x = result.x
            best_val = result.fun
    except Exception:
        pass
    
    if time.time() > deadline:
        best_x = np.clip(best_x, -10.0, 10.0)
        best_x = np.nan_to_num(best_x, nan=0.0, posinf=10.0, neginf=-10.0)
        return best_x.tolist()
    
    try:
        result2 = minimize(_objective, best_x, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10,
                                    'adaptive': True})
        if result2.fun < best_val:
            best_x = result2.x
            best_val = result2.fun
    except Exception:
        pass
    
    best_x = np.clip(best_x, -10.0, 10.0)
    best_x = np.nan_to_num(best_x, nan=0.0, posinf=10.0, neginf=-10.0)
    return best_x.tolist()
# EVOLVE-BLOCK-END

"""Improved growth-index approximation with optimized coefficients."""
import numpy as np
from scipy.optimize import minimize, least_squares
from scipy.integrate import quad
import time

# EVOLVE-BLOCK-START
def _Om_of_a(a, Om0, w0, wa):
    """Matter density parameter at scale factor a."""
    Omega_DE0 = 1.0 - Om0
    ratio_DE = (a ** (-3*(1+w0+wa))) * np.exp(3*wa*(a-1))
    Om_a = Om0 * a**(-3) / (Om0 * a**(-3) + Omega_DE0 * ratio_DE)
    return Om_a

def _compute_gamma(a, coeffs, Om0, w0, wa):
    """Compute gamma(a) from the six-coefficient ansatz."""
    c0, c1, c2, c3, c4, c5 = coeffs
    Om_a = _Om_of_a(a, Om0, w0, wa)
    q = 1.0 - Om_a
    u = 1.0 + w0 + wa * (1.0 - a)
    gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
    gamma = np.clip(gamma, 0.1, 1.2)
    return gamma

def _f_of_a(a, coeffs, Om0, w0, wa):
    """Compute f(a) = Om(a)^gamma(a)."""
    Om_a = _Om_of_a(a, Om0, w0, wa)
    gamma = _compute_gamma(a, coeffs, Om0, w0, wa)
    return Om_a ** gamma

def _D_normalized(a, coeffs, Om0, w0, wa):
    """Compute D(a)/D(1) = exp(-integral from ln(a) to 0 of f(exp(x)) dx)."""
    if a >= 1.0:
        return 1.0
    ln_a = np.log(a)
    def integrand(x):
        a_x = np.exp(x)
        return _f_of_a(a_x, coeffs, Om0, w0, wa)
    integral_val, _ = quad(integrand, ln_a, 0.0)
    return np.exp(-integral_val)

def _solve_growth_ODE(a_grid, Om0, w0, wa, n_steps=2000):
    """Solve the growth ODE numerically on a grid."""
    a_arr = a_grid
    n = len(a_arr)
    
    # Precompute Om(a) on grid
    Om_arr = np.zeros(n)
    for i, ai in enumerate(a_arr):
        Om_arr[i] = _Om_of_a(ai, Om0, w0, wa)
    
    # H^2/H0^2 = Om(a) * a^-3
    H2 = Om_arr * a_arr**(-3)
    
    # q(a) = 0.5 * d ln H2 / d ln a
    q_arr = np.zeros(n)
    for i in range(1, n-1):
        da = a_arr[i+1] - a_arr[i-1]
        dH2 = H2[i+1] - H2[i-1]
        if abs(H2[i]) > 1e-15:
            q_arr[i] = 0.5 * a_arr[i] * dH2 / (da * H2[i])
        else:
            q_arr[i] = 0.0
    q_arr[0] = q_arr[1]
    q_arr[-1] = q_arr[-2]
    
    # Solve ODE: D' = v, v' = -(3/a - q(a))v + (3/2)Om(a)D
    D = np.zeros(n)
    v = np.zeros(n)
    
    # Boundary condition at early times: D ~ a, v ~ 1
    D[0] = a_arr[0]
    v[0] = 1.0
    
    dt = a_arr[1] - a_arr[0]
    for i in range(n - 1):
        ai = a_arr[i]
        Di = D[i]
        vi = v[i]
        qi = q_arr[i]
        Omi = Om_arr[i]
        
        def rhs(D_val, v_val):
            D_prime = v_val
            v_prime = -(3.0/ai - qi) * v_val + (1.5 * Omi) * D_val
            return D_prime, v_prime
        
        k1_D, k1_v = rhs(Di, vi)
        k2_D, k2_v = rhs(Di + 0.5*dt*k1_D, vi + 0.5*dt*k1_v)
        k3_D, k3_v = rhs(Di + 0.5*dt*k2_D, vi + 0.5*dt*k2_v)
        k4_D, k4_v = rhs(Di + dt*k3_D, vi + dt*k3_v)
        
        D[i+1] = Di + (dt/6.0) * (k1_D + 2*k2_D + 2*k3_D + k4_D)
        v[i+1] = vi + (dt/6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
    
    # Normalize so D(1) = 1
    D_norm = D / D[-1]
    v_norm = v / D[-1]
    
    return D_norm, v_norm

def _reference_f_and_D(a_eval, Om0, w0, wa, a_grid=None):
    """Get reference f and D at a given a value."""
    if a_grid is None:
        a_grid = np.linspace(0.01, 1.0, 2000)
    
    D_arr, v_arr = _solve_growth_ODE(a_grid, Om0, w0, wa)
    
    # Interpolate
    idx = np.searchsorted(a_grid, a_eval)
    if idx <= 0:
        D_val = D_arr[0]
        f_val = v_arr[0] / D_arr[0]
    elif idx >= len(a_grid):
        D_val = D_arr[-1]
        f_val = v_arr[-1] / D_arr[-1]
    else:
        t = (a_eval - a_grid[idx-1]) / (a_grid[idx] - a_grid[idx-1])
        D_val = D_arr[idx-1] + t * (D_arr[idx] - D_arr[idx-1])
        v_val = v_arr[idx-1] + t * (v_arr[idx] - v_arr[idx-1])
        f_val = v_val / D_val
    
    return f_val, D_val

def _check_gamma_bounds(coeffs, test_models, a_values):
    """Check if gamma lies in [0.1, 1.2] for all test models and a values."""
    for (w0, wa, Om0) in test_models:
        for a in a_values:
            c0, c1, c2, c3, c4, c5 = coeffs
            Om_a = _Om_of_a(a, Om0, w0, wa)
            q = 1.0 - Om_a
            u = 1.0 + w0 + wa * (1.0 - a)
            gamma = c0 + c1*q + c2*q**2 + c3*u + c4*wa*a + c5*u*q
            if gamma < 0.1 or gamma > 1.2:
                return False
    return True

def _build_design_matrix(a_values, Om0, w0, wa):
    """Build the 6-column design matrix for the gamma ansatz."""
    n = len(a_values)
    A = np.zeros((n, 6))
    for i, a in enumerate(a_values):
        Om_a = _Om_of_a(a, Om0, w0, wa)
        q = 1.0 - Om_a
        u = 1.0 + w0 + wa * (1.0 - a)
        A[i, 0] = 1.0       # c0
        A[i, 1] = q         # c1
        A[i, 2] = q**2      # c2
        A[i, 3] = u         # c3
        A[i, 4] = wa * a    # c4
        A[i, 5] = u * q     # c5
    return A

def _fit_gamma_taylor(test_models, a_values, ref_cache):
    """
    Fit coefficients using least-squares on log(f) to match exact gamma.
    We use: log(f) = gamma * log(Om(a)), so gamma = log(f)/log(Om(a)).
    This gives a direct target for the gamma ansatz.
    """
    all_A = []
    all_b = []
    
    for (w0, wa, Om0) in test_models:
        A = _build_design_matrix(a_values, Om0, w0, wa)
        for i, a in enumerate(a_values):
            if a >= 1.0:
                continue
            f_ref, D_ref = ref_cache[(w0, wa, Om0)][a]
            Om_a = _Om_of_a(a, Om0, w0, wa)
            if Om_a <= 0 or Om_a >= 1 or f_ref <= 0:
                continue
            gamma_target = np.log(f_ref) / np.log(Om_a)
            all_A.append(A[i])
            all_b.append(gamma_target)
    
    if len(all_A) == 0:
        return np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    A_mat = np.array(all_A)
    b_vec = np.array(all_b)
    
    # Weighted least squares: weight by how close a is to 0.5 (mid-range)
    weights = np.ones(len(b_vec))
    
    try:
        # Solve A * coeffs = b via least squares
        coeffs, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        coeffs = np.clip(coeffs, -10.0, 10.0)
        
        # Check bounds
        if _check_gamma_bounds(coeffs, test_models, a_values):
            return coeffs
        else:
            # Try with clipping and re-solve
            return np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    except Exception:
        return np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])

def construct():
    """Find optimal six coefficients by minimizing max error over test cosmologies."""
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Test cosmologies (w0, wa, Om0) - expanded for better coverage
    test_models = [
        (-1.0, 0.0, 0.3),
        (-1.0, 0.1, 0.3),
        (-0.9, 0.0, 0.3),
        (-1.1, 0.0, 0.3),
        (-1.0, 0.3, 0.3),
        (-0.8, -0.2, 0.3),
        (-1.2, 0.5, 0.3),
        (-1.0, 0.0, 0.27),
        (-0.95, 0.1, 0.28),
        (-1.05, -0.1, 0.32),
        (-1.0, 0.2, 0.3),
        (-0.95, -0.1, 0.3),
        (-1.05, 0.15, 0.3),
        (-1.0, 0.0, 0.35),
        (-1.0, 0.0, 0.25),
    ]
    
    # a values to check - denser for better fitting
    a_values = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
                          0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    
    # Precompute reference values
    print("Precomputing references...")
    ref_cache = {}
    for (w0, wa, Om0) in test_models:
        ref_cache[(w0, wa, Om0)] = {}
        for a in a_values:
            f_ref, D_ref = _reference_f_and_D(a, Om0, w0, wa)
            ref_cache[(w0, wa, Om0)][a] = (f_ref, D_ref)
    print("References precomputed.")
    
    def objective(coeffs):
        """Max relative error over all test models and a values, with penalty for bounds violation."""
        # Check bounds first
        if not _check_gamma_bounds(coeffs, test_models, a_values):
            return 10.0  # Large penalty
        
        max_err = 0.0
        for (w0, wa, Om0) in test_models:
            for a in a_values:
                f_approx = _f_of_a(a, coeffs, Om0, w0, wa)
                D_approx = _D_normalized(a, coeffs, Om0, w0, wa)
                
                f_ref, D_ref = ref_cache[(w0, wa, Om0)][a]
                
                if abs(f_ref) > 1e-10:
                    err_f = abs(f_approx - f_ref) / abs(f_ref)
                else:
                    err_f = abs(f_approx - f_ref)
                
                if abs(D_ref) > 1e-10:
                    err_D = abs(D_approx - D_ref) / abs(D_ref)
                else:
                    err_D = abs(D_approx - D_ref)
                
                max_err = max(max_err, err_f, err_D)
        
        return max_err
    
    # Strategy 1: Taylor/least-squares fit to exact gamma
    taylor_coeffs = _fit_gamma_taylor(test_models, a_values, ref_cache)
    taylor_err = objective(taylor_coeffs)
    print(f"Taylor fit error: {taylor_err:.6f}")
    
    # Strategy 2: Start from constant gamma
    x0 = np.array([0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
    const_err = objective(x0)
    
    best_coeffs = taylor_coeffs if taylor_err < const_err else x0.copy()
    best_err = min(taylor_err, const_err)
    
    # Try Nelder-Mead optimization from best starting point
    try:
        result = minimize(
            objective,
            best_coeffs.copy(),
            method='Nelder-Mead',
            options={'maxiter': 1500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
        )
        
        # Only accept if valid
        if result.fun < best_err and _check_gamma_bounds(result.x, test_models, a_values):
            best_coeffs = result.x
            best_err = result.fun
        
        # Check deadline
        if time.time() > deadline - 10.0:
            return np.clip(best_coeffs, -10.0, 10.0).tolist()
        
        # Try additional local optimizations from diverse starting points
        starting_points = [
            taylor_coeffs.tolist(),
            [0.5, 0.1, 0.0, 0.0, 0.0, 0.0],
            [0.6, -0.1, 0.0, 0.0, 0.0, 0.0],
            [0.55, 0.0, 0.05, 0.0, 0.0, 0.0],
            [0.55, 0.0, 0.0, 0.1, 0.0, 0.0],
            [0.55, 0.0, 0.0, 0.0, 0.1, 0.0],
            [0.55, 0.0, 0.0, 0.0, 0.0, 0.1],
            [0.55, 0.0, 0.0, -0.1, 0.0, 0.0],
            [0.55, 0.0, -0.05, 0.0, 0.0, 0.0],
            [0.55, 0.05, 0.0, 0.05, 0.0, 0.0],
            [0.55, -0.05, 0.0, -0.05, 0.0, 0.0],
        ]
        
        for sp in starting_points:
            if time.time() > deadline - 15.0:
                break
            sp_arr = np.array(sp)
            if not _check_gamma_bounds(sp_arr, test_models, a_values):
                continue
            try:
                res = minimize(
                    objective,
                    sp_arr,
                    method='Nelder-Mead',
                    options={'maxiter': 800, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
                )
                if res.fun < best_err and _check_gamma_bounds(res.x, test_models, a_values):
                    best_coeffs = res.x
                    best_err = res.fun
            except Exception:
                pass
        
    except Exception:
        pass
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 5.0:
        try:
            res_final = minimize(
                objective,
                best_coeffs.copy(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
            )
            if res_final.fun < best_err and _check_gamma_bounds(res_final.x, test_models, a_values):
                best_coeffs = res_final.x
                best_err = res_final.fun
        except Exception:
            pass
    
    # Ensure coefficients are in [-10, 10] and bounds are met
    best_coeffs = np.clip(best_coeffs, -10.0, 10.0)
    if not _check_gamma_bounds(best_coeffs, test_models, a_values):
        # Fallback to Taylor fit if valid, else seed
        if _check_gamma_bounds(taylor_coeffs, test_models, a_values):
            best_coeffs = taylor_coeffs
        else:
            best_coeffs = x0.copy()
    
    return best_coeffs.tolist()
# EVOLVE-BLOCK-END

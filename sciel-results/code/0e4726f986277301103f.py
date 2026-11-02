"""Improved program: spherical parameterization + Fibonacci lattice + gradient refinement."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_hermite
import time

_DEADLINE = 195  # safety margin before 210s kill

def _hermite_h(n, x):
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    elif n == 2:
        return 4 * x**2 - 2
    elif n == 3:
        return 8 * x**3 - 12 * x
    elif n == 4:
        return 16 * x**4 - 48 * x**2 + 12
    elif n == 7:
        return 128*x**7 - 4480*x**5 + 40320*x**3 - 67200*x
    elif n == 8:
        return 256*x**8 - 4480*x**6 + 26880*x**4 - 67200*x**2 + 40320
    elif n == 11:
        return (2048*x**11 - 147456*x**9 + 3133440*x**7 
                - 30198720*x**5 + 122305920*x**3 - 145152000*x)
    elif n == 12:
        return (4096*x**12 - 147456*x**10 + 2580480*x**8 
                - 20321280*x**6 + 81729600*x**4 
                - 172972800*x**2 + 103219200)
    else:
        raise ValueError(f"Hermite H_{n} not implemented")

_quad_data = None

def _get_quad_data():
    global _quad_data
    if _quad_data is None:
        n_quad = 300
        x_quad, w_quad = roots_hermite(n_quad)
        
        H0 = _hermite_h(0, x_quad)
        H4 = _hermite_h(4, x_quad)
        H8 = _hermite_h(8, x_quad)
        H12 = _hermite_h(12, x_quad)
        
        dH0 = np.zeros_like(x_quad)
        dH4 = 8 * _hermite_h(3, x_quad)
        dH8 = 16 * _hermite_h(7, x_quad)
        dH12 = 24 * _hermite_h(11, x_quad)
        
        x2 = x_quad**2
        x2w = x2 * w_quad
        
        d_d0 = -H0_0 / H12_0
        d_d4 = -H4_0 / H12_0
        d_d8 = -H8_0 / H12_0
        
        Q_base = dH12 - x_quad * H12
        P0 = H0 + d_d0 * H12
        P4 = H4 + d_d4 * H12
        P8 = H8 + d_d8 * H12
        Q0 = dH0 + d_d0 * Q_base
        Q4 = dH4 + d_d4 * Q_base
        Q8 = dH8 + d_d8 * Q_base
        
        _quad_data = (x_quad, w_quad, H0, H4, H8, H12, dH0, dH4, dH8, dH12, x2, x2w,
                      P0, P4, P8, Q0, Q4, Q8)
    return _quad_data

H0_0 = 1.0
H4_0 = 12.0
H8_0 = 40320.0
H12_0 = 103219200.0

def _compute_bound_and_grad(coeffs):
    """
    Compute log(var_x) + log(var_k) and its gradient.
    Returns (log_val, grad) or (inf, None).
    """
    a0, a4, a8 = coeffs
    
    x_quad, w_quad, H0, H4, H8, H12, dH0, dH4, dH8, dH12, x2, x2w, \
        P0, P4, P8, Q0, Q4, Q8 = _get_quad_data()
    
    P = a0*P0 + a4*P4 + a8*P8
    P2 = P * P
    
    Q = a0*Q0 + a4*Q4 + a8*Q8
    Q2 = Q * Q
    
    norm2 = np.sum(w_quad * P2)
    if norm2 <= 1e-15:
        return float('inf'), None
    
    Sx = np.sum(x2w * P2)
    if Sx <= 1e-15:
        return float('inf'), None
    
    Sd = np.sum(w_quad * Q2)
    if Sd <= 1e-15:
        return float('inf'), None
    
    var_x = Sx / norm2
    var_k = Sd / norm2
    
    log_val = np.log(var_x) + np.log(var_k)
    
    dS0_0 = 2.0 * np.sum(w_quad * P * P0)
    dS0_4 = 2.0 * np.sum(w_quad * P * P4)
    dS0_8 = 2.0 * np.sum(w_quad * P * P8)
    
    dSx_0 = 2.0 * np.sum(x2w * P * P0)
    dSx_4 = 2.0 * np.sum(x2w * P * P4)
    dSx_8 = 2.0 * np.sum(x2w * P * P8)
    
    dSd_0 = 2.0 * np.sum(w_quad * Q * Q0)
    dSd_4 = 2.0 * np.sum(w_quad * Q * Q4)
    dSd_8 = 2.0 * np.sum(w_quad * Q * Q8)
    
    gx0 = dSx_0 / Sx - dS0_0 / norm2
    gx4 = dSx_4 / Sx - dS0_4 / norm2
    gx8 = dSx_8 / Sx - dS0_8 / norm2
    
    gk0 = dSd_0 / Sd - dS0_0 / norm2
    gk4 = dSd_4 / Sd - dS0_4 / norm2
    gk8 = dSd_8 / Sd - dS0_8 / norm2
    
    grad = np.array([gx0 + gk0, gx4 + gk4, gx8 + gk8])
    
    return log_val, grad

def _spherical_to_cartesian(theta, phi):
    """Convert spherical angles to unit vector on S^2."""
    st = np.sin(theta)
    return np.array([st * np.cos(phi), st * np.sin(phi), np.cos(theta)])

def _cartesian_to_spherical(v):
    """Convert unit vector to (theta, phi)."""
    x, y, z = v
    r = np.sqrt(x*x + y*y)
    if r < 1e-15:
        theta = 0.0 if z > 0 else np.pi
        phi = 0.0
    else:
        theta = np.arccos(np.clip(z, -1.0, 1.0))
        phi = np.arctan2(y, x)
    return theta, phi

def _fibonacci_sphere(n_points):
    """Generate n_points on unit sphere using Fibonacci lattice."""
    points = []
    golden = (1 + np.sqrt(5)) / 2
    for i in range(n_points):
        z = 1 - 2 * (i + 0.5) / n_points
        r = np.sqrt(max(0, 1 - z*z))
        phi = 2 * np.pi * i / golden
        points.append(np.array([r * np.cos(phi), r * np.sin(phi), z]))
    return np.array(points)

def _spherical_objective(params):
    """Objective on sphere: params = (theta, phi), returns log bound."""
    if time.time() > _DEADLINE:
        return float('inf')
    theta, phi = params
    v = _spherical_to_cartesian(theta, phi)
    # Scale to original coefficient scale (the objective is homogeneous of degree 0)
    coeffs = v  # unit vector is fine since objective is scale-invariant
    val, _ = _compute_bound_and_grad(coeffs)
    return val

def _spherical_objective_grad(params):
    """Objective and gradient on sphere."""
    if time.time() > _DEADLINE:
        return float('inf'), np.zeros(2)
    theta, phi = params
    v = _spherical_to_cartesian(theta, phi)
    val, grad_cart = _compute_bound_and_grad(v)
    if val == float('inf'):
        return 999.0, np.zeros(2)
    # Transform gradient from Cartesian to spherical coordinates
    # d/dtheta = [-sin(theta)*cos(phi), -sin(theta)*sin(phi), -cos(theta)]
    # d/dphi = [-sin(phi), cos(phi), 0]
    st = np.sin(theta)
    ct = np.cos(theta)
    sp = np.sin(phi)
    cp = np.cos(phi)
    grad_theta = -st*cp*grad_cart[0] - st*sp*grad_cart[1] - ct*grad_cart[2]
    grad_phi = -sp*grad_cart[0] + cp*grad_cart[1]
    return val, np.array([grad_theta, grad_phi])

def construct():
    start_time = time.time()
    
    seed = np.array([-113/100, 1/25, 1.0/3240])
    
    best_result = seed.copy()
    best_log_val = _compute_bound_and_grad(seed)[0]
    
    if best_log_val == float('inf'):
        best_log_val = 999.0
    
    def _try_update(cand, val):
        nonlocal best_result, best_log_val
        if val < best_log_val:
            best_log_val = val
            best_result = cand.copy()
    
    # Phase 1: BFGS from seed in Cartesian space
    def log_objective(params):
        if time.time() > _DEADLINE:
            return float('inf')
        val, _ = _compute_bound_and_grad(params)
        return val
    
    def log_objective_grad(params):
        if time.time() > _DEADLINE:
            return float('inf'), np.zeros(3)
        val, grad = _compute_bound_and_grad(params)
        if val == float('inf'):
            return 999.0, np.zeros(3)
        return val, grad
    
    try:
        result = minimize(log_objective_grad, seed, method='L-BFGS-B',
                         jac=True, options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-13})
        _try_update(result.x, result.fun)
    except Exception:
        pass
    
    # Phase 2: Spherical Fibonacci lattice global search
    n_fib = 2000
    fib_points = _fibonacci_sphere(n_fib)
    
    # Evaluate all Fibonacci points and keep top candidates
    fib_vals = []
    for i in range(n_fib):
        if time.time() > _DEADLINE - 60:
            break
        v = fib_points[i]
        val, _ = _compute_bound_and_grad(v)
        fib_vals.append((val, i))
    
    # Sort by value and take top candidates for local refinement
    fib_vals.sort(key=lambda x: x[0])
    n_top = min(30, len(fib_vals))
    
    for rank in range(n_top):
        if time.time() > _DEADLINE - 40:
            break
        val, idx = fib_vals[rank]
        if val >= best_log_val - 1e-6:
            continue
        _try_update(fib_points[idx], val)
        
        # Local refinement from this point using spherical coordinates
        theta, phi = _cartesian_to_spherical(fib_points[idx])
        start_sphere = np.array([theta, phi])
        
        try:
            result_s = minimize(_spherical_objective_grad, start_sphere, method='L-BFGS-B',
                               jac=True, options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12})
            v_refined = _spherical_to_cartesian(result_s.x[0], result_s.x[1])
            v_refined = v_refined / np.linalg.norm(v_refined)  # renormalize
            val_refined, _ = _compute_bound_and_grad(v_refined)
            _try_update(v_refined, val_refined)
        except Exception:
            pass
    
    # Phase 3: Cartesian BFGS from best spherical result
    if time.time() < _DEADLINE - 30:
        try:
            result = minimize(log_objective_grad, best_result, method='L-BFGS-B',
                             jac=True, options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-13})
            _try_update(result.x, result.fun)
        except Exception:
            pass
    
    # Phase 4: Nelder-Mead in Cartesian from best
    if time.time() < _DEADLINE - 25:
        try:
            result = minimize(log_objective, best_result, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            _try_update(result.x, result.fun)
        except Exception:
            pass
    
    # Phase 5: Spherical Nelder-Mead from best
    if time.time() < _DEADLINE - 20:
        theta_best, phi_best = _cartesian_to_spherical(best_result / np.linalg.norm(best_result))
        try:
            result_s = minimize(_spherical_objective, np.array([theta_best, phi_best]), 
                               method='Nelder-Mead',
                               options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            v_nm = _spherical_to_cartesian(result_s.x[0], result_s.x[1])
            v_nm = v_nm / np.linalg.norm(v_nm)
            val_nm, _ = _compute_bound_and_grad(v_nm)
            _try_update(v_nm, val_nm)
        except Exception:
            pass
    
    # Phase 6: More random spherical restarts
    np.random.seed(42)
    n_random = 20
    if time.time() < _DEADLINE - 15:
        for i in range(n_random):
            if time.time() > _DEADLINE - 15:
                break
            # Random point on sphere
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2*np.pi)
            v_rand = _spherical_to_cartesian(theta, phi)
            
            # Quick BFGS from this random spherical point
            start_sphere = np.array([theta, phi])
            try:
                result_s = minimize(_spherical_objective_grad, start_sphere, method='L-BFGS-B',
                                   jac=True, options={'maxiter': 800, 'ftol': 1e-13, 'gtol': 1e-11})
                v_refined = _spherical_to_cartesian(result_s.x[0], result_s.x[1])
                v_refined = v_refined / np.linalg.norm(v_refined)
                val_refined, _ = _compute_bound_and_grad(v_refined)
                _try_update(v_refined, val_refined)
            except Exception:
                pass
    
    # Phase 7: Final high-precision polish in Cartesian
    if time.time() < _DEADLINE - 10:
        try:
            result = minimize(log_objective_grad, best_result, method='L-BFGS-B',
                             jac=True, options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-15})
            _try_update(result.x, result.fun)
        except Exception:
            pass
    
    # Phase 8: Final spherical polish
    if time.time() < _DEADLINE - 5:
        theta_best, phi_best = _cartesian_to_spherical(best_result / np.linalg.norm(best_result))
        try:
            result_s = minimize(_spherical_objective_grad, np.array([theta_best, phi_best]), 
                               method='L-BFGS-B',
                               jac=True, options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-15})
            v_final = _spherical_to_cartesian(result_s.x[0], result_s.x[1])
            v_final = v_final / np.linalg.norm(v_final)
            val_final, _ = _compute_bound_and_grad(v_final)
            _try_update(v_final, val_final)
        except Exception:
            pass
    
    return np.array(best_result)

# EVOLVE-BLOCK-END

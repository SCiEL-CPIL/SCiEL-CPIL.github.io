"""Analytic uncertainty bound with precomputed constants + spherical reparameterization + Nelder-Mead."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 195  # 15s safety margin before 210s kill


def _precompute_constants():
    """Precompute integration constants and derivative matrices once."""
    I = np.zeros(15)
    for n in range(15):
        if n == 0:
            I[n] = np.sqrt(np.pi)
        else:
            log_val = 0.5 * np.log(np.pi)
            for m in range(1, 2*n + 1):
                log_val += np.log(m)
            log_val -= n * np.log(4.0)
            for m in range(1, n + 1):
                log_val -= np.log(m)
            I[n] = np.exp(log_val)
    
    M = np.zeros((3, 7, 7))
    for k in range(3):
        for i in range(7):
            for j in range(7):
                idx = k + i + j
                if idx < 15:
                    M[k, i, j] = I[idx]
    
    signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    
    D = np.array([
        [1.0, 3.0, 105.0, 945.0],
        [0.0, -6.0, -420.0, -7560.0],
        [0.0, 1.0, 210.0, 13860.0],
        [0.0, 0.0, -28.0, -6930.0],
        [0.0, 0.0, 1.0, 1155.0],
        [0.0, 0.0, 0.0, -66.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    da12 = np.array([-1.0/945.0, -3.0/945.0, -105.0/945.0])
    
    return I, M, D, da12, signs


_I, _M, _D, _da12, _signs = _precompute_constants()


def _coeffs_to_params(a0, a4, a8):
    """Convert free coefficients to full polynomial coefficients p[0..6]."""
    a12 = _da12[0]*a0 + _da12[1]*a4 + _da12[2]*a8
    a_full = np.array([a0, a4, a8, a12])
    return _D @ a_full


def _bound_from_coeffs(p):
    """Compute uncertainty product from polynomial coefficients p[0..6]."""
    q = p * _signs
    
    m = np.zeros(3)
    for k in range(3):
        m[k] = np.einsum('i,ij,j->', p, _M[k], p)
    
    n = np.zeros(3)
    for k in range(3):
        n[k] = 2.0 * np.pi * np.einsum('i,ij,j->', q, _M[k], q)
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18
    
    return np.sqrt(vf * vF)


def _bound_from_angles(theta, phi, r):
    """
    Spherical reparameterization:
    a0 = r * cos(theta)
    a4 = r * sin(theta) * cos(phi)
    a8 = r * sin(theta) * sin(phi)
    
    This ensures we explore the full 3D coefficient space radially.
    """
    a0 = r * np.cos(theta)
    a4 = r * np.sin(theta) * np.cos(phi)
    a8 = r * np.sin(theta) * np.sin(phi)
    
    p = _coeffs_to_params(a0, a4, a8)
    return _bound_from_coeffs(p)


def _objective_angles(params):
    """Objective function for angle-space optimization."""
    theta, phi, r = params[0], params[1], params[2]
    return _bound_from_angles(theta, phi, r)


def _objective_grad(params):
    """Gradient-based objective for L-BFGS-B in coefficient space."""
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = _da12[0]*a0 + _da12[1]*a4 + _da12[2]*a8
    a_full = np.array([a0, a4, a8, a12])
    p = _D @ a_full
    q = p * _signs
    
    m = np.zeros(3)
    for k in range(3):
        m[k] = np.einsum('i,ij,j->', p, _M[k], p)
    
    n = np.zeros(3)
    for k in range(3):
        n[k] = 2.0 * np.pi * np.einsum('i,ij,j->', q, _M[k], q)
    
    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18, np.array([1e18, 1e18, 1e18])
    
    bound = np.sqrt(vf * vF)
    
    # Analytic gradient
    S_m = np.zeros((3, 7))
    for k in range(3):
        for i in range(7):
            S_m[k, i] = np.dot(p, _M[k, i, :])
    
    dm = np.zeros((3, 3))
    for k in range(3):
        for c in range(3):
            dp_da = _D[:, c] + _D[:, 3] * _da12[c]
            dm[k, c] = 2.0 * np.dot(dp_da, S_m[k, :])
    
    S_n = np.zeros((3, 7))
    for k in range(3):
        for i in range(7):
            S_n[k, i] = np.dot(q, _M[k, i, :])
    
    dn = np.zeros((3, 3))
    for k in range(3):
        for c in range(3):
            dq_da = (_D[:, c] + _D[:, 3] * _da12[c]) * _signs
            dn[k, c] = 2.0 * 2.0 * np.pi * np.dot(dq_da, S_n[k, :])
    
    dvm2 = (dm[1, :] * m0 - m2 * dm[0, :]) / (m0 * m0)
    dvn2 = (dn[1, :] * n0 - n2 * dn[0, :]) / (n0 * n0)
    
    grad = (vF * dvm2 + vf * dvn2) / (2.0 * bound)
    
    return bound, grad


def construct():
    # Phase 1: Nelder-Mead in angle-space (derivative-free, global exploration)
    # Initialize with multiple starting points in (theta, phi, r) space
    rng = np.random.default_rng(42)
    
    # Convert known good point to angle-space
    a0_seed, a4_seed, a8_seed = -1.13, 0.04, 0.000308642
    r_seed = np.sqrt(a0_seed**2 + a4_seed**2 + a8_seed**2)
    theta_seed = np.arccos(a0_seed / r_seed) if r_seed > 0 else 0
    phi_seed = np.arctan2(a8_seed, a4_seed) if (a4_seed**2 + a8_seed**2) > 1e-12 else 0
    
    angle_starts = [
        [theta_seed, phi_seed, r_seed],
        [np.pi/4, np.pi/4, 1.5],
        [np.pi/3, np.pi/6, 1.0],
        [np.pi/2, 0, 1.0],
        [np.pi/6, np.pi/2, 0.8],
        [2*np.pi/3, np.pi/3, 1.2],
        [np.pi/5, 3*np.pi/4, 0.9],
        [3*np.pi/4, np.pi/8, 1.1],
    ]
    
    # Add random starts
    for _ in range(15):
        theta = rng.uniform(0, np.pi)
        phi = rng.uniform(-np.pi, np.pi)
        r = rng.uniform(0.1, 3.0)
        angle_starts.append([theta, phi, r])
    
    best_val = np.inf
    best_params = None
    best_angle_params = None
    
    for start in angle_starts:
        if time.time() > _DEADLINE - 60:
            break
        try:
            res = minimize(_objective_angles, start, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-16,
                                   'adaptive': True})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_angle_params = res.x.copy()
        except Exception:
            pass
    
    # Convert best angle-space solution back to coefficient space
    if best_angle_params is not None:
        theta, phi, r = best_angle_params
        a0 = r * np.cos(theta)
        a4 = r * np.sin(theta) * np.cos(phi)
        a8 = r * np.sin(theta) * np.sin(phi)
        best_params = np.array([a0, a4, a8])
    else:
        best_params = np.array([a0_seed, a4_seed, a8_seed])
        best_val = _objective_grad(best_params)[0]
    
    # Phase 2: L-BFGS-B polish in coefficient space (gradient-based, local refinement)
    if time.time() < _DEADLINE - 30:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-15})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Basin hopping in angle-space with L-BFGS-B local minimization
    if time.time() < _DEADLINE - 20:
        temp = 0.005
        step_scale = 0.2
        current_val = best_val
        current_params = best_params.copy()
        
        # Convert current to angle-space for perturbations
        def coeffs_to_angles(a0, a4, a8):
            r = np.sqrt(a0**2 + a4**2 + a8**2)
            if r < 1e-12:
                return np.array([0.0, 0.0, 1e-12])
            theta = np.arccos(np.clip(a0/r, -1, 1))
            phi = np.arctan2(a8, a4)
            return np.array([theta, phi, r])
        
        def angles_to_coeffs(theta, phi, r):
            a0 = r * np.cos(theta)
            a4 = r * np.sin(theta) * np.cos(phi)
            a8 = r * np.sin(theta) * np.sin(phi)
            return np.array([a0, a4, a8])
        
        current_angles = coeffs_to_angles(*current_params)
        
        for it in range(100):
            if time.time() > _DEADLINE - 15:
                break
            
            # Perturb in angle-space
            step = rng.normal(0, step_scale, 3)
            perturbed_angles = current_angles + step
            # Wrap phi to [-pi, pi]
            perturbed_angles[1] = (perturbed_angles[1] + np.pi) % (2*np.pi) - np.pi
            perturbed_angles[0] = np.clip(perturbed_angles[0], 0.01, np.pi - 0.01)
            
            perturbed_params = angles_to_coeffs(*perturbed_angles)
            
            # Local minimize in coefficient space
            try:
                res = minimize(_objective_grad, perturbed_params, method='L-BFGS-B',
                              jac=True, options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-13})
                new_params = res.x
                new_val = res.fun
            except Exception:
                new_params = perturbed_params
                new_val = _objective_grad(new_params)[0]
            
            # Metropolis acceptance
            delta = new_val - current_val
            if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-10)):
                current_params = new_params
                current_val = new_val
                current_angles = coeffs_to_angles(*current_params)
                if new_val < best_val:
                    best_val = new_val
                    best_params = new_params.copy()
            
            if it % 10 == 0:
                step_scale *= 0.95
                temp *= 0.9
    
    # Final high-precision polish
    if time.time() < _DEADLINE - 10:
        try:
            res = minimize(_objective_grad, best_params, method='L-BFGS-B',
                          jac=True, options={'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-17})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END

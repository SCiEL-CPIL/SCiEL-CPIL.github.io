"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150
import numpy as np
import time
import math
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _fibonacci_sphere(n):
    """Generate points on the sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_energy(x):
    """
    Compute logarithmic energy efficiently.
    E = sum_{i<j} log(1/||x_i - x_j||)
    """
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    mask = ~np.eye(N, dtype=bool)
    
    safe_dist = np.where(mask, dist, 1.0)
    log_terms = -np.log(safe_dist)
    energy = 0.5 * np.sum(log_terms[mask])
    
    return energy

def _compute_grad(x):
    """
    Compute gradient of logarithmic energy.
    grad_i = - sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    """
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    mask = ~np.eye(N, dtype=bool)
    
    inv_dist_sq = np.zeros_like(dist)
    inv_dist_sq[mask] = 1.0 / (dist[mask] ** 2)
    
    weighted_diff = diff * inv_dist_sq[:, :, np.newaxis]
    
    grad = -np.sum(weighted_diff, axis=1)
    
    return grad

def _exp_map(x, p, v):
    """
    Exponential map on the sphere.
    x: current position (N, 3)
    p: point index
    v: tangent vector (3,)
    Returns new point at x[p] + v
    """
    xp = x[p]
    theta = np.linalg.norm(v)
    if theta < 1e-12:
        return xp
    v_hat = v / theta
    # Rodrigues rotation formula
    # v is tangent to xp, so xp . v = 0
    # new_p = xp * cos(theta) + v_hat * sin(theta)
    new_p = xp * np.cos(theta) + v_hat * np.sin(theta)
    # Normalize for numerical stability
    new_p /= np.linalg.norm(new_p)
    return new_p

def _riemannian_cg(x, max_iter=50, tol=1e-10, time_limit=10.0):
    """
    Riemannian Conjugate Gradient method.
    """
    start_time = time.time()
    grad = _compute_grad(x)
    # Project gradient to tangent space: grad - (grad . x) * x
    # Since x is on sphere, grad should be nearly tangent, but let's be safe.
    grad = grad - np.sum(grad * x, axis=1, keepdims=True) * x
    
    d = -grad
    d = d - np.sum(d * x, axis=1, keepdims=True) * x
    
    x_new = x.copy()
    
    for k in range(max_iter):
        if time.time() > start_time + time_limit:
            break
        
        # Line search
        # We want to minimize f(x + exp_map(d))
        # Simple backtracking line search
        alpha = 1.0
        f0 = _compute_energy(x_new)
        g0 = _compute_grad(x_new)
        g0 = g0 - np.sum(g0 * x_new, axis=1, keepdims=True) * x_new
        
        # Approximate directional derivative
        # df/dalpha at alpha=0 is <grad, d>
        df0 = np.sum(g0 * d)
        
        if df0 >= 0:
            # Direction not descent, restart CG
            d = -g0
            d = d - np.sum(d * x_new, axis=1, keepdims=True) * x_new
            continue
            
        # Backtracking
        for ls_iter in range(20):
            # Move points
            x_trial = x_new.copy()
            # Apply step
            # For efficiency, we can just update the whole array
            # But exp_map is per point.
            # d is (N, 3).
            # We need to apply exp_map for each point.
            # This is O(N) exp_maps.
            
            # Optimization: vectorized exp map?
            # theta = norm(d, axis=1)
            # d_hat = d / theta
            # new_x = x_new * cos(theta) + d_hat * sin(theta)
            
            theta = np.linalg.norm(d, axis=1)
            # Avoid division by zero
            theta_safe = np.where(theta < 1e-12, 1.0, theta)
            d_hat = d / theta_safe[:, np.newaxis]
            
            x_trial = x_new * np.cos(alpha * theta_safe)[:, np.newaxis] + d_hat * np.sin(alpha * theta_safe)[:, np.newaxis]
            x_trial /= np.linalg.norm(x_trial, axis=1, keepdims=True)
            
            f_trial = _compute_energy(x_trial)
            
            if f_trial < f0 + 1e-4 * alpha * df0:
                x_new = x_trial
                break
            else:
                alpha *= 0.5
        else:
            # Line search failed
            break
            
        # Update gradient
        grad_new = _compute_grad(x_new)
        grad_new = grad_new - np.sum(grad_new * x_new, axis=1, keepdims=True) * x_new
        
        # Check convergence
        if np.linalg.norm(grad_new) < tol:
            break
            
        # Update direction
        # Polak-Ribiere or Hestenes-Stiefel
        # beta = <grad_new - grad, d> / <d, d>
        # Since we are on manifold, we need to project differences?
        # Simple beta:
        num = np.sum((grad_new - g0) * d)
        den = np.sum(d * d)
        if den < 1e-12:
            beta = 0
        else:
            beta = num / den
            if beta < 0:
                beta = 0 # Non-negative beta for CG
                
        d_new = -grad_new + beta * d
        # Project to tangent space
        d_new = d_new - np.sum(d_new * x_new, axis=1, keepdims=True) * x_new
        
        # Check if d_new is descent
        if np.sum(grad_new * d_new) >= 0:
            d_new = -grad_new
            d_new = d_new - np.sum(d_new * x_new, axis=1, keepdims=True) * x_new
            
        d = d_new
        g0 = grad_new
        
    return x_new

def _rotate_points(x, i, j, angle):
    """
    Rotate points i and j around the axis defined by their difference vector.
    """
    x_new = x.copy()
    v = x[i] - x[j]
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-12:
        return x_new
    
    axis = v / norm_v
    
    c = np.cos(angle)
    s = np.sin(angle)
    k = axis
    v1 = x[i]
    v2 = x[j]
    
    v1_rot = v1 * c + np.cross(k, v1) * s + k * np.dot(k, v1) * (1 - c)
    v2_rot = v2 * c + np.cross(k, v2) * s + k * np.dot(k, v2) * (1 - c)
    
    v1_rot /= np.linalg.norm(v1_rot)
    v2_rot /= np.linalg.norm(v2_rot)
    
    x_new[i] = v1_rot
    x_new[j] = v2_rot
    
    return x_new

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Initial configuration: Fibonacci sphere
    x = _fibonacci_sphere(N)
    best_x = x.copy()
    best_energy = _compute_energy(x)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Riemannian CG
    # Run for a fixed time or iterations
    x = _riemannian_cg(x, max_iter=100, time_limit=10.0)
    e = _compute_energy(x)
    if e < best_energy:
        best_energy = e
        best_x = x.copy()
        
    # Phase 2: Basin Hopping with R-CG restarts
    # Use random perturbations, then R-CG
    T = 5.0
    T_end = 0.1
    n_hops = 2000 # Will be limited by time
    
    current_x = best_x.copy()
    current_e = best_energy
    
    for hop in range(n_hops):
        if time.time() > deadline:
            break
            
        # Perturb
        # Random rotation of a few points
        n_perturb = 5
        indices = rng.choice(N, n_perturb, replace=False)
        x_perturb = current_x.copy()
        for idx in indices:
            # Random small rotation
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            angle = rng.uniform(-0.1, 0.1)
            c = np.cos(angle)
            s = np.sin(angle)
            v = x_perturb[idx]
            v_rot = v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1 - c)
            v_rot /= np.linalg.norm(v_rot)
            x_perturb[idx] = v_rot
            
        # Optimize
        x_opt = _riemannian_cg(x_perturb, max_iter=20, time_limit=2.0)
        e_opt = _compute_energy(x_opt)
        
        # Accept
        delta_e = e_opt - current_e
        if delta_e < 0:
            accept = True
        else:
            if T < 1e-10:
                accept = False
            else:
                accept = rng.random() < math.exp(-delta_e / T)
                
        if accept:
            current_x = x_opt
            current_e = e_opt
            if current_e < best_energy:
                best_energy = current_e
                best_x = current_x.copy()
                
        # Cool
        T *= 0.995
        
    # Phase 3: Final Polish with SA (small steps)
    # This helps escape tiny local minima that R-CG might miss due to line search
    x = best_x.copy()
    e = best_energy
    T = 0.1
    T_end = 0.01
    max_iter_sa = 50000
    alpha = (T_end / T) ** (1.0 / max_iter_sa)
    
    for it in range(max_iter_sa):
        if time.time() > deadline:
            break
        T = T * (alpha ** it)
        
        i, j = rng.choice(N, 2, replace=False)
        angle = rng.uniform(-math.pi, math.pi)
        
        x_new = _rotate_points(x, i, j, angle)
        
        # Incremental energy update
        d_ij_old = np.linalg.norm(x[i] - x[j])
        d_i_old = np.linalg.norm(x[i] - x, axis=1)
        d_j_old = np.linalg.norm(x[j] - x, axis=1)
        
        d_ij_new = np.linalg.norm(x_new[i] - x_new[j])
        d_i_new = np.linalg.norm(x_new[i] - x_new, axis=1)
        d_j_new = np.linalg.norm(x_new[j] - x_new, axis=1)
        
        mask_i = np.ones(N, dtype=bool)
        mask_i[i] = False
        mask_i[j] = False
        mask_j = np.ones(N, dtype=bool)
        mask_j[i] = False
        mask_j[j] = False
        
        E_old_part = np.sum(-np.log(d_i_old[mask_i])) + np.sum(-np.log(d_j_old[mask_j])) - np.log(d_ij_old)
        E_new_part = np.sum(-np.log(d_i_new[mask_i])) + np.sum(-np.log(d_j_new[mask_j])) - np.log(d_ij_new)
        
        delta_E = E_new_part - E_old_part
        
        if delta_E < 0:
            accept = True
        else:
            if T < 1e-10:
                accept = False
            else:
                accept = rng.random() < math.exp(-delta_E / T)
                
        if accept:
            x = x_new
            e += delta_E
            if e < best_energy:
                best_energy = e
                best_x = x.copy()
                
    best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
    return best_x

# EVOLVE-BLOCK-END

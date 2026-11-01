"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150
import numpy as np
import time
import math
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _log_energy_and_grad_sphere(x):
    """
    Computes the logarithmic energy and its Euclidean gradient for points on the sphere.
    x: (N, 3) array of points on the unit sphere.
    Returns: energy, grad (N, 3)
    """
    # diff is (N, N, 3) where diff[i,j] = x_i - x_j
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    mask = ~np.eye(N, dtype=bool)
    
    # Energy: 0.5 * sum_{i!=j} -log(dist)
    safe_dist = np.where(mask, dist, 1.0)
    log_terms = -np.log(safe_dist)
    energy = 0.5 * np.sum(log_terms[mask])
    
    # Gradient: grad_i = - sum_{j != i} (x_i - x_j) / dist_ij^2
    inv_dist_sq = np.zeros_like(dist)
    inv_dist_sq[mask] = 1.0 / (dist[mask] ** 2)
    
    weighted_diff = diff * inv_dist_sq[:, :, np.newaxis]
    grad = -np.sum(weighted_diff, axis=1)
    
    return energy, grad

def _stereo_to_sphere(p, pole):
    """
    Map 2D stereographic coordinates to unit sphere relative to a specific pole.
    p: (N, 2) array.
    pole: (3,) array, the point on the sphere from which we project.
    
    We rotate the sphere so the pole is at the North Pole (0,0,1), 
    apply standard stereographic projection, then rotate back.
    """
    # Rotation matrix that maps 'pole' to (0,0,1)
    # If pole is (0,0,1), rotation is identity.
    # Otherwise, we need a rotation.
    # For efficiency, we precompute rotations for the two poles: North (0,0,1) and South (0,0,-1).
    
    u, v = p[:, 0], p[:, 1]
    denom = 1.0 + u**2 + v**2
    x_local = 2.0 * u / denom
    y_local = 2.0 * v / denom
    z_local = (1.0 - u**2 - v**2) / denom
    
    x_sphere = np.column_stack([x_local, y_local, z_local])
    
    # Rotate back from local North Pole to 'pole'
    # If pole is North (0,0,1), R is I.
    # If pole is South (0,0,-1), R maps (0,0,1) to (0,0,-1).
    # R = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]] maps (0,0,1) to (0,0,1)?? No.
    # Let's define rotations explicitly for the two charts.
    
    if np.allclose(pole, [0, 0, 1]):
        return x_sphere
    elif np.allclose(pole, [0, 0, -1]):
        # Map local North (0,0,1) to South (0,0,-1)
        # This is a 180 degree rotation around X axis? 
        # (x, y, z) -> (x, -y, -z) maps (0,0,1) to (0,0,-1).
        x_sphere[:, 1] *= -1
        x_sphere[:, 2] *= -1
        return x_sphere
    else:
        # Fallback: general rotation
        # This case shouldn't happen if we stick to N/S poles
        raise ValueError("Only North and South poles supported for efficiency")

def _sphere_to_stereo(x, pole):
    """
    Map unit sphere to 2D stereographic coordinates relative to a specific pole.
    """
    if np.allclose(pole, [0, 0, 1]):
        # Standard inverse stereographic from North Pole
        # u = x / (1 - z)
        # v = y / (1 - z)
        denom = 1.0 - x[:, 2]
        # Handle case where z is close to 1
        denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
        u = x[:, 0] / denom
        v = x[:, 1] / denom
        return np.column_stack([u, v])
    elif np.allclose(pole, [0, 0, -1]):
        # First rotate so South Pole becomes North Pole
        # (x, y, z) -> (x, -y, -z)
        x_rot = x.copy()
        x_rot[:, 1] *= -1
        x_rot[:, 2] *= -1
        # Now x_rot is in the frame where the original South Pole is at North Pole
        denom = 1.0 - x_rot[:, 2]
        denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
        u = x_rot[:, 0] / denom
        v = x_rot[:, 1] / denom
        return np.column_stack([u, v])
    else:
        raise ValueError("Only North and South poles supported for efficiency")

def _energy_and_grad_stereo_multi(p, pole_indices):
    """
    Computes energy and gradient for points parameterized by stereographic coords 
    with potentially different poles for each point.
    p: (N, 2) array of stereographic parameters.
    pole_indices: (N,) array of 0 (North) or 1 (South).
    """
    x = np.zeros((N, 3))
    J = np.zeros((N, 3, 2))
    
    # Process each point
    for i in range(N):
        if pole_indices[i] == 0:
            pole = np.array([0.0, 0.0, 1.0])
            x[i] = _stereo_to_sphere(p[i:i+1, :], pole)[0]
            # Jacobian for North Pole chart
            u, v = p[i, 0], p[i, 1]
            D = 1.0 + u**2 + v**2
            D2 = D**2
            J[i, 0, 0] = 2.0 * (1.0 - u**2 + v**2) / D2
            J[i, 0, 1] = -4.0 * u * v / D2
            J[i, 1, 0] = -4.0 * u * v / D2
            J[i, 1, 1] = 2.0 * (1.0 + u**2 - v**2) / D2
            J[i, 2, 0] = -4.0 * u / D2
            J[i, 2, 1] = -4.0 * v / D2
        else:
            pole = np.array([0.0, 0.0, -1.0])
            x[i] = _stereo_to_sphere(p[i:i+1, :], pole)[0]
            # Jacobian for South Pole chart
            # x_sphere = (x_local, -y_local, -z_local)
            # x_local = 2u/D, y_local = 2v/D, z_local = (1-u^2-v^2)/D
            # x_sphere_x = 2u/D
            # x_sphere_y = -2v/D
            # x_sphere_z = -(1-u^2-v^2)/D = (u^2+v^2-1)/D
            
            u, v = p[i, 0], p[i, 1]
            D = 1.0 + u**2 + v**2
            D2 = D**2
            
            # dx/du
            J[i, 0, 0] = 2.0 * (1.0 - u**2 + v**2) / D2
            J[i, 0, 1] = -4.0 * u * v / D2
            
            # dy/du (where y is the second component of x_sphere)
            # y_sphere = -2v/D
            # dy/du = -(-4uv/D^2) = 4uv/D^2 ? No.
            # d(-2v/D)/du = -2 * (0*D - v*2u)/D^2 = 4uv/D^2
            J[i, 1, 0] = 4.0 * u * v / D2
            # dy/dv
            # d(-2v/D)/dv = -2 * (1*D - v*2v)/D^2 = -2(1-u^2-v^2)/D^2
            J[i, 1, 1] = -2.0 * (1.0 - u**2 - v**2) / D2
            
            # dz/du (where z is the third component)
            # z_sphere = (u^2+v^2-1)/D
            # dz/du = (2u*D - (u^2+v^2-1)*2u)/D^2 = 2u(D - u^2 - v^2 + 1)/D^2 = 2u(1+u^2+v^2 - u^2 - v^2 + 1)/D^2 = 4u/D^2
            J[i, 2, 0] = 4.0 * u / D2
            # dz/dv
            # dz/dv = 4v/D^2
            J[i, 2, 1] = 4.0 * v / D2
            
    energy, grad_sphere = _log_energy_and_grad_sphere(x)
    
    # grad_p = J^T grad_sphere
    grad_p = np.einsum('ikl,ik->il', J, grad_sphere)
    
    return energy, grad_p

def _assign_poles(x):
    """
    Assign each point to the nearest pole (North or South) to avoid singularity.
    Returns an array of indices: 0 for North, 1 for South.
    """
    # Distance to North Pole (0,0,1)
    dist_n = np.linalg.norm(x - np.array([0, 0, 1]), axis=1)
    # Distance to South Pole (0,0,-1)
    dist_s = np.linalg.norm(x - np.array([0, 0, -1]), axis=1)
    
    # Assign to the pole that is FARTHER away? 
    # Stereographic projection from Pole P is singular at P.
    # We want to project from the pole that is NOT close to the point.
    # So if point is close to North, project from South.
    # If point is close to South, project from North.
    
    pole_indices = np.where(dist_n < dist_s, 1, 0)
    return pole_indices

def _fibonacci_sphere(n):
    """Generate a Fibonacci spiral distribution on the sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return x

def _halton_sphere(n, dims=3):
    """Generate a Halton sequence distribution on the sphere (approx)."""
    # This is a simpler heuristic: use Halton in 2D and map to sphere via inverse CDF or just use Fibonacci which is better.
    # For now, let's just use random and Fibonacci.
    rng = np.random.default_rng(123)
    x = rng.normal(size=(n, 3))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    best_energy = float('inf')
    best_x = None
    
    # Initial configurations
    inits = [
        _fibonacci_sphere(N),
        _halton_sphere(N),
        np.random.default_rng(42).normal(size=(N, 3)) / np.linalg.norm(np.random.default_rng(42).normal(size=(N, 3)), axis=1, keepdims=True)
    ]
    
    # Normalize initial configs
    for i in range(len(inits)):
        inits[i] /= np.linalg.norm(inits[i], axis=1, keepdims=True)
    
    for init_idx, x_init in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        # Assign poles
        pole_indices = _assign_poles(x_init)
        
        # Convert to stereographic parameters
        p0 = np.zeros((N, 2))
        for i in range(N):
            if pole_indices[i] == 0:
                pole = np.array([0.0, 0.0, 1.0])
            else:
                pole = np.array([0.0, 0.0, -1.0])
            p0[i] = _sphere_to_stereo(x_init[i:i+1, :], pole)[0]
        
        p_flat0 = p0.flatten()
        
        def obj_flat(p_flat):
            p = p_flat.reshape(N, 2)
            e, g = _energy_and_grad_stereo_multi(p, pole_indices)
            return e, g.flatten()
            
        try:
            # Use L-BFGS-B
            # We don't have bounds, but L-BFGS-B handles unbounded well.
            res = minimize(obj_flat, p_flat0, method='L-BFGS-B', jac=True, 
                           options={'maxiter': 300, 'maxfun': 300, 'ftol': 1e-12, 'gtol': 1e-8})
            
            p_res = res.x.reshape(N, 2)
            
            # Convert back to sphere
            x_res = np.zeros((N, 3))
            for i in range(N):
                if pole_indices[i] == 0:
                    pole = np.array([0.0, 0.0, 1.0])
                else:
                    pole = np.array([0.0, 0.0, -1.0])
                x_res[i] = _stereo_to_sphere(p_res[i:i+1, :], pole)[0]
                
            # Normalize
            x_res /= np.linalg.norm(x_res, axis=1, keepdims=True)
            
            e_res, _ = _log_energy_and_grad_sphere(x_res)
            
            if e_res < best_energy:
                best_energy = e_res
                best_x = x_res
                
        except Exception:
            pass
    
    # If no valid result, return initial best guess
    if best_x is None:
        best_x = inits[0]
        
    # Final polish: run L-BFGS again on the best result found
    if time.time() < deadline - 5:
        pole_indices = _assign_poles(best_x)
        p_curr = np.zeros((N, 2))
        for i in range(N):
            if pole_indices[i] == 0:
                pole = np.array([0.0, 0.0, 1.0])
            else:
                pole = np.array([0.0, 0.0, -1.0])
            p_curr[i] = _sphere_to_stereo(best_x[i:i+1, :], pole)[0]
            
        p_flat_curr = p_curr.flatten()
        
        def obj_flat_final(p_flat):
            p = p_flat.reshape(N, 2)
            e, g = _energy_and_grad_stereo_multi(p, pole_indices)
            return e, g.flatten()
            
        try:
            res_final = minimize(obj_flat_final, p_flat_curr, method='L-BFGS-B', jac=True, 
                                 options={'maxiter': 200, 'maxfun': 200, 'ftol': 1e-14, 'gtol': 1e-10})
            p_final = res_final.x.reshape(N, 2)
            x_final = np.zeros((N, 3))
            for i in range(N):
                if pole_indices[i] == 0:
                    pole = np.array([0.0, 0.0, 1.0])
                else:
                    pole = np.array([0.0, 0.0, -1.0])
                x_final[i] = _stereo_to_sphere(p_final[i:i+1, :], pole)[0]
                
            x_final /= np.linalg.norm(x_final, axis=1, keepdims=True)
            e_final, _ = _log_energy_and_grad_sphere(x_final)
            
            if e_final < best_energy:
                best_energy = e_final
                best_x = x_final
        except Exception:
            pass
            
    return best_x

# EVOLVE-BLOCK-END

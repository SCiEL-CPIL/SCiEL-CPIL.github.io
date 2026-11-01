"""
Strategy: Basin Hopping with L-BFGS-B local minimization and stochastic global moves.
This approach combines the speed of L-BFGS-B (measured to converge faster than Riemannian CG)
with the global exploration capability of basin hopping to escape local minima.
"""

N = 150
import numpy as np
import time
import math
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _energy_and_grad_stereo(params):
    """
    Compute energy and gradient in stereographic projection coordinates.
    params: (2N,) array, interleaved (u_i, v_i) for each point.
    """
    u = params[0::2]
    v = params[1::2]
    
    # Stereographic projection: (u, v) -> (x, y, z) on unit sphere
    # x = 2u / (1+u^2+v^2), y = 2v / (1+u^2+v^2), z = (1-u^2-v^2)/(1+u^2+v^2)
    # This maps the sphere minus north pole (0,0,1) to the plane.
    # Points near north pole have large u, v.
    
    denom = 1.0 + u*u + v*v
    x = 2.0 * u / denom
    y = 2.0 * v / denom
    z = (1.0 - u*u - v*v) / denom
    
    # Compute pairwise distances
    # x, y, z are (N,)
    diff_x = x[:, None] - x[None, :]
    diff_y = y[:, None] - y[None, :]
    diff_z = z[:, None] - z[None, :]
    
    dist_sq = diff_x**2 + diff_y**2 + diff_z**2
    
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    safe_dist_sq = np.where(mask, dist_sq, 1.0)
    
    # Energy: sum_{i<j} log(1/d) = -0.5 * sum_{i<j} log(d^2)
    log_terms = -0.5 * np.log(safe_dist_sq)
    energy = np.sum(log_terms[mask])
    
    # Gradient in R^3 (ambient space)
    # grad_i = - sum_{j!=i} (x_i - x_j) / ||x_i - x_j||^2
    # Note: this is the gradient of -log(d) = -0.5 log(d^2) w.r.t x_i?
    # d/dx_i (-0.5 log(||x_i - x_j||^2)) = -0.5 * (2(x_i - x_j)) / ||x_i - x_j||^2 = -(x_i - x_j) / d^2
    inv_dist_sq = np.zeros_like(dist_sq)
    inv_dist_sq[mask] = 1.0 / dist_sq[mask]
    
    # grad_r3[i] = - sum_j (x_i - x_j) * inv_dist_sq[i,j]
    # Let's compute this vectorized.
    # diff = x_i - x_j. We want - sum_j diff_ij * inv_dist_sq_ij
    # weighted_diff_x = diff_x * inv_dist_sq
    # grad_r3_x = - sum(weighted_diff_x, axis=1)
    
    weighted_diff_x = diff_x * inv_dist_sq
    weighted_diff_y = diff_y * inv_dist_sq
    weighted_diff_z = diff_z * inv_dist_sq
    
    grad_r3_x = -np.sum(weighted_diff_x, axis=1)
    grad_r3_y = -np.sum(weighted_diff_y, axis=1)
    grad_r3_z = -np.sum(weighted_diff_z, axis=1)
    
    # Project gradient to tangent space of sphere?
    # The energy is defined on the sphere. The gradient in R^3 has a radial component.
    # We need the Riemannian gradient projected onto the tangent space.
    # grad_tangent = grad_r3 - (grad_r3 . x) x
    # x, y, z are the point coordinates.
    dot = grad_r3_x * x + grad_r3_y * y + grad_r3_z * z
    grad_r3_x -= dot * x
    grad_r3_y -= dot * y
    grad_r3_z -= dot * z
    
    # Now map gradient from R^3 tangent space to stereographic parameter space (u, v)
    # Chain rule: dE/dparam = (dE/dx, dE/dy, dE/dz) . (dx/dparam, dy/dparam, dz/dparam)
    # 
    # x = 2u / D, y = 2v / D, z = (1 - u^2 - v^2) / D, where D = 1 + u^2 + v^2
    # 
    # dx/du = (2D - 2u(2u)) / D^2 = 2(1 - u^2 + v^2) / D^2
    # dx/dv = (0 - 2u(2v)) / D^2 = -4uv / D^2
    # 
    # dy/du = -4uv / D^2
    # dy/dv = 2(1 + u^2 - v^2) / D^2
    # 
    # dz/du = (-2u D - (1-u^2-v^2)(2u)) / D^2 = (-2u(1+u^2+v^2) - 2u(1-u^2-v^2)) / D^2
    #        = -2u(1+u^2+v^2 + 1 - u^2 - v^2) / D^2 = -4u / D^2
    # dz/dv = -4v / D^2
    
    D2 = denom * denom
    
    dx_du = 2.0 * (1.0 - u*u + v*v) / D2
    dx_dv = -4.0 * u * v / D2
    dy_du = -4.0 * u * v / D2
    dy_dv = 2.0 * (1.0 + u*u - v*v) / D2
    dz_du = -4.0 * u / D2
    dz_dv = -4.0 * v / D2
    
    grad_u = grad_r3_x * dx_du + grad_r3_y * dy_du + grad_r3_z * dz_du
    grad_v = grad_r3_x * dx_dv + grad_r3_y * dy_dv + grad_r3_z * dz_dv
    
    grad_params = np.zeros(2 * N)
    grad_params[0::2] = grad_u
    grad_params[1::2] = grad_v
    
    return energy, grad_params

def _sphere_to_stereo(x):
    """
    Convert points on sphere (N, 3) to stereographic coordinates (2N,).
    x: (N, 3)
    Returns: (2N,) interleaved u, v
    """
    # Avoid singularity at north pole (0, 0, 1)
    # If z is close to 1, u, v will be large.
    # We can handle this by checking z < 1 - eps.
    # If z >= 1 - eps, we can use the other projection (from south pole) or just clip.
    # For robustness, let's clip z to 1 - 1e-9.
    z = x[:, 2]
    z_safe = np.clip(z, -1.0, 1.0 - 1e-9)
    
    u = x[:, 0] / (1.0 - z_safe)
    v = x[:, 1] / (1.0 - z_safe)
    
    params = np.zeros(2 * N)
    params[0::2] = u
    params[1::2] = v
    return params

def _stereo_to_sphere(params):
    """
    Convert stereographic coordinates (2N,) to points on sphere (N, 3).
    """
    u = params[0::2]
    v = params[1::2]
    denom = 1.0 + u*u + v*v
    x = 2.0 * u / denom
    y = 2.0 * v / denom
    z = (1.0 - u*u - v*v) / denom
    return np.column_stack([x, y, z])

def _local_minimize(x, time_limit):
    """
    Perform local minimization using L-BFGS-B on stereographic coordinates.
    x: (N, 3) initial points
    time_limit: seconds to spend on this minimization
    Returns: (x_opt, energy_opt)
    """
    start_time = time.time()
    params = _sphere_to_stereo(x)
    
    # Use a small maxiter and maxfun to respect time limit, or rely on scipy's timeout if available?
    # scipy minimize doesn't have a built-in wall-clock timeout.
    # We will run it and check time afterwards. If it takes too long, we might cut it off?
    # Actually, L-BFGS-B is fast. For N=150, 100-200 iterations should be < 1s.
    # Let's set a reasonable cap.
    
    res = minimize(_energy_and_grad_stereo, params, method='L-BFGS-B', jac=True,
                   options={'maxiter': 200, 'maxfun': 400, 'gtol': 1e-8, 'ftol': 1e-12})
    
    x_opt = _stereo_to_sphere(res.x)
    # Ensure normalization
    x_opt /= np.linalg.norm(x_opt, axis=1, keepdims=True)
    
    # Calculate final energy
    e_opt, _ = _energy_and_grad_stereo(_sphere_to_stereo(x_opt))
    
    return x_opt, e_opt

def _random_rotation(x, num_points, angle):
    """
    Apply a random rotation to a subset of points.
    x: (N, 3)
    num_points: number of points to rotate
    angle: maximum rotation angle
    Returns: new x
    """
    x_new = x.copy()
    indices = np.random.choice(N, num_points, replace=False)
    
    # Generate random rotation matrix
    # Use QR decomposition of a random Gaussian matrix
    M = np.random.normal(size=(3, 3))
    Q, R = np.linalg.qr(M)
    # Ensure determinant is +1
    d = np.sign(np.linalg.det(Q))
    Q[:, 0] *= d
    R[0, :] *= d
    # Q is now a random rotation matrix
    
    # Apply rotation to selected points
    # We can mix the rotated points with the original? No, just rotate them.
    # To make it a "move" in basin hopping, we rotate them by a small angle.
    # Actually, a full random rotation of a subset is a big jump.
    # Better: generate a rotation close to identity.
    # Or, just add a small random vector and renormalize?
    # Let's stick to a small random rotation.
    
    # Generate a small rotation axis and angle
    axis = np.random.normal(size=3)
    axis /= np.linalg.norm(axis)
    theta = np.random.uniform(0, angle)
    
    # Rodrigues' rotation formula
    c = np.cos(theta)
    s = np.sin(theta)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R_rot = np.eye(3) + s * K + (1 - c) * (K @ K)
    
    x_new[indices] = x[indices] @ R_rot.T
    
    return x_new

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # 1. Initialize with Fibonacci sphere (better coverage than Halton for some metrics)
    # Or Halton? Halton was in the seed. Let's use Halton as it was in the seed and worked.
    # Actually, Fibonacci sphere is very uniform. Let's use Halton to be consistent with seed's init.
    u = np.zeros((N, 2))
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    for i in range(N):
        u[i, 0] = halton(i + 1, 2)
        u[i, 1] = halton(i + 1, 3)
    
    z = 2.0 * u[:, 0] - 1.0
    phi = 2.0 * np.pi * u[:, 1]
    r = np.sqrt(1.0 - z * z)
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Initial local minimization
    x, e = _local_minimize(x, deadline - start_time)
    best_x = x.copy()
    best_e = e
    
    # 2. Basin Hopping
    # Schedule:
    # Start with a larger step size (angle) and decrease it.
    # Or use a reheat schedule.
    
    # Let's try a simple cooling schedule for the rotation angle.
    # Angle starts at 0.5 radians and decreases to 0.01 radians.
    # Number of iterations depends on time.
    
    current_angle = 0.5
    min_angle = 0.01
    cooling_rate = 0.95
    
    # We will run until time is up.
    # Each iteration: 
    # 1. Generate candidate (rotate subset)
    # 2. Local minimize candidate
    # 3. Accept if better or with probability exp(-deltaE / T)
    
    # Temperature T?
    # Energy scale: log(1/d). Typical energy ~ -2000.
    # Differences might be small.
    # Let's use a fixed temperature or anneal it.
    # T starts at 10.0 and decreases to 0.1.
    T = 10.0
    T_min = 0.1
    T_cooling = 0.98
    
    while time.time() < deadline - 2.0: # Leave 2s for final check
        # Generate move
        # Rotate 10-30 random points
        num_points = np.random.randint(10, 30)
        x_cand = _random_rotation(best_x, num_points, current_angle)
        
        # Local minimize
        x_cand, e_cand = _local_minimize(x_cand, deadline - start_time - 5.0)
        
        # Acceptance criterion
        delta_e = e_cand - best_e
        if delta_e < 0:
            # Always accept if better
            best_x = x_cand
            best_e = e_cand
            # Reset angle and temperature if we found a better solution?
            # Or just continue cooling.
        else:
            # Metropolis criterion
            if np.random.rand() < np.exp(-delta_e / T):
                best_x = x_cand
                best_e = e_cand
        
        # Update schedule
        current_angle *= cooling_rate
        if current_angle < min_angle:
            current_angle = min_angle
        T *= T_cooling
        if T < T_min:
            T = T_min
            # Reheat?
            # If we've been stuck for a while, reheat.
            # Simple heuristic: if no improvement in last 10 iterations, reheat.
            # We don't track that here. Let's just keep cooling.
            # Or, periodically reheat to escape deep minima.
            # Let's add a simple reheat: every 20 iterations, if T is low, boost it.
            pass

    # Final local minimization on best_x
    if time.time() < deadline - 1.0:
        best_x, best_e = _local_minimize(best_x, deadline - start_time - 0.5)
        
    # Final normalization
    best_x /= np.linalg.norm(best_x, axis=1, keepdims=True)
    
    return best_x

# EVOLVE-BLOCK-END

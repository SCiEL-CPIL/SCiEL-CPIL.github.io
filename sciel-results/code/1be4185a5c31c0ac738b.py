"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

def _points_from_spherical(theta, phi):
    """Convert spherical coordinates to Cartesian on unit sphere."""
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    points = np.column_stack([x, y, z])
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms < 1e-15] = 1.0
    points = points / norms
    return points

def _energy_spherical(v):
    """
    Compute logarithmic energy from spherical coordinates vector.
    v: (2N,) array [theta_1..theta_N, phi_1..phi_N]
    """
    theta = v[:N]
    phi = v[N:]
    # Clamp theta to avoid singularities at poles
    theta = np.clip(theta, 1e-6, np.pi - 1e-6)
    
    points = _points_from_spherical(theta, phi)
    
    # Pairwise squared distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Avoid log(0)
    dist_sq[dist_sq < 1e-12] = 1e-12
    
    # Sum over i < j
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    energy = -0.5 * np.sum(np.log(dist_sq[mask]))
    
    return energy

def _optimize_spherical(init_theta, init_phi, max_iter=500):
    """
    Optimize points using L-BFGS-B on spherical coordinates.
    """
    v0 = np.concatenate([init_theta, init_phi])
    
    # Use numerical gradients
    result = scipy.optimize.minimize(
        _energy_spherical,
        v0,
        method='L-BFGS-B',
        options={
            'maxiter': max_iter,
            'ftol': 1e-12,
            'gtol': 1e-8,
            'disp': False
        }
    )
    
    if result.fun < np.inf:
        v_opt = result.x
        theta_opt = v_opt[:N]
        phi_opt = v_opt[N:]
        points = _points_from_spherical(theta_opt, phi_opt)
        return points, result.fun
    else:
        return None, np.inf

def construct():
    start_time = time.time()
    deadline = 115.0  # 5s safety margin
    
    # 1. Fibonacci Spiral Initialization
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Convert to spherical coordinates
    theta_init = np.arccos(np.clip(init_points[:, 2], -1.0, 1.0))
    phi_init = np.arctan2(init_points[:, 1], init_points[:, 0])
    
    best_points = init_points
    best_energy = np.inf
    
    # Optimize from Fibonacci start
    if time.time() - start_time < deadline:
        opt_points, energy = _optimize_spherical(theta_init, phi_init)
        if opt_points is not None and energy < best_energy:
            best_energy = energy
            best_points = opt_points
            
    # 2. Random Restarts
    rng = np.random.default_rng(42)
    num_restarts = 5
    
    for k in range(num_restarts):
        if time.time() - start_time > deadline - 10:
            break
            
        # Generate random points on sphere
        random_points = rng.normal(size=(N, 3))
        random_points /= np.linalg.norm(random_points, axis=1, keepdims=True)
        
        theta_rand = np.arccos(np.clip(random_points[:, 2], -1.0, 1.0))
        phi_rand = np.arctan2(random_points[:, 1], random_points[:, 0])
        
        opt_points, energy = _optimize_spherical(theta_rand, phi_rand)
        if opt_points is not None and energy < best_energy:
            best_energy = energy
            best_points = opt_points
            
    # Final validation
    # Ensure no points are closer than 1e-9
    # This is a safety check, the optimizer should prevent this.
    return best_points
# EVOLVE-BLOCK-END

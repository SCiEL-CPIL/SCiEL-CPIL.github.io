import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    # 1. Generate initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    
    # Convert to spherical coordinates (theta, phi)
    # x = r cos(phi), y = r sin(phi), z = z
    # theta = arccos(z), phi = arctan2(y, x)
    theta = np.arccos(z)
    phi_sph = np.arctan2(r * np.sin(phi), r * np.cos(phi))
    # Ensure phi is in [0, 2pi)
    phi_sph = phi_sph % (2 * np.pi)
    
    # Flatten params: [theta_0, phi_0, theta_1, phi_1, ...]
    x0 = np.zeros(2 * N)
    x0[0::2] = theta
    x0[1::2] = phi_sph
    
    # Objective function
    def energy(params):
        thetas = params[0::2]
        phis = params[1::2]
        
        # Convert to Cartesian
        sin_t = np.sin(thetas)
        x = sin_t * np.cos(phis)
        y = sin_t * np.sin(phis)
        z = np.cos(thetas)
        
        pts = np.column_stack((x, y, z))
        
        # Calculate pairwise distances
        # dist[i, j] = ||p_i - p_j||
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # We only need i < j
        # Mask for upper triangle
        iu = np.triu_indices(N, k=1)
        d = dists[iu]
        
        # Avoid log(0)
        # If any distance is too small, return large penalty
        if np.any(d < 1e-8):
            return 1e10
        
        # Energy = sum log(1/d) = -sum log(d)
        # We minimize this.
        return -np.sum(np.log(d))
        
    # We want to minimize -sum(log(d))
    # Wait, the task says "minimize ... sum log(1/d)".
    # log(1/d) = -log(d).
    # So we minimize -sum(log(d)).
    # This is what the function above returns.
    
    bounds = []
    for i in range(N):
        # theta in (0, pi)
        bounds.append((1e-8, np.pi - 1e-8))
        # phi in [0, 2pi)
        bounds.append((0, 2 * np.pi))
        
    # Use L-BFGS-B
    res = minimize(energy, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 5000, 'ftol': 1e-10, 'gtol': 1e-8})
    
    if res.success or res.fun < energy(x0):
        best_params = res.x
    else:
        best_params = x0
        
    # Convert best_params back to Cartesian
    thetas = best_params[0::2]
    phis = best_params[1::2]
    
    sin_t = np.sin(thetas)
    x = sin_t * np.cos(phis)
    y = sin_t * np.sin(phis)
    z = np.cos(thetas)
    
    pts = np.column_stack((x, y, z))
    
    # Normalize to ensure they are exactly on the sphere (numerical stability)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    return pts

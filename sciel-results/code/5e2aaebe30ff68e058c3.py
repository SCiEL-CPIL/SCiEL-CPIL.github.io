import numpy as np
from scipy.optimize import minimize

def construct():
    N = 50
    # Initialize on a Fibonacci sphere
    # This provides a good uniform distribution on a sphere
    # We will optimize the spherical coordinates (theta, phi) and the radius R
    # Variables: [theta_1, phi_1, ..., theta_N, phi_N, R]
    # Total 2*N + 1 = 101 variables
    
    # Initial guess
    # Fibonacci sphere points
    theta = np.zeros(N)
    phi = np.zeros(N)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(N):
        theta[i] = np.arccos(1.0 - 2.0 * (i + 0.5) / N)
        phi[i] = i * golden_angle
    
    R_init = 0.3 # Initial radius of the sphere containing the centers
    
    x0 = np.concatenate([theta, phi, [R_init]])
    
    # Objective function to minimize
    # We want to maximize the minimum distance.
    # Minimizing sum of inverse square distances is a standard proxy.
    # We must ensure points stay inside the cube.
    # Penalty for being outside [-0.5, 0.5]
    
    def objective(x):
        th = x[:N]
        ph = x[N:2*N]
        R = x[2*N]
        
        # Convert to Cartesian
        x_coords = R * np.sin(th) * np.cos(ph)
        y_coords = R * np.sin(th) * np.sin(ph)
        z_coords = R * np.cos(th)
        
        centers = np.column_stack((x_coords, y_coords, z_coords))
        
        # Check bounds
        # If any coordinate is outside [-0.5, 0.5], apply penalty
        # We use a soft penalty to allow gradient flow
        # Penalty = sum of (violation)^2
        min_coord = np.min(centers)
        max_coord = np.max(centers)
        
        penalty = 0.0
        if min_coord < -0.5:
            penalty += (min_coord + 0.5) ** 2 * 1000
        if max_coord > 0.5:
            penalty += (max_coord - 0.5) ** 2 * 1000
        
        # Calculate pairwise distances
        # diff[i, j] = centers[i] - centers[j]
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        
        # Avoid division by zero
        # dist_sq[i, i] is 0, so we mask it
        np.fill_diagonal(dist_sq, 1e-9)
        
        # Sum of inverse square distances
        # This encourages points to spread out
        repulsion = np.sum(1.0 / dist_sq)
        
        # We want to minimize repulsion (which corresponds to maximizing distances)
        # Wait, if distances are large, 1/d^2 is small. So minimizing sum(1/d^2) maximizes distances.
        # Yes.
        
        return repulsion + penalty

    # Bounds for theta, phi, R
    # theta in [0, pi]
    # phi in [0, 2*pi]
    # R in [0, 0.5] (max radius for center to be in box is 0.5, but effectively less)
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, np.pi))
    for i in range(N):
        bounds.append((0.0, 2.0 * np.pi))
    bounds.append((0.0, 0.5))
    
    # Optimize
    # L-BFGS-B is suitable for bounded problems
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12})
    
    # Extract final centers
    th = result.x[:N]
    ph = result.x[N:2*N]
    R = result.x[2*N]
    
    centers = np.zeros((N, 3))
    centers[:, 0] = R * np.sin(th) * np.cos(ph)
    centers[:, 1] = R * np.sin(th) * np.sin(ph)
    centers[:, 2] = R * np.cos(th)
    
    # Final check and cleanup
    # Clip to ensure strict validity if numerical errors occurred
    centers = np.clip(centers, -0.5, 0.5)
    
    return centers

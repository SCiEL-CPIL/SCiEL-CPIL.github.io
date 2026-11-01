import numpy as np
from scipy.optimize import minimize

def construct():
    N = 50
    
    # Initial guess: Uniform distribution on sphere (Fibonacci lattice)
    # This is a good starting point for points on a sphere.
    def fibonacci_sphere(N):
        points = np.zeros((N, 2))
        golden_angle = np.pi * (3. - np.sqrt(5.))
        for i in range(N):
            points[i, 0] = 2. * np.arcsin((i + 0.5) / N) # theta (polar)
            points[i, 1] = golden_angle * i               # phi (azimuthal)
        return points

    init_theta_phi = fibonacci_sphere(N)
    # Initial R: guess based on simple estimate
    # r ~ 0.13, R ~ 0.36
    init_R = 0.35
    
    x0 = np.concatenate([init_theta_phi.flatten(), [init_R]])
    
    def objective(x):
        theta_phi = x[:2*N].reshape(N, 2)
        R = x[2*N]
        
        # Ensure R is positive and within bounds
        # R must be < 0.5
        if R <= 0 or R >= 0.5:
            return 1e6
            
        # Convert spherical to cartesian on sphere of radius R
        # theta is polar (0 to pi), phi is azimuthal (0 to 2pi)
        # Standard physics convention:
        # x = R sin(theta) cos(phi)
        # y = R sin(theta) sin(phi)
        # z = R cos(theta)
        
        theta = theta_phi[:, 0]
        phi = theta_phi[:, 1]
        
        centers = np.zeros((N, 3))
        centers[:, 0] = R * np.sin(theta) * np.cos(phi)
        centers[:, 1] = R * np.sin(theta) * np.sin(phi)
        centers[:, 2] = R * np.cos(theta)
        
        # Calculate pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        # Exclude self-distances
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        
        # Wall distance
        # Since centers are on sphere radius R centered at origin,
        # the distance to the nearest wall (x=0.5, etc) is 0.5 - R
        # This is true because the sphere is inscribed in the cube.
        # The closest point on the sphere to the wall x=0.5 is at (R,0,0).
        # Distance is 0.5 - R.
        wall_dist = 0.5 - R
        
        r = min(min_pair_dist / 2.0, wall_dist)
        
        # We want to maximize r, so minimize -r
        if not np.isfinite(r):
            return 1e6
            
        return -r

    # Bounds
    # theta: [0, pi]
    # phi: [0, 2pi]
    # R: [0.1, 0.49] (avoid 0 and 0.5)
    bounds = []
    for i in range(N):
        bounds.append((0, np.pi)) # theta
        bounds.append((0, 2 * np.pi)) # phi
    bounds.append((0.01, 0.49)) # R

    # Optimize
    # Use L-BFGS-B for bounded optimization
    # maxiter might need to be high, but 120s limit.
    # 101 variables.
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-10})
    
    # Extract best solution
    x_best = res.x
    theta_phi = x_best[:2*N].reshape(N, 2)
    R_best = x_best[2*N]
    
    theta = theta_phi[:, 0]
    phi = theta_phi[:, 1]
    
    centers = np.zeros((N, 3))
    centers[:, 0] = R_best * np.sin(theta) * np.cos(phi)
    centers[:, 1] = R_best * np.sin(theta) * np.sin(phi)
    centers[:, 2] = R_best * np.cos(theta)
    
    return centers

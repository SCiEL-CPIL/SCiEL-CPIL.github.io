import numpy as np
from scipy.optimize import minimize

def construct():
    N = 50
    
    # 1. Initialize angles using Fibonacci sphere
    # Returns theta, phi arrays of length N
    # theta in [0, pi], phi in [0, 2*pi]
    
    # Fibonacci lattice on sphere
    def fibonacci_sphere(n):
        points = np.zeros((n, 3))
        phi = np.arccos(1 - 2 * (np.arange(n) + 0.5) / n)
        theta = np.pi * (1 + 5**0.5) * np.arange(n)
        points[:, 0] = np.sin(phi) * np.cos(theta)
        points[:, 1] = np.sin(phi) * np.sin(theta)
        points[:, 2] = np.cos(phi)
        return points

    # Get initial unit vectors
    u_init = fibonacci_sphere(N)
    
    # Convert to angles
    # x = sin(theta)cos(phi), y = sin(theta)sin(phi), z = cos(theta)
    # theta = arccos(z)
    # phi = arctan2(y, x)
    theta_init = np.arccos(u_init[:, 2])
    phi_init = np.arctan2(u_init[:, 1], u_init[:, 0])
    
    # Flatten to 1D array for optimizer
    x0 = np.concatenate([theta_init, phi_init])
    
    def objective(x):
        theta = x[:N]
        phi = x[N:]
        
        # Compute unit vectors
        u = np.zeros((N, 3))
        u[:, 0] = np.sin(theta) * np.cos(phi)
        u[:, 1] = np.sin(theta) * np.sin(phi)
        u[:, 2] = np.cos(theta)
        
        # Compute min distance D_unit
        # diff[i, j, k]
        diff = u[:, np.newaxis, :] - u[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        # Set diagonal to inf
        np.fill_diagonal(dist, np.inf)
        D_unit = np.min(dist)
        
        # Compute max U
        # U_i = max(|x_i|, |y_i|, |z_i|)
        U_max = np.max(np.abs(u))
        
        # We want to maximize D_unit / U_max
        # So minimize - (D_unit / U_max)
        # Avoid division by zero (shouldn't happen for valid sphere points)
        if U_max < 1e-9:
            return 0.0
        return - (D_unit / U_max)
        
    # Optimize
    # L-BFGS-B is good for this size (100 vars)
    # Bounds: theta in [0, pi], phi in [0, 2*pi]
    bounds = [(0, np.pi)] * N + [(0, 2 * np.pi)] * N
    
    # Use multiple restarts to avoid local minima
    best_score = -np.inf
    best_x = x0.copy()
    
    for restart in range(5):
        if restart > 0:
            # Add noise to initial guess
            noise = np.random.normal(0, 0.1, size=100)
            x_start = x0 + noise
            # Wrap phi
            x_start[N:] = np.mod(x_start[N:], 2*np.pi)
            # Clip theta
            x_start[:N] = np.clip(x_start[:N], 0, np.pi)
        else:
            x_start = x0
        
        res = minimize(objective, x_start, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if res.fun < best_score:
            best_score = res.fun
            best_x = res.x
            
    # Get best angles
    theta_best = best_x[:N]
    phi_best = best_x[N:]
    
    # Compute unit vectors
    u_best = np.zeros((N, 3))
    u_best[:, 0] = np.sin(theta_best) * np.cos(phi_best)
    u_best[:, 1] = np.sin(theta_best) * np.sin(phi_best)
    u_best[:, 2] = np.cos(theta_best)
    
    # Calculate optimal R
    D_unit = np.min([np.linalg.norm(u_best[i] - u_best[j]) for i in range(N) for j in range(i+1, N)])
    U_max = np.max(np.abs(u_best))
    
    # r = 0.5 * D_unit / (D_unit + 2 * U_max)
    # R * D_unit / 2 = 0.5 - R * U_max
    # R (D_unit/2 + U_max) = 0.5
    # R = 0.5 / (D_unit/2 + U_max)
    R = 0.5 / (D_unit / 2.0 + U_max)
    
    # Compute centers
    centers = u_best * R
    
    return centers

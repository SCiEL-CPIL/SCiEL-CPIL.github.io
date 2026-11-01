import numpy as np
from scipy.optimize import minimize

def construct():
    N = 50
    
    def objective(params):
        R = params[0]
        if R >= 0.5 or R < 0:
            return 1e6
        
        theta = params[1::2]
        phi = params[2::2]
        
        x = R * np.sin(theta) * np.cos(phi)
        y = R * np.sin(theta) * np.sin(phi)
        z = R * np.cos(theta)
        
        centers = np.column_stack((x, y, z))
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        
        wall_dist = 0.5 - R
        
        r = min(min_pair_dist / 2.0, wall_dist)
        
        return -r

    R0 = 0.4
    n = N
    i = np.arange(n)
    # Fibonacci sphere for uniform initial distribution
    phi_init = np.arccos(1 - 2 * (i + 0.5) / n)
    theta_init = np.pi * (1 + 5**0.5) * i
    
    x0 = np.zeros(1 + 2 * N)
    x0[0] = R0
    x0[1::2] = phi_init
    x0[2::2] = theta_init
    
    b = [(0.0, 0.49)]
    for _ in range(N):
        b.append((0.0, np.pi))
        b.append((0.0, 2 * np.pi))
        
    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5})
    
    best_params = res.x
    R_opt = best_params[0]
    theta_opt = best_params[1::2]
    phi_opt = best_params[2::2]
    
    x = R_opt * np.sin(theta_opt) * np.cos(phi_opt)
    y = R_opt * np.sin(theta_opt) * np.sin(phi_opt)
    z = R_opt * np.cos(theta_opt)
    
    centers = np.column_stack((x, y, z))
    
    return centers

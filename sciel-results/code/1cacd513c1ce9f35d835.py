import numpy as np
from scipy.optimize import minimize, differential_evolution

def construct():
    N = 50
    
    # Initial Fibonacci sphere points on unit sphere
    # We will optimize R and the angles
    # Angles: theta (0 to pi), phi (0 to 2*pi)
    
    # Generate initial angles
    # Using Fibonacci lattice for uniform distribution
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    thetas = np.zeros(N)
    phis = np.zeros(N)
    for i in range(N):
        # i is 0 to N-1
        # z = 1 - 2*(i+0.5)/N
        # theta = arccos(z)
        z = 1.0 - 2.0 * (i + 0.5) / N
        thetas[i] = np.arccos(np.clip(z, -1.0, 1.0))
        phis[i] = i * golden_angle
    
    # Flatten variables: [R, theta_0, phi_0, ..., theta_49, phi_49]
    x0 = np.concatenate(([0.25], thetas, phis)) # R=0.25 initial guess
    
    def objective(x):
        R = x[0]
        if R < 0 or R > 0.5:
            return 1e6 # Penalty
        
        # Extract angles
        theta = x[1:1+N]
        phi = x[1+N:1+2*N]
        
        # Compute unit vectors
        # u = [sin(theta)cos(phi), sin(theta)sin(phi), cos(theta)]
        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        cos_p = np.cos(phi)
        sin_p = np.sin(phi)
        
        u = np.column_stack((sin_t * cos_p, sin_t * sin_p, cos_t))
        
        # Wall constraint term
        # max |u_ik|
        max_abs_u = np.max(np.abs(u))
        r_wall = 0.5 - R * max_abs_u
        
        # Pair constraint term
        # min |u_i - u_j|
        # Compute pairwise distances
        # dist = sqrt(2 - 2 * dot(u_i, u_j))
        dot = u @ u.T
        np.fill_diagonal(dot, 1.0)
        # Clip for numerical stability
        dot = np.clip(dot, -1.0, 1.0)
        dist_sq = 2.0 - 2.0 * dot
        # Avoid negative due to precision
        dist_sq = np.clip(dist_sq, 0.0, None)
        dist = np.sqrt(dist_sq)
        # Get min off-diagonal
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        min_dist = np.min(dist[mask])
        
        r_pair = (R / 2.0) * min_dist
        
        r = min(r_wall, r_pair)
        
        # We want to maximize r, so minimize -r
        return -r

    # Use Powell's method for local optimization
    # It doesn't require gradients and handles the non-smooth min well enough
    # Bounds: R in [0, 0.5], theta in [0, pi], phi in [0, 2*pi]
    bounds = [(0, 0.5)] + [(0, np.pi)] * N + [(0, 2 * np.pi)] * N
    
    # Run multiple restarts to avoid local minima
    best_r = -1e9
    best_x = x0
    
    # Initial run
    res = minimize(objective, x0, method='Powell', bounds=bounds, options={'maxiter': 5000, 'maxfev': 50000})
    if res.fun < -best_r:
        best_r = -res.fun
        best_x = res.x
    
    # Try a few random restarts or perturbations
    # Since N=50, 101 vars, Powell can be slow.
    # Maybe use differential evolution for a global search first?
    # DE with 101 vars might be too slow if not tuned.
    
    # Let's try a simpler approach: Optimize R for a fixed set of angles, then optimize angles?
    # Or just run Powell with a good initial guess.
    
    # To improve performance, we can also try to optimize the 3D positions directly
    # but the directive says "constrained to lie on the surface".
    
    # Let's add a second optimization phase with a different initial guess
    # Rotate the initial sphere?
    
    # Actually, let's just return the result from the constrained optimization.
    # We need to convert back to centers.
    
    R_opt = best_x[0]
    theta_opt = best_x[1:1+N]
    phi_opt = best_x[1+N:1+2*N]
    
    sin_t = np.sin(theta_opt)
    cos_t = np.cos(theta_opt)
    cos_p = np.cos(phi_opt)
    sin_p = np.sin(phi_opt)
    
    u_opt = np.column_stack((sin_t * cos_p, sin_t * sin_p, cos_t))
    
    centers = R_opt * u_opt
    
    # Ensure centers are within [-0.5, 0.5]
    centers = np.clip(centers, -0.5, 0.5)
    
    return centers

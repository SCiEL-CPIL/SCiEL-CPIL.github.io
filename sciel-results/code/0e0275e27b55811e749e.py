import numpy as np
from scipy.optimize import minimize

N = 130

# EVOLVE-BLOCK-START
def construct():
    # 1. Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    
    # Convert to theta, phi parameters for optimization
    # z = cos(theta) => theta = arccos(z)
    # x = r cos(phi), y = r sin(phi)
    theta_init = np.arccos(np.clip(z, -1.0, 1.0))
    phi_init = phi
    
    x0 = np.concatenate([theta_init, phi_init])
    
    # 2. Define the energy function
    # We minimize the sum of inverse powers of distances to encourage uniform distribution.
    # This is a proxy for minimizing the covering radius.
    def energy(params):
        theta = params[:N]
        phi = params[N:]
        
        # Calculate Cartesian coordinates
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        pts = np.column_stack([x, y, z])
        
        # Calculate pairwise distances
        # To avoid memory issues for large N, we can do it in chunks or just use broadcasting for N=130
        # N=130 is small enough for (130, 130, 3)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        
        # Mask the diagonal (distance 0)
        np.fill_diagonal(dist_sq, 1.0)
        
        # Use a high power to emphasize close pairs (maximizing min distance)
        # p=10 is a good balance for N=130
        p = 10.0
        inv_dist_p = np.power(dist_sq, -p/2.0)
        
        # Sum over all pairs (each pair counted twice, but constant factor doesn't matter for minimization)
        return np.sum(inv_dist_p)
    
    # 3. Optimize
    # L-BFGS-B is good for this. We set a reasonable maxiter.
    # We can also try Powell if L-BFGS-B struggles, but L-BFGS-B is faster.
    # We'll run a few iterations to be safe within time limits.
    
    result = minimize(energy, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
    
    theta_opt = result.x[:N]
    phi_opt = result.x[N:]
    
    # 4. Convert back to Cartesian
    x = np.sin(theta_opt) * np.cos(phi_opt)
    y = np.sin(theta_opt) * np.sin(phi_opt)
    z = np.cos(theta_opt)
    pts = np.column_stack([x, y, z])
    
    # Normalize to ensure they are on the unit sphere
    norms = np.linalg.norm(pts, axis=1)
    pts = pts / norms[:, np.newaxis]
    
    return pts
# EVOLVE-BLOCK-END

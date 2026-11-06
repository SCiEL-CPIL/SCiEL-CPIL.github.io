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
    
    # Convert to theta, phi parameters
    theta_init = np.arccos(np.clip(z, -1.0, 1.0))
    phi_init = phi
    
    def to_cartesian(theta, phi):
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.column_stack([x, y, z])
        
    def compute_dist_sq(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        return dist_sq

    theta = theta_init.copy()
    phi = phi_init.copy()
    pts = to_cartesian(theta, phi)
    
    # Initial Distances and Energy
    # Riesz energy proxy: Sum of inverse power distances
    p = 10.0
    power = -p / 2.0
    
    dist_sq = compute_dist_sq(pts)
    # E = sum_{i,j} dist_sq[i,j]^power. Diagonal is 1.0^power = 1.0.
    # We minimize this sum.
    E = np.sum(np.power(dist_sq, power))
    
    # 2. Simulated Annealing
    T = 1.0
    T_min = 1e-6
    alpha = 0.9995
    max_iter = 200000
    
    rng = np.random.default_rng(42)
    
    for _ in range(max_iter):
        i = rng.integers(0, N)
        
        # Propose move
        d_theta = rng.normal(0, 0.05)
        d_phi = rng.normal(0, 0.05)
        
        old_theta_i = theta[i]
        old_phi_i = phi[i]
        
        theta[i] += d_theta
        phi[i] += d_phi
        
        # Calculate new Cartesian for point i
        x_new = np.sin(theta[i]) * np.cos(phi[i])
        y_new = np.sin(theta[i]) * np.sin(phi[i])
        z_new = np.cos(theta[i])
        p_new = np.array([x_new, y_new, z_new])
        
        # Calculate new distances from i to all others
        diffs = pts - p_new
        new_dist_sq_i = np.sum(diffs ** 2, axis=1)
        new_dist_sq_i[i] = 1.0 # Self distance
        
        # Old distances from i to all others
        old_dist_sq_i = dist_sq[i]
        
        # Delta E
        new_sum = np.sum(np.power(new_dist_sq_i, power))
        old_sum = np.sum(np.power(old_dist_sq_i, power))
        
        dE = new_sum - old_sum
        
        if dE < 0 or rng.random() < np.exp(-dE / T):
            # Accept
            pts[i] = p_new
            dist_sq[i] = new_dist_sq_i
            dist_sq[:, i] = new_dist_sq_i # Symmetry
            E += dE
        else:
            # Reject
            theta[i] = old_theta_i
            phi[i] = old_phi_i
            
        T *= alpha
        if T < T_min:
            break

    # 3. Final Polish with L-BFGS-B
    x0 = np.concatenate([theta, phi])
    
    def energy_lbfgs(params):
        th = params[:N]
        ph = params[N:]
        p_pts = to_cartesian(th, ph)
        dsq = compute_dist_sq(p_pts)
        return np.sum(np.power(dsq, power))
        
    result = minimize(energy_lbfgs, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
    
    theta_opt = result.x[:N]
    phi_opt = result.x[N:]
    
    # 4. Convert back to Cartesian
    pts_opt = to_cartesian(theta_opt, phi_opt)
    
    # Normalize
    norms = np.linalg.norm(pts_opt, axis=1)
    pts_opt = pts_opt / norms[:, np.newaxis]
    
    return pts_opt
# EVOLVE-BLOCK-END

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    
    def cartesian_from_spherical(theta, phi):
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        return np.column_stack((x, y, z))

    def objective(sph_params):
        theta = sph_params[:n]
        phi = sph_params[n:]
        
        # Clip phi to valid range to avoid numerical issues in sin/cos if optimizer drifts
        # though Nelder-Mead usually stays in bounds if initialized well, 
        # but spherical coords have singularities at poles.
        # It's safer to let it be, but ensure normalization.
        
        pts = cartesian_from_spherical(theta, phi)
        
        # Normalize to unit sphere to handle any numerical drift in sin/cos
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        # If a point is at the pole, sin(phi) is 0, norm is 1.
        # If phi is out of bounds, norms might be off.
        # Actually, sin(phi)cos(theta) etc are always unit length if phi, theta are real.
        # So normalization is a safety check.
        pts /= norms
        
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max < 1e-10:
            return 0.0
            
        ratio = d_min / d_max
        return -ratio

    best_score = -1.0
    best_params = None
    
    np.random.seed(42)
    
    # Multi-start Nelder-Mead
    # 14 points -> 28 parameters
    n_starts = 50
    
    for i in range(n_starts):
        # Random initialization
        # theta in [0, 2*pi)
        # phi in [0, pi]
        # Avoid poles to prevent gradient issues? 
        # Spherical coordinates are singular at poles. 
        # Cartesian optimization on sphere is better, but directive says spherical.
        # To mitigate pole issues, we can initialize phi in [0.1, pi-0.1]
        theta_init = np.random.uniform(0, 2 * np.pi, n)
        phi_init = np.random.uniform(0.1, np.pi - 0.1, n)
        
        init_params = np.concatenate([theta_init, phi_init])
        
        try:
            res = minimize(objective, init_params, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            
            score = -res.fun
            if score > best_score:
                best_score = score
                best_params = res.x
                
        except Exception:
            pass
            
    if best_params is None:
        # Fallback
        theta = np.random.uniform(0, 2 * np.pi, n)
        phi = np.random.uniform(0.1, np.pi - 0.1, n)
        best_params = np.concatenate([theta, phi])
        
    theta = best_params[:n]
    phi = best_params[n:]
    
    pts = cartesian_from_spherical(theta, phi)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    
    return pts

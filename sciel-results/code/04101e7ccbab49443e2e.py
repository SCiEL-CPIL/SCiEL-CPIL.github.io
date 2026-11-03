import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Helper to generate Fibonacci spiral
    def fibonacci_spiral(offset=0.5):
        i = np.arange(N) + offset
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        pts = np.column_stack((x, y, z))
        return pts

    def log_energy(params):
        pts = params.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        pts = pts / norms
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, k=1)
        d = dists[iu]
        d = np.clip(d, 1e-8, None)
        return -np.sum(np.log(d))

    def repulsive_energy(params):
        pts = params.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        pts = pts / norms
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, k=1)
        d = dists[iu]
        d = np.clip(d, 1e-8, None)
        return np.sum(1.0 / (d * d))

    bounds = [(-10, 10)] * (3 * N)
    
    best_energy = np.inf
    best_pts = None
    
    # Generate diverse initializations
    inits = []
    
    # 1. Pure Fibonacci
    inits.append(fibonacci_spiral())
    
    # 2. Random Uniform
    for _ in range(3):
        pts = np.random.randn(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        inits.append(pts)
        
    # 3. Fibonacci with noise
    for _ in range(3):
        pts0 = fibonacci_spiral()
        noise = np.random.normal(0, 0.1, (N, 3))
        pts = pts0 + noise
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        inits.append(pts)

    for init in inits:
        x0 = init.flatten()
        
        # Pre-optimization: Spread points using repulsive energy
        # This helps avoid local minima where points are clustered
        res_pre = minimize(repulsive_energy, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 20})
        
        # Main optimization: Minimize log energy
        res = minimize(log_energy, res_pre.x, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 20})
        
        if res.fun < best_energy:
            best_energy = res.fun
            best_pts = res.x

    # Final polish with more iterations on the best result
    if best_pts is not None:
        res_final = minimize(log_energy, best_pts, method='L-BFGS-B', bounds=bounds, 
                             options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
        if res_final.fun < best_energy:
            best_pts = res_final.x
            best_energy = res_final.fun

    # Convert best_params back to Cartesian and normalize
    pts = best_pts.reshape(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    pts = pts / norms
    
    return pts

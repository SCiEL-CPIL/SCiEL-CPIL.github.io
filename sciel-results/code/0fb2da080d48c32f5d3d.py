import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    # 1. Generate initial FCC lattice
    # Approximate cell size for N=100 in unit cube
    a = (4.0 / N) ** (1.0 / 3.0)
    
    # Generate FCC points
    basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5]
    ]) * a
    
    n_cells = int(np.ceil(1.0 / a)) + 2
    start_idx = -n_cells // 2
    
    points = []
    for i in range(start_idx, start_idx + n_cells):
        for j in range(start_idx, start_idx + n_cells):
            for k in range(start_idx, start_idx + n_cells):
                cell_origin = np.array([i, j, k]) * a
                for b in basis:
                    p = cell_origin + b
                    if np.all(p >= -0.5) and np.all(p <= 0.5):
                        points.append(p)
    
    points = np.array(points)
    
    if len(points) < N:
        np.random.seed(42)
        points = np.random.uniform(-0.4, 0.4, (N, 3))
    else:
        dists = np.linalg.norm(points, axis=1)
        sorted_indices = np.argsort(dists)
        points = points[sorted_indices[:N]]
        
    # 2. Optimize
    lambda_rep = 10.0
    
    def objective(x):
        pts = x.reshape(-1, 3)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        pair_energy = np.sum(np.exp(-lambda_rep * dists))
        
        wall_dists = 0.5 - np.abs(pts)
        wall_energy = np.sum(np.exp(-lambda_rep * wall_dists))
        
        return pair_energy + wall_energy

    bounds = [(-0.5, 0.5)] * (N * 3)
    x0 = points.flatten()
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
        if res.success:
            x_opt = res.x
        else:
            res = minimize(objective, x0, method='Powell', options={'maxiter': 100, 'maxfev': 20000})
            x_opt = res.x
    except Exception:
        x_opt = x0
        
    final_points = x_opt.reshape(N, 3)
    final_points = np.clip(final_points, -0.5, 0.5)
    
    return final_points

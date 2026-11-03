import numpy as np
from scipy.optimize import minimize
import time

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    def calc_radius(points):
        dists_to_origin = np.linalg.norm(points, axis=1)
        min_boundary = 1.0 - np.min(dists_to_origin)
        
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pairwise = np.min(dists) / 2.0
        
        return min(min_boundary, min_pairwise)

    def energy(x):
        points = x.reshape(N, 2)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, 1.0)
        
        i, j = np.triu_indices(N, k=1)
        pair_dists_sq = dists_sq[i, j]
        e_pair = np.sum(1.0 / pair_dists_sq)
        
        r = np.linalg.norm(points, axis=1)
        e_bound = np.sum(np.maximum(0.0, r - 1.0)**2) * 100.0
        
        return e_pair + e_bound

    def energy_grad(x):
        points = x.reshape(N, 2)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists_sq_safe = np.where(dists_sq < 1e-12, 1e-12, dists_sq)
        
        inv_d4 = 1.0 / (dists_sq_safe**2)
        np.fill_diagonal(inv_d4, 0.0)
        
        grad_pair = -2.0 * np.sum(diff * inv_d4[:, :, np.newaxis], axis=1)
        
        r = np.linalg.norm(points, axis=1)
        grad_bound = np.zeros_like(points)
        mask = r > 1.0
        if np.any(mask):
            r_masked = r[mask]
            p_masked = points[mask]
            factor = 200.0 * (r_masked - 1.0) / r_masked
            grad_bound[mask] = p_masked * factor[:, np.newaxis]
            
        return (grad_pair + grad_bound).flatten()

    def get_hex_points(spacing):
        points = []
        k = int(1.0 / spacing) + 2
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                points.append([x, y])
        points = np.array(points)
        r = np.linalg.norm(points, axis=1)
        mask = r <= 1.0 - spacing/2.0
        points = points[mask]
        points = points[np.argsort(np.linalg.norm(points, axis=1))]
        if len(points) >= N:
            return points[:N]
        return None

    best_radius = 0.0
    best_points = None
    
    # Try different initial spacings and perturbations
    spacings = [0.1, 0.095, 0.105, 0.09, 0.11]
    
    for sp in spacings:
        if time.time() > deadline:
            break
            
        init_points = get_hex_points(sp)
        if init_points is None:
            continue
        
        # Try 3 perturbations per spacing
        for _ in range(3):
            if time.time() > deadline:
                break
                
            # Small noise
            noisy_points = init_points + np.random.normal(0, 0.002, init_points.shape)
            
            try:
                res = minimize(
                    energy,
                    noisy_points.flatten(),
                    jac=energy_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-7}
                )
                
                if res.fun < 1e6:
                    opt_points = res.x.reshape(N, 2)
                    rad = calc_radius(opt_points)
                    
                    # Check validity: no NaN, all inside
                    if np.all(np.isfinite(opt_points)) and rad > 0:
                        if rad > best_radius:
                            best_radius = rad
                            best_points = opt_points
                            
            except Exception:
                pass

    if best_points is None:
        init_points = get_hex_points(0.1)
        if init_points is not None:
            best_points = init_points
        else:
            best_points = np.random.uniform(-0.9, 0.9, (N, 2))
            
    return best_points

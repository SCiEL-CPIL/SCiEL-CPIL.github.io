# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def _compute_radius(centers):
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
        
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    min_wall_dist = np.min(wall_dists)
    
    if min_wall_dist < 0:
        return 0.0

    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _energy_and_gradient(x, R):
    centers = x.reshape(-1, 3)
    n = len(centers)
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # 1. Pairwise terms
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    d_ij = dists[i_idx, j_idx]
    
    v_ij = np.maximum(0.0, 2.0 * R - d_ij)
    
    if np.any(v_ij > 0):
        energy += np.sum(v_ij**2)
        
        vecs = diff[i_idx, j_idx]
        d_safe = np.where(d_ij > 1e-10, d_ij, 1.0)
        factor = -2.0 * v_ij / d_safe
        
        forces_k = factor[:, np.newaxis] * vecs
        
        np.add.at(grad, i_idx, forces_k)
        np.add.at(grad, j_idx, -forces_k)

    # 2. Wall terms
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_abs
    v_wall = np.maximum(0.0, R - wall_dists)
    
    if np.any(v_wall > 0):
        energy += np.sum(v_wall**2)
        
        max_indices = np.argmax(abs_centers, axis=1)
        signs = np.sign(centers[np.arange(n), max_indices])
        
        mask = v_wall > 0
        if np.any(mask):
            grad[np.arange(n)[mask], max_indices[mask]] += 2.0 * v_wall[mask] * signs[mask]
            
    return energy, grad.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_radius = 0.0
    best_centers = None
    
    rng = np.random.RandomState(42)
    
    def make_grid(nx, ny, nz):
        xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
        ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
        zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
        grid = []
        for z in zs:
            for y in ys:
                for x in xs:
                    grid.append([x, y, z])
        return np.array(grid)
        
    inits = []
    inits.append(make_grid(5, 4, 5))
    inits.append(make_grid(4, 5, 5))
    inits.append(make_grid(5, 5, 4))
    
    for _ in range(5):
        inits.append(rng.uniform(-0.45, 0.45, size=(N, 3)))
        
    base = make_grid(5, 4, 5)
    for scale in [0.01, 0.02, 0.05]:
        inits.append(base + rng.normal(0, scale, size=(N, 3)))

    for init in inits:
        if time.time() > deadline - 15.0:
            break
            
        current_centers = init.copy()
        current_r = _compute_radius(current_centers)
        
        if current_r > best_radius:
            best_radius = current_r
            best_centers = current_centers.copy()
            
        target_R = max(current_r * 1.01, 0.05)
        stuck_count = 0
        
        while time.time() < deadline - 10.0:
            def obj_grad(x):
                e, g = _energy_and_gradient(x, target_R)
                return e, g
                
            bounds = [(-0.5, 0.5)] * (N * 3)
            
            try:
                res = minimize(
                    obj_grad,
                    current_centers.flatten(),
                    jac=True,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'disp': False, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                new_centers = res.x.reshape(-1, 3)
                energy = res.fun
            except Exception:
                break
                
            if energy < 1e-6:
                new_r = _compute_radius(new_centers)
                if new_r > best_radius:
                    best_radius = new_r
                    best_centers = new_centers.copy()
                    
                current_centers = new_centers
                current_r = new_r
                target_R = current_r * 1.01
                stuck_count = 0
            else:
                stuck_count += 1
                if stuck_count > 3:
                    break
                target_R *= 0.99

    if best_centers is not None and time.time() < deadline - 5.0:
        target_R = best_radius * 1.001
        def obj_grad(x):
            e, g = _energy_and_gradient(x, target_R)
            return e, g
            
        try:
            res = minimize(
                obj_grad,
                best_centers.flatten(),
                jac=True,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 200, 'disp': False, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            polished = res.x.reshape(-1, 3)
            r_polished = _compute_radius(polished)
            if r_polished > best_radius:
                best_radius = r_polished
                best_centers = polished
        except Exception:
            pass

    if best_centers is None:
        best_centers = make_grid(5, 4, 5)
        
    best_centers = np.clip(best_centers, -0.49999, 0.49999)
    return best_centers
# EVOLVE-BLOCK-END

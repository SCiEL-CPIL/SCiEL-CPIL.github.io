import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    def energy_and_grad(x):
        centers = x.reshape(N, 3)
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        d2 = np.sum(diff**2, axis=2)
        np.fill_diagonal(d2, 1e-9)
        
        # Energy from pairs: sum 1/d^2
        d2_sq = d2**2
        np.fill_diagonal(d2_sq, 1e-9)
        
        coeff = -2.0 / d2_sq
        np.fill_diagonal(coeff, 0.0)
        
        grad_pair = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
        
        grad_wall = np.zeros_like(centers)
        e_wall = 0.0
        for k in range(3):
            c = centers[:, k]
            d_pos = 0.5 - c
            d_neg = 0.5 + c
            
            d_pos = np.maximum(d_pos, 1e-6)
            d_neg = np.maximum(d_neg, 1e-6)
            
            e_wall += np.sum(1.0 / d_pos**2) + np.sum(1.0 / d_neg**2)
            
            grad_wall[:, k] += 2.0 / d_pos**3 - 2.0 / d_neg**3
            
        e_pair = 0.5 * np.sum(1.0 / d2)
        
        total_grad = grad_pair + grad_wall
        total_energy = e_pair + e_wall
        
        return total_energy, total_grad.flatten()

    def get_radius(centers):
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        d2 = np.sum(diff**2, axis=2)
        np.fill_diagonal(d2, 1e9)
        d_min = np.sqrt(np.min(d2))
        
        w_min = np.min(0.5 - np.abs(centers))
        
        return min(d_min / 2.0, w_min)

    inits = []
    
    # 1. Jittered Grid
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    grid = grid[:N]
    for _ in range(5):
        jitter = np.random.normal(0, 0.02, grid.shape)
        inits.append(np.clip(grid + jitter, -0.49, 0.49))
        
    # 2. Random
    for _ in range(5):
        inits.append(np.random.uniform(-0.4, 0.4, (N, 3)))
        
    # 3. Lattice with different offsets
    m1, m2, m3 = 5, 5, 4
    ticks1 = (np.arange(m1) + 0.5) / m1 - 0.5
    ticks2 = (np.arange(m2) + 0.5) / m2 - 0.5
    ticks3 = (np.arange(m3) + 0.5) / m3 - 0.5
    grid2 = np.array([(x, y, z) for z in ticks3 for y in ticks2 for x in ticks1])
    grid2 = grid2[:N]
    for _ in range(5):
        jitter = np.random.normal(0, 0.02, grid2.shape)
        inits.append(np.clip(grid2 + jitter, -0.49, 0.49))

    best_r = 0.0
    best_centers = None
    start_time = time.time()
    time_limit = 110.0

    for init in inits:
        if time.time() - start_time > time_limit:
            break
            
        x0 = init.flatten()
        
        try:
            res = minimize(
                energy_and_grad, 
                x0, 
                jac=True, 
                method='L-BFGS-B', 
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
        except Exception:
            continue
            
        centers = res.x.reshape(N, 3)
        r = get_radius(centers)
        
        if r > best_r:
            best_r = r
            best_centers = centers

    if best_centers is not None and time.time() - start_time < time_limit:
        x0 = best_centers.flatten()
        try:
            res = minimize(
                energy_and_grad, 
                x0, 
                jac=True, 
                method='L-BFGS-B', 
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            centers = res.x.reshape(N, 3)
            r = get_radius(centers)
            if r > best_r:
                best_r = r
                best_centers = centers
        except Exception:
            pass

    return best_centers

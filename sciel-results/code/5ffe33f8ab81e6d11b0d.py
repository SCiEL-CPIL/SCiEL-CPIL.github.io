import numpy as np
from scipy.optimize import minimize, minimize_scalar
import time

N = 38

def _lj_energy(pos):
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    inv_r6 = 1.0 / (dist ** 6)
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6)
    return 0.5 * energy

def _lj_grad(pos):
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    inv_dist_sq = 1.0 / dist_sq
    G = 24.0 * (2.0 * (inv_dist_sq ** 7) - (inv_dist_sq ** 4))
    np.fill_diagonal(G, 0.0)
    sum_G = np.sum(G, axis=1, keepdims=True)
    G_pos = G @ pos
    grad = sum_G * pos - G_pos
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Generate FCC lattice points
    points = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            for k in range(-5, 6):
                if (i + j + k) % 2 == 0:
                    points.append((i, j, k))
    
    points.sort(key=lambda p: p[0]**2 + p[1]**2 + p[2]**2)
    pts = np.array(points[:N], dtype=float)
    
    # Scale to LJ minimum
    scale = (2.0 ** (1.0/6.0)) / np.sqrt(2.0)
    pts = pts * scale
    pts -= np.mean(pts, axis=0)
    
    # Relax
    if time.time() < deadline - 10.0:
        res = minimize(
            _lj_energy, 
            pts, 
            method='L-BFGS-B', 
            jac=_lj_grad, 
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if res.success:
            pts = res.x
    
    # Scale optimization
    if time.time() < deadline - 2.0:
        def energy_scaled(s):
            return _lj_energy(pts * s)
        
        res_scale = minimize_scalar(
            energy_scaled, 
            bounds=(0.9, 1.2), 
            method='bounded', 
            options={'xatol': 1e-8}
        )
        if res_scale.success:
            pts = pts * res_scale.x
    
    return pts

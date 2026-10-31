import numpy as np
from scipy.optimize import minimize
import time

N = 38
TARGET_ENERGY = -173.928427

def _lj_energy_and_grad(pos):
    """
    Compute total Lennard-Jones energy and gradient.
    pos: (N, 3) array.
    Returns: (energy, grad) where grad is (N, 3).
    """
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(r2, 1.0)
    
    r6 = r2 ** 3
    r12 = r6 ** 2
    
    # Pairwise energy
    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = np.sum(pair_energy)
    
    # Gradient
    # d/dr (4*(r^-12 - r^-6)) = 4*(-12*r^-13 + 6*r^-7) = 24*(2*r^-13 - r^-7)
    # d/dr2 (4*(r^-12 - r^-6)) = 2 * d/dr (4*(r^-12 - r^-6)) / (2*r) * 2? No.
    # Let's use the standard vector form.
    # U = 4 * (r^-12 - r^-6)
    # F = -dU/dr * r_hat = -dU/dr * (diff / r)
    # dU/dr = 4 * (-12 * r^-13 - (-6) * r^-7) = 4 * (-12 * r^-13 + 6 * r^-7)
    # F = -4 * (-12 * r^-13 + 6 * r^-7) * diff / r
    # F = 4 * (12 * r^-13 - 6 * r^-7) * diff / r
    # F = 4 * (12 * r^-14 - 6 * r^-8) * diff
    # F = 24 * (2 * r^-14 - r^-8) * diff
    
    # Using r2:
    # r^-14 = (r2)^-7
    # r^-8 = (r2)^-4
    # coeff = 24 * (2 * (r2)^-7 - (r2)^-4)
    # coeff shape (N, N)
    
    coeff = 24.0 * (2.0 / r12 / r2 - 1.0 / r6) # Wait.
    # r^-14 = r^-12 * r^-2 = (1/r12) * (1/r2)
    # r^-8 = r^-6 * r^-2 = (1/r6) * (1/r2)
    # So coeff = 24 * (2/r12/r2 - 1/r6/r2)
    
    # Let's re-verify.
    # U = 4(r^-12 - r^-6)
    # dU/dri = 2 * (U_r / r) * (ri - rj) ? No.
    # dU/dri = 24 * (2/r^14 - 1/r^8) * (ri - rj)
    # (ri - rj) is diff[i, j].
    # coeff[i, j] = 24 * (2/r^14 - 1/r^8)
    
    coeff = 24.0 * (2.0 / (r12 * r2) - 1.0 / (r6 * r2))
    np.fill_diagonal(coeff, 0.0)
    
    # grad[i] = sum_j coeff[i, j] * diff[i, j]
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad

def _make_fcc_truncated_octahedron(n_atoms):
    """
    Generate initial coordinates for an FCC truncated octahedron cluster.
    """
    r_min = 2.0 ** (1.0 / 6.0)
    a = r_min * np.sqrt(2.0)
    
    points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                if (i + j + k) % 2 == 0:
                    x = i * a
                    y = j * a
                    z = k * a
                    points.append([x, y, z])
    
    points = np.array(points)
    dists = np.sum(points**2, axis=1)
    sorted_indices = np.argsort(dists)
    
    selected_points = points[sorted_indices[:n_atoms]]
    return selected_points

def _relax(pos, deadline):
    """
    Relax positions using L-BFGS-B with analytic gradient.
    """
    def obj_grad(x):
        p = x.reshape(N, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.flatten()
    
    bounds = [(-10.0, 10.0)] * (N * 3)
    x0 = pos.flatten()
    
    res = minimize(
        obj_grad, 
        x0, 
        jac=True,
        method='L-BFGS-B', 
        bounds=bounds, 
        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    if time.time() > deadline:
        return pos, res.fun
    
    final_pos = res.x.reshape(N, 3)
    return final_pos, res.fun

def _ensure_min_sep(pos, min_sep=0.5):
    """
    Ensure all atoms are at least min_sep apart.
    If not, push them apart slightly.
    """
    pos = pos.copy()
    for _ in range(10):
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, 1.0)
        
        # Find pairs closer than min_sep
        close_mask = dists < min_sep
        if not np.any(close_mask):
            break
        
        # For each close pair, push them apart
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                if close_mask[i, j]:
                    d = pos[i] - pos[j]
                    dist = np.linalg.norm(d)
                    if dist < 1e-6:
                        d = np.array([1e-6, 0, 0])
                        dist = 1e-6
                    d_hat = d / dist
                    # Push apart by (min_sep - dist) / 2
                    push = (min_sep - dist) / 2.0
                    pos[i] += d_hat * push
                    pos[j] -= d_hat * push
                    changed = True
        if not changed:
            break
            
    return pos

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin for 150s limit
    
    # 1. Generate initial guess: FCC truncated octahedron
    initial_pos = _make_fcc_truncated_octahedron(N)
    
    # 2. Relax
    pos, energy = _relax(initial_pos, deadline)
    
    # 3. Polish with tighter tolerance if time permits
    if time.time() < deadline - 5.0:
        def obj_grad(x):
            p = x.reshape(N, 3)
            e, g = _lj_energy_and_grad(p)
            return e, g.flatten()
        
        x0 = pos.flatten()
        bounds = [(-10.0, 10.0)] * (N * 3)
        res = minimize(
            obj_grad, 
            x0, 
            jac=True,
            method='L-BFGS-B', 
            bounds=bounds, 
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        pos = res.x.reshape(N, 3)
        energy = res.fun
    
    # 4. Ensure minimum separation
    pos = _ensure_min_sep(pos)
    
    # Recalculate final energy to be safe
    energy, _ = _lj_energy_and_grad(pos)
    
    return pos

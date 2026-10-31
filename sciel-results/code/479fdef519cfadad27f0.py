import numpy as np
from scipy.optimize import minimize
import time

N = 38
START_TIME = time.time()
TIME_LIMIT = 115.0  # Safety margin for 120s limit

def lj_energy_and_grad(pos):
    """
    Calculate Lennard-Jones energy and gradient.
    pos: (N, 3) array
    """
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=2)
    # Avoid division by zero for self-interaction
    np.fill_diagonal(r2, 1.0)
    
    r6 = r2**(-3)
    r12 = r6**2
    
    # Energy: 4 * sum(r^-12 - r^-6) over i < j
    # Sum over all i != j is 2 * sum over i < j
    E = 2.0 * np.sum(4.0 * (r12 - r6))
    
    # Gradient: dE/dx_i
    # dE/dr = 4 * (-12*r^-13 + 6*r^-7)
    # dr/dx_i = (x_i - x_j) / r
    # dE/dx_i = sum_j 4 * (-12*r^-13 + 6*r^-7) * (x_i - x_j) / r
    # = sum_j 4 * (-12*r^-14 + 6*r^-8) * (x_i - x_j)
    
    # r^-14 = r2^-7
    # r^-8 = r2^-4
    grad_factor = 4.0 * (-12.0 * r2**(-7) + 6.0 * r2**(-4))
    grad = np.sum(grad_factor[:, :, np.newaxis] * diff, axis=1)
    
    return E, grad

def relax(pos):
    """
    Relax a configuration to a local minimum using L-BFGS.
    """
    def objective(x):
        e, g = lj_energy_and_grad(x.reshape(N, 3))
        return e, g

    res = minimize(objective, pos, method='L-BFGS-B', jac=True, options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
    return res.x, res.fun

def basin_hopping(initial_pos, max_iters=500):
    """
    Perform Basin Hopping starting from initial_pos.
    """
    current_pos, current_energy = relax(initial_pos)
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    sigma = 0.5  # Initial step size
    T = 5.0      # Initial temperature
    
    for i in range(max_iters):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        # Perturb
        delta = np.random.normal(0, sigma, (N, 3))
        perturbed_pos = current_pos + delta
        
        # Relax
        try:
            new_pos, new_energy = relax(perturbed_pos)
        except Exception:
            continue
        
        # Acceptance criterion (Metropolis)
        if new_energy < current_energy:
            current_pos = new_pos
            current_energy = new_energy
        else:
            delta_E = new_energy - current_energy
            if np.random.rand() < np.exp(-delta_E / T):
                current_pos = new_pos
                current_energy = new_energy
        
        # Update best
        if current_energy < best_energy:
            best_pos = current_pos.copy()
            best_energy = current_energy
        
        # Anneal
        sigma *= 0.995
        T *= 0.995
        
        # If sigma becomes too small, stop or reset
        if sigma < 0.01:
            break
            
    return best_pos, best_energy

def construct():
    # Generate diverse initial seeds
    seeds = []
    
    # Seed 1: Random
    seeds.append(np.random.uniform(-2, 2, (N, 3)))
    
    # Seed 2: Cubic lattice fragment
    side = int(np.ceil(N ** (1/3)))
    points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
    points.sort(key=lambda p: (p[0] - (side-1)/2)**2 + (p[1] - (side-1)/2)**2 + (p[2] - (side-1)/2)**2)
    seeds.append(np.array(points[:N], dtype=float) * 1.12)
    
    # Seed 3: Icosahedral-like (1 center, 12 shell, rest random nearby)
    ico_seed = np.zeros((N, 3))
    # 12 icosahedral vertices
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    ico_verts /= np.linalg.norm(ico_verts[0]) # Normalize
    ico_verts *= 1.12
    ico_seed[1:13] = ico_verts
    # Remaining 25 atoms: random in a sphere of radius 2.5
    remaining = np.random.uniform(-2.5, 2.5, (N-13, 3))
    ico_seed[13:] = remaining
    seeds.append(ico_seed)
    
    # Seed 4: FCC-like fragment
    # Generate FCC points
    fcc_points = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # Simple cubic
                fcc_points.append([i, j, k])
                # Face centers
                if i < 2: fcc_points.append([i+0.5, j+0.5, k])
                if j < 2: fcc_points.append([i+0.5, j, k+0.5])
                if k < 2: fcc_points.append([i, j+0.5, k+0.5])
    fcc_points = np.array(fcc_points)
    # Center and scale
    fcc_points -= fcc_points.mean(axis=0)
    fcc_points *= 1.12
    # Select closest 38 to center
    dists = np.linalg.norm(fcc_points, axis=1)
    idx = np.argsort(dists)[:N]
    seeds.append(fcc_points[idx])
    
    best_pos = None
    best_energy = np.inf
    
    # Run Basin Hopping from each seed
    for seed in seeds:
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        # Relax the seed first to get a good starting point for BH
        try:
            pos, energy = relax(seed)
        except Exception:
            continue
            
        if energy < best_energy:
            best_pos = pos
            best_energy = energy
            
        # Run Basin Hopping
        bh_pos, bh_energy = basin_hopping(pos, max_iters=100)
        
        if bh_energy < best_energy:
            best_pos = bh_pos
            best_energy = bh_energy
            
    # Final polish with tighter tolerance
    if best_pos is not None:
        try:
            final_pos, final_energy = relax(best_pos)
            if final_energy < best_energy:
                best_pos = final_pos
        except Exception:
            pass
            
    return best_pos

# Ensure the function exists and is called correctly
# The task requires construct() to return the array.


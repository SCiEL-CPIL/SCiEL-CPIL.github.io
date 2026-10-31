import numpy as np
from scipy.optimize import minimize
import time
import random

N = 38

def _lj_energy_and_grad(pos):
    """
    Calculate Lennard-Jones energy and gradient for a set of positions.
    pos: (N, 3) array of coordinates.
    Returns: energy, gradient (N, 3)
    """
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Avoid division by zero for i==j
    np.fill_diagonal(dist_sq, 1.0)
    
    inv_r2 = 1.0 / dist_sq
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    
    # Energy
    pair_energy = inv_r12 - inv_r6
    np.fill_diagonal(pair_energy, 0.0)
    energy = 2.0 * np.sum(pair_energy)
    
    # Gradient
    inv_r8 = inv_r2 ** 4
    inv_r14 = inv_r2 ** 7
    
    grad_factor = 4.0 * (-12.0 * inv_r14 + 6.0 * inv_r8)
    np.fill_diagonal(grad_factor, 0.0)
    
    grad = np.sum(grad_factor[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad

def _make_fcc_cluster_variant(seed):
    """
    Generate an FCC lattice fragment of 38 atoms.
    Uses a specific random seed to select which 19 atoms from the outer shell (24 total) to include.
    """
    random.seed(seed)
    
    # Generate FCC points
    v1 = np.array([1.0, 1.0, 0.0])
    v2 = np.array([1.0, 0.0, 1.0])
    v3 = np.array([0.0, 1.0, 1.0])
    
    points = []
    for n1 in range(-4, 5):
        for n2 in range(-4, 5):
            for n3 in range(-4, 5):
                p = n1 * v1 + n2 * v2 + n3 * v3
                points.append(p)
                
    points = np.array(points)
    dists_sq = np.sum(points**2, axis=1)
    
    # Sort by distance
    sorted_indices = np.argsort(dists_sq)
    
    # We know the structure of distances:
    # 1 point at d2=0
    # 12 points at d2=2
    # 6 points at d2=4
    # 24 points at d2=6
    # We need 38 points.
    # The first 19 (indices 0..18) are the inner shells (d2 <= 4).
    # The next 24 (indices 19..42) are the outer shell (d2 = 6).
    # We must take all 19 inner points.
    # We must take 19 of the 24 outer points.
    
    inner_indices = sorted_indices[:19]
    outer_indices = sorted_indices[19:43]
    
    # Randomly select 19 from the 24 outer indices
    selected_outer = random.sample(outer_indices.tolist(), 19)
    
    selected_indices = np.concatenate([inner_indices, selected_outer])
    
    cluster = points[selected_indices]
    
    # Center the cluster
    centroid = np.mean(cluster, axis=0)
    cluster = cluster - centroid
    
    # Scale to LJ minimum
    # Nearest neighbor distance in this lattice is sqrt(2).
    # LJ min is 2^(1/6).
    scale = (2.0 ** (1.0/6.0)) / np.sqrt(2.0)
    cluster = cluster * scale
    
    return cluster

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_pos = None
    best_e = np.inf
    
    # Try multiple variants to ensure we find the global minimum
    # The number of variants is limited by time.
    # L-BFGS on 38 atoms is fast (< 1s per run usually).
    # We can afford ~50-100 runs.
    
    num_tries = 50
    for i in range(num_tries):
        if time.time() > deadline:
            break
            
        # Use a deterministic seed for reproducibility of the search process,
        # but vary the variant seed.
        pos = _make_fcc_cluster_variant(seed=i)
        
        try:
            res = minimize(
                _lj_energy_and_grad, 
                pos, 
                jac=True, 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'gtol': 1e-12, 'ftol': 1e-14}
            )
            
            if res.fun < best_e:
                best_e = res.fun
                best_pos = res.x.copy()
                
        except Exception:
            pass

    if best_pos is None:
        # Fallback to a single run if all failed (unlikely)
        pos = _make_fcc_cluster_variant(seed=0)
        res = minimize(
            _lj_energy_and_grad, 
            pos, 
            jac=True, 
            method='L-BFGS-B', 
            options={'maxiter': 1000}
        )
        best_pos = res.x
        best_e = res.fun

    return best_pos

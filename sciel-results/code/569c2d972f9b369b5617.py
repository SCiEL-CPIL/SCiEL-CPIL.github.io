"""Program for Lennard-Jones 55-atom cluster global minimum search.

Uses the known Mackay icosahedron geometry, applies diverse perturbations,
and relaxes with L-BFGS to converge to the global minimum energy of ~-279.248.
Includes strict time management and high-precision polishing.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 55

def _get_icosahedron_vectors():
    """
    Returns 12 unit vectors along the 5-fold axes (icosahedron vertices)
    and 30 unit vectors along the 2-fold axes (icosahedron edge midpoints).
    """
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of icosahedron
    verts = np.array([
        [0, 1, phi],
        [0, -1, phi],
        [0, 1, -phi],
        [0, -1, -phi],
        [1, phi, 0],
        [-1, phi, 0],
        [1, -phi, 0],
        [-1, -phi, 0],
        [phi, 0, 1],
        [-phi, 0, 1],
        [phi, 0, -1],
        [-phi, 0, -1],
    ], dtype=float)
    
    # Normalize to unit length
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # 30 edge midpoints
    dists = np.linalg.norm(verts[:, np.newaxis, :] - verts[np.newaxis, :, :], axis=2)
    min_dist = np.min(dists[dists > 1e-8])
    
    edge_mids = []
    for i in range(12):
        for j in range(i+1, 12):
            if np.isclose(dists[i,j], min_dist, atol=1e-8):
                mid = (verts[i] + verts[j]) / 2
                mid = mid / np.linalg.norm(mid)
                edge_mids.append(mid)
                
    edge_mids = np.array(edge_mids)
    return verts, edge_mids

def _mackay_icosahedron_55(r1=1.135, r2=1.162, r3=1.283):
    """
    Construct the 55-atom two-shell Mackay icosahedron.
    1 center, 12 at 5-fold (r1), 30 at 2-fold (r2), 12 at 5-fold (r3).
    """
    v5, v2 = _get_icosahedron_vectors()
    
    points = [np.zeros(3)]
    
    # Add 12 atoms at 5-fold axes, radius r1
    for v in v5:
        points.append(v * r1)
        
    # Add 30 atoms at 2-fold axes, radius r2
    for v in v2:
        points.append(v * r2)
        
    # Add 12 atoms at 5-fold axes, radius r3
    for v in v5:
        points.append(v * r3)
        
    points = np.array(points, dtype=float)
    return points

def _lj_energy_and_grad(pos):
    """Compute total Lennard-Jones energy and gradient."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    
    # Avoid division by zero
    mask = r2 > 1e-10
    r2_safe = np.where(mask, r2, 1.0)
    
    # r^-6 and r^-12
    r6 = r2_safe ** (-3)
    r12 = r6 ** 2
    
    # Energy: 4 * sum(r^-12 - r^-6)
    # Note: The sum is over all pairs i,j. Since the matrix is symmetric and diagonal is 0,
    # summing the whole matrix counts each pair twice.
    # Standard LJ energy is sum_{i<j} 4(r^-12 - r^-6).
    # So we compute sum over all i!=j and divide by 2.
    energy_matrix = np.where(mask, r12 - r6, 0)
    energy = 2.0 * np.sum(energy_matrix) # 4 * (1/2) * sum = 2 * sum
    
    # Gradient:
    # dE/dx_i = sum_{j != i} 4 * (-12*r^-14 + 6*r^-8) * (x_i - x_j)
    # r^-14 = r2^-7, r^-8 = r2^-4
    r2_inv4 = r2_safe ** (-4)
    r2_inv7 = r2_safe ** (-7)
    
    coeff = 4 * (-12 * r2_inv7 + 6 * r2_inv4)
    coeff = np.where(mask, coeff, 0)
    
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return energy, grad

def _is_valid(pos):
    """Check if all inter-atomic distances are >= 0.5."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= 0.5

def _relax(pos, maxiter=5000, ftol=1e-12, gtol=1e-10):
    """Relax positions using L-BFGS-B with analytical gradient."""
    def energy_func(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(pos_flat)
        return e, g.flatten()
        
    try:
        result = minimize(
            energy_func,
            pos.flatten(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
        )
        return result.x.reshape(-1, 3)
    except Exception:
        return pos

def _polish(pos, deadline):
    """High-precision polish pass with tighter tolerance."""
    def energy_func(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(pos_flat)
        return e, g.flatten()
    
    current_pos = pos
    best_energy = float('inf')
    
    # Multiple stages of increasing strictness
    stages = [
        {'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12},
        {'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14},
        {'maxiter': 10000, 'ftol': 1e-18, 'gtol': 1e-16},
    ]
    
    for opts in stages:
        if time.time() > deadline - 5.0:
            break
            
        try:
            result = minimize(
                energy_func,
                current_pos.flatten(),
                jac=True,
                method='L-BFGS-B',
                options=opts
            )
            new_pos = result.x.reshape(-1, 3)
            e, _ = _lj_energy_and_grad(new_pos)
            
            if e < best_energy and _is_valid(new_pos):
                best_energy = e
                current_pos = new_pos
        except Exception:
            pass
            
    return current_pos

def construct():
    """
    Construct the 55-atom LJ cluster by perturbing the Mackay icosahedron
    and relaxing with L-BFGS.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    
    # Base geometry: Standard Mackay Icosahedron
    base = _mackay_icosahedron_55(1.135, 1.162, 1.283)
    
    best_energy = float('inf')
    best_pos = base.copy()
    
    # Strategy 1: Relax the exact Mackay geometry
    relaxed = _relax(base)
    if _is_valid(relaxed):
        e, _ = _lj_energy_and_grad(relaxed)
        if e < best_energy:
            best_energy = e
            best_pos = relaxed.copy()
            
    # Strategy 2: Multi-start with small perturbations
    # We try a few different noise scales and radial perturbations
    noise_scales = [0.01, 0.05, 0.1]
    radial_perts = [0.99, 1.0, 1.01]
    
    for scale in noise_scales:
        for rp in radial_perts:
            if time.time() > deadline - 20.0:
                break
                
            # Perturb radii slightly
            r1 = 1.135 * rp
            r2 = 1.162 * rp
            r3 = 1.283 * rp
            pos = _mackay_icosahedron_55(r1, r2, r3)
            
            # Add Gaussian noise
            noise = rng.normal(0, scale, size=pos.shape)
            pos = pos + noise
            
            relaxed = _relax(pos)
            
            if not _is_valid(relaxed):
                continue
                
            e, _ = _lj_energy_and_grad(relaxed)
            
            if e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
                
    # Final high-precision polish on the best candidate found
    if time.time() < deadline - 5.0:
        polished = _polish(best_pos, deadline)
        if _is_valid(polished):
            e, _ = _lj_energy_and_grad(polished)
            if e < best_energy:
                best_energy = e
                best_pos = polished
    
    return best_pos

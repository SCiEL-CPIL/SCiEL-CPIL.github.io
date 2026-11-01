"""
Lennard-Jones 55-atom cluster global minimum search.

Constructs the two-shell Mackay icosahedron geometry and relaxes it using
L-BFGS-B to converge to the global minimum energy (~-279.248470).
Includes multi-start logic to ensure robustness against local minima traps,
though the Mackay icosahedron is the known global minimum for N=55.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 55
TIME_LIMIT = 150.0  # Seconds, hard limit
SAFETY_MARGIN = 5.0 # Seconds to spare

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
    # Calculate distances between all pairs
    dists = np.linalg.norm(verts[:, np.newaxis, :] - verts[np.newaxis, :, :], axis=2)
    # Find the minimum distance (edge length)
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
    """Compute total Lennard-Jones energy and gradient efficiently."""
    n = len(pos)
    # Compute all pairwise differences
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    
    # Mask diagonal to avoid division by zero
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    
    r2_safe = np.where(mask, r2, 1.0)
    
    # Compute r^-6 and r^-12
    r6 = r2_safe ** (-3)
    r12 = r6 ** 2
    
    # Energy: 4 * sum (r^-12 - r^-6)
    # Note: The sum includes both (i,j) and (j,i), so we sum over all i != j
    # This is 2 * sum_{i<j} 4(r^-12 - r^-6)
    energy = 4 * np.sum(np.where(mask, r12 - r6, 0))
    
    # Gradient:
    # dE/dx_i = sum_{j != i} 4 * (-12 * r^-14 + 6 * r^-8) * (x_i - x_j) / r
    # Note: (x_i - x_j) / r = diff[i,j] / sqrt(r2)
    # So coeff = 4 * (-12 * r^-14 + 6 * r^-8) / r = 4 * (-12 * r^-15 + 6 * r^-9)
    # Alternatively: 4 * (-12 * r2^-7 + 6 * r2^-4) * diff / sqrt(r2)
    # Let's use the r2 form for numerical stability
    r2_inv4 = r2_safe ** (-4)
    r2_inv7 = r2_safe ** (-7)
    
    # Coefficient for each pair (i,j) acting on diff[i,j]
    # dE/dx_i = sum_j 4 * (-12 * r^-14 + 6 * r^-8) * (x_i - x_j) / r
    # = sum_j 4 * (-12 * r2^-7 * r^-1 + 6 * r2^-4 * r^-1) * (x_i - x_j)
    # Wait, r^-14 = r2^-7. r^-8 = r2^-4.
    # dE/dx_i = sum_j 4 * (-12 * r2^-7 + 6 * r2^-4) * (x_i - x_j) / r
    # Let C_ij = 4 * (-12 * r2^-7 + 6 * r2^-4) / sqrt(r2)
    # grad_i = sum_j C_ij * diff[i,j]
    
    r = np.sqrt(r2_safe)
    coeff = 4 * (-12 * r2_inv7 + 6 * r2_inv4) / r
    coeff = np.where(mask, coeff, 0)
    
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return energy, grad

def _is_valid(pos):
    """Check if all inter-atomic distances are >= 0.5."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= 0.5

def _relax(pos, maxiter=10000, ftol=1e-14, gtol=1e-12):
    """Relax positions using L-BFGS-B with analytical gradient."""
    def energy_func(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(pos_flat)
        return e, g.flatten()
        
    result = minimize(
        energy_func,
        pos.flatten(),
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
    )
    return result.x.reshape(-1, 3)

def construct():
    """
    Construct the 55-atom LJ cluster at the global minimum.
    """
    start_time = time.time()
    
    # 1. Generate the exact Mackay icosahedron geometry
    # These radii are chosen to be close to the relaxed equilibrium distances
    base = _mackay_icosahedron_55(r1=1.135, r2=1.162, r3=1.283)
    
    best_pos = base.copy()
    best_energy = _lj_energy_and_grad(base)[0]
    
    # 2. Relax the geometry
    # Since the Mackay icosahedron is the global minimum, a single high-precision
    # relaxation from the ideal symmetric structure should suffice.
    
    if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
        return best_pos
    
    try:
        relaxed = _relax(base)
        if _is_valid(relaxed):
            e = _lj_energy_and_grad(relaxed)[0]
            if e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
    except Exception:
        pass
    
    # 3. If time remains, try a few small perturbations just in case
    # This implements the multi-start strategy to ensure we are in the global basin.
    if time.time() - start_time < TIME_LIMIT - SAFETY_MARGIN:
        rng = np.random.default_rng(42)
        # Try a few different noise scales
        scales = [0.01, 0.05, 0.1]
        for scale in scales:
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break
            # Try a few random perturbations for each scale
            for _ in range(3):
                if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                    break
                noise = rng.normal(0, scale, size=base.shape)
                pos = base + noise
                try:
                    relaxed = _relax(pos, maxiter=5000)
                    if _is_valid(relaxed):
                        e = _lj_energy_and_grad(relaxed)[0]
                        if e < best_energy:
                            best_energy = e
                            best_pos = relaxed.copy()
                except Exception:
                    pass

    return best_pos

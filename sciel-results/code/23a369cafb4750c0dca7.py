"""
Program for Lennard-Jones 55-atom cluster global minimum search.

Strategy:
1. Construct the Mackay icosahedron geometry with variable shell radii.
2. Optimize the three shell radii (r1, r2, r3) using L-BFGS-B to minimize 
   the energy of the rigid radial structure. This is a low-dimensional 
   optimization problem (3 variables) which is very fast and robust.
3. After finding optimal radii, perform a full 3D relaxation of the 55 
   coordinates to converge to the precise global minimum.
4. Use multiple starting points for the radii to ensure global convergence.
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

def _mackay_icosahedron_55(r1, r2, r3):
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

def _lj_energy(pos):
    """Compute total Lennard-Jones energy."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    mask = r2 > 1e-10
    r2_safe = np.where(mask, r2, 1.0)
    r6 = r2_safe ** (-3)
    r12 = r6 ** 2
    energy = 4 * np.sum(np.where(mask, r12 - r6, 0))
    return energy

def _lj_energy_and_grad(pos):
    """Compute total Lennard-Jones energy and gradient."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    mask = r2 > 1e-10
    r2_safe = np.where(mask, r2, 1.0)
    r6 = r2_safe ** (-3)
    r12 = r6 ** 2
    energy = 4 * np.sum(np.where(mask, r12 - r6, 0))
    
    # Gradient: dE/dx_i = sum_j 4*(-12*r^-14 + 6*r^-8) * (x_i - x_j)/r
    # Simplified: grad = 4 * sum_j (-12*r2^-7 + 6*r2^-4) * diff
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

def _relax_full(pos, maxiter=10000, ftol=1e-12, gtol=1e-10):
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

def _optimize_radii(r0, maxiter=200, ftol=1e-12, gtol=1e-8):
    """
    Optimize the three shell radii for the Mackay icosahedron.
    """
    def objective(params):
        r1, r2, r3 = params
        if r1 < 0.5 or r2 < 0.5 or r3 < 0.5:
            return 1e10
        pos = _mackay_icosahedron_55(r1, r2, r3)
        return _lj_energy(pos)
    
    # Use BFGS for the 3-parameter optimization
    result = minimize(
        objective,
        r0,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
    )
    
    if result.success:
        return result.x
    else:
        return r0

def construct():
    """
    Construct the 55-atom LJ cluster by optimizing the Mackay icosahedron 
    shell radii and then relaxing the full structure.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Ideal Mackay icosahedron radii are approximately:
    # r1 ~ 1.135, r2 ~ 1.162, r3 ~ 1.283
    # We will try several starting points around these values.
    
    initial_radii_list = [
        [1.135, 1.162, 1.283],
        [1.10, 1.15, 1.30],
        [1.15, 1.17, 1.27],
        [1.12, 1.16, 1.29],
        [1.14, 1.16, 1.28],
        [1.0, 1.2, 1.4],  # Wider range
        [1.2, 1.1, 1.3],  # Swapped
    ]
    
    best_energy = float('inf')
    best_pos = None
    
    # 1. Optimize radii for each starting point
    optimized_radii = []
    for r0 in initial_radii_list:
        if time.time() > deadline:
            break
        opt_r = _optimize_radii(r0)
        # Ensure radii are positive
        opt_r = np.maximum(opt_r, 0.5)
        optimized_radii.append(opt_r)
        
    # 2. For each optimized radius set, construct and fully relax
    for r_opt in optimized_radii:
        if time.time() > deadline:
            break
            
        pos = _mackay_icosahedron_55(r_opt[0], r_opt[1], r_opt[2])
        
        # Check validity before relaxation
        if not _is_valid(pos):
            continue
            
        # Full 3D relaxation
        relaxed = _relax_full(pos, maxiter=5000, ftol=1e-12, gtol=1e-10)
        
        if not _is_valid(relaxed):
            continue
            
        energy = _lj_energy(relaxed)
        
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed.copy()
            
    # 3. Final refinement if we have a good candidate
    if best_pos is not None and time.time() < deadline - 5.0:
        refined = _relax_full(best_pos, maxiter=10000, ftol=1e-14, gtol=1e-12)
        if _is_valid(refined):
            e = _lj_energy(refined)
            if e < best_energy:
                best_energy = e
                best_pos = refined

    # Fallback: return the best known Mackay icosahedron if relaxation failed
    if best_pos is None:
        pos = _mackay_icosahedron_55(1.135, 1.162, 1.283)
        return pos
        
    return best_pos

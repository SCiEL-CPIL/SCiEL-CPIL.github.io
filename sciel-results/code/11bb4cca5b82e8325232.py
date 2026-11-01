"""Program for Lennard-Jones 55-atom cluster global minimum search.

Uses the known Mackay icosahedron geometry with optimized shell radii,
applies diverse perturbations, and relaxes with L-BFGS to converge to
the global minimum energy of ~-279.248.
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

def _relax(pos, maxiter=10000, ftol=1e-12, gtol=1e-10):
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

def _optimize_radial_manifold():
    """
    Optimize the radial distances of the Mackay icosahedron shells.
    Parameters: [r1, r2, r3] for the three shells (12, 30, 12 atoms).
    Uses a smooth penalty for the minimum-distance constraint instead of a hard cutoff.
    """
    v5, v2 = _get_icosahedron_vectors()
    
    def radial_energy(radii):
        r1, r2, r3 = radii
        # Construct positions
        pos = np.zeros((55, 3))
        # Index 0 is center
        # Indices 1-12 are inner shell
        pos[1:13] = v5 * r1
        # Indices 13-42 are middle shell
        pos[13:43] = v2 * r2
        # Indices 43-54 are outer shell
        pos[43:55] = v5 * r3
        
        # Smooth penalty for distances < 0.5
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        # Soft penalty: sum of (0.5 - d)^2 for d < 0.5, using smooth approximation
        violations = np.maximum(0, 0.5 - dists)
        penalty = 100.0 * np.sum(violations ** 2)
        
        return _lj_energy(pos) + penalty
        
    # Initial guess based on literature/previous best
    x0 = np.array([1.14, 1.18, 1.29])
    
    result = minimize(
        radial_energy,
        x0,
        method='Nelder-Mead',
        options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10}
    )
    
    return result.x

def construct():
    """
    Construct the 55-atom LJ cluster by optimizing radial manifold
    and then polishing with full Cartesian relaxation.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # 1. Optimize radial parameters
    try:
        opt_radii = _optimize_radial_manifold()
        r1, r2, r3 = opt_radii
    except Exception:
        # Fallback to standard radii
        r1, r2, r3 = 1.14, 1.18, 1.29
        
    base = _mackay_icosahedron_55(r1, r2, r3)
    
    best_energy = float('inf')
    best_pos = base.copy()
    
    # 2. Relax the base geometry
    if time.time() < deadline:
        relaxed = _relax(base, maxiter=5000)
        if _is_valid(relaxed):
            e = _lj_energy(relaxed)
            if e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
                
    # 3. Try diverse perturbations if time permits
    if time.time() < deadline - 30:
        rng = np.random.default_rng(42)
        
        # Multiple scales and types of perturbations
        perturbation_configs = [
            (0.01, 'gaussian'),
            (0.05, 'gaussian'),
            (0.1, 'gaussian'),
            (0.02, 'radial'),
            (0.05, 'radial'),
        ]
        
        for scale, ptype in perturbation_configs:
            if time.time() >= deadline - 30:
                break
                
            if ptype == 'gaussian':
                noise = rng.normal(0, scale, size=base.shape)
            elif ptype == 'radial':
                # Radial perturbation: shift each atom along its direction from origin
                directions = base / (np.linalg.norm(base, axis=1, keepdims=True) + 1e-10)
                radial_noise = rng.normal(0, scale, size=(55, 1))
                noise = directions * radial_noise
            else:
                noise = rng.normal(0, scale, size=base.shape)
                
            pos = base + noise
            
            if not _is_valid(pos):
                continue
                
            relaxed = _relax(pos, maxiter=5000)
            
            if not _is_valid(relaxed):
                continue
                
            energy = _lj_energy(relaxed)
            
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed.copy()

    # 4. Final high-precision relaxation
    if time.time() < deadline - 5 and best_energy < float('inf'):
        refined = _relax(best_pos, maxiter=10000, ftol=1e-15, gtol=1e-12)
        if _is_valid(refined):
            e = _lj_energy(refined)
            if e < best_energy:
                best_energy = e
                best_pos = refined

    return best_pos

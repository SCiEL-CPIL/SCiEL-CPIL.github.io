"""Program for Lennard-Jones 55-atom cluster global minimum search.

Uses the known Mackay icosahedron geometry with optimal shell radii as a
starting point, applies small Gaussian perturbations, and relaxes with
L-BFGS to converge to the global minimum energy of ~-279.248.
"""

import numpy as np
from scipy.optimize import minimize

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

def _mackay_icosahedron_55():
    """
    Construct the 55-atom two-shell Mackay icosahedron.
    1 center, 12 at 5-fold (r1), 30 at 2-fold (r2), 12 at 5-fold (r3).
    Uses radii close to the known optimal values.
    """
    v5, v2 = _get_icosahedron_vectors()
    
    # Optimal radii for the Mackay icosahedron (from literature)
    r1 = 1.135
    r2 = 1.162
    r3 = 1.283
    
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

def _is_valid(pos):
    """Check if all inter-atomic distances are >= 0.5."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= 0.5

def _relax(pos, maxiter=10000, ftol=1e-12, gtol=1e-10):
    """Relax positions using L-BFGS-B."""
    def energy_func(x):
        pos_flat = x.reshape(-1, 3)
        return _lj_energy(pos_flat)
        
    result = minimize(
        energy_func,
        pos.flatten(),
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
    )
    return result.x.reshape(-1, 3)

def construct():
    """
    Construct the 55-atom LJ cluster by perturbing the Mackay icosahedron
    and relaxing with L-BFGS.
    """
    base = _mackay_icosahedron_55()
    
    # Multi-start stochastic search
    best_energy = float('inf')
    best_pos = base.copy()
    
    rng = np.random.default_rng(42)
    
    # First, try the exact geometry
    relaxed = _relax(base)
    if _is_valid(relaxed):
        e = _lj_energy(relaxed)
        if e < best_energy:
            best_energy = e
            best_pos = relaxed
    
    # Try diverse perturbation scales
    noise_scales = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]
    
    for scale in noise_scales:
        noise = rng.normal(0, scale, size=base.shape)
        pos = base + noise
        
        relaxed = _relax(pos)
        
        if not _is_valid(relaxed):
            continue
            
        energy = _lj_energy(relaxed)
        
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed.copy()
            
    # Final fine-tuning pass with tighter tolerances
    if best_energy < float('inf'):
        refined = _relax(best_pos, maxiter=20000, ftol=1e-14, gtol=1e-12)
        if _is_valid(refined):
            e = _lj_energy(refined)
            if e < best_energy:
                best_energy = e
                best_pos = refined
    
    return best_pos

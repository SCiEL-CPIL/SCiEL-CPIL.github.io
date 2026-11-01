"""Lennard-Jones 55-atom cluster global minimum search.

Uses the Mackay icosahedron geometry with optimized shell radii,
applies diverse perturbations, and relaxes with L-BFGS to converge
to the global minimum energy of ~-279.248.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist
import time

N = 55
_DEADLINE = 145.0  # Safety margin before 150s kill
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _DEADLINE - _elapsed()


def _get_icosahedron_vectors():
    """
    Returns 12 unit vectors along 5-fold axes (icosahedron vertices)
    and 30 unit vectors along 2-fold axes (edge midpoints).
    """
    phi = (1 + np.sqrt(5)) / 2
    
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    D = cdist(verts, verts)
    np.fill_diagonal(D, np.inf)
    min_dist = D.min()
    
    edge_mids = []
    for i in range(12):
        for j in range(i + 1, 12):
            if np.isclose(D[i, j], min_dist, atol=1e-10):
                mid = (verts[i] + verts[j]) / 2
                mid = mid / np.linalg.norm(mid)
                edge_mids.append(mid)
    
    edge_mids = np.array(edge_mids)
    return verts, edge_mids


def _mackay_icosahedron_55(r1=1.125, r2=1.155, r3=1.275):
    """
    Construct the 55-atom two-shell Mackay icosahedron.
    1 center, 12 at 5-fold (r1), 30 at 2-fold (r2), 12 at 5-fold (r3).
    """
    v5, v2 = _get_icosahedron_vectors()
    
    points = [np.zeros(3)]
    
    for v in v5:
        points.append(v * r1)
        
    for v in v2:
        points.append(v * r2)
        
    for v in v5:
        points.append(v * r3)
        
    return np.array(points, dtype=float)


def _lj_energy(pos):
    """Compute total Lennard-Jones energy using pdist for speed."""
    d = pdist(pos)
    d = np.maximum(d, 1e-10)
    r6 = d ** -6
    return 4.0 * np.sum(r6 ** 2 - r6)


def _lj_grad(pos):
    """Compute gradient of LJ energy using vectorized operations."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    mask = r2 > 1e-12
    r2_safe = np.where(mask, r2, 1.0)
    
    r6 = r2_safe ** -3
    r12 = r6 ** 2
    coeff = 4.0 * (-12.0 * r12 + 6.0 * r6) / r2_safe
    coeff = np.where(mask, coeff, 0.0)
    
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return grad


def _lj_energy_and_grad(pos):
    """Combined energy and gradient for L-BFGS-B."""
    return _lj_energy(pos), _lj_grad(pos)


def _is_valid(pos, min_d=0.5):
    """Check if all inter-atomic distances are >= min_d."""
    d = pdist(pos)
    return d.min() >= min_d - 1e-9


def _relax(pos, maxiter=10000, ftol=1e-12, gtol=1e-10):
    """Relax positions using L-BFGS-B with analytical gradient."""
    if _remaining() < 5.0:
        return pos
    
    def energy_func(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(pos_flat)
        return e, g.flatten()
        
    result = minimize(
        energy_func,
        pos.flatten(),
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxls': 20}
    )
    return result.x.reshape(-1, 3)


def _optimize_radii(radii, maxiter=5000):
    """Optimize the three shell radii by relaxing the full geometry."""
    def objective(r):
        if _remaining() < 10.0:
            return 1e10
        pos = _mackay_icosahedron_55(r[0], r[1], r[2])
        relaxed = _relax(pos, maxiter=2000, ftol=1e-10, gtol=1e-8)
        if not _is_valid(relaxed):
            return 1e10
        return _lj_energy(relaxed)
    
    result = minimize(objective, radii, method='Nelder-Mead', 
                      options={'maxiter': 100, 'xatol': 1e-3, 'fatol': 1e-6})
    return result.x


def construct():
    """
    Construct the 55-atom LJ cluster by perturbing the Mackay icosahedron
    and relaxing with L-BFGS.
    """
    global _start_time
    _start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_energy = float('inf')
    best_pos = None
    
    # Step 1: Optimize the shell radii (quick pass)
    initial_radii = np.array([1.125, 1.155, 1.275])
    
    if _remaining() > 30.0:
        optimal_radii = _optimize_radii(initial_radii)
    else:
        optimal_radii = initial_radii
    
    # Construct base geometry with optimized radii
    base = _mackay_icosahedron_55(*optimal_radii)
    
    # First, try the exact optimized geometry
    if _remaining() > 15.0:
        relaxed = _relax(base)
        if _is_valid(relaxed):
            e = _lj_energy(relaxed)
            if e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
    
    # Check time budget
    if _remaining() < 30.0:
        if best_pos is not None:
            return best_pos
        return base
    
    # Try diverse perturbation strategies
    # Strategy 1: Gaussian noise at various scales (focused on small perturbations)
    noise_scales = [0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08]
    for scale in noise_scales:
        if _remaining() < 40.0:
            break
        
        noise = rng.normal(0, scale, size=base.shape)
        pos = base + noise
        
        relaxed = _relax(pos)
        
        if not _is_valid(relaxed):
            continue
            
        energy = _lj_energy(relaxed)
        
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed.copy()
    
    # Strategy 2: Radial perturbations (scale each shell differently)
    for trial in range(15):
        if _remaining() < 50.0:
            break
        
        r1_pert = optimal_radii[0] * (1 + rng.normal(0, 0.015))
        r2_pert = optimal_radii[1] * (1 + rng.normal(0, 0.015))
        r3_pert = optimal_radii[2] * (1 + rng.normal(0, 0.015))
        pos = _mackay_icosahedron_55(r1_pert, r2_pert, r3_pert)
        
        # Add small Gaussian noise
        noise = rng.normal(0, 0.008, size=pos.shape)
        pos = pos + noise
        
        relaxed = _relax(pos)
        
        if not _is_valid(relaxed):
            continue
            
        energy = _lj_energy(relaxed)
        
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed.copy()
    
    # Strategy 3: Angular perturbations - rotate shells slightly
    for trial in range(10):
        if _remaining() < 60.0:
            break
        
        # Random small rotations
        theta = rng.normal(0, 0.02)
        axis = rng.normal(size=3)
        axis = axis / np.linalg.norm(axis)
        
        # Rodrigues rotation
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        
        pos = base @ R.T
        noise = rng.normal(0, 0.005, size=pos.shape)
        pos = pos + noise
        
        relaxed = _relax(pos)
        
        if not _is_valid(relaxed):
            continue
            
        energy = _lj_energy(relaxed)
        
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed.copy()
    
    # Final fine-tuning pass with tighter tolerances
    if best_pos is not None and _remaining() > 10.0:
        refined = _relax(best_pos, maxiter=20000, ftol=1e-14, gtol=1e-12)
        if _is_valid(refined):
            e = _lj_energy(refined)
            if e < best_energy:
                best_energy = e
                best_pos = refined
    
    if best_pos is None:
        best_pos = base
    
    return best_pos

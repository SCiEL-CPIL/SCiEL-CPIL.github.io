"""Program for Lennard-Jones 55-atom cluster global minimum search.

Uses the known Mackay icosahedron geometry, applies diverse perturbations,
and relaxes with L-BFGS to converge to the global minimum energy of ~-279.248.
Optimized for speed and reliability within the 120s time limit.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 55
START_TIME = time.time()
TIME_LIMIT = 110.0  # Leave 10s buffer

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

def _relax(pos, maxiter=5000, ftol=1e-12, gtol=1e-10):
    """Relax positions using L-BFGS-B with analytical gradient."""
    def energy_func(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(pos_flat)
        return e, g.flatten()
        
    # Check time limit
    if time.time() - START_TIME > TIME_LIMIT:
        return pos
    
    result = minimize(
        energy_func,
        pos.flatten(),
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
    )
    
    if result.success or result.status in [0, 1, 2, 3]:
        return result.x.reshape(-1, 3)
    else:
        # If optimization fails or is slow, return the best point found so far
        # L-BFGS-B usually returns the last iterate even if not fully converged
        return result.x.reshape(-1, 3)

def construct():
    """
    Construct the 55-atom LJ cluster by perturbing the Mackay icosahedron
    and relaxing with L-BFGS.
    """
    rng = np.random.default_rng(42)
    
    # Base geometry: Standard Mackay Icosahedron
    # Radii are chosen to be close to the known minimum
    base = _mackay_icosahedron_55(1.135, 1.162, 1.283)
    
    best_energy = float('inf')
    best_pos = base.copy()
    
    # List of perturbations to try
    # We prioritize the unperturbed case first, then small perturbations
    perturbations = []
    
    # 1. Unperturbed
    perturbations.append(None)
    
    # 2. Small Gaussian noise
    for scale in [0.001, 0.005, 0.01, 0.02, 0.05]:
        perturbations.append(rng.normal(0, scale, size=base.shape))
        
    # 3. Radial perturbations
    for trial in range(10):
        r1_pert = 1.135 * (1 + rng.normal(0, 0.01))
        r2_pert = 1.162 * (1 + rng.normal(0, 0.01))
        r3_pert = 1.283 * (1 + rng.normal(0, 0.01))
        pos_pert = _mackay_icosahedron_55(r1_pert, r2_pert, r3_pert) - base
        perturbations.append(pos_pert)
        
    # 4. Angular perturbations (small rotations of shells)
    for trial in range(10):
        pos_pert = np.zeros_like(base)
        # Rotate inner shell (atoms 1-12)
        angle1 = rng.normal(0, 0.01)
        axis1 = rng.normal(size=3)
        axis1 = axis1 / (np.linalg.norm(axis1) + 1e-10)
        c1, s1 = np.cos(angle1), np.sin(angle1)
        K1 = np.array([[0, -axis1[2], axis1[1]],
                       [axis1[2], 0, -axis1[0]],
                       [-axis1[1], axis1[0], 0]])
        R1 = c1 * np.eye(3) + s1 * K1 + (1 - c1) * np.outer(axis1, axis1)
        pos_pert[1:13] = R1 @ base[1:13].T - base[1:13].T
        
        # Rotate middle shell (atoms 13-42)
        angle2 = rng.normal(0, 0.01)
        axis2 = rng.normal(size=3)
        axis2 = axis2 / (np.linalg.norm(axis2) + 1e-10)
        c2, s2 = np.cos(angle2), np.sin(angle2)
        K2 = np.array([[0, -axis2[2], axis2[1]],
                       [axis2[2], 0, -axis2[0]],
                       [-axis2[1], axis2[0], 0]])
        R2 = c2 * np.eye(3) + s2 * K2 + (1 - c2) * np.outer(axis2, axis2)
        pos_pert[13:43] = R2 @ base[13:43].T - base[13:43].T
        
        # Rotate outer shell (atoms 43-54)
        angle3 = rng.normal(0, 0.01)
        axis3 = rng.normal(size=3)
        axis3 = axis3 / (np.linalg.norm(axis3) + 1e-10)
        c3, s3 = np.cos(angle3), np.sin(angle3)
        K3 = np.array([[0, -axis3[2], axis3[1]],
                       [axis3[2], 0, -axis3[0]],
                       [-axis3[1], axis3[0], 0]])
        R3 = c3 * np.eye(3) + s3 * K3 + (1 - c3) * np.outer(axis3, axis3)
        pos_pert[43:55] = R3 @ base[43:55].T - base[43:55].T
        
        perturbations.append(pos_pert)

    # Sort perturbations by magnitude (smallest first) to likely find good minima early
    # None is treated as 0 magnitude
    def get_magnitude(p):
        if p is None:
            return 0
        return np.linalg.norm(p)
        
    perturbations.sort(key=get_magnitude)
    
    # Try each perturbation
    for pert in perturbations:
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
            
        if pert is None:
            pos = base
        else:
            pos = base + pert
        
        # Relax
        relaxed = _relax(pos, maxiter=5000, ftol=1e-12, gtol=1e-10)
        
        if not _is_valid(relaxed):
            continue
            
        energy = _lj_energy(relaxed)
        
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed.copy()
            
            # Early termination if we hit the known global minimum (within floating point error)
            if best_energy < -279.248470 + 1e-6:
                break

    # Final fine-tuning pass with tighter tolerances if time permits
    if best_energy < float('inf') and time.time() - START_TIME < TIME_LIMIT - 5:
        refined = _relax(best_pos, maxiter=20000, ftol=1e-14, gtol=1e-12)
        if _is_valid(refined):
            e = _lj_energy(refined)
            if e < best_energy:
                best_energy = e
                best_pos = refined
    
    return best_pos

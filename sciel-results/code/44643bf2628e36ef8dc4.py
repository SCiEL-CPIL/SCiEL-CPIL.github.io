"""Program for Lennard-Jones 55-atom cluster global minimum search.

Uses the known Mackay icosahedron geometry with optimized shell radii,
applies diverse perturbations, and relaxes with L-BFGS to converge to
the global minimum energy of ~-279.248470.
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
    
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
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
    for v in v5:
        points.append(v * r1)
    for v in v2:
        points.append(v * r2)
    for v in v5:
        points.append(v * r3)
        
    return np.array(points, dtype=float)

def _lj_energy_and_grad(pos):
    """Compute total Lennard-Jones energy and gradient."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    mask = r2 > 1e-10
    r2_safe = np.where(mask, r2, 1.0)
    
    r6 = r2_safe ** (-3)
    r12 = r6 ** 2
    
    energy = 2.0 * np.sum(np.where(mask, r12 - r6, 0))
    
    r2_inv4 = r2_safe ** (-4)
    r2_inv7 = r2_safe ** (-7)
    coeff = 4.0 * (-12.0 * r2_inv7 + 6.0 * r2_inv4)
    coeff = np.where(mask, coeff, 0)
    
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return energy, grad

def _is_valid(pos):
    """Check if all inter-atomic distances are >= 0.5."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= 0.5

def _relax(pos, maxiter=5000, ftol=1e-12, gtol=1e-10, deadline=None):
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

def _optimize_radii(initial_radii, deadline):
    """
    Optimize the three shell radii using Nelder-Mead on the energy
    of the fully-relaxed Mackay geometry at given radii.
    """
    def objective(r):
        if time.time() > deadline - 10.0:
            return 1e10
        pos = _mackay_icosahedron_55(r[0], r[1], r[2])
        relaxed = _relax(pos, maxiter=3000, ftol=1e-10, gtol=1e-8)
        if not _is_valid(relaxed):
            return 1e10
        e, _ = _lj_energy_and_grad(relaxed)
        return e
    
    try:
        result = minimize(
            objective,
            initial_radii,
            method='Nelder-Mead',
            options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-8, 'adaptive': True}
        )
        return result.x
    except Exception:
        return initial_radii

def _polish(pos, deadline):
    """High-precision polish with multiple stages of increasing strictness."""
    def energy_func(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(pos_flat)
        return e, g.flatten()
    
    current_pos = pos
    best_energy = float('inf')
    
    stages = [
        {'maxiter': 5000, 'ftol': 1e-13, 'gtol': 1e-11},
        {'maxiter': 8000, 'ftol': 1e-14, 'gtol': 1e-12},
        {'maxiter': 10000, 'ftol': 1e-15, 'gtol': 1e-13},
    ]
    
    for opts in stages:
        if time.time() > deadline - 3.0:
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
    Construct the 55-atom LJ cluster by optimizing the Mackay icosahedron
    shell radii, applying diverse perturbations, and relaxing with L-BFGS.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    rng = np.random.default_rng(42)
    
    # Step 1: Optimize shell radii
    initial_radii = np.array([1.135, 1.162, 1.283])
    optimal_radii = _optimize_radii(initial_radii, deadline)
    
    # Construct base geometry with optimized radii
    base = _mackay_icosahedron_55(*optimal_radii)
    
    best_energy = float('inf')
    best_pos = base.copy()
    
    # Step 2: Relax the exact optimized geometry
    relaxed = _relax(base, maxiter=5000, ftol=1e-12, gtol=1e-10)
    if _is_valid(relaxed):
        e, _ = _lj_energy_and_grad(relaxed)
        if e < best_energy:
            best_energy = e
            best_pos = relaxed.copy()
    
    # Step 3: Multi-start with diverse perturbations
    if time.time() < deadline - 30.0:
        # Strategy A: Gaussian noise at various scales
        noise_scales = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]
        for scale in noise_scales:
            if time.time() > deadline - 25.0:
                break
            noise = rng.normal(0, scale, size=base.shape)
            pos = base + noise
            relaxed = _relax(pos, maxiter=5000, ftol=1e-12, gtol=1e-10)
            if _is_valid(relaxed):
                e, _ = _lj_energy_and_grad(relaxed)
                if e < best_energy:
                    best_energy = e
                    best_pos = relaxed.copy()
        
        # Strategy B: Radial perturbations (scale each shell differently)
        for trial in range(20):
            if time.time() > deadline - 25.0:
                break
            r1_pert = optimal_radii[0] * (1 + rng.normal(0, 0.015))
            r2_pert = optimal_radii[1] * (1 + rng.normal(0, 0.015))
            r3_pert = optimal_radii[2] * (1 + rng.normal(0, 0.015))
            pos = _mackay_icosahedron_55(r1_pert, r2_pert, r3_pert)
            noise = rng.normal(0, 0.005, size=pos.shape)
            pos = pos + noise
            relaxed = _relax(pos, maxiter=5000, ftol=1e-12, gtol=1e-10)
            if _is_valid(relaxed):
                e, _ = _lj_energy_and_grad(relaxed)
                if e < best_energy:
                    best_energy = e
                    best_pos = relaxed.copy()
        
        # Strategy C: Angular perturbations (small random rotations per atom)
        for trial in range(15):
            if time.time() > deadline - 25.0:
                break
            pos = base.copy()
            for i in range(1, len(pos)):
                angle = rng.normal(0, 0.01)
                axis = rng.normal(size=3)
                norm = np.linalg.norm(axis)
                if norm > 1e-10:
                    axis = axis / norm
                    c = np.cos(angle)
                    s = np.sin(angle)
                    K = np.array([[0, -axis[2], axis[1]],
                                  [axis[2], 0, -axis[0]],
                                  [-axis[1], axis[0], 0]])
                    R = c * np.eye(3) + s * K + (1 - c) * np.outer(axis, axis)
                    pos[i] = R @ pos[i]
            relaxed = _relax(pos, maxiter=5000, ftol=1e-12, gtol=1e-10)
            if _is_valid(relaxed):
                e, _ = _lj_energy_and_grad(relaxed)
                if e < best_energy:
                    best_energy = e
                    best_pos = relaxed.copy()
    
    # Step 4: Final high-precision polish
    if time.time() < deadline - 5.0:
        polished = _polish(best_pos, deadline)
        if _is_valid(polished):
            e, _ = _lj_energy_and_grad(polished)
            if e < best_energy:
                best_energy = e
                best_pos = polished
    
    return best_pos

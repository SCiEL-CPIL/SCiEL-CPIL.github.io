"""Program for Lennard-Jones 55-atom cluster global minimum search.

Uses the known Mackay icosahedron geometry with a topological graph-based
parameterization. Atoms are indexed by their connectivity in the Mackay
icosahedron. We optimize radial distances from center and inter-shell
couplings, then polish in full Cartesian space with L-BFGS.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 55
_DEADLINE = 140.0  # safety margin before 150s kill

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

def _build_mackay_topology():
    """
    Build the topological graph of the Mackay icosahedron.
    Returns:
        - directions: (55, 3) unit direction vectors for each atom
        - shell_ids: (55,) array with 0=center, 1=inner shell, 2=middle shell, 3=outer shell
        - neighbors: adjacency list for connectivity
    """
    v5, v2 = _get_icosahedron_vectors()
    
    # Atom indices:
    # 0: center
    # 1-12: inner shell (5-fold), 12 atoms
    # 13-42: middle shell (2-fold), 30 atoms
    # 43-54: outer shell (5-fold), 12 atoms
    
    directions = np.zeros((55, 3))
    shell_ids = np.zeros(55, dtype=int)
    
    directions[0] = 0  # center
    shell_ids[0] = 0
    
    for i in range(12):
        directions[1+i] = v5[i]
        shell_ids[1+i] = 1
    
    for i in range(30):
        directions[13+i] = v2[i]
        shell_ids[13+i] = 2
    
    for i in range(12):
        directions[43+i] = v5[i]
        shell_ids[43+i] = 3
    
    return directions, shell_ids

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
        
    points = np.array(points, dtype=float)
    return points

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
    
    r2_inv4 = r2_safe ** (-4)
    r2_inv7 = r2_safe ** (-7)
    coeff = 4 * (-12 * r2_inv7 + 6 * r2_inv4)
    coeff = np.where(mask, coeff, 0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return energy, grad

def _lj_energy_only(pos):
    """Compute total Lennard-Jones energy only."""
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

def _relax(pos, maxiter=5000, ftol=1e-12, gtol=1e-10):
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

def _optimize_radial_manifold(directions, shell_ids, t0=None, maxiter=2000, ftol=1e-14, gtol=1e-12):
    """
    Optimize over radial distances for each atom, keeping directions fixed.
    This is a 54-parameter optimization (excluding center).
    """
    if t0 is None:
        # Initial radii based on shell
        t0 = np.zeros(55)
        t0[0] = 0
        t0[1:13] = 1.135
        t0[13:43] = 1.162
        t0[43:55] = 1.283
    
    def radial_energy_and_grad(t):
        pos = directions * t[:, np.newaxis]
        e, g = _lj_energy_and_grad(pos)
        # Chain rule: dE/dt_i = sum_k g_ik * direction_ik
        grad_t = np.sum(g * directions, axis=1)
        return e, grad_t
    
    result = minimize(
        radial_energy_and_grad,
        t0,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
    )
    return result.x, result.x * directions[np.newaxis, :].T

def construct():
    """
    Construct the 55-atom LJ cluster using topological graph-based
    parameterization and multi-stage optimization.
    """
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    directions, shell_ids = _build_mackay_topology()
    
    # Standard Mackay radii
    base_radii = np.zeros(55)
    base_radii[1:13] = 1.135
    base_radii[13:43] = 1.162
    base_radii[43:55] = 1.283
    base_pos = directions * base_radii[:, np.newaxis]
    
    best_energy = float('inf')
    best_pos = base_pos.copy()
    
    # Stage 1: Radial manifold optimization from standard radii
    if time.time() - start_time < _DEADLINE - 30:
        radii_opt, pos_radial = _optimize_radial_manifold(directions, shell_ids, base_radii)
        if _is_valid(pos_radial):
            e = _lj_energy_only(pos_radial)
            if e < best_energy:
                best_energy = e
                best_pos = pos_radial.copy()
    
    # Stage 2: Full Cartesian relaxation from radial-optimized position
    if time.time() - start_time < _DEADLINE - 20:
        relaxed = _relax(best_pos, maxiter=5000, ftol=1e-13, gtol=1e-11)
        if _is_valid(relaxed):
            e = _lj_energy_only(relaxed)
            if e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
    
    # Stage 3: Multi-start with small perturbations on radial manifold
    if time.time() - start_time < _DEADLINE - 15:
        noise_scales = [0.005, 0.01, 0.02, 0.05]
        for scale in noise_scales:
            if time.time() - start_time > _DEADLINE - 10:
                break
            noise = rng.normal(0, scale, size=55)
            noise[0] = 0  # keep center fixed
            t_perturbed = base_radii + noise
            radii_pert, pos_pert = _optimize_radial_manifold(directions, shell_ids, t_perturbed)
            if _is_valid(pos_pert):
                # Quick Cartesian polish
                polished = _relax(pos_pert, maxiter=2000, ftol=1e-13, gtol=1e-11)
                if _is_valid(polished):
                    e = _lj_energy_only(polished)
                    if e < best_energy:
                        best_energy = e
                        best_pos = polished.copy()
    
    # Stage 4: Final high-precision polish
    if time.time() - start_time < _DEADLINE - 5:
        refined = _relax(best_pos, maxiter=10000, ftol=1e-15, gtol=1e-13)
        if _is_valid(refined):
            e = _lj_energy_only(refined)
            if e < best_energy:
                best_energy = e
                best_pos = refined
    
    # Ensure validity
    if not _is_valid(best_pos):
        # Fallback to base Mackay
        best_pos = base_pos
    
    return best_pos

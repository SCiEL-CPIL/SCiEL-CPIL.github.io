"""Lennard-Jones 55-atom global minimum: Mackay icosahedron seed + L-BFGS relaxation."""
import numpy as np
import time
import sys

N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _lj_energy(pos):
    """Compute total LJ energy for a set of positions (vectorized)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** 3
    e = 4.0 * (1.0 / (r6 * r6) - 1.0 / r6)
    return np.sum(e) / 2.0

def _lj_energy_grad(pos):
    """Compute total LJ energy and gradient (vectorized, numerically stable)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    
    r6 = r2 * r2 * r2
    r12 = r6 * r6
    
    e_pair = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(e_pair, 0.0)
    energy = np.sum(e_pair) / 2.0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = 4.0 * (-12.0 / (r2 ** 7) + 6.0 / (r2 ** 4))
    np.fill_diagonal(factor, 0.0)
    
    sum_factor = np.sum(factor, axis=1, keepdims=True)
    sum_factor_pos = np.sum(factor[:, :, np.newaxis] * pos[np.newaxis, :, :], axis=1)
    grad = pos * sum_factor - sum_factor_pos
    
    return energy, grad

def _get_icosahedron_data():
    """
    Returns icosahedron vertex unit vectors and edge list.
    12 vertices, 30 edges.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
    # Find edges: pairs of vertices at minimum distance
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            dists[i, j] = np.linalg.norm(verts_unit[i] - verts_unit[j])
    
    min_dist = np.min(dists[dists > 1e-6])
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                edges.append((i, j))
    
    return verts_unit, edges

def _get_55_directions(verts_unit, edges):
    """
    Return 55 unit direction vectors for the Mackay icosahedron:
    - index 0: origin (dummy, never used for radial optimization)
    - indices 1-12: icosahedron vertices (shell 1)
    - indices 13-42: edge midpoints normalized (shell 2)
    - indices 43-54: icosahedron vertices again (shell 3)
    """
    dirs = np.zeros((55, 3))
    dirs[0] = np.array([1.0, 0.0, 0.0])  # dummy for center
    
    # Shell 1: 12 vertices
    dirs[1:13] = verts_unit
    
    # Shell 2: 30 edge directions (normalized midpoints)
    for k, (i, j) in enumerate(edges):
        mid = verts_unit[i] + verts_unit[j]
        norm = np.linalg.norm(mid)
        if norm > 1e-10:
            dirs[13 + k] = mid / norm
        else:
            dirs[13 + k] = verts_unit[i]
    
    # Shell 3: 12 vertices (same as shell 1)
    dirs[43:55] = verts_unit
    
    return dirs

def _radial_to_positions(radii, dirs):
    """Convert 55 radial distances to Cartesian positions. Center atom at origin."""
    pos = np.zeros((55, 3))
    # Center atom stays at origin
    # All other atoms: pos[i] = radii[i] * dirs[i]
    for i in range(1, 55):
        pos[i] = radii[i] * dirs[i]
    return pos

def _radial_energy_grad(radii, dirs):
    """
    Compute LJ energy and gradient w.r.t. radial parameters.
    radii: array of shape (55,), radii[0] is ignored (center at origin).
    dirs: array of shape (55, 3) unit direction vectors.
    Returns (energy, grad_radii) where grad_radii has shape (55,).
    """
    pos = _radial_to_positions(radii, dirs)
    e, grad_pos = _lj_energy_grad(pos)
    
    # grad_radii[i] = dot(grad_pos[i], dirs[i]) for i >= 1
    grad_r = np.zeros(55)
    grad_r[1:] = np.sum(grad_pos[1:] * dirs[1:], axis=1)
    
    return e, grad_r

def _relax_radial(radii0, dirs, maxiter=3000, gtol=1e-12):
    """Relax in the 55D radial manifold using L-BFGS-B."""
    def obj_and_grad(r):
        e, g = _radial_energy_grad(r, dirs)
        return e, g
    
    bounds = [(0.0, 5.0)] + [(0.1, 5.0)] * 54
    
    try:
        result = minimize(
            obj_and_grad, radii0, jac=True, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 200, 'ftol': 1e-15}
        )
        return result.x, result.fun
    except Exception:
        return radii0, _radial_energy_grad(radii0, dirs)[0]

def _relax_lbfgs(pos, maxiter=5000, gtol=1e-12):
    """Relax using L-BFGS-B with tight tolerance in full 3D Cartesian space."""
    best_energy = _lj_energy(pos)
    best_pos = pos.copy()
    
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    x0 = pos.flatten()
    
    try:
        result = minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 200, 'ftol': 1e-15}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    return best_pos, best_energy

def construct():
    """Construct 55-atom LJ cluster using 55D radial manifold + full 3D relaxation."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (150s limit)
    
    # 1. Build icosahedron data and 55 directions
    verts_unit, edges = _get_icosahedron_data()
    dirs = _get_55_directions(verts_unit, edges)
    
    # 2. Initial radial guess: Mackay icosahedron shell radii
    # Shell 1 (12 atoms): ~1.0, Shell 2 (30 atoms): ~1.5, Shell 3 (12 atoms): ~2.0
    radii0 = np.zeros(55)
    radii0[0] = 0.0  # center
    radii0[1:13] = 1.0   # shell 1
    radii0[13:43] = 1.5  # shell 2
    radii0[43:55] = 2.0  # shell 3
    
    # 3. Optimize in 55D radial manifold
    radii_opt, e_radial = _relax_radial(radii0, dirs, maxiter=3000, gtol=1e-12)
    pos_seed = _radial_to_positions(radii_opt, dirs)
    
    best_pos = pos_seed.copy()
    best_energy = _lj_energy(pos_seed)
    
    # 4. Full 3D L-BFGS relaxation from radial manifold result
    if time.time() < deadline - 30.0:
        pos_relaxed, energy_relaxed = _relax_lbfgs(pos_seed, maxiter=5000, gtol=1e-12)
        if energy_relaxed < best_energy:
            best_energy = energy_relaxed
            best_pos = pos_relaxed
    
    # 5. Multi-start: perturb radial parameters and re-relax
    rng = np.random.default_rng(42)
    search_deadline = deadline - 15.0
    
    for trial in range(5):
        if time.time() >= search_deadline:
            break
        
        # Perturb radial parameters
        noise = rng.normal(0, 0.05, size=55)
        noise[0] = 0.0  # center stays at origin
        radii_pert = np.clip(radii_opt + noise, 0.1, 3.0)
        radii_pert[0] = 0.0
        
        radii_opt2, e2 = _relax_radial(radii_pert, dirs, maxiter=1500, gtol=1e-10)
        pos2 = _radial_to_positions(radii_opt2, dirs)
        
        # Full 3D relaxation
        if time.time() < search_deadline:
            pos_relaxed, energy_relaxed = _relax_lbfgs(pos2, maxiter=2000, gtol=1e-12)
            if energy_relaxed < best_energy:
                best_energy = energy_relaxed
                best_pos = pos_relaxed
    
    # 6. Final high-precision polish in 3D
    if time.time() < deadline - 5.0:
        pos_polished, energy_polished = _relax_lbfgs(best_pos, maxiter=10000, gtol=1e-14)
        if energy_polished < best_energy:
            best_energy = energy_polished
            best_pos = pos_polished
    
    # Final min-distance check
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist_found = np.min(dists)
    if min_dist_found < 0.5:
        best_pos = best_pos * (0.5 / min_dist_found * 1.01)
    
    return best_pos
# EVOLVE-BLOCK-END

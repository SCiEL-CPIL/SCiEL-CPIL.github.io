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
    np.fill_diagonal(r2, 1.0) # Avoid div by 0 in calculation, handled later
    
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

def _get_mackay_directions():
    """
    Returns unit vectors for the 55 positions in a Mackay icosahedron.
    1 center (0,0,0)
    12 first shell (icosahedron vertices)
    30 second shell (icosahedron edge midpoints)
    12 third shell (icosahedron vertices, same direction as first shell)
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
    # Find edges of the icosahedron
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
    
    # Edge midpoints (30 atoms)
    edge_mids = []
    for i, j in edges:
        mid = (verts_unit[i] + verts_unit[j]) / 2.0
        norm = np.linalg.norm(mid)
        if norm > 1e-6:
            edge_mids.append(mid / norm)
    edge_mids = np.array(edge_mids)
    
    # Construct full unit vectors
    # Order: Center, Shell1 (12), Shell2 (30), Shell3 (12)
    center = np.array([[0.0, 0.0, 0.0]])
    shell1_dir = verts_unit
    shell2_dir = edge_mids
    shell3_dir = verts_unit
    
    return center, shell1_dir, shell2_dir, shell3_dir

def _radial_to_positions(radii, dirs):
    """Convert 54 radial distances to 54 positions (plus center at origin)."""
    pos_noncenter = dirs * radii[:, np.newaxis]
    center = np.array([[0.0, 0.0, 0.0]])
    return np.vstack([center, pos_noncenter])

def _optimize_3d_radii(center, s1, s2, s3):
    """Optimize the 3 shell radii using L-BFGS-B."""
    def radii_energy_and_grad(radii):
        r1, r2, r3 = radii
        # Ensure radii are positive
        r1 = max(r1, 0.1)
        r2 = max(r2, 0.1)
        r3 = max(r3, 0.1)
        
        pos = np.vstack([center, s1 * r1, s2 * r2, s3 * r3])
        e, grad_pos = _lj_energy_grad(pos)
        
        # Gradient w.r.t radii
        # grad_r1 = sum(grad_pos[1:13] * s1, axis=1)
        # grad_r2 = sum(grad_pos[13:43] * s2, axis=1)
        # grad_r3 = sum(grad_pos[43:55] * s3, axis=1)
        
        grad_r = np.zeros(3)
        grad_r[0] = np.sum(grad_pos[1:13] * s1)
        grad_r[1] = np.sum(grad_pos[13:43] * s2)
        grad_r[2] = np.sum(grad_pos[43:55] * s3)
        
        return e, grad_r
    
    # Initial guess
    x0 = np.array([1.0, 1.5, 2.0])
    
    try:
        result = minimize(
            radii_energy_and_grad, x0, jac=True, method='L-BFGS-B',
            bounds=[(0.1, 5.0)] * 3,
            options={'maxiter': 1000, 'gtol': 1e-12, 'ftol': 1e-14}
        )
        return result.x
    except Exception:
        return x0

def _relax_lbfgs(pos, maxiter=5000, gtol=1e-12):
    """Relax using L-BFGS-B with tight tolerance."""
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
    """Construct 55-atom LJ cluster using 3D radial optimization + full relaxation."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Build direction manifold
    center, s1, s2, s3 = _get_mackay_directions()
    
    # 2. Optimize 3 shell radii
    radii_opt = _optimize_3d_radii(center, s1, s2, s3)
    pos_seed = np.vstack([center, s1 * radii_opt[0], s2 * radii_opt[1], s3 * radii_opt[2]])
    
    best_pos = pos_seed.copy()
    best_energy = _lj_energy(pos_seed)
    
    # 3. Full 3D L-BFGS relaxation from seed
    if time.time() < deadline - 40.0:
        pos_relaxed, energy_relaxed = _relax_lbfgs(pos_seed, maxiter=5000, gtol=1e-12)
        if energy_relaxed < best_energy:
            best_energy = energy_relaxed
            best_pos = pos_relaxed
    
    # 4. Multi-start with small perturbations
    rng = np.random.default_rng(42)
    search_deadline = deadline - 20.0
    
    for scale in [0.01, 0.05, 0.1]:
        if time.time() >= search_deadline - 5.0:
            break
        
        noise = rng.normal(0, scale, size=best_pos.shape)
        perturbed = best_pos + noise
        
        # Enforce min distance
        diff = perturbed[:, np.newaxis, :] - perturbed[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_d = np.min(dists)
        if min_d < 0.5:
            perturbed = perturbed * (0.5 / min_d * 1.01)
        
        pos_relaxed, energy_relaxed = _relax_lbfgs(perturbed, maxiter=2000, gtol=1e-12)
        if energy_relaxed < best_energy:
            best_energy = energy_relaxed
            best_pos = pos_relaxed
    
    # 5. Final high-precision polish
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

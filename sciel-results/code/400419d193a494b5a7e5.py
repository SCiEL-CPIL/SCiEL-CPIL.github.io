"""Lennard-Jones 55-atom global minimum: icosahedral manifold + full Cartesian relaxation."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55

# EVOLVE-BLOCK-START

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
        # dE/dr2 = 4*(-6/r2^7 + 3/r2^4)
        factor = 4.0 * (-6.0 / (r2 ** 7) + 3.0 / (r2 ** 4))
    np.fill_diagonal(factor, 0.0)
    
    # grad_i = sum_j factor_ij * (pos_i - pos_j)
    # = pos_i * sum_j factor_ij - sum_j factor_ij * pos_j
    sum_factor = np.sum(factor, axis=1, keepdims=True)
    sum_factor_pos = np.sum(factor[:, :, np.newaxis] * pos[np.newaxis, :, :], axis=1)
    grad = pos * sum_factor - sum_factor_pos
    
    return energy, grad

def _get_mackay_icosahedron_dirs():
    """
    Returns:
      center: shape (1,3) - origin
      shell1_dir: shape (12,3) - icosahedron vertex unit vectors
      shell2_dir: shape (30,3) - edge midpoint unit vectors
      shell3_dir: shape (12,3) - same as shell1
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
    # Find edges
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
    
    center = np.array([[0.0, 0.0, 0.0]])
    return center, verts_unit, edge_mids, verts_unit

def _build_full_dirs():
    """Build 54 unit direction vectors for non-center atoms."""
    center, s1, s2, s3 = _get_mackay_icosahedron_dirs()
    dirs = np.vstack([s1, s2, s3])
    return center, dirs

def _radial_to_positions(radii, dirs, center):
    """Convert 54 radial distances to 55 positions."""
    pos_noncenter = dirs * radii[:, np.newaxis]
    return np.vstack([center, pos_noncenter])

def _radial_energy_and_grad(radii, dirs, center):
    """Energy and gradient w.r.t. radial parameters on the direction manifold."""
    pos = _radial_to_positions(radii, dirs, center)
    e, grad_pos = _lj_energy_grad(pos)
    grad_rad = np.sum(grad_pos[1:] * dirs, axis=1)
    return e, grad_rad

def _relax_lbfgs(pos, maxiter=3000, gtol=1e-12):
    """Relax using L-BFGS-B."""
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
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100, 'ftol': 1e-14}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    return best_pos, best_energy

def _check_min_dist(pos, min_d=0.5):
    """Check and fix minimum distance constraint."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_d_found = np.min(dists)
    if min_d_found < min_d:
        scale = min_d / min_d_found * 1.01
        pos = pos * scale
    return pos

def construct():
    """
    Construct 55-atom LJ cluster.
    
    Strategy:
    1. Build Mackay icosahedron direction manifold (54 fixed directions + center).
    2. Optimize 54 individual radial distances via L-BFGS-B on the manifold.
    3. Full 165D Cartesian L-BFGS relaxation.
    4. Multi-start perturbations around best structure.
    5. Final high-precision polish.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Build direction manifold
    center, dirs = _build_full_dirs()
    
    # Step 1: 3-shell radius optimization for initial guess
    r1_init, r2_init, r3_init = 1.05, 1.75, 1.95
    try:
        if time.time() < deadline - 100.0:
            def radii_energy(radii):
                r1, r2, r3 = radii
                pos = np.vstack([center, dirs[:12] * r1, dirs[12:42] * r2, dirs[42:54] * r3])
                return _lj_energy(pos)
            result = minimize(radii_energy, [1.05, 1.75, 1.95], method='Nelder-Mead',
                              options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 1000})
            r1_init, r2_init, r3_init = result.x
    except Exception:
        pass
    
    # Step 2: Full 54D radial manifold optimization
    x0 = np.zeros(54)
    x0[:12] = r1_init
    x0[12:42] = r2_init
    x0[42:54] = r3_init
    
    try:
        if time.time() < deadline - 80.0:
            result = minimize(
                lambda x: _radial_energy_and_grad(x, dirs, center),
                x0, jac=True, method='L-BFGS-B',
                bounds=[(0.3, 5.0)] * 54,
                options={'maxiter': 2000, 'gtol': 1e-12, 'ftol': 1e-14, 'maxls': 200}
            )
            radii_opt = result.x
        else:
            radii_opt = x0
    except Exception:
        radii_opt = x0
    
    manifold_pos = _radial_to_positions(radii_opt, dirs, center)
    best_pos = manifold_pos.copy()
    best_energy = _lj_energy(manifold_pos)
    
    # Step 3: Full 3D L-BFGS relaxation from manifold result
    if time.time() < deadline - 20.0:
        pos_relaxed, energy_relaxed = _relax_lbfgs(manifold_pos, maxiter=3000, gtol=1e-12)
        if energy_relaxed < best_energy:
            best_energy = energy_relaxed
            best_pos = pos_relaxed
    
    # Step 4: Multi-start with perturbations
    rng = np.random.default_rng(42)
    search_deadline = deadline - 15.0
    
    for scale in [0.03, 0.05, 0.1, 0.15]:
        if time.time() >= search_deadline - 10.0:
            break
        noise = rng.normal(0, scale, size=best_pos.shape)
        perturbed = best_pos + noise
        perturbed = _check_min_dist(perturbed)
        
        pos_relaxed, energy_relaxed = _relax_lbfgs(perturbed, maxiter=1500, gtol=1e-12)
        if energy_relaxed < best_energy:
            best_energy = energy_relaxed
            best_pos = pos_relaxed
    
    # Step 5: Final high-precision polish
    if time.time() < deadline - 5.0:
        pos_polished, energy_polished = _relax_lbfgs(best_pos, maxiter=5000, gtol=1e-14)
        if energy_polished < best_energy:
            best_energy = energy_polished
            best_pos = pos_polished
    
    # Final min-distance check
    best_pos = _check_min_dist(best_pos)
    
    return best_pos

# EVOLVE-BLOCK-END

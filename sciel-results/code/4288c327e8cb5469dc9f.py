"""Lennard-Jones 55-atom global minimum: Mackay icosahedron + multi-stage optimization."""
import numpy as np
import time
import sys

N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time

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

def _get_mackay_icosahedron_coords():
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
    
    center = np.array([[0.0, 0.0, 0.0]])
    shell1_dir = verts_unit
    shell2_dir = edge_mids
    shell3_dir = verts_unit
    
    return center, shell1_dir, shell2_dir, shell3_dir

def _make_mackay_icosahedron(r1=1.05, r2=1.75, r3=1.95):
    """Construct a two-shell Mackay icosahedron for 55 atoms."""
    center, s1, s2, s3 = _get_mackay_icosahedron_coords()
    
    first_shell = s1 * r1
    second_inner = s2 * r2
    second_outer = s3 * r3
    
    all_pos = np.vstack([center, first_shell, second_inner, second_outer])
    return all_pos

def _min_distance_check(pos):
    """Check if minimum distance constraint is satisfied."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists)

def _ensure_min_distance(pos, min_d=0.5):
    """Ensure minimum distance constraint by scaling if needed."""
    md = _min_distance_check(pos)
    if md < min_d:
        scale = min_d / md * 1.01
        return pos * scale
    return pos

def _relax_lbfgs(pos, maxiter=5000, gtol=1e-12):
    """Relax using L-BFGS-B."""
    from scipy.optimize import minimize
    
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

def _optimize_radial(initial_radii, directions, maxiter=2000, gtol=1e-10):
    """Optimize over radial coordinates."""
    from scipy.optimize import minimize
    
    def _radial_energy_grad(radii):
        pos = directions * radii[:, np.newaxis]
        e, grad_pos = _lj_energy_grad(pos)
        grad_r = np.sum(grad_pos * directions, axis=1)
        return e, grad_r
    
    best_energy = _radial_energy_grad(initial_radii)[0]
    best_radii = initial_radii.copy()
    
    try:
        result = minimize(
            _radial_energy_grad, initial_radii, jac=True, method='L-BFGS-B',
            bounds=[(0.01, 10.0)] * len(initial_radii),
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100, 'ftol': 1e-14}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_radii = result.x
    except Exception:
        pass
    
    return best_radii, best_energy

def construct():
    """Construct 55-atom LJ cluster near global minimum."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # 1. Get icosahedral directions
    center, s1, s2, s3 = _get_mackay_icosahedron_coords()
    directions = np.vstack([center, s1, s2, s3])
    
    # 2. Initial radial coordinates - try multiple starting points
    initial_radii_sets = [
        np.concatenate([[0.0], [1.05] * 12, [1.75] * 30, [1.95] * 12]),
        np.concatenate([[0.0], [1.0] * 12, [1.7] * 30, [1.9] * 12]),
        np.concatenate([[0.0], [1.1] * 12, [1.8] * 30, [2.0] * 12]),
        np.concatenate([[0.0], [0.95] * 12, [1.65] * 30, [1.85] * 12]),
    ]
    
    best_pos = None
    best_energy = np.inf
    
    # 3. Radial manifold optimization for each initial set
    for radii0 in initial_radii_sets:
        if time.time() >= deadline - 60.0:
            break
        
        radii_opt, energy_radial = _optimize_radial(radii0, directions, maxiter=3000, gtol=1e-12)
        
        if energy_radial < best_energy:
            best_energy = energy_radial
            best_pos = directions * radii_opt[:, np.newaxis]
    
    # 4. Full Cartesian relaxation from best radial solution
    if best_pos is not None:
        if time.time() < deadline - 50.0:
            pos_optimized, energy_optimized = _relax_lbfgs(
                best_pos, maxiter=10000, gtol=1e-14
            )
            if energy_optimized < best_energy:
                best_energy = energy_optimized
                best_pos = pos_optimized
    
    # 5. Multi-start stochastic search with meaningful perturbations
    rng = np.random.default_rng(42)
    
    perturbation_scales = [0.15, 0.25, 0.4, 0.6, 1.0, 1.5]
    
    search_deadline = deadline - 35.0
    
    for scale in perturbation_scales:
        if time.time() >= search_deadline - 10.0:
            break
        
        n_perturbations = 3 if scale < 0.5 else 2
        
        for _ in range(n_perturbations):
            if time.time() >= search_deadline - 5.0:
                break
            
            noise = rng.normal(0, scale, size=best_pos.shape)
            perturbed_pos = best_pos + noise
            
            # Check and enforce minimum distance
            perturbed_pos = _ensure_min_distance(perturbed_pos)
            
            # Relax
            pos_relaxed, energy_relaxed = _relax_lbfgs(
                perturbed_pos, maxiter=5000, gtol=1e-12
            )
            
            if energy_relaxed < best_energy:
                best_energy = energy_relaxed
                best_pos = pos_relaxed
    
    # 6. Final high-precision polish with progressively tighter tolerance
    if time.time() < deadline - 15.0:
        pos_polished, energy_polished = _relax_lbfgs(
            best_pos, maxiter=20000, gtol=1e-15
        )
        if energy_polished < best_energy:
            best_energy = energy_polished
            best_pos = pos_polished
    
    # Second polish pass
    if time.time() < deadline - 8.0:
        pos_polished2, energy_polished2 = _relax_lbfgs(
            best_pos, maxiter=10000, gtol=1e-16
        )
        if energy_polished2 < best_energy:
            best_energy = energy_polished2
            best_pos = pos_polished2
    
    # Third polish pass if time permits
    if time.time() < deadline - 3.0:
        pos_polished3, energy_polished3 = _relax_lbfgs(
            best_pos, maxiter=5000, gtol=1e-17
        )
        if energy_polished3 < best_energy:
            best_energy = energy_polished3
            best_pos = pos_polished3
    
    # Final: ensure minimum distance constraint
    best_pos = _ensure_min_distance(best_pos)
    
    return best_pos
# EVOLVE-BLOCK-END

"""Lennard-Jones 55-atom cluster: optimized Mackay icosahedron radii + multi-start L-BFGS relaxation."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55
DEADLINE_SECONDS = 115.0

# EVOLVE-BLOCK-START

def _lj_energy(pos):
    """Compute total Lennard-Jones energy for N atoms (vectorized)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    e = 4.0 * (r12 - r6).sum() / 2.0
    return e

def _lj_energy_grad(pos):
    """Compute LJ energy and gradient."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    e = 4.0 * (r12 - r6).sum() / 2.0
    coeff = 4.0 * (-12.0 * r2 ** (-7) + 6.0 * r2 ** (-4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.einsum('ij,ijk->ik', coeff, diff)
    return e, grad

def _get_icosahedral_directions():
    """Get 12 vertex directions and 30 edge-midpoint directions of regular icosahedron."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # 12 vertices of regular icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit radius
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
    # Find edges: pairs of vertices at minimum distance
    edge_mask = np.zeros((12, 12), dtype=bool)
    min_dist = np.inf
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts_unit[i] - verts_unit[j])
            if d < min_dist:
                min_dist = d
            edge_mask[i, j] = d < min_dist * 1.001
            edge_mask[j, i] = edge_mask[i, j]
    
    # Get edge midpoints (30 edges)
    edge_midpoints = []
    for i in range(12):
        for j in range(i + 1, 12):
            if edge_mask[i, j]:
                mid = (verts_unit[i] + verts_unit[j]) / 2.0
                mid_unit = mid / np.linalg.norm(mid)
                edge_midpoints.append(mid_unit)
    
    edge_midpoints = np.array(edge_midpoints)  # shape (30, 3)
    
    return verts_unit, edge_midpoints

def _rotation_matrix_from_euler(rx, ry, rz):
    """Construct rotation matrix from ZYX Euler angles."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    
    return Rz @ Ry @ Rx

def _mackay_icosahedron_55_params(params):
    """
    Construct 55-atom Mackay icosahedron from 6 parameters:
    [r_inner, r_edge, r_vertex, rx, ry, rz]
    - 1 atom at center
    - 12 atoms at inner shell radius r_inner (vertex directions)
    - 30 atoms at edge-midpoint radius r_edge
    - 12 atoms at vertex radius r_vertex
    All directions rotated by R = Rz(ry) @ Ry(ry) @ Rx(rx)
    """
    r_inner, r_edge, r_vertex, rx, ry, rz = params
    
    if r_inner < 0.3 or r_edge < 0.5 or r_vertex < 0.5:
        return None
    
    verts_unit, edge_midpoints = _get_icosahedral_directions()
    
    R = _rotation_matrix_from_euler(rx, ry, rz)
    
    points = [np.zeros(3)]  # center atom
    
    # Inner shell: 12 vertices at radius r_inner
    for v in verts_unit:
        points.append(r_inner * (R @ v))
    
    # Outer shell: 30 edge-midpoint directions at radius r_edge
    for v in edge_midpoints:
        points.append(r_edge * (R @ v))
    
    # Outer shell: 12 vertex directions at radius r_vertex
    for v in verts_unit:
        points.append(r_vertex * (R @ v))
    
    return np.array(points, dtype=float)

def _optimize_manifold():
    """Optimize 6 manifold parameters (3 radii + 3 Euler angles) via Nelder-Mead."""
    def energy_from_params(params):
        try:
            pos = _mackay_icosahedron_55_params(params)
            if pos is None:
                return 1e10
            return _lj_energy(pos)
        except Exception:
            return 1e10
    
    # Initial guess: known Mackay geometry, identity rotation
    r0 = np.array([1.054, 1.755, 2.0, 0.0, 0.0, 0.0])
    try:
        result = minimize(energy_from_params, r0, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14,
                                   'initial_simplex': None})
        if result.fun < 1e9:
            return result.x, result.fun
    except Exception:
        pass
    
    # Fallback: try a few different starting rotations
    rng = np.random.RandomState(0)
    best_params = r0.copy()
    best_energy = energy_from_params(r0)
    for _ in range(10):
        trial = np.array([1.054 + rng.normal(0, 0.02),
                          1.755 + rng.normal(0, 0.03),
                          2.0 + rng.normal(0, 0.03),
                          rng.normal(0, 0.3),
                          rng.normal(0, 0.3),
                          rng.normal(0, 0.3)])
        try:
            result = minimize(energy_from_params, trial, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
            if result.fun < best_energy:
                best_energy = result.fun
                best_params = result.x
        except Exception:
            pass
    
    return best_params, best_energy

def _relax(pos0, maxiter=5000, gtol=1e-10, ftol=1e-14):
    """Apply L-BFGS relaxation to a starting position."""
    def fun(x):
        x = x.reshape(N, 3)
        e, g = _lj_energy_grad(x)
        return e, g.reshape(-1)
    
    try:
        result = minimize(fun, pos0.reshape(-1), method='L-BFGS-B', jac=True,
                         options={'maxiter': maxiter, 'gtol': gtol, 'ftol': ftol})
        return result.x.reshape(N, 3), result.fun
    except Exception:
        return pos0, _lj_energy(pos0)

def construct():
    start_time = time.time()
    
    # Step 1: Optimize on the 6D symmetry-constrained manifold
    try:
        params, manifold_energy = _optimize_manifold()
    except Exception:
        params = np.array([1.054, 1.755, 2.0, 0.0, 0.0, 0.0])
        pos_tmp = _mackay_icosahedron_55_params(params)
        manifold_energy = _lj_energy(pos_tmp)
    
    pos0 = _mackay_icosahedron_55_params(params)
    if pos0 is None:
        pos0 = _mackay_icosahedron_55_params(np.array([1.054, 1.755, 2.0, 0.0, 0.0, 0.0]))
    best_pos = pos0.copy()
    best_energy = _lj_energy(pos0)
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 40:
        return best_pos
    
    # Step 2: Primary L-BFGS relaxation from manifold optimum
    pos_relaxed, energy_relaxed = _relax(pos0, maxiter=5000, gtol=1e-10, ftol=1e-14)
    if energy_relaxed < best_energy:
        best_energy = energy_relaxed
        best_pos = pos_relaxed.copy()
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 20:
        return best_pos
    
    # Step 3: Multi-start refinement with small perturbations around manifold result
    rng = np.random.RandomState(42)
    scales = [0.02, 0.04, 0.06, 0.1, 0.15, 0.2]
    for scale in scales:
        if time.time() - start_time > DEADLINE_SECONDS - 10:
            break
        pos_perturbed = best_pos + rng.normal(0, scale, best_pos.shape)
        pos_new, energy_new = _relax(pos_perturbed, maxiter=3000, gtol=1e-9, ftol=1e-12)
        if energy_new < best_energy:
            best_energy = energy_new
            best_pos = pos_new.copy()
    
    # Step 4: Final high-precision polish with very tight tolerance
    elapsed = time.time() - start_time
    if elapsed < DEADLINE_SECONDS - 3:
        pos_final, energy_final = _relax(best_pos, maxiter=10000, gtol=1e-13, ftol=1e-15)
        if energy_final < best_energy:
            best_pos = pos_final
    
    return best_pos

# EVOLVE-BLOCK-END

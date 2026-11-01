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

def _mackay_icosahedron_55(r_inner, r_outer):
    """
    Construct 55-atom Mackay icosahedron.
    - 1 atom at center
    - 12 atoms at inner shell radius r_inner (vertices of regular icosahedron)
    - 30 atoms at outer shell radius r_outer (edge midpoints)
    - 12 atoms at outer shell radius r_outer * vertex_ratio (vertex directions)
    """
    verts_unit, edge_midpoints = _get_icosahedral_directions()
    
    # Vertex-to-edge-midpoint radius ratio for Mackay icosahedron
    # The second shell of a Mackay icosahedron has 30 edge-midpoint atoms and 12 vertex atoms
    # The vertex atoms are slightly further out than the edge-midpoint atoms
    vertex_ratio = 1.1397  # approx 2.0 / 1.755
    
    points = [np.zeros(3)]  # center atom
    
    # Inner shell: 12 vertices at radius r_inner
    for v in verts_unit:
        points.append(r_inner * v)
    
    # Outer shell: 30 edge-midpoint directions at radius r_outer
    for v in edge_midpoints:
        points.append(r_outer * v)
    
    # Outer shell: 12 vertex directions at radius r_outer * vertex_ratio
    for v in verts_unit:
        points.append(r_outer * vertex_ratio * v)
    
    return np.array(points, dtype=float)

def _optimize_radii():
    """Optimize the two shell radii to minimize LJ energy."""
    def energy_from_radii(params):
        r_inner, r_outer = params
        if r_inner < 0.3 or r_outer < 0.5 or r_outer < r_inner:
            return 1e10
        try:
            pos = _mackay_icosahedron_55(r_inner, r_outer)
            return _lj_energy(pos)
        except Exception:
            return 1e10
    
    # Initial guess based on known Mackay geometry
    r0 = [1.054, 1.755]
    try:
        result = minimize(energy_from_radii, r0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
        if result.fun < 1e9:
            return result.x, result.fun
    except Exception:
        pass
    return r0, _lj_energy(_mackay_icosahedron_55(r0[0], r0[1]))

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
    
    # Step 1: Optimize shell radii
    try:
        radii, radius_energy = _optimize_radii()
    except Exception:
        radii = [1.054, 1.755]
        pos_tmp = _mackay_icosahedron_55(1.054, 1.755)
        radius_energy = _lj_energy(pos_tmp)
    
    pos0 = _mackay_icosahedron_55(radii[0], radii[1])
    best_pos = pos0.copy()
    best_energy = _lj_energy(pos0)
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 30:
        return best_pos
    
    # Step 2: Primary L-BFGS relaxation
    pos_relaxed, energy_relaxed = _relax(pos0, maxiter=5000, gtol=1e-10, ftol=1e-14)
    if energy_relaxed < best_energy:
        best_energy = energy_relaxed
        best_pos = pos_relaxed.copy()
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 15:
        return best_pos
    
    # Step 3: Multi-start refinement with Gaussian perturbations
    rng = np.random.RandomState(42)
    remaining = DEADLINE_SECONDS - elapsed - 8
    if remaining > 5:
        scales = [0.03, 0.06, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
        for scale in scales:
            if time.time() - start_time > DEADLINE_SECONDS - 8:
                break
            pos_perturbed = best_pos + rng.normal(0, scale, best_pos.shape)
            pos_new, energy_new = _relax(pos_perturbed, maxiter=3000, gtol=1e-9, ftol=1e-12)
            if energy_new < best_energy:
                best_energy = energy_new
                best_pos = pos_new.copy()
    
    # Step 4: Final high-precision polish
    elapsed = time.time() - start_time
    if elapsed < DEADLINE_SECONDS - 3:
        pos_final, energy_final = _relax(best_pos, maxiter=10000, gtol=1e-12, ftol=1e-16)
        if energy_final < best_energy:
            best_pos = pos_final
    
    return best_pos

# EVOLVE-BLOCK-END

"""Lennard-Jones 55-atom global minimum: Symmetry-reduced Mackay Icosahedron optimization."""
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
    """Compute total LJ energy and gradient (vectorized)."""
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
        factor = 24.0 * (r6 - 2.0) / (r2 ** 7)
    np.fill_diagonal(factor, 0.0)
    
    sum_factor = np.sum(factor, axis=1, keepdims=True)
    sum_factor_pos = np.sum(factor[:, :, np.newaxis] * pos[np.newaxis, :, :], axis=1)
    grad = pos * sum_factor - sum_factor_pos
    
    return energy, grad

def _get_icosahedron_data():
    """
    Returns unit vectors for:
    1. 12 vertices
    2. 30 edge midpoints
    3. 12 face centers
    All normalized to unit length.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
    # Find edges (distance between nearest neighbors)
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
    
    # 30 Edge Midpoints
    edge_mids = []
    for i, j in edges:
        mid = (verts_unit[i] + verts_unit[j]) / 2.0
        norm = np.linalg.norm(mid)
        if norm > 1e-6:
            edge_mids.append(mid / norm)
    edge_mids = np.array(edge_mids)
    
    # 12 Face Centers
    # Faces are triangles of vertices. We can find them by looking for triples of vertices
    # that form an equilateral triangle with side length min_dist.
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                for k in range(j+1, 12):
                    if abs(dists[i, k] - min_dist) < 1e-6 and abs(dists[j, k] - min_dist) < 1e-6:
                        faces.append((i, j, k))
    
    face_centers = []
    for i, j, k in faces:
        center = (verts_unit[i] + verts_unit[j] + verts_unit[k]) / 3.0
        norm = np.linalg.norm(center)
        if norm > 1e-6:
            face_centers.append(center / norm)
    face_centers = np.array(face_centers)
    
    return verts_unit, edge_mids, face_centers

def _generate_mackay_positions(params):
    """
    Generate 55 positions from 3 parameters: [r_inner, r_outer, r_outer]
    Actually, the Mackay icosahedron has two shells.
    Shell 1: 1 center + 12 vertices (radius r1)
    Shell 2: 30 edge midpoints + 12 face centers (radius r2)
    Total 1 + 12 + 42 = 55.
    
    We optimize r1 and r2. The center is fixed at 0.
    """
    r1, r2 = params[0], params[1]
    if r1 < 0.1: r1 = 0.1
    if r2 < 0.5: r2 = 0.5
    
    verts_unit, edge_mids, face_centers = _get_icosahedron_data()
    
    center = np.array([[0.0, 0.0, 0.0]])
    shell1 = verts_unit * r1
    shell2_inner = edge_mids * r2
    shell2_outer = face_centers * r2
    
    # Concatenate: 1 + 12 + 30 + 12 = 55
    pos = np.vstack([center, shell1, shell2_inner, shell2_outer])
    return pos

def _energy_from_params(params):
    pos = _generate_mackay_positions(params)
    return _lj_energy(pos)

def _relax_lbfgs(pos, deadline=None, maxiter=500, gtol=1e-10):
    """Relax using L-BFGS-B with deadline guard."""
    start_time = time.time()
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
    
    if deadline is not None and time.time() > deadline - 5.0:
        return best_pos, best_energy
    
    return best_pos, best_energy

def _polish(pos, deadline=None):
    """High-precision polish pass."""
    start_time = time.time()
    best_energy = _lj_energy(pos)
    best_pos = pos.copy()
    
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    x0 = pos.flatten()
    
    # Multi-stage polish
    stages = [
        {'maxiter': 1000, 'gtol': 1e-12, 'maxls': 200, 'ftol': 1e-16},
        {'maxiter': 5000, 'gtol': 1e-14, 'maxls': 500, 'ftol': 1e-16},
        {'maxiter': 10000, 'gtol': 1e-16, 'maxls': 1000, 'ftol': 1e-18},
    ]
    
    for stage_opts in stages:
        if deadline is not None and time.time() > deadline - 5.0:
            break
        try:
            result = minimize(
                obj_and_grad, x0, jac=True, method='L-BFGS-B',
                options=stage_opts
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_pos = result.x.reshape(-1, 3)
                x0 = result.x
        except Exception:
            pass
    
    return best_pos, best_energy

def construct():
    """Construct 55-atom LJ cluster near global minimum."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Optimize radial parameters of the Mackay Icosahedron
    # Initial guess: r1 ~ 1.05, r2 ~ 1.90
    x0 = np.array([1.05, 1.90])
    
    # Use Nelder-Mead for robustness on the 2D parameter space
    # It doesn't need gradients and is very effective for smooth 2D landscapes
    try:
        res = minimize(
            _energy_from_params, x0, method='Nelder-Mead',
            options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 1000, 'adaptive': True}
        )
        optimal_params = res.x
    except Exception:
        optimal_params = x0
    
    # Ensure we are at least somewhat close to expected values
    if optimal_params[0] < 0.5 or optimal_params[1] < 1.0:
        optimal_params = x0

    # 2. Generate Cartesian positions from optimal parameters
    pos = _generate_mackay_positions(optimal_params)
    
    # 3. Full Cartesian Relaxation (L-BFGS)
    # This handles any slight asymmetries or numerical drift
    relaxed_pos, energy = _relax_lbfgs(pos, deadline=deadline, maxiter=500, gtol=1e-10)
    
    # 4. High-Precision Polish
    if time.time() < deadline - 20.0:
        polished_pos, polished_energy = _polish(relaxed_pos, deadline=deadline)
        if polished_energy < energy:
            pos = polished_pos
            energy = polished_energy
        else:
            pos = relaxed_pos

    # 5. Final safety check for minimum distance
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist_found = np.min(dists)
    
    if min_dist_found < 0.5:
        # This shouldn't happen for the global minimum, but if it does, scale up
        # Note: Scaling up increases energy, but satisfies constraint.
        # A better fix would be to reject this candidate and try another, 
        # but given the high quality of the seed, this is a rare edge case.
        scale = 0.5 / min_dist_found * 1.01
        pos = pos * scale

    return pos
# EVOLVE-BLOCK-END

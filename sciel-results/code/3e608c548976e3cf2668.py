"""Lennard-Jones 55-atom global minimum: Mackay icosahedron with symmetry-constrained manifold optimization."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import distance

N = 55

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
    r6 = r2 ** 3
    r12 = r6 * r6
    
    e_pair = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(e_pair, 0.0)
    energy = np.sum(e_pair) / 2.0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = 4.0 * (-12.0 / (r2 ** 7) + 6.0 / (r2 ** 4))
    np.fill_diagonal(factor, 0.0)
    
    factor_pos = factor[:, :, np.newaxis] * pos[np.newaxis, :, :]
    grad = -np.sum(factor_pos, axis=1) + pos * np.sum(factor, axis=1, keepdims=True)
    
    return energy, grad

def _get_icosahedron_geometry():
    """Get icosahedron vertices, edge midpoints, and face centers as unit vectors."""
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
    
    # Edge midpoints (30)
    edge_mids = []
    for i, j in edges:
        mid = (verts_unit[i] + verts_unit[j]) / 2.0
        norm = np.linalg.norm(mid)
        if norm > 1e-6:
            edge_mids.append(mid / norm)
    edge_mids = np.array(edge_mids)
    
    # Face centers (20)
    face_centers = []
    edge_set = set()
    for i, j in edges:
        edge_set.add((i, j))
        edge_set.add((j, i))
    
    for i in range(12):
        for j in range(i + 1, 12):
            if (i, j) not in edge_set:
                continue
            for k in range(j + 1, 12):
                if (j, k) in edge_set and (i, k) in edge_set:
                    center = (verts_unit[i] + verts_unit[j] + verts_unit[k]) / 3.0
                    norm = np.linalg.norm(center)
                    if norm > 1e-6:
                        face_centers.append(center / norm)
    face_centers = np.array(face_centers)
    
    return verts_unit, edge_mids, face_centers

def _make_mackay_icosahedron_55():
    """Construct 55-atom Mackay icosahedron: 1 center + 12 verts + 30 edge mids + 12 face centers."""
    verts_unit, edge_mids, face_centers = _get_icosahedron_geometry()
    
    # Standard Mackay radii (approximate)
    r1 = 1.05  # inner shell (vertices)
    r2 = 1.75  # middle shell (edge midpoints)
    r3 = 1.95  # outer shell (face centers)
    
    center = np.array([[0.0, 0.0, 0.0]])
    shell1 = verts_unit * r1
    shell2 = edge_mids * r2
    
    # Take 12 face centers (break symmetry, but L-BFGS will relax)
    # Use deterministic selection: first 12
    shell3 = face_centers[:12] * r3
    
    all_pos = np.vstack([center, shell1, shell2, shell3])
    return all_pos

def _construct_from_params(params):
    """Construct 55-atom cluster from 3 shell radii: [r1, r2, r3]
    1 center + 12 (verts) + 30 (edges) + 12 (faces) = 55
    """
    verts_unit, edge_mids, face_centers = _get_icosahedron_geometry()
    r1, r2, r3 = params
    
    center = np.array([[0.0, 0.0, 0.0]])
    shell1 = verts_unit * r1
    shell2 = edge_mids * r2
    shell3 = face_centers[:12] * r3
    
    all_pos = np.vstack([center, shell1, shell2, shell3])
    return all_pos

def _manifold_objective(params):
    """Objective: energy of the cluster constructed from shell radii, after L-BFGS relaxation."""
    pos = _construct_from_params(params)
    
    # Quick L-BFGS relaxation
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    x0 = pos.flatten()
    try:
        result = minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            options={'maxiter': 100, 'gtol': 1e-8, 'maxls': 30}
        )
        energy = result.fun
    except Exception:
        energy = _lj_energy(pos)
    
    return energy

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
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 50}
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
    """High-precision polish pass with tighter tolerance and more iterations."""
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
            options={'maxiter': 2000, 'gtol': 1e-12, 'maxls': 100}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    if deadline is not None and time.time() > deadline - 3.0:
        return best_pos, best_energy
    
    return best_pos, best_energy

def construct():
    """Construct 55-atom LJ cluster near global minimum using symmetry-constrained manifold."""
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    # Step 1: Start with a good Mackay icosahedron structure
    pos = _make_mackay_icosahedron_55()
    
    # Step 2: Optimize shell radii using Nelder-Mead in 3D parameter space
    # Use multiple initial points for robustness
    initial_points = [
        np.array([1.05, 1.75, 1.95]),
        np.array([1.0, 1.7, 1.9]),
        np.array([1.1, 1.8, 2.0]),
        np.array([0.95, 1.65, 1.85]),
        np.array([1.02, 1.72, 1.92]),
    ]
    
    best_energy = _lj_energy(pos)
    best_pos = pos.copy()
    
    for ip in initial_points:
        if time.time() > deadline - 40.0:
            break
        try:
            result = minimize(
                _manifold_objective, ip, method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-5}
            )
            if result.fun < best_energy:
                best_energy = result.fun
                candidate = _construct_from_params(result.x)
                relaxed, e_relaxed = _relax_lbfgs(candidate, deadline=deadline, maxiter=200, gtol=1e-9)
                if e_relaxed < best_energy:
                    best_energy = e_relaxed
                    best_pos = relaxed
        except Exception:
            pass
    
    # Step 3: Full L-BFGS relaxation from best structure
    if time.time() < deadline - 30.0:
        relaxed, energy = _relax_lbfgs(best_pos, deadline=deadline, maxiter=500, gtol=1e-10)
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed
    
    # Step 4: Try a few perturbations to escape local minima
    np.random.seed(42)
    for scale in [0.01, 0.02, 0.03]:
        if time.time() > deadline - 20.0:
            break
        perturbed = best_pos + np.random.randn(N, 3) * scale
        relaxed_p, energy_p = _relax_lbfgs(perturbed, deadline=deadline, maxiter=300, gtol=1e-9)
        if energy_p < best_energy:
            best_energy = energy_p
            best_pos = relaxed_p
    
    # Step 5: Final high-precision polish
    if time.time() < deadline - 10.0:
        polished, polished_energy = _polish(best_pos, deadline=deadline)
        if polished_energy < best_energy:
            best_energy = polished_energy
            best_pos = polished
    
    # Final check: ensure minimum distance >= 0.5
    min_dist_found = float('inf')
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pos[i] - best_pos[j])
            if d < min_dist_found:
                min_dist_found = d
    
    if min_dist_found < 0.5:
        scale = 0.5 / min_dist_found * 1.01
        best_pos = best_pos * scale
    
    return best_pos

"""Lennard-Jones 55-atom global minimum search with Mackay icosahedron construction and relaxation."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55


def _lj_energy_vec(pos):
    """Compute total LJ energy using vectorized operations."""
    n = pos.shape[0]
    # Compute all pairwise differences
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    r2 = r2[mask]
    r6 = r2 ** 3
    energy = 4.0 * np.sum(1.0 / (r6 * r6) - 1.0 / r6)
    return energy


def _lj_energy_grad_vec(pos):
    """Compute total LJ energy and gradient using vectorized operations."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    # Avoid division by zero on diagonal
    r2[np.eye(n, dtype=bool)] = 1.0
    r6 = r2 ** 3
    r12 = r6 * r6
    
    # Energy: sum over i<j of 4*(r^-12 - r^-6)
    # We compute for all pairs and divide by 2
    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    pair_energy[np.eye(n, dtype=bool)] = 0.0
    energy = 0.5 * np.sum(pair_energy)
    
    # Gradient: grad_i = sum_j 24*(2/r^13 - 1/r^7) * (pos_i - pos_j)
    # In terms of r2: r^13 = r2^6.5, r^7 = r2^3.5
    # factor_ij = 24*(2/r^13 - 1/r^7) = 24*(2/r2^6.5 - 1/r2^3.5)
    r = np.sqrt(r2)
    factor = 24.0 * (2.0 / (r ** 13) - 1.0 / (r ** 7))
    factor[np.eye(n, dtype=bool)] = 0.0
    
    # grad_i = sum_j factor_ij * diff_ij
    grad = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad


def _make_mackay_icosahedron():
    """Construct a two-shell Mackay icosahedron for 55 atoms.
    
    Structure: 1 center + 12 first shell + 42 second shell
    Second shell: 30 atoms at edge midpoints (scaled) + 12 atoms at face center directions (scaled)
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # Standard icosahedron vertices
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
    # First shell: 12 atoms at icosahedron vertices, radius ~1.0
    r1 = 1.0
    first_shell = verts_unit * r1
    
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
    
    # 30 edge midpoints (normalized to unit vectors)
    edge_mids = []
    for i, j in edges:
        mid = (verts_unit[i] + verts_unit[j]) / 2.0
        norm = np.linalg.norm(mid)
        if norm > 1e-6:
            edge_mids.append(mid / norm)
    edge_mids = np.array(edge_mids)
    
    # 12 face center directions: for each face (triangle of 3 vertices), 
    # the face center direction is the normalized sum of the 3 vertex directions
    # We need to find the 20 faces. For a regular icosahedron, each face is
    # an equilateral triangle. Let's find all triples that form faces.
    # A face consists of 3 vertices that are all connected (each pair is an edge).
    face_centers = []
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                # Check if all three pairs are edges
                d_ij = dists[i, j]
                d_ik = dists[i, k]
                d_jk = dists[j, k]
                if (abs(d_ij - min_dist) < 1e-6 and 
                    abs(d_ik - min_dist) < 1e-6 and 
                    abs(d_jk - min_dist) < 1e-6):
                    # This is a face
                    fc = (verts_unit[i] + verts_unit[j] + verts_unit[k]) / 3.0
                    norm = np.linalg.norm(fc)
                    if norm > 1e-6:
                        face_centers.append(fc / norm)
    
    face_centers = np.array(face_centers)
    
    # For the second shell, we need 42 atoms.
    # 30 at edge midpoints, 12 at some other positions.
    # Actually, for Mackay icosahedron n=2, the second shell has:
    # - 30 atoms: these are at positions corresponding to the 30 edges
    # - 12 atoms: these are at positions corresponding to the 12 vertices but at a different radius
    # But that gives 30 + 12 = 42. However, the 12 "vertex" positions in the second shell
    # are actually at the face centers of a dual dodecahedron, or more precisely,
    # they are at the 12 directions of the icosahedron vertices but at a larger radius.
    
    # Let's use: 30 at edge midpoints (radius r2), 12 at vertex directions (radius r3)
    # But we need to choose r2 and r3 appropriately.
    # For LJ, the optimal second shell radii are roughly 1.6-1.8.
    
    r2 = 1.6
    r3 = 1.8
    
    second_shell_edges = edge_mids * r2
    second_shell_verts = verts_unit * r3
    
    # Combine: center + first shell + second shell edges + second shell verts
    center = np.array([[0.0, 0.0, 0.0]])
    all_pos = np.vstack([center, first_shell, second_shell_edges, second_shell_verts])
    
    # We have 1 + 12 + 30 + 12 = 55 atoms
    return all_pos


def _fix_min_distance(pos, min_dist=0.5):
    """Ensure all pairs are at least min_dist apart."""
    pos = pos.copy()
    for _ in range(200):
        min_d = np.inf
        min_pair = None
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_d:
                    min_d = d
                    min_pair = (i, j)
        if min_d >= min_dist:
            break
        i, j = min_pair
        direction = pos[i] - pos[j]
        if np.linalg.norm(direction) < 1e-8:
            direction = np.array([1.0, 0.0, 0.0])
        direction = direction / np.linalg.norm(direction)
        push = (min_dist - min_d) / 2.0 + 0.01
        pos[i] += push * direction
        pos[j] -= push * direction
    return pos


def _relax_scipy(pos, max_iter=1000, tol=1e-10):
    """Relax using scipy L-BFGS-B."""
    def fun(x):
        pos_x = x.reshape(N, 3)
        e, g = _lj_energy_grad_vec(pos_x)
        return e, g.reshape(-1)
    
    result = minimize(fun, pos.flatten(), jac=True, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8})
    
    if result.success or np.isfinite(result.fun):
        return result.x.reshape(N, 3), result.fun
    return pos, _lj_energy_vec(pos)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_pos = None
    best_energy = np.inf
    
    # Start with Mackay icosahedron
    try:
        mackay = _make_mackay_icosahedron()
        mackay = _fix_min_distance(mackay)
        e0 = _lj_energy_vec(mackay)
        if np.isfinite(e0):
            best_energy = e0
            best_pos = mackay.copy()
    except Exception:
        pass
    
    # If Mackay construction failed, use cubic lattice
    if best_pos is None:
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        best_pos = np.array(points[:N], dtype=float) * 1.12
        best_energy = _lj_energy_vec(best_pos)
    
    # Relax the initial structure with scipy
    if time.time() < deadline:
        try:
            relaxed, e_rel = _relax_scipy(best_pos, max_iter=2000, tol=1e-12)
            relaxed = _fix_min_distance(relaxed)
            e_rel = _lj_energy_vec(relaxed)
            if np.isfinite(e_rel) and e_rel < best_energy:
                best_energy = e_rel
                best_pos = relaxed
        except Exception:
            pass
    
    # Try multiple restarts with perturbations
    rng = np.random.RandomState(42)
    for restart in range(30):
        if time.time() > deadline - 20.0:
            break
        
        # Create perturbed version
        if restart == 0:
            pos = best_pos.copy()
        else:
            # Add random perturbation, increasing in magnitude
            noise_scale = 0.05 + 0.02 * restart
            pos = best_pos + noise_scale * rng.randn(N, 3)
        
        # Ensure minimum distance constraint
        pos = _fix_min_distance(pos)
        
        # Relax
        try:
            relaxed, e_rel = _relax_scipy(pos, max_iter=1000, tol=1e-10)
            relaxed = _fix_min_distance(relaxed)
            e_rel = _lj_energy_vec(relaxed)
            if np.isfinite(e_rel) and e_rel < best_energy:
                best_energy = e_rel
                best_pos = relaxed
        except Exception:
            pass
    
    # Final check: ensure minimum distance
    best_pos = _fix_min_distance(best_pos)
    
    return best_pos

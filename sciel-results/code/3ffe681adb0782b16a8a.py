"""Lennard-Jones 55-atom cluster: Mackay icosahedron with edge-fraction manifold + high-precision polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55

# EVOLVE-BLOCK-START

def _lj_energy(pos):
    """Compute total LJ energy for N atoms (vectorized)."""
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

def _make_icosahedron_vertices():
    """Return the 12 unit icosahedron vertices."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    return verts / norms

def _find_icosahedron_edges(verts):
    """Find the 30 edges of the icosahedron."""
    edge_len = np.linalg.norm(verts[0] - verts[1])
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if np.isclose(d, edge_len, atol=1e-10):
                edges.append((i, j))
    return edges

def _find_icosahedron_faces(verts, edges):
    """Find the 20 faces of the icosahedron."""
    edge_set = set()
    for i, j in edges:
        edge_set.add((min(i, j), max(i, j)))
    
    faces = []
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                if ((min(i, j), max(i, j)) in edge_set and
                    (min(j, k), max(j, k)) in edge_set and
                    (min(i, k), max(i, k)) in edge_set):
                    faces.append((i, j, k))
    return faces

def _make_mackay_icosahedron(r1=1.04, r2=2.08, r3=3.0):
    """Generate a Mackay icosahedral cluster with 55 atoms.
    
    Structure: 1 (center) + 12 (inner shell) + 30 (edge-midpoint shell) + 12 (face-center shell)
    For N=55: 1+12+30+12 = 55.
    """
    verts = _make_icosahedron_vertices()
    edges = _find_icosahedron_edges(verts)
    faces = _find_icosahedron_faces(verts, edges)
    
    # Edge midpoints (30 directions)
    midpoints = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid)
        midpoints.append(mid)
    midpoints = np.array(midpoints)
    
    # Face centers (use first 12)
    face_centers = []
    for i, j, k in faces[:12]:
        fc = (verts[i] + verts[j] + verts[k]) / 3.0
        fc = fc / np.linalg.norm(fc)
        face_centers.append(fc)
    face_centers = np.array(face_centers)
    
    inner = np.array([[0.0, 0.0, 0.0]])
    shell1 = verts * r1
    shell2 = midpoints * r2
    shell3 = face_centers * r3
    
    pos = np.vstack([inner, shell1, shell2, shell3])
    return pos

def _make_edge_fraction_mackay(r1=1.04, r2=2.08, r3=3.0, t_vals=None):
    """Generate a Mackay icosahedron where the 30 outer-shell atoms are parameterized
    by edge fractions t in [0,1] along the icosahedron edges.
    
    The 30 edge-midpoint atoms become atoms sliding along edges:
    atom_position = (1-t)*vertex_i + t*vertex_j, scaled to radius r2.
    
    This reduces the optimization to a low-dimensional manifold.
    """
    verts = _make_icosahedron_vertices()
    edges = _find_icosahedron_edges(verts)
    faces = _find_icosahedron_faces(verts, edges)
    
    if t_vals is None:
        t_vals = np.full(30, 0.5)
    
    # Edge-fraction positions (30 atoms)
    edge_positions = []
    for idx, (i, j) in enumerate(edges):
        t = t_vals[idx]
        pos = (1 - t) * verts[i] + t * verts[j]
        pos = pos / np.linalg.norm(pos)
        edge_positions.append(pos)
    edge_positions = np.array(edge_positions)
    
    # Face centers (use first 12)
    face_centers = []
    for i, j, k in faces[:12]:
        fc = (verts[i] + verts[j] + verts[k]) / 3.0
        fc = fc / np.linalg.norm(fc)
        face_centers.append(fc)
    face_centers = np.array(face_centers)
    
    inner = np.array([[0.0, 0.0, 0.0]])
    shell1 = verts * r1
    shell2 = edge_positions * r2
    shell3 = face_centers * r3
    
    pos = np.vstack([inner, shell1, shell2, shell3])
    return pos

def _optimize_edge_fractions(r1=1.04, r2=2.08, r3=3.0, maxiter=200):
    """Optimize the 30 edge-fraction parameters to minimize LJ energy.
    
    This is a low-dimensional optimization (30 variables) that finds the
    best configuration of atoms sliding along icosahedron edges.
    """
    def energy_from_t(t_flat):
        t_vals = np.clip(t_flat, 0.0, 1.0)
        pos = _make_edge_fraction_mackay(r1, r2, r3, t_vals)
        return _lj_energy(pos)
    
    t0 = np.full(30, 0.5)
    
    try:
        result = minimize(energy_from_t, t0, method='L-BFGS-B',
                         bounds=[(0.0, 1.0)] * 30,
                         options={'maxiter': maxiter, 'gtol': 1e-8, 'ftol': 1e-10})
        t_opt = np.clip(result.x, 0.0, 1.0)
        pos = _make_edge_fraction_mackay(r1, r2, r3, t_opt)
        e = _lj_energy(pos)
        return pos, e, t_opt
    except Exception:
        pos = _make_mackay_icosahedron(r1, r2, r3)
        e = _lj_energy(pos)
        return pos, e, t0

def _make_cubic_lattice():
    """Generate a cubic lattice fragment."""
    side = int(np.ceil(N ** (1 / 3)))
    points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
    points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
    return np.array(points[:N], dtype=float) * 1.12

def _make_random_cluster(seed):
    """Generate a random cluster."""
    rng = np.random.RandomState(seed)
    pos = rng.randn(N, 3) * 2.5
    return pos

def _make_perturbed_mackay(seed, scale=0.1):
    """Generate a perturbed Mackay icosahedron."""
    base = _make_mackay_icosahedron()
    rng = np.random.RandomState(seed)
    pos = base + rng.randn(N, 3) * scale
    return pos

def _relax_lbfgs(pos, maxiter=300, gtol=1e-8, ftol=1e-10):
    """Relax using L-BFGS-B with gradient."""
    def fun(x):
        x = x.reshape(N, 3)
        e, g = _lj_energy_grad(x)
        return e, g.reshape(-1)
    
    try:
        result = minimize(fun, pos.reshape(-1), method='L-BFGS-B', jac=True,
                         options={'maxiter': maxiter, 'gtol': gtol, 'ftol': ftol})
        return result.x.reshape(N, 3), result.fun
    except Exception:
        return pos, _lj_energy(pos)

def _polish_high_precision(pos, deadline=None):
    """High-precision polish pass with progressively tighter tolerances."""
    best_pos = pos.copy()
    best_e = _lj_energy(pos)
    x0 = pos.reshape(-1).copy()
    
    stages = [
        {'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-12, 'maxls': 100},
        {'maxiter': 1000, 'gtol': 1e-12, 'ftol': 1e-14, 'maxls': 200},
        {'maxiter': 2000, 'gtol': 1e-14, 'ftol': 1e-16, 'maxls': 300},
    ]
    
    for stage_opts in stages:
        if deadline is not None and time.time() > deadline - 10.0:
            break
        
        def fun(x):
            x = x.reshape(N, 3)
            e, g = _lj_energy_grad(x)
            return e, g.reshape(-1)
        
        try:
            result = minimize(fun, x0, method='L-BFGS-B', jac=True,
                             options=stage_opts)
            if result.fun < best_e:
                best_e = result.fun
                best_pos = result.x.reshape(N, 3).copy()
                x0 = result.x.copy()
        except Exception:
            pass
    
    return best_pos, best_e

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_pos = _make_mackay_icosahedron()
    best_e = _lj_energy(best_pos)
    
    # Strategy 1: Optimize edge fractions for the Mackay icosahedron
    # This is a low-dimensional manifold optimization (30 variables)
    if time.time() < deadline - 30.0:
        pos_ef, e_ef, t_opt = _optimize_edge_fractions(r1=1.04, r2=2.08, r3=3.0, maxiter=300)
        if e_ef < best_e:
            best_e = e_ef
            best_pos = pos_ef.copy()
    
    # Strategy 2: Try multiple radius combinations with edge-fraction optimization
    radius_combos = [
        (1.04, 2.08, 3.0),
        (1.0, 2.0, 2.9),
        (1.08, 2.12, 3.1),
        (0.98, 2.02, 2.85),
        (1.1, 2.15, 3.15),
    ]
    
    for r1, r2, r3 in radius_combos:
        if time.time() > deadline - 20.0:
            break
        
        # Quick edge-fraction optimization
        pos_ef, e_ef, _ = _optimize_edge_fractions(r1, r2, r3, maxiter=100)
        if e_ef < best_e:
            best_e = e_ef
            best_pos = pos_ef.copy()
    
    # Strategy 3: Full Cartesian relaxation of the best candidate
    if time.time() < deadline - 15.0:
        pos_relaxed, e_relaxed = _relax_lbfgs(best_pos, maxiter=500, gtol=1e-10, ftol=1e-12)
        if e_relaxed < best_e:
            best_e = e_relaxed
            best_pos = pos_relaxed.copy()
    
    # Strategy 4: High-precision polish
    if time.time() < deadline - 10.0:
        pos_final, e_final = _polish_high_precision(best_pos, deadline=deadline)
        if e_final < best_e:
            best_e = e_final
            best_pos = pos_final.copy()
    
    # Strategy 5: Try a few random restarts if time remains
    if time.time() < deadline - 15.0:
        for seed in range(5):
            if time.time() > deadline - 10.0:
                break
            pos_rand = _make_random_cluster(seed)
            pos_rel, e_rel = _relax_lbfgs(pos_rand, maxiter=200, gtol=1e-8, ftol=1e-10)
            if e_rel < best_e:
                best_e = e_rel
                best_pos = pos_rel.copy()
    
    # Final high-precision polish
    if time.time() < deadline - 5.0:
        pos_final, e_final = _polish_high_precision(best_pos, deadline=deadline)
        if e_final < best_e:
            best_e = e_final
            best_pos = pos_final.copy()
    
    return best_pos

# EVOLVE-BLOCK-END

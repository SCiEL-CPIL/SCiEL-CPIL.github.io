"""Lennard-Jones cluster global minimum, 55 atoms.
Strategy: construct a Mackay icosahedral seed, relax with L-BFGS,
and use a time-bounded multi-restart basin-hopping scheme to improve the energy.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 55
TIME_LIMIT = 110.0  # safety margin under 120s wall-clock (evaluator kills at 150s)

def _lj_energy_grad(pos):
    """Compute total LJ energy and gradient for a cluster of N atoms.
    Energy: 4 * sum_{i<j} (r^-12 - r^-6)
    """
    n = pos.shape[0]
    # Compute all pairwise differences
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (n, n, 3)
    r2 = np.sum(diff * diff, axis=2)  # (n, n)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero on diagonal
    r6 = r2 ** (-3)
    r12 = r6 * r6
    # Energy: sum over i<j
    E = 4.0 * np.sum(np.triu(r12 - r6, 1))
    # Gradient: dE/dx_i = 4 * sum_{j != i} (-12 r^-14 + 6 r^-8) * (x_i - x_j)
    # = 24 * sum_{j != i} (2 r^-14 - r^-8) * (x_i - x_j)
    # Using r2: r^-8 = r2^-4, r^-14 = r2^-7
    # grad_i = 24 * sum_j (2*r2^-7 - r2^-4) * (pos_i - pos_j)
    # Let's compute: factor = 24 * (2 * r2^-7 - r2^-4) for each pair
    # grad[i] = sum_{j != i} 24 * (2 * r2[i,j]^-7 - r2[i,j]^-4) * (pos[i] - pos[j])
    r2_inv4 = r2 ** (-4)
    r2_inv7 = r2 ** (-7)
    factor = 24.0 * (2.0 * r2_inv7 - r2_inv4)  # (n, n)
    np.fill_diagonal(factor, 0.0)
    grad = np.sum(factor[:, :, np.newaxis] * diff, axis=1)  # diff[i,j] = pos[i]-pos[j]
    return E, grad


def _lj_energy(pos):
    """Just energy, for quick checks."""
    E, _ = _lj_energy_grad(pos)
    return E


def _relax(pos, maxiter=500):
    """L-BFGS local relaxation."""
    res = minimize(_lj_energy_grad, pos, jac=True, method='L-BFGS-B',
                   options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10})
    return res.x


def _generate_mackay_icosahedron():
    """Generate a Mackay icosahedral cluster with 55 atoms.
    Shell 0: 1 atom at center
    Shell 1: 12 atoms at icosahedral vertices, radius r1
    Shell 2: 42 atoms: 30 on edges + 12 on faces
    """
    # Golden ratio
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # Icosahedral vertices (12)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts[0])  # normalize to unit sphere
    
    # Shell 1: 12 atoms at icosahedral vertices
    r1 = 1.0
    shell1 = verts * r1
    
    # For shell 2, we need 42 more atoms.
    # In a Mackay icosahedron, shell 2 has:
    # - 30 atoms on the edges of the icosahedron (midpoints scaled)
    # - 12 atoms on the faces (face centers scaled)
    
    # Icosahedron edges: 30 edges
    # An icosahedron has 12 vertices, 30 edges, 20 faces.
    # Each vertex connects to 5 others.
    
    # Let's find edges: pairs of vertices that are nearest neighbors
    # The distance between adjacent vertices in a unit icosahedron:
    # |verts[i] - verts[j]| for adjacent pairs
    
    # Compute all pairwise distances
    diffs = verts[:, np.newaxis, :] - verts[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # The minimum distance is the edge length
    min_dist = np.min(dists)
    # Find all edges (pairs with distance close to min_dist)
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                edges.append((i, j))
    
    # 30 edge midpoints
    edge_mids = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid)
        edge_mids.append(mid)
    edge_mids = np.array(edge_mids)
    
    # 20 face centers
    # An icosahedron has 20 triangular faces.
    # Each face is a triangle of 3 vertices.
    # Find faces: triples of mutually adjacent vertices
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                for k in range(j+1, 12):
                    if abs(dists[i, k] - min_dist) < 1e-6 and abs(dists[j, k] - min_dist) < 1e-6:
                        faces.append((i, j, k))
    
    face_centers = []
    for i, j, k in faces:
        fc = (verts[i] + verts[j] + verts[k]) / 3.0
        fc = fc / np.linalg.norm(fc)
        face_centers.append(fc)
    face_centers = np.array(face_centers)
    
    # Shell 2 radius: for Mackay icosahedron, r2 = 2*r1 (approximately, but need to optimize)
    # Actually, the optimal ratio is determined by energy minimization.
    # Let's use r2 = 2.0 as a starting point, then relax.
    r2 = 2.0
    
    shell2_edges = edge_mids * r2
    shell2_faces = face_centers * r2
    
    # Total: 1 + 12 + 30 + 12 = 55
    center = np.array([[0.0, 0.0, 0.0]])
    all_atoms = np.vstack([center, shell1, shell2_edges, shell2_faces])
    
    return all_atoms


def construct():
    start_time = time.time()
    
    # Start with Mackay icosahedron
    pos0 = _generate_mackay_icosahedron()
    
    # Relax the initial guess
    best_pos = _relax(pos0, maxiter=500)
    best_energy = _lj_energy(best_pos)
    
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT * 0.5:
        return best_pos
    
    # Basin hopping: perturb and relax
    rng = np.random.default_rng(42)
    
    # Try several perturbation scales
    scales = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    
    for scale in scales:
        for attempt in range(20):
            elapsed = time.time() - start_time
            if elapsed > TIME_LIMIT - 10:  # leave 10s safety margin
                break
            
            # Perturb
            noise = rng.normal(0, scale, size=best_pos.shape)
            perturbed = best_pos + noise
            
            # Relax
            relaxed = _relax(perturbed, maxiter=300)
            energy = _lj_energy(relaxed)
            
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed
    
    return best_pos

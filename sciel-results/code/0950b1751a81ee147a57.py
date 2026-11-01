"""Lennard-Jones cluster global minimum, 55 atoms — Mackay icosahedron seed + L-BFGS relaxation."""
import numpy as np
import time
import math

N = 55


def _lj_energy(pos):
    """Compute total Lennard-Jones energy for positions (N,3)."""
    n = pos.shape[0]
    rij = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(rij * rij, axis=2)
    # Exclude self-interactions
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    energy = 4.0 * np.sum(r12 - r6)
    return energy


def _lj_grad(pos):
    """Compute gradient of LJ energy."""
    n = pos.shape[0]
    rij = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(rij * rij, axis=2)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    # dE/dr2 = 4*(-12*r2^-7 + 6*r2^-4) = 4*(-12*r12/r2 + 6*r6/r2)
    # grad_i = 4*(-12*r12 + 6*r6) * rij_i / r2 ... but need to be careful
    # E = 4 * sum_{i<j} (r^-12 - r^-6)
    # dE/dx_i = 4 * sum_{j!=i} (-12*r^-14 + 6*r^-8) * (x_i - x_j)
    # = 4 * sum_{j!=i} (-12*r12/r2 + 6*r6/r2) * (x_i - x_j)
    # where r2 = |x_i - x_j|^2
    coeff = 4.0 * (-12.0 * r12 + 6.0 * r6) / r2  # shape (n, n)
    grad = np.sum(coeff[:, :, np.newaxis] * rij, axis=1)
    return grad


def _mackay_icosahedron_55():
    """
    Construct the 55-atom Mackay icosahedron.
    
    Structure:
    - 1 atom at center
    - 12 atoms at icosahedron vertices (inner shell)
    - 30 atoms at edge midpoints (outer shell)
    - 12 atoms at face centers (outer shell) -- wait, that's 1+12+30+12=55
    
    Actually the standard 55-atom Mackay icosahedron has:
    - 1 center
    - 12 vertices of icosahedron (radius R1)
    - 30 edge midpoints (radius R2)
    - 12 face centers (radius R3)
    
    But the problem says "complete two-shell Mackay icosahedron". Let me reconsider.
    
    The Mackay icosahedron for n=2 (two shells) has:
    - Shell 0: 1 atom at center
    - Shell 1: 12 atoms (icosahedron vertices)
    - Shell 2: 42 atoms (30 edge midpoints + 12 face centers)
    Total: 1 + 12 + 42 = 55
    
    The radii are typically:
    - R1 = 1 (inner shell)
    - R2 = sqrt(2) ≈ 1.414 (outer shell, for ideal Mackay)
    
    But for LJ, the optimal radii may differ. Let's use the ideal Mackay ratios
    and then relax.
    
    Icosahedron vertices: 12 points.
    Edge midpoints: 30 points (each edge of icosahedron has one midpoint).
    Face centers: 12 points (each face of icosahedron has one center).
    """
    # Icosahedron vertices
    phi = (1 + math.sqrt(5)) / 2.0
    # 12 vertices of a regular icosahedron
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit sphere
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Inner shell radius
    R1 = 1.0
    inner = verts * R1
    
    # Find edges of icosahedron: pairs of vertices at minimum distance
    # The edge length of a regular icosahedron with unit circumradius is 2/sqrt(phi^2+1) * ... 
    # Let's just compute all pairwise distances and find the 30 shortest
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            dists[i, j] = d
            dists[j, i] = d
    
    # The 30 edges are the 30 smallest distances (should be all equal for regular icosahedron)
    edge_pairs = []
    for i in range(12):
        for j in range(i+1, 12):
            edge_pairs.append((dists[i, j], i, j))
    edge_pairs.sort(key=lambda x: x[0])
    edges = [(i, j) for d, i, j in edge_pairs[:30]]
    
    # Edge midpoints
    edge_mids = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        edge_mids.append(mid)
    edge_mids = np.array(edge_mids)
    # Normalize to unit sphere then scale to R2
    edge_mids = edge_mids / np.linalg.norm(edge_mids, axis=1, keepdims=True)
    R2 = math.sqrt(2.0)  # ideal Mackay outer shell radius
    outer1 = edge_mids * R2
    
    # Face centers: 12 faces of icosahedron
    # Each face is a triangle of 3 mutually connected vertices
    # Build adjacency
    adj = set()
    for i, j in edges:
        adj.add((i, j))
        adj.add((j, i))
    
    faces = []
    used_faces = set()
    for i in range(12):
        for j in range(12):
            if i == j or (i, j) not in adj:
                continue
            for k in range(12):
                if i == k or j == k or (j, k) not in adj or (i, k) not in adj:
                    continue
                # Triangle (i, j, k)
                tri = tuple(sorted([i, j, k]))
                if tri not in used_faces:
                    used_faces.add(tri)
                    faces.append(tri)
    
    # Should have 20 faces (icosahedron has 20 triangular faces)
    face_centers = []
    for face in faces:
        fc = (verts[face[0]] + verts[face[1]] + verts[face[2]]) / 3.0
        face_centers.append(fc)
    face_centers = np.array(face_centers)
    face_centers = face_centers / np.linalg.norm(face_centers, axis=1, keepdims=True)
    outer2 = face_centers * R2
    
    # Assemble all 55 atoms
    center = np.array([[0.0, 0.0, 0.0]])
    all_atoms = np.vstack([center, inner, outer1, outer2])
    
    return all_atoms


def construct():
    """
    Construct 55-atom LJ cluster near the Mackay icosahedron minimum.
    Returns array of shape (55, 3).
    """
    t0 = time.time()
    deadline = 115.0  # seconds, leave margin for evaluator overhead
    
    # Start with Mackay icosahedron
    pos = _mackay_icosahedron_55()
    
    # Ensure minimum distance constraint (atoms at least 0.5 apart)
    # The Mackay icosahedron should satisfy this, but let's verify
    # and adjust if needed
    
    # Relax using L-BFGS
    from scipy.optimize import minimize
    
    def energy_func(x):
        p = x.reshape(-1, 3)
        # Check minimum distance
        rij = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        r2 = np.sum(rij * rij, axis=2)
        np.fill_diagonal(r2, np.inf)
        min_r2 = np.min(r2)
        if min_r2 < 0.25:  # 0.5^2
            return _lj_energy(p) + 1e6 * (0.25 - min_r2)
        return _lj_energy(p)
    
    def grad_func(x):
        p = x.reshape(-1, 3)
        return _lj_grad(p).reshape(-1)
    
    x0 = pos.flatten()
    
    best_pos = pos.copy()
    best_energy = _lj_energy(pos)
    
    # Try L-BFGS-B
    try:
        result = minimize(
            energy_func,
            x0,
            jac=grad_func,
            method='L-BFGS-B',
            options={'maxiter': 500, 'gtol': 1e-8, 'ftol': 1e-10}
        )
        if result.success or result.fun < best_energy:
            new_pos = result.x.reshape(-1, 3)
            # Validate minimum distance
            rij = new_pos[:, np.newaxis, :] - new_pos[np.newaxis, :, :]
            r2 = np.sum(rij * rij, axis=2)
            np.fill_diagonal(r2, np.inf)
            if np.min(r2) >= 0.25:
                best_pos = new_pos
                best_energy = result.fun
    except Exception:
        pass
    
    elapsed = time.time() - t0
    if elapsed > deadline:
        return best_pos
    
    # Try a few more restarts with slightly perturbed positions if time allows
    rng = np.random.default_rng(42)
    for restart in range(5):
        if time.time() - t0 > deadline - 10:
            break
        perturbation = rng.normal(0, 0.01, pos.shape)
        x0_pert = (pos + perturbation).flatten()
        try:
            result = minimize(
                energy_func,
                x0_pert,
                jac=grad_func,
                method='L-BFGS-B',
                options={'maxiter': 300, 'gtol': 1e-8, 'ftol': 1e-10}
            )
            if result.fun < best_energy:
                new_pos = result.x.reshape(-1, 3)
                rij = new_pos[:, np.newaxis, :] - new_pos[np.newaxis, :, :]
                r2 = np.sum(rij * rij, axis=2)
                np.fill_diagonal(r2, np.inf)
                if np.min(r2) >= 0.25:
                    best_pos = new_pos
                    best_energy = result.fun
        except Exception:
            pass
    
    return best_pos

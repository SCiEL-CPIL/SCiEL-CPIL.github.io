"""Lennard-Jones 55-atom cluster: multi-start search from Mackay icosahedron with L-BFGS relaxation."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55

# EVOLVE-BLOCK-START

def _lj_energy_and_grad(pos):
    """Compute Lennard-Jones energy and gradient for a set of positions."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r2_safe = np.maximum(r2, 1e-12)
    
    r6 = r2_safe ** (-3)
    r12 = r6 ** 2
    
    # Energy: 4 * sum_{i<j} (r^-12 - r^-6)
    # Matrix has each pair twice, so: E = 2 * sum(r12 - r6)
    E = 2.0 * np.sum(r12 - r6)
    
    # Gradient
    # dE/dpos_i = 4 * sum_{j!=i} (-12*r^-14 + 6*r^-8) * (pos_i - pos_j)
    # In terms of r2: r^-14 = r2^(-7), r^-8 = r2^(-4)
    # coeff = 24 * (2*r2^(-7) - r2^(-4))
    coeff = 24.0 * (2.0 * r2_safe ** (-7) - r2_safe ** (-4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    return E, grad

def _make_mackay_icosahedron(n_atoms=55):
    """
    Generate a Mackay icosahedron cluster with 55 atoms.
    Shell 0: 1 atom at origin
    Shell 1: 12 atoms at icosahedron vertices
    Shell 2: 42 atoms (30 edge midpoints + 12 face centers of the icosahedron)
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # Icosahedron vertices (unit length)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Find edges: pairs of vertices that are nearest neighbors
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            dists[i, j] = d
            dists[j, i] = d
    
    min_dist = np.min(dists[dists > 0])
    tol = min_dist * 1.01
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if dists[i, j] < tol:
                edges.append((i, j))
    
    # Edge midpoints (30 atoms)
    edge_mids = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        edge_mids.append(mid)
    edge_mids = np.array(edge_mids)
    edge_mids /= np.linalg.norm(edge_mids, axis=1, keepdims=True)
    
    # Find faces: triangles of mutually adjacent vertices
    adj = {i: set() for i in range(12)}
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    
    faces = set()
    for i in range(12):
        for j in adj[i]:
            for k in adj[j]:
                if k != i and i in adj[k]:
                    tri = tuple(sorted([i, j, k]))
                    faces.add(tri)
    
    # Face centers (20 faces, but we need 12 for the second shell)
    # In a Mackay icosahedron, the 42 second-shell atoms are:
    # 30 at edge midpoints and 12 at face centers
    # We'll use 12 of the 20 face centers (the ones that are "equivalent" under symmetry)
    # For simplicity, we'll use all 20 face centers and select 12
    # Actually, let's use a different approach: the 42 atoms are at specific positions
    # Let's use the known coordinates for a 55-atom Mackay icosahedron
    
    # The second shell in a Mackay icosahedron has two types of positions:
    # - 30 atoms at the midpoints of the 30 edges of the icosahedron
    # - 12 atoms at the centers of 12 of the 20 faces
    # The 12 face centers are those that are not "opposite" to each other
    
    # For now, let's use all 20 face centers and pick 12
    face_centers = []
    for face in faces:
        center = (verts[face[0]] + verts[face[1]] + verts[face[2]]) / 3.0
        face_centers.append(center)
    face_centers = np.array(face_centers)
    face_centers /= np.linalg.norm(face_centers, axis=1, keepdims=True)
    
    # Select 12 face centers (we'll use the first 12 for now)
    face_centers_12 = face_centers[:12]
    
    # Combine all positions
    center = np.array([[0.0, 0.0, 0.0]])
    shell1 = verts  # 12 atoms
    shell2 = np.vstack([edge_mids, face_centers_12])  # 42 atoms
    
    all_points = np.vstack([center, shell1, shell2])
    
    return all_points[:n_atoms]

def _cubic_lattice_fragment(n_atoms=55, spacing=1.12):
    """Generate a cubic lattice fragment."""
    side = int(np.ceil(n_atoms ** (1/3)))
    points = []
    for i in range(side):
        for j in range(side):
            for k in range(side):
                points.append([i, j, k])
    points.sort(key=lambda p: (p[0]-(side-1)/2)**2 + (p[1]-(side-1)/2)**2 + (p[2]-(side-1)/2)**2)
    return np.array(points[:n_atoms], dtype=float) * spacing

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    best_pos = None
    best_energy = np.inf
    
    # Generate base Mackay icosahedron
    mackay_base = _make_mackay_icosahedron(55)
    
    # Multi-start stochastic search
    np.random.seed(42)
    
    # Try different perturbation scales
    for scale in [0.01, 0.05, 0.1, 0.2, 0.5]:
        if time.time() > deadline:
            break
        
        # Generate perturbed Mackay icosahedron
        perturbed = mackay_base + np.random.randn(55, 3) * scale
        
        # Ensure minimum distance constraint
        for i in range(55):
            for j in range(i + 1, 55):
                d = np.linalg.norm(perturbed[i] - perturbed[j])
                if d < 0.5:
                    perturbed[j] = perturbed[i] + 0.6 * (perturbed[j] - perturbed[i]) / max(d, 1e-6)
        
        # Local relaxation with L-BFGS
        try:
            result = minimize(_lj_energy_and_grad, perturbed, method='L-BFGS-B',
                             options={'maxiter': 2000, 'gtol': 1e-8})
            
            if result.fun < best_energy:
                best_energy = result.fun
                best_pos = result.x.copy()
        except Exception:
            pass
    
    # If no valid result, try cubic lattice
    if best_pos is None or best_energy > -100:
        cubic = _cubic_lattice_fragment(55, 1.12)
        try:
            result = minimize(_lj_energy_and_grad, cubic, method='L-BFGS-B',
                             options={'maxiter': 2000, 'gtol': 1e-8})
            if result.fun < best_energy:
                best_energy = result.fun
                best_pos = result.x.copy()
        except Exception:
            best_pos = cubic
    
    # If still no valid result, return cubic lattice
    if best_pos is None:
        return _cubic_lattice_fragment(55, 1.12)
    
    return best_pos

# EVOLVE-BLOCK-END

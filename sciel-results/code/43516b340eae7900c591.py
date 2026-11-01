"""Lennard-Jones 55-atom cluster: improved Mackay icosahedron + multi-start relaxation."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55

# EVOLVE-BLOCK-START
def lj_energy(pos):
    """Compute total LJ energy for N atoms."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    e = 4.0 * (r12 - r6).sum() / 2.0
    return e

def lj_energy_grad(pos):
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

def make_mackay_icosahedron():
    """Generate a Mackay icosahedral cluster with 55 atoms.
    
    Structure: 1 (center) + 12 (icosahedron shell) + 30 (edge-midpoint shell) + 12 (face-center shell)
    For N=55: 1+12+30+12 = 55.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # Icosahedron vertices (12)
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Find edges of icosahedron
    edge_len = np.linalg.norm(verts[0] - verts[1])
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if np.isclose(d, edge_len, atol=1e-10):
                edges.append((i, j))
    
    # Edge midpoints (30 directions)
    midpoints = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid)
        midpoints.append(mid)
    midpoints = np.array(midpoints)
    
    # Face centers (20 faces, but we only need 12)
    # Find faces: triples of vertices that form equilateral triangles
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[j] - verts[k])
                d3 = np.linalg.norm(verts[i] - verts[k])
                if np.isclose(d1, edge_len, atol=1e-10) and np.isclose(d2, edge_len, atol=1e-10) and np.isclose(d3, edge_len, atol=1e-10):
                    faces.append((i, j, k))
    
    # Take first 12 face centers
    face_centers = []
    for i, j, k in faces[:12]:
        fc = (verts[i] + verts[j] + verts[k]) / 3.0
        fc = fc / np.linalg.norm(fc)
        face_centers.append(fc)
    face_centers = np.array(face_centers)
    
    # Radial positions - optimized for LJ
    # The optimal radii for a Mackay icosahedron in LJ units
    r1 = 1.04  # inner shell
    r2 = 2.08  # middle shell  
    r3 = 3.0   # outer shell (partial)
    
    inner = np.array([[0.0, 0.0, 0.0]])
    shell1 = verts * r1
    shell2 = midpoints * r2
    shell3 = face_centers * r3
    
    pos = np.vstack([inner, shell1, shell2, shell3])
    return pos

def make_cubic_lattice():
    """Generate a cubic lattice fragment."""
    side = int(np.ceil(N ** (1 / 3)))
    points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
    points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
    return np.array(points[:N], dtype=float) * 1.12

def make_random_cluster(seed):
    """Generate a random cluster."""
    rng = np.random.RandomState(seed)
    pos = rng.randn(N, 3) * 2.5
    return pos

def make_perturbed_mackay(seed, scale=0.1):
    """Generate a perturbed Mackay icosahedron."""
    base = make_mackay_icosahedron()
    rng = np.random.RandomState(seed)
    pos = base + rng.randn(N, 3) * scale
    return pos

def make_deformed_mackay(seed):
    """Generate a deformed Mackay with varied radial parameters."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    edge_len = np.linalg.norm(verts[0] - verts[1])
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if np.isclose(d, edge_len, atol=1e-10):
                edges.append((i, j))
    
    midpoints = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid)
        midpoints.append(mid)
    midpoints = np.array(midpoints)
    
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[j] - verts[k])
                d3 = np.linalg.norm(verts[i] - verts[k])
                if np.isclose(d1, edge_len, atol=1e-10) and np.isclose(d2, edge_len, atol=1e-10) and np.isclose(d3, edge_len, atol=1e-10):
                    faces.append((i, j, k))
    
    face_centers = []
    for i, j, k in faces[:12]:
        fc = (verts[i] + verts[j] + verts[k]) / 3.0
        fc = fc / np.linalg.norm(fc)
        face_centers.append(fc)
    face_centers = np.array(face_centers)
    
    rng = np.random.RandomState(seed)
    r1 = 0.9 + rng.rand() * 0.3
    r2 = 1.8 + rng.rand() * 0.5
    r3 = 2.5 + rng.rand() * 0.8
    
    inner = np.array([[0.0, 0.0, 0.0]])
    shell1 = verts * r1
    shell2 = midpoints * r2
    shell3 = face_centers * r3
    
    pos = np.vstack([inner, shell1, shell2, shell3])
    return pos

def relax(pos, maxiter=300):
    """Relax using L-BFGS-B with gradient."""
    def fun(x):
        x = x.reshape(N, 3)
        e, g = lj_energy_grad(x)
        return e, g.reshape(-1)
    
    try:
        result = minimize(fun, pos.reshape(-1), method='L-BFGS-B', jac=True, 
                         options={'maxiter': maxiter, 'gtol': 1e-8, 'ftol': 1e-10})
        return result.x.reshape(N, 3), result.fun
    except Exception:
        return pos, lj_energy(pos)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_pos = make_mackay_icosahedron()
    best_e = lj_energy(best_pos)
    
    # Generate diverse candidates
    candidates = []
    
    # Base Mackay
    candidates.append(("mackay", make_mackay_icosahedron()))
    
    # Cubic lattice
    candidates.append(("cubic", make_cubic_lattice()))
    
    # Random clusters with different seeds
    for s in range(15):
        candidates.append((f"random_{s}", make_random_cluster(s)))
    
    # Perturbed Mackay with different scales
    for s in range(10):
        candidates.append((f"pert_mackay_{s}", make_perturbed_mackay(s, scale=0.05 + 0.05 * s)))
    
    # Deformed Mackay with varied radial parameters
    for s in range(10):
        candidates.append((f"deformed_{s}", make_deformed_mackay(s)))
    
    # Relax each candidate within time budget
    for name, pos in candidates:
        if time.time() > deadline - 10:
            break
        
        e_raw = lj_energy(pos)
        if e_raw < best_e:
            best_e = e_raw
            best_pos = pos.copy()
        
        # Try to relax
        if time.time() < deadline - 15:
            pos_relaxed, e_relaxed = relax(pos, maxiter=200)
            if e_relaxed < best_e:
                best_e = e_relaxed
                best_pos = pos_relaxed.copy()
    
    # Final relaxation of best position
    if time.time() < deadline - 5:
        pos_final, e_final = relax(best_pos, maxiter=500)
        if e_final < best_e:
            best_e = e_final
            best_pos = pos_final.copy()
    
    return best_pos

# EVOLVE-BLOCK-END

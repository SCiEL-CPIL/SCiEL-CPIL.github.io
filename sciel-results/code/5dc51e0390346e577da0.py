"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 110.0  # Safety margin before 120s kill

def lj_energy(pos):
    """Compute total Lennard-Jones energy."""
    n = len(pos)
    # Compute all pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    # Avoid division by zero (diagonal)
    np.fill_diagonal(r2, 1.0)
    r = np.sqrt(r2)
    r6 = r ** 6
    r12 = r6 ** 2
    # Energy per pair: 4 * (r^-12 - r^-6)
    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    # Sum over all pairs (upper triangle only, exclude diagonal)
    iu = np.triu_indices(n, k=1)
    return np.sum(pair_energy[iu])

def lj_gradient(pos):
    """Compute gradient of LJ energy."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 ** 3
    r12 = r6 ** 2
    # Force on atom i from atom j: 4 * (12/r^13 * r_vec - 6/r^7 * r_vec)
    # = 4 * r_vec * (12/r^13 - 6/r^7)
    # Gradient = -Force
    # Let's compute: for each pair (i,j), contribution to grad_i
    # E = sum_{i<j} 4*(r^-12 - r^-6)
    # dE/dx_i = sum_{j!=i} 4 * (-12*r^-13 + 6*r^-7) * (x_i - x_j)/r
    # = sum_{j!=i} 4 * (x_i - x_j) * (-12/r^14 + 6/r^8)
    # = sum_{j!=i} 4 * (x_i - x_j) / r * (-12/r^13 + 6/r^7)
    # = sum_{j!=i} 4 * (x_i - x_j) * (-12/r^14 + 6/r^8)
    
    # More carefully:
    # E_pair = 4*(r^-12 - r^-6)
    # dE_pair/dx_i = 4*(-12*r^-13*dr/dx_i + 6*r^-7*dr/dx_i)
    # dr/dx_i = (x_i - x_j)/r
    # So dE_pair/dx_i = 4*(-12*r^-13 + 6*r^-7) * (x_i - x_j)/r
    # = 4*(-12*r^-14 + 6*r^-8) * (x_i - x_j)
    
    coeff = 4.0 * (-12.0 / (r2 ** 7) + 6.0 / (r2 ** 4))
    # coeff shape: (n, n)
    # Set diagonal to 0
    np.fill_diagonal(coeff, 0.0)
    # grad_i = sum_j coeff[i,j] * (x_i - x_j)
    # = sum_j coeff[i,j] * x_i - sum_j coeff[i,j] * x_j
    # = x_i * sum_j coeff[i,j] - sum_j coeff[i,j] * x_j
    row_sum = np.sum(coeff, axis=1, keepdims=True)
    grad = row_sum * pos - coeff @ pos
    return grad

def mackay_icosahedron_55():
    """Generate Mackay icosahedron with 55 atoms (1 + 12 + 42)."""
    # The Mackay icosahedron has:
    # - 1 atom at center
    # - 12 atoms at vertices of icosahedron (radius r1)
    # - 42 atoms: 30 on edges (midpoints of icosahedron edges) + 12 on faces (face centers)
    # Actually for 55: 1 + 12 + 30 + 12 = 55
    # The 30 edge-midpoint atoms and 12 face-center atoms form the second shell
    
    # Icosahedron vertices (12 vertices)
    # Standard icosahedron vertices with edge length a:
    # (0, ±1, ±φ), (±1, ±φ, 0), (±φ, 0, ±1) where φ = golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # Unit icosahedron vertices (normalized to have circumradius 1)
    verts = []
    for x in [0, 1, phi]:
        for y in [0, 1, phi]:
            for z in [0, 1, phi]:
                if x == 0 or y == 0 or z == 0:
                    pass
                else:
                    continue
    
    # Let me use the standard construction
    # 12 vertices of regular icosahedron:
    # (0, ±1, ±φ), (±1, ±φ, 0), (±φ, 0, ±1)
    # These have circumradius sqrt(1+φ^2)
    
    r_circ = np.sqrt(1 + phi**2)
    
    vertices = []
    # (0, ±1, ±φ)
    vertices.append([0, 1, phi])
    vertices.append([0, -1, phi])
    vertices.append([0, 1, -phi])
    vertices.append([0, -1, -phi])
    # (±1, ±φ, 0)
    vertices.append([1, phi, 0])
    vertices.append([-1, phi, 0])
    vertices.append([1, -phi, 0])
    vertices.append([-1, -phi, 0])
    # (±φ, 0, ±1)
    vertices.append([phi, 0, 1])
    vertices.append([phi, 0, -1])
    vertices.append([-phi, 0, 1])
    vertices.append([-phi, 0, -1])
    
    vertices = np.array(vertices, dtype=float)
    vertices = vertices / r_circ  # Normalize to unit circumradius
    
    # Now we need to determine the radii for the two shells
    # In a Mackay icosahedron, the inner shell (12 atoms) is at radius r1
    # and the outer shell (42 atoms) is at radius r2
    # The 30 edge-midpoint atoms are at the midpoints of the icosahedron edges
    # The 12 face-center atoms are at the centers of the 20 triangular faces
    
    # Let's parameterize: inner shell radius = r1, outer shell radius = r2
    # The edge midpoints of the unit icosahedron are at distance from center:
    # Edge length of unit circumradius icosahedron: a = 2 * sin(π/5) * ... 
    # Actually, for an icosahedron with circumradius R, edge length a = R * sqrt(10 - 2*sqrt(5)) / 2 * ... 
    # Let me compute directly.
    
    # Find all edges of the icosahedron
    # Two vertices are connected if their distance equals the edge length
    # Edge length for unit circumradius icosahedron:
    d01 = np.linalg.norm(vertices[0] - vertices[1])
    edge_len = d01
    
    # Find all edges
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(vertices[i] - vertices[j])
            if np.isclose(d, edge_len, atol=1e-9):
                edges.append((i, j))
    
    # 30 edges expected
    assert len(edges) == 30, f"Expected 30 edges, got {len(edges)}"
    
    # Edge midpoints (on unit icosahedron)
    edge_midpoints = []
    for i, j in edges:
        mid = (vertices[i] + vertices[j]) / 2
        edge_midpoints.append(mid)
    edge_midpoints = np.array(edge_midpoints)
    
    # The distance from center to edge midpoints:
    edge_mid_radius = np.linalg.norm(edge_midpoints[0])
    
    # Find faces (20 triangular faces)
    # A face is a set of 3 mutually connected vertices
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                dij = np.linalg.norm(vertices[i] - vertices[j])
                djk = np.linalg.norm(vertices[j] - vertices[k])
                dki = np.linalg.norm(vertices[k] - vertices[i])
                if np.isclose(dij, edge_len, atol=1e-9) and \
                   np.isclose(djk, edge_len, atol=1e-9) and \
                   np.isclose(dki, edge_len, atol=1e-9):
                    faces.append((i, j, k))
    
    assert len(faces) == 20, f"Expected 20 faces, got {len(faces)}"
    
    # Face centers
    face_centers = []
    for i, j, k in faces:
        center = (vertices[i] + vertices[j] + vertices[k]) / 3
        face_centers.append(center)
    face_centers = np.array(face_centers)
    
    face_center_radius = np.linalg.norm(face_centers[0])
    
    # Now construct the 55-atom cluster:
    # Atom 0: center at origin
    # Atoms 1-12: inner shell at radius r1 * vertices
    # Atoms 13-42: edge midpoints at radius r2 * edge_midpoints (normalized)
    # Atoms 43-54: face centers at radius r3 * face_centers (normalized)
    
    # Actually, in the Mackay icosahedron, all outer shell atoms are at the same radius
    # The edge midpoints and face centers, when scaled, should be at the same radius
    # Let's normalize edge_midpoints and face_centers to unit vectors, then scale
    
    edge_mid_unit = edge_midpoints / np.linalg.norm(edge_midpoints, axis=1, keepdims=True)
    face_center_unit = face_centers / np.linalg.norm(face_centers, axis=1, keepdims=True)
    
    # The key parameters are r1 (inner shell radius) and r2 (outer shell radius)
    # Optimal values: r1 ≈ 1.12 (LJ minimum), r2 ≈ 2.0-2.1
    
    # Let's start with reasonable initial guesses
    r1 = 1.12
    r2 = 2.05
    
    positions = np.zeros((55, 3))
    positions[0] = [0, 0, 0]
    positions[1:13] = r1 * vertices
    positions[13:43] = r2 * edge_mid_unit
    positions[43:55] = r2 * face_center_unit
    
    return positions, r1, r2

def construct():
    global START_TIME
    START_TIME = time.time()
    
    # Generate initial Mackay icosahedron
    pos, r1, r2 = mackay_icosahedron_55()
    
    # Optimize the radii r1 and r2 first using the symmetric structure
    def symmetric_energy(params):
        r1_val, r2_val = params
        phi = (1 + np.sqrt(5)) / 2
        r_circ = np.sqrt(1 + phi**2)
        
        vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ], dtype=float)
        vertices = vertices / r_circ
        
        edges = []
        edge_len = np.linalg.norm(vertices[0] - vertices[1])
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(vertices[i] - vertices[j])
                if np.isclose(d, edge_len, atol=1e-9):
                    edges.append((i, j))
        
        edge_midpoints = np.array([(vertices[i] + vertices[j]) / 2 for i, j in edges])
        edge_mid_unit = edge_midpoints / np.linalg.norm(edge_midpoints, axis=1, keepdims=True)
        
        faces = []
        for i in range(12):
            for j in range(i+1, 12):
                for k in range(j+1, 12):
                    dij = np.linalg.norm(vertices[i] - vertices[j])
                    djk = np.linalg.norm(vertices[j] - vertices[k])
                    dki = np.linalg.norm(vertices[k] - vertices[i])
                    if np.isclose(dij, edge_len, atol=1e-9) and \
                       np.isclose(djk, edge_len, atol=1e-9) and \
                       np.isclose(dki, edge_len, atol=1e-9):
                        faces.append((i, j, k))
        
        face_centers = np.array([(vertices[i] + vertices[j] + vertices[k]) / 3 for i, j, k in faces])
        face_center_unit = face_centers / np.linalg.norm(face_centers, axis=1, keepdims=True)
        
        positions = np.zeros((55, 3))
        positions[0] = [0, 0, 0]
        positions[1:13] = r1_val * vertices
        positions[13:43] = r2_val * edge_mid_unit
        positions[43:55] = r2_val * face_center_unit
        
        return lj_energy(positions)
    
    # Optimize radii
    try:
        res = minimize(symmetric_energy, [r1, r2], method='Nelder-Mead',
                      options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
        r1_opt, r2_opt = res.x
        pos, _, _ = mackay_icosahedron_55()
        # Reconstruct with optimized radii
        phi = (1 + np.sqrt(5)) / 2
        r_circ = np.sqrt(1 + phi**2)
        vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ], dtype=float)
        vertices = vertices / r_circ
        
        edge_len = np.linalg.norm(vertices[0] - vertices[1])
        edges = []
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(vertices[i] - vertices[j])
                if np.isclose(d, edge_len, atol=1e-9):
                    edges.append((i, j))
        
        edge_midpoints = np.array([(vertices[i] + vertices[j]) / 2 for i, j in edges])
        edge_mid_unit = edge_midpoints / np.linalg.norm(edge_midpoints, axis=1, keepdims=True)
        
        faces = []
        for i in range(12):
            for j in range(i+1, 12):
                for k in range(j+1, 12):
                    dij = np.linalg.norm(vertices[i] - vertices[j])
                    djk = np.linalg.norm(vertices[j] - vertices[k])
                    dki = np.linalg.norm(vertices[k] - vertices[i])
                    if np.isclose(dij, edge_len, atol=1e-9) and \
                       np.isclose(djk, edge_len, atol=1e-9) and \
                       np.isclose(dki, edge_len, atol=1e-9):
                        faces.append((i, j, k))
        
        face_centers = np.array([(vertices[i] + vertices[j] + vertices[k]) / 3 for i, j, k in faces])
        face_center_unit = face_centers / np.linalg.norm(face_centers, axis=1, keepdims=True)
        
        pos = np.zeros((55, 3))
        pos[0] = [0, 0, 0]
        pos[1:13] = r1_opt * vertices
        pos[13:43] = r2_opt * edge_mid_unit
        pos[43:55] = r2_opt * face_center_unit
    except Exception:
        pass
    
    best_pos = pos.copy()
    best_energy = lj_energy(pos)
    
    # Now do full Cartesian relaxation
    if time.time() - START_TIME < TIME_LIMIT:
        try:
            res = minimize(lj_energy, pos, jac=lj_gradient, method='L-BFGS-B',
                         options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12})
            if res.success or res.fun < best_energy:
                best_pos = res.x
                best_energy = res.fun
        except Exception:
            pass
    
    # Try a few perturbations and relaxations if time permits
    if time.time() - START_TIME < TIME_LIMIT - 30:
        rng = np.random.default_rng(42)
        for trial in range(5):
            if time.time() - START_TIME >= TIME_LIMIT - 10:
                break
            # Perturb the current best
            perturbed = best_pos + rng.normal(0, 0.05, size=(55, 3))
            try:
                res = minimize(lj_energy, perturbed, jac=lj_gradient, method='L-BFGS-B',
                             options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-11})
                if res.fun < best_energy:
                    best_energy = res.fun
                    best_pos = res.x
            except Exception:
                pass
    
    # Final check: ensure minimum distance >= 0.5
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, 1e10)
    min_dist = np.min(dists)
    
    if min_dist < 0.5:
        # Scale up slightly
        scale = 0.5 / min_dist * 1.01
        best_pos = best_pos * scale
        best_energy = lj_energy(best_pos)
    
    return best_pos
# EVOLVE-BLOCK-END

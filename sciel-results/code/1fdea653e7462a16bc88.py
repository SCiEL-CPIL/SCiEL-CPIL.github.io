"""Improved program for the suite task; constructs a Mackay icosahedron and relaxes it."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55

# EVOLVE-BLOCK-START

def _lennard_jones_energy(pos):
    """Compute total LJ energy for N atoms."""
    n = pos.shape[0]
    # Compute all pair distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    # Avoid division by zero (diagonal)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    energy = 4.0 * np.sum(r12 - r6)
    # Subtract the N self-interaction terms (which we set to r=1, so energy per self-term is 4*(1-1)=0)
    # Actually 4*(1^-12 - 1^-6) = 4*(1-1) = 0, so no correction needed
    return 0.5 * energy  # Each pair counted twice

def _lennard_jones_energy_and_grad(pos):
    """Compute total LJ energy and gradient for N atoms."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    # energy = 4 * sum over pairs (r^-12 - r^-6)
    # Using the full matrix, each pair is counted twice, so multiply by 0.5
    energy = 2.0 * np.sum(r12 - r6)
    
    # Gradient: d/dr of 4*(r^-12 - r^-6) = 4*(-12*r^-13 + 6*r^-7)
    # In terms of r2: dE/dr2 = 4*(-6*r^-14 + 3*r^-8) = 24*r^-8 - 24*r^-14
    # grad_i = sum_j 24*(r_ij^-8 - r_ij^-14) * (pos_i - pos_j)
    # Wait, let me be more careful.
    # E = 4 * sum_{i<j} (r_ij^-12 - r_ij^-6)
    # dE/dx_i = 4 * sum_{j != i} (-12 * r_ij^-14 + 6 * r_ij^-8) * (x_i - x_j)
    # In terms of r2: r^-14 = r2^-7, r^-8 = r2^-4
    # dE/dx_i = 4 * sum_{j != i} (-12 * r2_ij^-7 + 6 * r2_ij^-4) * (x_i - x_j)
    
    # Using the full matrix approach (each pair counted twice):
    # grad = 2 * sum over all j != i of 4 * (-12 * r2^-7 + 6 * r2^-4) * (x_i - x_j)
    # Wait, let me just compute it directly.
    
    # coefficient for each pair (i,j) with i != j
    coeff = 4.0 * (-12.0 * r2 ** (-7) + 6.0 * r2 ** (-4))
    np.fill_diagonal(coeff, 0.0)
    
    # grad_i = sum_j coeff[i,j] * (x_i - x_j) = sum_j coeff[i,j] * diff[i,j]
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad

def _make_icosahedron():
    """Construct a Mackay icosahedron with 55 atoms (1+12+42)."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0  # golden ratio
    
    # Shell 0: 1 atom at origin
    shell0 = np.array([[0.0, 0.0, 0.0]])
    
    # Shell 1: 12 vertices of icosahedron
    # Icosahedron vertices: (0, ±1, ±phi), (±1, ±phi, 0), (±phi, 0, ±1)
    verts = []
    for a in [1, -1]:
        for b in [1, -1]:
            verts.append([0.0, a, b * phi])
            verts.append([a, b * phi, 0.0])
            verts.append([a * phi, 0.0, b])
    shell1 = np.array(verts, dtype=float)
    # Normalize to unit length
    norms = np.linalg.norm(shell1, axis=1, keepdims=True)
    shell1 = shell1 / norms
    
    # Shell 2: 42 atoms
    # The second shell of the Mackay icosahedron consists of:
    # - 30 atoms at edge midpoints direction, scaled
    # - 12 atoms at face centers direction, scaled
    # 
    # For a Mackay icosahedron with shell radii 1 and 2 (in units of first shell radius):
    # The 30 edge-midpoint directions are normalized edge midpoints
    # The 12 face-center directions are normalized face normals
    #
    # Let's construct them properly.
    # 
    # Icosahedron has 30 edges and 20 faces.
    # Edge midpoints: for each edge (v_i, v_j), midpoint direction = normalize(v_i + v_j)
    # Face centers: for each face (v_i, v_j, v_k), normal direction = normalize(cross(v_i, v_j))
    
    # Get all unique edges
    edges = set()
    for i in range(12):
        for j in range(i + 1, 12):
            # Check if they form an edge (distance between adjacent vertices)
            d = np.linalg.norm(shell1[i] - shell1[j])
            if d < 0.5:  # threshold for adjacency
                edges.add((i, j))
    
    # Actually, let me just use known directions for the 42 second-shell atoms.
    # The Mackay icosahedron second shell has:
    # 30 atoms along edge directions (midpoints of icosahedron edges), at radius 2 * (edge_midpoint_length / vertex_length)
    # 12 atoms along face normal directions, at radius 2 * (face_center_distance / vertex_length)
    
    # For a regular icosahedron with vertices at distance 1 from center:
    # Edge length = 2 * sin(pi/5) ≈ 1.1756 (wait, let me compute)
    # Actually for our unit icosahedron, edge length = distance between adjacent vertices
    
    # Let me just compute the 30 edge midpoint directions and 12 face center directions
    
    # Find all edges (pairs of vertices at minimum distance)
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            dists[i, j] = np.linalg.norm(shell1[i] - shell1[j])
    
    min_dist = np.min(dists[dists > 0.01])
    
    edge_pairs = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                edge_pairs.append((i, j))
    
    # 30 edge midpoint directions
    edge_dirs = []
    for i, j in edge_pairs:
        mid = shell1[i] + shell1[j]
        norm = np.linalg.norm(mid)
        if norm > 1e-10:
            edge_dirs.append(mid / norm)
    edge_dirs = np.array(edge_dirs)
    
    # 12 face center directions (face normals)
    # Find all triangular faces
    face_dirs = []
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                # Check if i, j, k form a face (all three pairwise distances = min_dist)
                if (abs(dists[i, j] - min_dist) < 1e-6 and
                    abs(dists[j, k] - min_dist) < 1e-6 and
                    abs(dists[i, k] - min_dist) < 1e-6):
                    normal = np.cross(shell1[j] - shell1[i], shell1[k] - shell1[i])
                    norm = np.linalg.norm(normal)
                    if norm > 1e-10:
                        face_dirs.append(normal / norm)
    
    # Deduplicate face directions (each face might be found multiple times with different orientations)
    # Keep only unique directions
    unique_face_dirs = []
    for fd in face_dirs:
        is_dup = False
        for ud in unique_face_dirs:
            if abs(np.dot(fd, ud)) > 0.999:  # same or opposite direction
                is_dup = True
                break
        if not is_dup:
            unique_face_dirs.append(fd)
    
    unique_face_dirs = np.array(unique_face_dirs)
    
    # For Mackay icosahedron, the second shell radius is 2x the first shell radius
    # The edge midpoint atoms are at radius 2 * |midpoint| / |vertex|
    # For unit vertices, edge midpoint distance from center = |v_i + v_j| / 2 ... no
    # The midpoint of two unit vectors v_i, v_j has distance |v_i + v_j| / 2 from center? No.
    # Midpoint = (v_i + v_j) / 2, distance from origin = |v_i + v_j| / 2
    # |v_i + v_j|^2 = |v_i|^2 + |v_j|^2 + 2*v_i.v_j = 2 + 2*cos(theta)
    # For icosahedron edge, cos(theta) = (|v_i - v_j|^2 - 2) / 2 = (min_dist^2 - 2) / 2
    # So |v_i + v_j| = sqrt(2 + 2*cos(theta)) = sqrt(4 - min_dist^2)
    # Midpoint distance = sqrt(4 - min_dist^2) / 2
    
    # For Mackay icosahedron, the second shell is at radius 2 (in units of first shell = 1)
    # So we scale the edge midpoint directions to radius 2
    # And face center directions to radius 2
    
    # Actually, in the Mackay icosahedron, the radial distances are:
    # Shell 1: r = 1 (12 atoms)
    # Shell 2: r = 2 (42 atoms) - but the 30 edge atoms and 12 face atoms are all at r = 2
    
    shell2 = np.vstack([edge_dirs * 2.0, unique_face_dirs * 2.0])
    
    # Combine all shells
    all_atoms = np.vstack([shell0, shell1, shell2])
    
    # Scale to LJ minimum distance
    # The nearest neighbor distance in the icosahedron should be ~1.122
    # In our construction, the first shell atoms are at distance 1 from center
    # The distance from center to shell 1 atom = 1.0
    # We want nearest-neighbor distance = 1.122
    # The nearest neighbors of the central atom are the 12 shell-1 atoms at distance 1.0
    # So we scale by 1.122
    scale = 1.122
    all_atoms = all_atoms * scale
    
    return all_atoms

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    # Construct Mackay icosahedron
    pos = _make_icosahedron()
    
    # Verify we have exactly 55 atoms
    if pos.shape[0] != N:
        # Fallback: use a simpler construction
        # Just create 55 points in a dense arrangement
        pos = np.random.randn(N, 3) * 2.0
        # Center it
        pos -= np.mean(pos, axis=0)
    
    # Ensure minimum distance constraint
    def _enforce_min_dist(p, min_d=0.5):
        """Nudge atoms to satisfy minimum distance."""
        p = p.copy()
        for _ in range(100):
            moved = False
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.linalg.norm(p[i] - p[j])
                    if d < min_d:
                        # Push apart
                        if d < 1e-10:
                            p[i] += np.random.randn(3) * 0.1
                        else:
                            direction = (p[i] - p[j]) / d
                            shift = (min_d - d) / 2.0 + 0.01
                            p[i] += direction * shift
                            p[j] -= direction * shift
                        moved = True
            if not moved:
                break
        return p
    
    pos = _enforce_min_dist(pos)
    
    # Relax using L-BFGS
    best_pos = pos.copy()
    best_energy = _lennard_jones_energy(pos)
    
    try:
        result = minimize(
            _lennard_jones_energy_and_grad,
            pos,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 5000}
        )
        if result.success or result.fun < best_energy:
            pos_relaxed = result.x
            # Enforce minimum distance after relaxation
            pos_relaxed = _enforce_min_dist(pos_relaxed)
            e_relaxed = _lennard_jones_energy(pos_relaxed)
            if e_relaxed < best_energy:
                best_pos = pos_relaxed
                best_energy = e_relaxed
    except Exception:
        pass
    
    # Check time
    if time.time() < deadline - 5.0:
        # Try a few more restarts with small perturbations if time permits
        rng = np.random.RandomState(42)
        for restart in range(5):
            if time.time() > deadline - 10.0:
                break
            perturbed = best_pos + rng.randn(N, 3) * 0.05
            perturbed = _enforce_min_dist(perturbed)
            try:
                result = minimize(
                    _lennard_jones_energy_and_grad,
                    perturbed,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 3000}
                )
                if result.fun < best_energy:
                    p_new = _enforce_min_dist(result.x)
                    e_new = _lennard_jones_energy(p_new)
                    if e_new < best_energy:
                        best_pos = p_new
                        best_energy = e_new
            except Exception:
                pass
    
    return best_pos

# EVOLVE-BLOCK-END

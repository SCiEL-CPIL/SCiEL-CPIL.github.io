"""Lennard-Jones 55-atom cluster: Mackay icosahedron + local relaxation."""
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
    r = np.sqrt(r2)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    # Force on atom i from atom j: 4*(12*r^-12 - 6*r^-6) * r_vec / r
    # Gradient of energy w.r.t. pos[i] = -force_i
    # E = 4*sum(r^-12 - r^-6)
    # dE/dpos[i] = 4 * sum_j ( -12*r^-13 + 6*r^-7 ) * (pos[i] - pos[j]) / r
    #           = 4 * sum_j ( -12*r^-14 + 6*r^-8 ) * (pos[i] - pos[j])
    # Let's use: dE/dpos[i] = 24 * sum_j (2*r^-14 - r^-8) * (pos[i] - pos[j])
    # Actually: dE/dr = 4*(-12*r^-13 + 6*r^-7)
    # dE/dpos[i] = dE/dr * dr/dpos[i] = 4*(-12*r^-13 + 6*r^-7) * (pos[i]-pos[j])/r
    #            = 4*(-12*r^-14 + 6*r^-8) * (pos[i]-pos[j])
    #            = 24*(2*r^-14 - r^-8) * (pos[i]-pos[j])
    # Wait, 4*(-12*r^-14 + 6*r^-8) = -48*r^-14 + 24*r^-8 = 24*(r^-8 - 2*r^-14)
    
    # Let me recompute carefully.
    # E_pair = 4*(r^-12 - r^-6)
    # dE_pair/dr = 4*(-12*r^-13 + 6*r^-7)
    # dr/dpos[i] = (pos[i] - pos[j]) / r
    # dE_pair/dpos[i] = 4*(-12*r^-13 + 6*r^-7) * (pos[i] - pos[j]) / r
    #                 = 4*(-12*r^-14 + 6*r^-8) * (pos[i] - pos[j])
    #                 = (-48*r^-14 + 24*r^-8) * (pos[i] - pos[j])
    
    # For the total energy, we sum over all pairs. Each pair (i,j) contributes to both dE/dpos[i] and dE/dpos[j].
    # Using the matrix form:
    # coeff[i,j] = -48*r^-14 + 24*r^-8  for i != j
    # grad[i] = sum_j coeff[i,j] * (pos[i] - pos[j])
    
    r8 = r6 * r6  # r^-12... no, r6 = r^-6, so r8 should be r^-8
    # Let me recompute powers
    # r2 = r^2
    # r^-6 = r2^-3
    # r^-8 = r2^-4
    # r^-14 = r2^-7
    
    r2_safe = r2.copy()
    np.fill_diagonal(r2_safe, 1.0)  # avoid division by zero, we'll zero out diagonal later
    
    r2_neg3 = r2_safe ** (-3)  # r^-6
    r2_neg4 = r2_safe ** (-4)  # r^-8
    r2_neg7 = r2_safe ** (-7)  # r^-14
    
    # coeff[i,j] = -48*r^-14 + 24*r^-8
    coeff = -48.0 * r2_neg7 + 24.0 * r2_neg4
    np.fill_diagonal(coeff, 0.0)
    
    # grad[i] = sum_j coeff[i,j] * (pos[i] - pos[j])
    # = sum_j coeff[i,j] * pos[i] - sum_j coeff[i,j] * pos[j]
    # = pos[i] * sum_j coeff[i,j] - (coeff @ pos)[i]
    row_sums = coeff.sum(axis=1, keepdims=True)
    grad = row_sums * pos - coeff @ pos
    
    # Energy
    r6_matrix = r2_safe ** (-3)
    r12_matrix = r6_matrix ** 2
    np.fill_diagonal(r6_matrix, 0.0)
    np.fill_diagonal(r12_matrix, 0.0)
    e = 4.0 * (r12_matrix - r6_matrix).sum() / 2.0
    
    return e, grad

def make_mackay_icosahedron():
    """Generate Mackay icosahedron with 55 atoms."""
    # Mackay icosahedron: 1 center + 12 inner shell + 42 outer shell
    # Inner shell radius: a, outer shell radius: a * sqrt(5/3) ≈ a * 1.291
    # Standard Mackay icosahedron uses specific distances
    
    # Icosahedron vertices (12 points)
    # Using the standard icosahedron with vertices at:
    # (0, ±1, ±φ), (±1, ±φ, 0), (±φ, 0, ±1)
    # where φ = (1+sqrt(5))/2
    
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of icosahedron
    verts = []
    for x, y, z in [(0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
                     (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
                     (phi, 0, 1), (-phi, 0, 1), (phi, 0, -1), (-phi, 0, -1)]:
        verts.append([x, y, z])
    verts = np.array(verts, dtype=float)
    
    # Normalize to unit length
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Inner shell: 12 atoms at radius a
    # Outer shell: 42 atoms
    # For Mackay icosahedron, the inner shell is at radius a and outer at a*sqrt(5/3)
    # But we need to determine the actual positions
    
    # The 42 outer shell atoms consist of:
    # - 30 atoms at face centers of the icosahedron faces (scaled)
    # - 12 atoms at edge midpoints (scaled)
    # Actually, the Mackay icosahedron has:
    # Shell 1: 12 atoms (icosahedron vertices)
    # Shell 2: 30 atoms (face centers) + 12 atoms (edge midpoints) = 42 atoms
    
    # Let's construct the outer shell properly
    # The icosahedron has 20 faces and 30 edges
    
    # For a proper Mackay icosahedron, the inner shell radius is a and outer is a*sqrt(5/3)
    # Let's set a = 1.0 for now and scale later
    
    # Inner shell positions
    inner = verts.copy()  # 12 atoms
    
    # For the outer shell, we need face centers and edge midpoints
    # Icosahedron faces: each face is a triangle
    # We need to find all 20 faces
    
    # The icosahedron adjacency can be determined by distances
    # Let's use a known construction
    
    # Alternative: use the known coordinates for Mackay icosahedron
    # From literature, the 55-atom Mackay icosahedron has:
    # - 1 atom at center
    # - 12 atoms at distance a (icosahedron)
    # - 30 atoms at distance a*sqrt(5/3) (face centers, scaled)
    # - 12 atoms at distance a*sqrt(5/3) (edge midpoints, scaled)
    
    # Let me use a simpler approach: generate the icosahedron and its dual structure
    
    # Actually, let me just use the standard Mackay icosahedron coordinates
    # The ratio of outer to inner shell radius is sqrt(5/3)
    
    # For simplicity, let's construct:
    # Center: (0,0,0)
    # Inner shell: 12 icosahedron vertices at radius 1
    # Outer shell: 42 points at radius sqrt(5/3)
    
    # The 42 outer points are the face centers (30) and edge midpoints (12) of the icosahedron,
    # but scaled to be at radius sqrt(5/3)
    
    # Let's find the faces and edges of the icosahedron
    # Compute pairwise distances
    from scipy.spatial import ConvexHull
    
    hull = ConvexHull(inner)
    faces = hull.simplices  # 20 faces, each with 3 vertices
    
    # Face centers
    face_centers = []
    for face in faces:
        fc = inner[face].mean(axis=0)
        face_centers.append(fc)
    face_centers = np.array(face_centers)
    # Normalize and scale
    fc_norms = np.linalg.norm(face_centers, axis=1, keepdims=True)
    face_centers = face_centers / fc_norms * np.sqrt(5.0/3.0)
    
    # Edge midpoints
    # Find all edges from the hull
    edges = set()
    for face in faces:
        for i in range(3):
            j = (i + 1) % 3
            edge = tuple(sorted([face[i], face[j]]))
            edges.add(edge)
    
    edge_midpoints = []
    for e in edges:
        em = (inner[e[0]] + inner[e[1]]) / 2.0
        edge_midpoints.append(em)
    edge_midpoints = np.array(edge_midpoints)
    # Normalize and scale
    em_norms = np.linalg.norm(edge_midpoints, axis=1, keepdims=True)
    edge_midpoints = edge_midpoints / em_norms * np.sqrt(5.0/3.0)
    
    # Combine all positions
    center = np.array([[0.0, 0.0, 0.0]])
    outer = np.vstack([face_centers, edge_midpoints])
    
    positions = np.vstack([center, inner, outer])
    
    # Scale so that the nearest-neighbor distance is around 1.12 (LJ minimum)
    # The inner shell atoms are at distance 1 from center, and inner-outer distance is sqrt(5/3) - 1... 
    # Actually, let's compute the minimum distance and scale accordingly
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    min_dist = np.sqrt(r2.min())
    
    # Scale so min distance is ~1.12
    scale = 1.12 / min_dist
    positions = positions * scale
    
    return positions

def construct():
    """Construct initial positions for 55-atom LJ cluster."""
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    # Start with Mackay icosahedron
    pos = make_mackay_icosahedron()
    
    # Relax using L-BFGS
    best_pos = pos.copy()
    best_energy = lj_energy(pos)
    
    # First relaxation
    result = minimize(
        lj_energy_grad, 
        pos, 
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    if result.fun < best_energy:
        best_energy = result.fun
        best_pos = result.x.copy()
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 20.0:
        return best_pos
    
    # Try multiple random restarts with small perturbations
    rng = np.random.default_rng(42)
    n_restarts = 5
    
    for i in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # Perturb the best position
        perturbation = rng.normal(0, 0.1, size=best_pos.shape)
        trial_pos = best_pos + perturbation
        
        # Relax
        result = minimize(
            lj_energy_grad,
            trial_pos,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.copy()
    
    return best_pos

# EVOLVE-BLOCK-END

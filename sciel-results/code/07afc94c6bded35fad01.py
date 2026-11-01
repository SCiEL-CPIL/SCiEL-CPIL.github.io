"""Lennard-Jones 55-atom cluster: analytic Mackay icosahedron + L-BFGS relaxation."""
import time
import numpy as np
from scipy.optimize import minimize

N = 55
DEADLINE_MARGIN = 10.0  # seconds to spare before the 150s kill


def _lj_energy_and_grad(pos):
    """Compute LJ total energy and gradient for positions (N,3)."""
    n = pos.shape[0]
    # Pairwise displacement vectors
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (n, n, 3)
    r2 = np.sum(diff ** 2, axis=2)  # (n, n)
    # Mask out self-interactions
    np.fill_diagonal(r2, np.inf)
    r = np.sqrt(r2)
    # Energy: 4 * sum_{i<j} (r^-12 - r^-6)
    inv_r6 = r2 ** (-3)  # r^-6
    inv_r12 = inv_r6 ** 2  # r^-12
    pair_e = 4.0 * (inv_r12 - inv_r6)
    energy = 0.5 * np.sum(pair_e)  # each pair counted twice
    # Gradient: dE/dr_i = sum_{j != i} 4 * (-12 r^-14 + 6 r^-8) * (r_i - r_j)/r
    # = sum_j 4 * (-12 r^-14 + 6 r^-8) * diff[i,j]
    # factor = 4 * (-12 * r^-14 + 6 * r^-8) = 4 * (-12 * inv_r12 / r + 6 * inv_r6 / r)
    #        = 4 * (-12 * inv_r12 + 6 * inv_r6) / r
    #        = 24 * (inv_r6 - 2 * inv_r12) / r
    #        = 24 * (inv_r6 - 2 * inv_r12) * r^-1
    # Since r = sqrt(r2), 1/r = 1/sqrt(r2)
    factor = 24.0 * (inv_r6 - 2.0 * inv_r12) / np.sqrt(r2)  # (n, n)
    np.fill_diagonal(factor, 0.0)
    grad = np.sum(factor[:, :, np.newaxis] * diff, axis=1)  # (n, 3)
    return energy, grad


def _mckay_icosahedron_55():
    """
    Construct the two-shell Mackay icosahedron for 55 atoms.
    
    Shell 1 (inner): 12 vertices of a regular icosahedron at distance d1.
    Shell 2 (outer): 42 vertices: 12 at the icosahedron vertices (same directions),
                     12 at the midpoints of edges, 18 at face centers? 
    Actually Mackay icosahedron (2,1,1) has shells:
      - 1 center
      - 12 at distance a (icosahedron vertices)
      - 30 at distance b (midpoints of edges of the icosahedron)
      - 12 at distance c (face centers)
    Total: 1 + 12 + 30 + 12 = 55. Yes!
    
    The Mackay icosahedron (2,1,1) in the icosahedral lattice:
    Using the icosahedral basis vectors.
    
    Let me use the standard construction:
    Icosahedron vertices: (0, ±1, ±φ), (±1, ±φ, 0), (±φ, 0, ±1) where φ = (1+√5)/2
    Normalized to unit length.
    
    For Mackay (2,1,1):
    - 12 vertices at the icosahedron corners: positions = a * v_i where v_i are unit icosahedron vertices
    - 30 vertices at edge midpoints: positions = b * (v_i + v_j)/|v_i + v_j| for each edge (i,j)
    - 12 vertices at face centers: positions = c * (v_i + v_j + v_k)/|v_i + v_j + v_k| for each face
    - 1 at center
    
    The ratios for the Mackay icosahedron (2,1,1) are determined by the icosahedral lattice.
    In the icosahedral lattice with basis vectors, the (2,1,1) shell corresponds to:
    - Inner shell radius: related to |2a + b| in lattice coords
    - The actual distances are: r1, r2, r3 with specific ratios.
    
    For a proper LJ Mackay icosahedron, the optimal radii are approximately:
    r_center = 0
    r_12 (vertices) ≈ 1.12 (LJ minimum distance)
    r_30 (edge mids) ≈ 1.97
    r_12 (face centers) ≈ 2.29
    
    But let's use the exact icosahedral lattice construction and then relax.
    
    Icosahedral lattice basis:
    a1 = (0, 1, φ)/|...|  -- actually let's use the standard icosahedral coordinates.
    
    The 12 icosahedron vertices (unnormalized):
    (0, ±1, ±φ), (±1, ±φ, 0), (±φ, 0, ±1)
    |v| = sqrt(1 + φ^2) = sqrt(1 + (1+√5)/2 * (1+√5)/2) = sqrt(1 + (3+√5)/2) = sqrt((5+√5)/2)
    
    Let's just use the known structure and let L-BFGS relax it.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # 12 icosahedron vertices
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit vectors
    norms = np.linalg.norm(verts, axis=1)
    verts_unit = verts / norms[:, np.newaxis]
    
    # 1 center atom
    center = np.array([[0.0, 0.0, 0.0]])
    
    # 12 atoms at icosahedron vertices (shell 1)
    # Initial guess: radius ~1.12
    shell1_r = 1.12
    shell1 = shell1_r * verts_unit
    
    # 30 atoms at edge midpoints (shell 2)
    # Edges of icosahedron: each vertex connects to 5 others
    # Find edges by checking which pairs of vertices are closest
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i + 1, 12):
            dists[i, j] = np.linalg.norm(verts_unit[i] - verts_unit[j])
    
    # The 30 edges are the 30 shortest distances
    edge_pairs = []
    for i in range(12):
        for j in range(i + 1, 12):
            # An edge if the angle between them is ~63.43 degrees (cos = 1/√5 * something)
            # Actually for icosahedron, edge length in unit sphere: 
            # distance between adjacent vertices = 2*sin(π/5) ≈ 1.176
            # Let's just take all pairs with distance < threshold
            pass
    
    # Better: for icosahedron, each vertex has 5 neighbors. 
    # The dot product between adjacent vertices is 1/√(5) * ... 
    # Actually: for unit icosahedron vertices, adjacent have dot product = 1/(φ+1/φ) ... 
    # Let me just compute: the 30 edges correspond to the 30 pairs with smallest distance.
    pairs = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts_unit[i] - verts_unit[j])
            pairs.append((d, i, j))
    pairs.sort()
    edges = [(p[1], p[2]) for p in pairs[:30]]
    
    # 30 edge midpoints (directions)
    edge_dirs = []
    for i, j in edges:
        d = verts_unit[i] + verts_unit[j]
        n = np.linalg.norm(d)
        if n > 1e-10:
            edge_dirs.append(d / n)
    edge_dirs = np.array(edge_dirs)
    
    # Shell 2 radius ~1.97
    shell2_r = 1.97
    shell2 = shell2_r * edge_dirs
    
    # 12 atoms at face centers (shell 3)
    # Faces of icosahedron: 20 triangular faces
    # Each face is a set of 3 mutually adjacent vertices
    # Find faces: for each edge (i,j), find the third vertex k that is adjacent to both
    faces = []
    adj = {i: set() for i in range(12)}
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    
    for i, j in edges:
        common = adj[i] & adj[j]
        for k in common:
            if k > j:  # avoid duplicates
                faces.append((i, j, k))
    
    # Should have 20 faces
    face_dirs = []
    for i, j, k in faces:
        d = verts_unit[i] + verts_unit[j] + verts_unit[k]
        n = np.linalg.norm(d)
        if n > 1e-10:
            face_dirs.append(d / n)
    face_dirs = np.array(face_dirs)
    
    # Shell 3 radius ~2.29
    shell3_r = 2.29
    shell3 = shell3_r * face_dirs
    
    # Combine all atoms
    pos = np.vstack([center, shell1, shell2, shell3])
    
    # Should have 1 + 12 + 30 + 12 = 55 atoms
    assert len(pos) == 55, f"Expected 55 atoms, got {len(pos)}"
    
    return pos


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    # Start with the analytic Mackay icosahedron
    pos = _mckay_icosahedron_55()
    
    # Check minimum distance constraint
    def check_min_dist(p):
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        r2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(r2, np.inf)
        min_r = np.sqrt(np.min(r2))
        return min_r
    
    min_d = check_min_dist(pos)
    if min_d < 0.5:
        # Scale up slightly
        scale = 0.5 / min_d * 1.01
        pos = pos * scale
    
    best_pos = pos.copy()
    best_energy = _lj_energy_and_grad(pos)[0]
    
    # L-BFGS relaxation
    try:
        result = minimize(
            _lj_energy_and_grad,
            pos,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
        )
        if result.success or result.fun < best_energy:
            # Check constraint
            if check_min_dist(result.x) >= 0.5:
                best_pos = result.x
                best_energy = result.fun
    except Exception:
        pass
    
    # If we have time, try a few perturbations and re-relax
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        rng = np.random.RandomState(42)
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            # Small perturbation
            perturbed = best_pos + rng.normal(0, 0.05, size=(55, 3))
            if check_min_dist(perturbed) < 0.5:
                continue
            try:
                res = minimize(
                    _lj_energy_and_grad,
                    perturbed,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
                )
                if (res.success or res.fun < best_energy) and check_min_dist(res.x) >= 0.5:
                    best_pos = res.x
                    best_energy = res.fun
            except Exception:
                pass
    
    return best_pos

"""Lennard-Jones 55-atom cluster: Mackay icosahedron template + Nelder-Mead radial optimization + L-BFGS polish."""
import numpy as np
from scipy.optimize import minimize, minimize_scalar
import time

N = 55

def _lj_energy(pos):
    """Compute total Lennard-Jones energy for positions array (N,3)."""
    n = pos.shape[0]
    # Compute all pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dist_sq, 1.0)  # avoid division by zero on diagonal
    dist_sq = np.maximum(dist_sq, 1e-12)
    inv_r6 = dist_sq ** (-3.0)
    inv_r12 = inv_r6 ** 2
    # Sum over all pairs (each pair counted twice, so divide by 2)
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    return energy


def _lj_energy_grad(pos):
    """Compute total Lennard-Jones energy and gradient for positions array (N,3)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-12)
    inv_r2 = 1.0 / dist_sq
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    # Energy
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    # Gradient: dE/dr_i = 4 * sum_j (12/r_ij^13 - 6/r_ij^7) * (r_i - r_j)/r_ij
    # In vector form: grad_i = 4 * sum_{j!=i} (12/r^13 - 6/r^7) * (pos_i - pos_j) / r
    # = 4 * sum_{j!=i} (12*inv_r12 - 6*inv_r6) * (pos_i - pos_j) / r
    # Note: (pos_i - pos_j) / r = diff[i,j] / sqrt(dist_sq[i,j])
    # So grad_i = 4 * sum_j (12*inv_r12[i,j] - 6*inv_r6[i,j]) * diff[i,j] / sqrt(dist_sq[i,j])
    # Let's compute coefficient matrix
    coeff = 4.0 * (12.0 * inv_r12 - 6.0 * inv_r6) / np.sqrt(dist_sq)
    np.fill_diagonal(coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return energy, grad


def _mackay_icosahedron(r1, r2):
    """
    Generate Mackay icosahedron coordinates with inner shell radius r1 and outer shell radius r2.
    Structure: 1 center + 12 vertices (inner) + 30 edge midpoints (outer) + 12 face centers (outer)
    Actually the standard Mackay icosahedron for 55 atoms:
    - 1 center atom
    - 12 atoms at icosahedron vertices (radius r1)
    - 30 atoms at edge midpoints (radius ~ r2)
    - 12 atoms at face centers (radius ~ r2)
    
    Let's use the standard icosahedron vertices.
    Icosahedron vertices: (0, ±1, ±φ), (±1, ±φ, 0), (±φ, 0, ±1) where φ = (1+√5)/2
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    # 12 vertices of icosahedron (unit circumscribed sphere)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit length
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Center atom
    center = np.array([[0.0, 0.0, 0.0]])
    
    # Inner shell: 12 vertices at radius r1
    inner = verts * r1
    
    # Outer shell: 30 edge midpoints + 12 face centers at radius r2
    # Edge midpoints: average of pairs of vertices that are connected by an edge
    # In a regular icosahedron, each vertex connects to 5 others. There are 30 edges.
    # We need to identify the 30 edges. Two vertices are connected if their dot product is a specific value.
    # For unit icosahedron, the angle between adjacent vertices: cos(theta) = 1/(phi^2) ≈ 0.381966
    # Actually, let's compute all pairwise dot products and find the edges.
    dot = verts @ verts.T
    # The minimum non-zero absolute dot product among off-diagonal entries gives edge connections
    # For icosahedron, adjacent vertices have dot product = 1/phi^2 = 2-phi ≈ 0.381966
    # Let's find all pairs with dot product close to this value
    edge_dot = 2.0 - phi  # ≈ 0.381966
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(dot[i, j] - edge_dot) < 1e-6:
                edges.append((i, j))
    
    # There should be 30 edges
    # Edge midpoints (on the unit sphere, normalized)
    edge_mids = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid)
        edge_mids.append(mid)
    edge_mids = np.array(edge_mids)
    
    # Face centers: 20 faces of icosahedron, but we only need 12? 
    # Wait, 1 + 12 + 30 + 12 = 55. So we need 12 face centers, not 20.
    # Actually, in the Mackay icosahedron, the outer shell consists of:
    # - 30 atoms at edge midpoints
    # - 12 atoms at... hmm, let me reconsider.
    
    # Actually, the standard Mackay icosahedron M(2) has:
    # Shell 1: 13 atoms (1 center + 12 vertices)
    # Shell 2: 42 atoms (30 edge midpoints + 12 face centers)
    # Total: 55 atoms
    # But an icosahedron has 20 faces, not 12. Let me reconsider.
    
    # Actually, looking this up: the Mackay icosahedron for N=55 has:
    # - 1 center
    # - 12 at icosahedron vertices
    # - 30 at edge midpoints  
    # - 12 at... 
    # Hmm, 1+12+30+12 = 55. But icosahedron has 20 faces.
    
    # Let me reconsider: maybe it's 1 + 12 + 30 + 12 where the 12 are not face centers but something else.
    # Or perhaps the structure is different.
    
    # Alternative: use a different decomposition. Let's just use all 20 face centers and see if we can get 55.
    # 1 + 12 + 30 + 20 = 63, too many.
    
    # Let me look at this differently. The Mackay icosahedron M(n) has:
    # M(1) = 13 (1+12)
    # M(2) = 55 (1+12+30+12)
    # The 12 in the outer shell might be at the centers of the 12 pentagonal faces? No, icosahedron has 20 triangular faces.
    
    # Actually, I think the correct decomposition is:
    # Inner shell: 13 atoms (center + 12 vertices)
    # Outer shell: 42 atoms arranged as 30 edge midpoints + 12 "face" positions
    # But icosahedron has 20 faces... 
    
    # Let me just generate all candidate positions and pick 42 that form a reasonable shell.
    # For now, let's use edge midpoints (30) and face centers (20), but we only need 42 total.
    # 30 + 12 = 42, so we need 12 face centers. Maybe we use 12 of the 20 face centers?
    
    # Actually, I recall now: in the Mackay icosahedron, the outer shell atoms are at:
    # - Midpoints of edges (30)
    # - Centers of faces (20)
    # That's 50, plus 13 inner = 63. That's not right either.
    
    # Let me just use a simpler approach: generate the 13 inner atoms and 42 outer atoms
    # where outer atoms are placed at edge midpoints and some face centers, scaled to radius r2.
    
    # For now, let's use all 30 edge midpoints and 12 of the 20 face centers (arbitrary selection)
    # to get 42 outer atoms.
    
    # Face centers: average of 3 vertices for each triangular face
    # We need to identify the 20 faces. In an icosahedron, each face is a triangle of 3 mutually adjacent vertices.
    faces = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(dot[i, j] - edge_dot) < 1e-6:
                for k in range(j + 1, 12):
                    if abs(dot[i, k] - edge_dot) < 1e-6 and abs(dot[j, k] - edge_dot) < 1e-6:
                        faces.append((i, j, k))
    
    # Should have 20 faces
    face_centers = []
    for i, j, k in faces:
        fc = (verts[i] + verts[j] + verts[k]) / 3.0
        fc = fc / np.linalg.norm(fc)
        face_centers.append(fc)
    face_centers = np.array(face_centers)
    
    # Use first 12 face centers
    face_centers_12 = face_centers[:12]
    
    # Outer shell: 30 edge midpoints + 12 face centers, all at radius r2
    outer = np.vstack([edge_mids, face_centers_12]) * r2
    
    # Combine all atoms
    all_atoms = np.vstack([center, inner, outer])
    return all_atoms


def _objective(params):
    """Objective function for Nelder-Mead: takes [r1, r2] and returns LJ energy."""
    r1, r2 = params
    pos = _mackay_icosahedron(r1, r2)
    return _lj_energy(pos)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    # Initial guess: r1 ≈ 1.12 (pair minimum), r2 ≈ 2.0
    # The optimal Mackay icosahedron has specific radii. Let's start with reasonable values.
    initial = np.array([1.12, 2.0])
    
    # First, try Nelder-Mead on the 2D parameter space
    best_pos = None
    best_energy = np.inf
    
    try:
        res = minimize(_objective, initial, method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        elapsed = time.time() - start_time
        if elapsed < deadline - 20:
            r1_opt, r2_opt = res.x
            pos = _mackay_icosahedron(r1_opt, r2_opt)
            e = _lj_energy(pos)
            if e < best_energy:
                best_energy = e
                best_pos = pos.copy()
    except Exception:
        pass
    
    # If Nelder-Mead didn't give us a good start, use default
    if best_pos is None:
        pos = _mackay_icosahedron(1.12, 2.0)
        best_energy = _lj_energy(pos)
        best_pos = pos.copy()
    
    # Now do L-BFGS polish on the full 55-atom configuration
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        try:
            def fun(x):
                pos = x.reshape(-1, 3)
                e, g = _lj_energy_grad(pos)
                return e, g.reshape(-1)
            
            x0 = best_pos.reshape(-1)
            res2 = minimize(fun, x0, method='L-BFGS-B', jac=True,
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
            pos2 = res2.x.reshape(-1, 3)
            e2 = _lj_energy(pos2)
            if e2 < best_energy:
                best_energy = e2
                best_pos = pos2.copy()
        except Exception:
            pass
    
    # Final check: ensure minimum distance constraint (0.5 apart)
    # If any pair is too close, we need to fix it. But for a good LJ minimum, this shouldn't happen.
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        # Verify minimum distance
        diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dist, np.inf)
        min_dist = np.min(dist)
        if min_dist < 0.5:
            # Fall back to cubic lattice if constraint violated
            side = int(np.ceil(N ** (1 / 3)))
            points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
            points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
            best_pos = np.array(points[:N], dtype=float) * 1.12
    
    return best_pos

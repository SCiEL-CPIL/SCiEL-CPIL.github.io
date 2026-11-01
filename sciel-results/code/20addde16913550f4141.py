"""
Lennard-Jones cluster global minimum, 55 atoms.
Strategy: Start from a Mackay icosahedron geometry (13 inner + 42 outer),
then perform local relaxation using L-BFGS-B to refine the structure.
"""
import numpy as np
import time
from scipy.optimize import minimize
import math

N = 55

# Constants
SIGMA = 1.0
EPSILON = 1.0

def lj_energy_and_grad(pos):
    """
    Compute total Lennard-Jones energy and gradient.
    pos: (N, 3) array
    Returns: energy, gradient (N, 3)
    """
    # pos shape (N, 3)
    # Compute pairwise differences: (N, N, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    # Squared distances
    r2 = np.sum(diff**2, axis=2)
    
    # Avoid division by zero on diagonal
    # We will mask the diagonal
    mask = ~np.eye(N, dtype=bool)
    
    # r2 is symmetric, we can use upper triangle to save computation, 
    # but for gradient it's easier to compute full and sum.
    # However, calculating full N^2 matrix is fine for N=55.
    
    # r^-6 and r^-12
    # r2 > 0 for i != j
    r2_inv = np.zeros_like(r2)
    r2_inv[mask] = 1.0 / r2[mask]
    
    r6_inv = r2_inv**3
    r12_inv = r6_inv**2
    
    # Energy term: 4 * (r^-12 - r^-6)
    # Sum over all pairs. Note that the pair (i,j) is counted twice in the sum over all i,j.
    # The standard LJ potential sum is usually defined as sum over i<j.
    # E = 4 * sum_{i<j} (r_ij^-12 - r_ij^-6)
    # If we sum over all i,j, we get 2 * E.
    # So E = 2 * sum_{i,j} (r_ij^-12 - r_ij^-6) ? No.
    # Let S = sum_{i,j} (r_ij^-12 - r_ij^-6).
    # S = sum_{i != j} ... = 2 * sum_{i<j} ...
    # So sum_{i<j} ... = S / 2.
    # E = 4 * (S / 2) = 2 * S.
    
    S = np.sum(r12_inv[mask] - r6_inv[mask])
    energy = 2.0 * S
    
    # Gradient
    # dE/dr_ij = 4 * (-12 r^-13 + 6 r^-7) = 24 * (2 r^-13 - r^-7)
    # Vector form:
    # F_ij = 24 * (2 r^-13 - r^-7) * (r_ij_vec / r_ij)
    #       = 24 * (2 r^-14 - r^-8) * r_ij_vec
    #       = 24 * (2 r2^-7 - r2^-4) * r_ij_vec
    # 
    # Note: r_ij_vec = pos[i] - pos[j]
    # dE/dpos[i] = sum_{j != i} F_ij
    # dE/dpos[j] = - sum_{i != j} F_ij
    #
    # Using the full matrix approach:
    # coeff = 24 * (2 * r2^-7 - r2^-4)
    # coeff has shape (N, N)
    # grad[i] = sum_j coeff[i,j] * (pos[i] - pos[j])
    
    r2_inv_4 = r2_inv**4
    r2_inv_7 = r2_inv**7
    
    coeff = 24.0 * (2.0 * r2_inv_7 - r2_inv_4)
    # Zero out diagonal to be safe (though r2_inv is 0 there)
    coeff[~mask] = 0.0
    
    # grad[i] = sum_j coeff[i,j] * (pos[i] - pos[j])
    # = sum_j coeff[i,j] * pos[i] - sum_j coeff[i,j] * pos[j]
    # = pos[i] * sum_j coeff[i,j] - sum_j (coeff[i,j] * pos[j])
    
    # sum_j coeff[i,j]
    coeff_sum = np.sum(coeff, axis=1) # shape (N,)
    
    # sum_j (coeff[i,j] * pos[j])
    # This is matrix multiplication: coeff @ pos
    # coeff is (N, N), pos is (N, 3) -> result (N, 3)
    term2 = coeff @ pos
    
    # term1 = pos * coeff_sum[:, np.newaxis]
    term1 = pos * coeff_sum[:, np.newaxis]
    
    grad = term1 - term2
    
    return energy, grad

def mackay_icosahedron_55():
    """
    Generate coordinates for a 55-atom Mackay icosahedron.
    13 atoms in the inner shell (icosahedron).
    42 atoms in the outer shell (truncated icosahedron / fullerene-like, but specifically Mackay).
    
    Mackay icosahedron (2,2,2):
    - Center: 1
    - 1st shell: 12 (icosahedron vertices)
    - 2nd shell: 42 (faces, edges, vertices of the icosahedron scaled up)
    
    Actually, the standard Mackay icosahedron with 55 atoms is:
    1 (center) + 12 (1st shell) + 42 (2nd shell) = 55.
    
    The 2nd shell consists of:
    - 12 atoms at the vertices of the icosahedron (scaled by factor 2)
    - 30 atoms at the midpoints of the edges (scaled by factor ?)
    - 20 atoms at the centers of the faces (scaled by factor ?)
    
    Let's construct it properly.
    Icosahedron vertices (unit radius):
    (0, ±1, ±φ), (±1, ±φ, 0), (±φ, 0, ±1)
    where φ = (1 + sqrt(5)) / 2
    
    Normalize to radius R1.
    Then scale by factor 2 for the outer shell vertices.
    The edge midpoints and face centers need to be placed on the surface of the outer icosahedron.
    For a Mackay icosahedron, the outer shell atoms lie on the great circles.
    
    A simpler way to generate the 55-atom Mackay icosahedron is to use the known coordinates or generate them by projecting points onto the icosahedron surface.
    
    Let's use the explicit coordinates for the 12 vertices of the icosahedron.
    Then, the 42 outer atoms are:
    - 12 at the vertices of the outer icosahedron (radius 2*R1)
    - 30 at the edge centers of the outer icosahedron
    - 20 at the face centers of the outer icosahedron
    
    Wait, the Mackay icosahedron is a specific packing.
    The 2nd shell is a truncated icosahedron? No, it's a icosahedron with atoms on the edges and faces.
    
    Let's calculate the positions.
    1. Generate 12 vertices of a regular icosahedron.
    2. Normalize them to have length R1 = 1.
    3. Inner shell: 12 atoms at these vertices.
    4. Outer shell:
       - 12 atoms at 2 * vertices.
       - 30 atoms at the midpoints of the edges of the icosahedron, scaled to radius 2? 
         The midpoint of an edge of a unit icosahedron has length sqrt(1 - 1/(2*phi))? 
         Let's just project the midpoint onto the sphere of radius 2.
       - 20 atoms at the centers of the faces, projected onto the sphere of radius 2.
       
    This "projected" approach might not yield the exact Mackay geometry, but it's a very good starting point for relaxation.
    The true Mackay icosahedron has the 2nd shell atoms lying on the same great circles as the 1st shell.
    
    Let's refine the outer shell generation:
    The icosahedron has 12 vertices, 30 edges, 20 faces.
    Outer shell vertices: 2 * V_i
    Outer shell edge centers: Project (V_i + V_j) / 2 to radius 2, where V_i, V_j are connected.
    Outer shell face centers: Project (V_i + V_j + V_k) / 3 to radius 2, where V_i, V_j, V_k form a face.
    
    This should give a reasonable approximation.
    """
    phi = (1 + math.sqrt(5)) / 2
    
    # 12 vertices of icosahedron
    verts = [
        (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
        (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
        (phi, 0, 1), (-phi, 0, 1), (phi, 0, -1), (-phi, 0, -1)
    ]
    verts = np.array(verts, dtype=float)
    
    # Normalize to unit length
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis]
    
    # Inner shell: 12 atoms
    inner = verts.copy()
    
    # Center atom
    center = np.array([[0.0, 0.0, 0.0]])
    
    # Outer shell: 42 atoms
    # 1. 12 vertices scaled by 2
    outer_verts = 2.0 * verts
    
    # 2. 30 edge midpoints projected to radius 2
    # Find edges: pairs of vertices that are close (distance ~ 2/sqrt(1+phi^2) * something)
    # Distance between connected vertices in unit icosahedron is 2 / sqrt(1 + phi^2) ?
    # Let's just find all pairs with distance < 1.5 (since max distance is 2, min is ~1.05)
    # Actually, in unit icosahedron, edge length is 2 / sqrt(1 + phi^2) * 2? No.
    # Let's compute pairwise distances and find the smallest 30.
    
    dists = np.linalg.norm(verts[:, np.newaxis, :] - verts[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Get indices of the 30 smallest distances
    flat_dists = dists.flatten()
    sorted_indices = np.argsort(flat_dists)
    edge_indices = sorted_indices[:30]
    
    edge_pairs = []
    for idx in edge_indices:
        i, j = np.unravel_index(idx, (12, 12))
        if i < j:
            edge_pairs.append((i, j))
        else:
            edge_pairs.append((j, i))
            
    # Ensure we have 30 unique pairs
    # The 30 smallest distances should correspond to the 30 edges.
    # Let's verify uniqueness.
    unique_pairs = set()
    final_edge_pairs = []
    for i, j in edge_pairs:
        if (i, j) not in unique_pairs:
            unique_pairs.add((i, j))
            final_edge_pairs.append((i, j))
            
    # If we don't have 30, something went wrong, but for icosahedron it should be fine.
    # Pad if necessary (should not happen)
    while len(final_edge_pairs) < 30:
        final_edge_pairs.append((0, 1)) # Dummy
        
    outer_edges = []
    for i, j in final_edge_pairs:
        mid = (verts[i] + verts[j]) / 2.0
        norm = np.linalg.norm(mid)
        if norm > 0:
            mid = mid / norm * 2.0
        else:
            mid = np.array([2.0, 0.0, 0.0])
        outer_edges.append(mid)
    outer_edges = np.array(outer_edges)
    
    # 3. 20 face centers projected to radius 2
    # Faces are triangles of vertices.
    # We can find them by looking at triples that form a face.
    # A face (i,j,k) exists if i,j and j,k and k,i are all edges.
    edge_set = set(final_edge_pairs)
    # Also add reversed
    for i, j in edge_set:
        edge_set.add((j, i))
        
    face_centers = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                if (i, j) in edge_set and (j, k) in edge_set and (i, k) in edge_set:
                    center = (verts[i] + verts[j] + verts[k]) / 3.0
                    norm = np.linalg.norm(center)
                    if norm > 0:
                        center = center / norm * 2.0
                    else:
                        center = np.array([2.0, 0.0, 0.0])
                    face_centers.append(center)
                    
    # Should have 20 faces
    # Pad if necessary
    while len(face_centers) < 20:
        face_centers.append(np.array([0.0, 0.0, 2.0]))
        
    outer_faces = np.array(face_centers)
    
    # Combine all
    all_pos = np.vstack([center, inner, outer_verts, outer_edges, outer_faces])
    
    # Scale to appropriate size for LJ
    # LJ minimum is at r = 2^(1/6) ~ 1.1225
    # The inner shell radius is 1. The distance between inner shell atoms is ~1.05.
    # We need to scale the whole structure so that nearest neighbor distance is ~1.1225.
    # Current min distance is between inner shell atoms.
    # Let's compute min distance and scale.
    
    # Compute pairwise distances
    dists_all = np.linalg.norm(all_pos[:, np.newaxis, :] - all_pos[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists_all, np.inf)
    min_dist = np.min(dists_all)
    
    # Scale factor
    scale = 2**(1/6) / min_dist
    all_pos = all_pos * scale
    
    return all_pos

def construct():
    start_time = time.time()
    deadline = 150.0  # seconds
    
    # Generate initial Mackay icosahedron
    pos = mackay_icosahedron_55()
    
    # Ensure minimum distance constraint (0.5)
    # The Mackay icosahedron should satisfy this, but let's check.
    # If not, we might need to adjust.
    
    # Local relaxation using L-BFGS-B
    # We don't have explicit bounds, but L-BFGS-B requires them.
    # We can use a large box, e.g., [-10, 10] in each dimension.
    bounds = [(-10.0, 10.0)] * (N * 3)
    
    best_pos = pos
    best_energy = lj_energy_and_grad(pos)[0]
    
    # Relaxation
    try:
        res = minimize(
            lambda x: lj_energy_and_grad(x.reshape(N, 3))[0],
            pos.flatten(),
            jac=lambda x: lj_energy_and_grad(x.reshape(N, 3))[1].flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        
        if res.success:
            new_pos = res.x.reshape(N, 3)
            new_energy = res.fun
            
            # Check validity: min distance
            dists = np.linalg.norm(new_pos[:, np.newaxis, :] - new_pos[np.newaxis, :, :], axis=2)
            np.fill_diagonal(dists, np.inf)
            if np.min(dists) >= 0.5:
                best_pos = new_pos
                best_energy = new_energy
        
    except Exception as e:
        # If optimization fails, keep the initial position
        pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 5.0:
        # If we are close to the deadline, return immediately
        return best_pos
    
    # Optional: Try a few more restarts if time permits
    # But for now, one good relaxation should be sufficient.
    # The Mackay icosahedron is very close to the global minimum.
    
    return best_pos

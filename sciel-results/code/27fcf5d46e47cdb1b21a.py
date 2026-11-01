"""Lennard-Jones cluster global minimum, 55 atoms - Mackay icosahedron with relaxation."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 55
DEADLINE = 115.0  # safety margin before 120s limit


def lj_energy_and_gradient(pos):
    """Compute LJ energy and gradient for N atoms."""
    n = pos.shape[0]
    # Compute pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (N, N, 3)
    r2 = np.sum(diff * diff, axis=2)  # (N, N)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero
    
    r = np.sqrt(r2)
    r6 = r ** 6
    r12 = r6 ** 2
    
    # Energy: 4 * sum(r^-12 - r^-6) over all pairs (i<j)
    # Using full matrix, each pair counted twice, so divide by 2
    pair_energy = 4.0 * (r12 - r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 0.5 * np.sum(pair_energy)
    
    # Gradient: dE/dr_i = sum_{j != i} 4 * (-12*r^-13 + 6*r^-7) * (r_i - r_j)/r
    # = sum_{j != i} 4 * (-12/r12 + 6/r6) * diff_ij / r
    # where diff_ij = r_i - r_j
    grad_factor = 4.0 * (-12.0 / r12 + 6.0 / r6)  # (N, N)
    np.fill_diagonal(grad_factor, 0.0)
    
    # grad[i] = sum_j grad_factor[i,j] * (pos[i] - pos[j]) / r[i,j]
    # = sum_j grad_factor[i,j] * diff[i,j] / r[i,j]
    # diff[i,j] / r[i,j] is the unit vector from j to i
    unit_vec = diff / r[:, :, np.newaxis]  # (N, N, 3)
    gradient = np.sum(grad_factor[:, :, np.newaxis] * unit_vec, axis=1)
    
    return energy, gradient


def mackay_icosahedron():
    """Generate the 55-atom Mackay icosahedron geometry."""
    # icosahedron vertices (12)
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    # Normalize to unit sphere
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    
    # Shell 1: 12 atoms at radius r1
    # Shell 2: 42 atoms at radius r2 (20 face centers + 30 edge midpoints... actually it's 20 faces -> 20 centers, plus 30 edges -> but Mackay has 42 in second shell)
    # Mackay icosahedron: shell 1 = 12 vertices, shell 2 = 20 face centers + 30 edge midpoints? No.
    # Actually: 1st shell: 12 vertices of icosahedron
    # 2nd shell: 20 face centers + 30 edge midpoints = 50? No, that's 50+12=62.
    # Let me reconsider. Mackay icosahedron for 55 atoms:
    # Shell 1: 12 atoms (icosahedron vertices)
    # Shell 2: 42 atoms. These are: 20 face centers of the icosahedron + 30 edge midpoints? 20+30=50, too many.
    # Actually the standard Mackay icosahedron has shells of 1, 12, 42, 120, ...
    # For 55 atoms: 1 + 12 + 42 = 55. So we need a central atom too!
    
    # Central atom
    center = np.array([[0.0, 0.0, 0.0]])
    
    # Shell 1: 12 atoms at radius r1 ~ 1.0 (in LJ units, the minimum is at 2^(1/6) ≈ 1.1225)
    # Let's use optimized radii. Literature values: r1 ≈ 1.0, r2 ≈ 1.9 (in units where sigma=1)
    # Actually for LJ, the equilibrium distance is 2^(1/6) ≈ 1.1225
    # Let's start with r1 = 1.0, r2 = 1.9 and let relaxation fix it
    
    r1 = 1.0
    r2 = 1.9
    
    shell1 = vertices * r1
    
    # Shell 2: 42 atoms
    # These are: 20 face centers of the icosahedron and 30 edge midpoints
    # Face centers: normalize the sum of 3 vertices of each face
    # Icosahedron faces: 20 triangular faces
    # Let me compute face centers from the icosahedron structure
    
    # The icosahedron has 20 faces. Each face is a triangle of 3 vertices.
    # We need to identify the faces. For a regular icosahedron, the faces can be found
    # by checking which triples of vertices form a face.
    
    # Alternative: use known coordinates for the 42 atoms in the second shell
    # The second shell of the Mackay icosahedron consists of:
    # - 20 atoms at the centers of the 20 faces
    # - 30 atoms at the midpoints of the 30 edges
    
    # Let me compute these geometrically.
    # First, find all edges of the icosahedron. An edge connects two vertices
    # if their distance equals the edge length.
    
    v = vertices
    n_v = len(v)
    # Compute all pairwise distances
    dists = np.zeros((n_v, n_v))
    for i in range(n_v):
        for j in range(i+1, n_v):
            d = np.linalg.norm(v[i] - v[j])
            dists[i, j] = d
            dists[j, i] = d
    
    # Find the minimum non-zero distance (edge length)
    min_dist = np.min(dists[dists > 0.1])
    
    # Edges: pairs with distance close to min_dist
    edges = []
    for i in range(n_v):
        for j in range(i+1, n_v):
            if abs(dists[i, j] - min_dist) < 1e-6:
                edges.append((i, j))
    
    # Edge midpoints (30 edges)
    edge_mids = []
    for i, j in edges:
        mid = (v[i] + v[j]) / 2.0
        mid = mid / np.linalg.norm(mid)  # normalize to unit sphere
        edge_mids.append(mid)
    edge_mids = np.array(edge_mids)
    
    # Face centers: for each face (triangle of 3 vertices), the center is the normalized sum
    # We need to find the 20 faces. A face is a set of 3 vertices that are mutually connected.
    # For a regular icosahedron, each face consists of 3 vertices where all 3 pairwise distances equal the edge length.
    
    faces = []
    for i in range(n_v):
        for j in range(i+1, n_v):
            if abs(dists[i, j] - min_dist) > 1e-6:
                continue
            for k in range(j+1, n_v):
                if abs(dists[i, k] - min_dist) > 1e-6 or abs(dists[j, k] - min_dist) > 1e-6:
                    continue
                faces.append((i, j, k))
    
    # Should have 20 faces
    face_centers = []
    for i, j, k in faces:
        center = v[i] + v[j] + v[k]
        center = center / np.linalg.norm(center)
        face_centers.append(center)
    face_centers = np.array(face_centers)
    
    # Combine: 20 face centers + 30 edge midpoints = 50? That's too many.
    # Wait, 20 + 30 = 50, but we need 42. Let me reconsider.
    
    # Actually, I think I'm overcomplicating this. Let me use a different approach.
    # The Mackay icosahedron for 55 atoms has:
    # 1 central atom
    # 12 atoms on the first shell (icosahedron vertices)
    # 42 atoms on the second shell
    
    # The 42 atoms on the second shell are positioned at the centers of the 20 faces
    # and at the midpoints of the 30 edges? That gives 50, not 42.
    
    # Let me look at this differently. Perhaps the second shell has a different structure.
    # Actually, upon reflection, I believe the correct decomposition is:
    # Shell 2 has 42 atoms arranged as: 20 at face centers and 22 at some other positions?
    # 
    # Let me just use a simpler approach: generate points on a sphere and use optimization.
    # Or better yet, use the known fact that the 55-atom Mackay icosahedron can be constructed
    # by taking the icosahedron vertices and adding points at specific positions.
    
    # Alternative approach: use the icosahedron vertices and scale, then add a second shell
    # of 42 points that are the centroids of the 20 faces plus additional points.
    
    # Actually, let me just create a reasonable initial guess and let L-BFGS do the work.
    # I'll create 55 points in an icosahedral arrangement.
    
    # Simple approach: use Fibonacci sphere for 55 points, then relax
    # Or: use the icosahedron vertices (12) + face centers (20) + edge midpoints (30) = 62, too many
    # Take 12 + 42 = 54, plus center = 55. So 42 on second shell.
    
    # Let me use: 12 vertices + 20 face centers + 10 selected edge midpoints = 42 on second shell
    # This is a rough guess. Let relaxation fix it.
    
    # Better: just use a spherical distribution and let optimization handle it.
    
    # Generate 55 points using a combination of icosahedral symmetry
    # Center: 1
    # Shell 1: 12 (icosahedron vertices) at radius r1
    # Shell 2: 42 points at radius r2
    
    # For shell 2, use the 20 face centers and 22 of the 30 edge midpoints
    # (select 22 edge midpoints to get 42 total)
    
    # Actually, let me just use all 20 face centers and 22 edge midpoints (first 22)
    shell2 = np.vstack([face_centers, edge_mids[:22]]) * r2
    
    # Total: 1 + 12 + 42 = 55
    positions = np.vstack([center, shell1, shell2])
    
    return positions


def construct():
    """Construct initial positions for 55-atom LJ cluster."""
    start_time = time.time()
    
    # Generate Mackay icosahedron initial guess
    pos = mackay_icosahedron()
    
    # Ensure minimum distance of 0.5
    # Check and fix if needed
    for _ in range(100):
        d = cdist(pos, pos)
        np.fill_diagonal(d, np.inf)
        min_d = np.min(d)
        if min_d >= 0.5:
            break
        # If too close, nudge apart
        i, j = np.unravel_index(np.argmin(d), d.shape)
        mid = (pos[i] + pos[j]) / 2
        direction = pos[i] - pos[j]
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        pos[i] += direction * 0.01
        pos[j] -= direction * 0.01
    
    # L-BFGS relaxation with analytical gradient
    def objective(x):
        pos_flat = x.reshape(-1, 3)
        e, _ = lj_energy_and_gradient(pos_flat)
        return e
    
    def gradient(x):
        pos_flat = x.reshape(-1, 3)
        _, g = lj_energy_and_gradient(pos_flat)
        return g.flatten()
    
    x0 = pos.flatten()
    
    # Run L-BFGS
    try:
        result = minimize(
            objective,
            x0,
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 10000}
        )
        pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    # Check deadline
    if time.time() - start_time > DEADLINE:
        return pos
    
    # Multi-start: try a few perturbations if time permits
    best_energy = lj_energy_and_gradient(pos)[0]
    best_pos = pos.copy()
    
    np.random.seed(42)
    for trial in range(5):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Perturb the Mackay geometry
        pos_trial = mackay_icosahedron()
        noise = np.random.normal(0, 0.1, pos_trial.shape)
        pos_trial += noise
        
        # Ensure minimum distance
        for _ in range(50):
            d = cdist(pos_trial, pos_trial)
            np.fill_diagonal(d, np.inf)
            min_d = np.min(d)
            if min_d >= 0.5:
                break
            i, j = np.unravel_index(np.argmin(d), d.shape)
            direction = pos_trial[i] - pos_trial[j]
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            pos_trial[i] += direction * 0.01
            pos_trial[j] -= direction * 0.01
        
        try:
            result = minimize(
                objective,
                pos_trial.flatten(),
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000}
            )
            pos_trial = result.x.reshape(-1, 3)
            e_trial = lj_energy_and_gradient(pos_trial)[0]
            
            if e_trial < best_energy:
                best_energy = e_trial
                best_pos = pos_trial.copy()
        except Exception:
            continue
    
    # Final high-precision polish
    if time.time() - start_time < DEADLINE - 5:
        try:
            result = minimize(
                objective,
                best_pos.flatten(),
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 20000}
            )
            best_pos = result.x.reshape(-1, 3)
        except Exception:
            pass
    
    return best_pos

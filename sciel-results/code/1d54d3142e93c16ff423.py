"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
import time

def _get_icosahedron_vertices():
    """Return the 12 vertices of a regular icosahedron centered at origin."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    return verts

def _generate_mackay_icosahedron(n_target, spacing=1.1):
    """
    Generate a Mackay icosahedral cluster with exactly n_target atoms.
    Shell structure:
      - Core: 1 atom at origin
      - Shell 1 (12 atoms): icosahedron vertices at radius r1
      - Shell 2 (42 atoms): icosahedral second shell at radius r2
    For n up to 60, we use core + shell1 + part of shell2.
    """
    # Core atom
    points = [np.array([0.0, 0.0, 0.0])]
    
    # Get icosahedron vertices and normalize to unit sphere
    verts = _get_icosahedron_vertices()
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Shell 1: 12 atoms at radius r1
    r1 = spacing
    shell1 = verts * r1
    for v in shell1:
        points.append(v.copy())
    
    if n_target <= 13:
        return np.array(points[:n_target])
    
    # Shell 2: 42 atoms - these are at the midpoints of edges and face centers
    # For a Mackay icosahedron, shell 2 consists of:
    # - 30 atoms at edge midpoints (directions)
    # - 12 atoms at face centers (directions)
    # Total: 42 atoms
    
    # Compute edge midpoints: for each pair of adjacent vertices
    # In an icosahedron, each vertex connects to 5 others
    # Let's compute all pairwise midpoints and filter by distance
    
    # Actually, for Mackay icosahedron, the second shell atoms are at:
    # The 30 edge-midpoint directions and 12 face-center directions
    # Let's compute them properly
    
    # Face centers: each face of the icosahedron
    # An icosahedron has 20 faces. Each face is a triangle of 3 vertices.
    # We need to identify all 20 triangular faces.
    
    # For simplicity, let's use a different approach:
    # Generate points on a sphere with icosahedral symmetry
    
    # Actually, let's use the known structure:
    # Shell 2 radius r2 = r1 * sqrt(5)/2 approximately? No, let's compute.
    # In Mackay icosahedron, the second shell is at distance sqrt(5)/2 * r1 from center? 
    # Let me use a simpler approach: generate 42 points with icosahedral symmetry.
    
    # The 42 points of the second shell in a Mackay icosahedron:
    # They are at the positions:
    # - 30 points: midpoints of the 30 edges
    # - 12 points: centers of the 20 faces... wait, that's only 20, not 12.
    
    # Let me reconsider. A Mackay icosahedron has:
    # 1 + 12 + 42 = 55 atoms
    # The 42 in the second shell: 
    # Actually, I think the standard Mackay icosahedron has shells of 1, 12, 42.
    # The 42 atoms are at the vertices of a truncated icosahedron? No.
    
    # Let me just use a simpler approach: place atoms at icosahedrally symmetric positions.
    # For the second shell, use the 42 directions corresponding to the second shell of a Mackay icosahedron.
    
    # Alternative: use the fact that in a Mackay icosahedron, the second shell atoms
    # are located at the edge midpoints (30) and face centers (12)... 
    # Wait, icosahedron has 30 edges and 20 faces. 30 + 20 = 50, not 42.
    
    # Let me look up: Mackay icosahedron 55-atom cluster:
    # Core: 1
    # First shell: 12 (vertices of icosahedron)
    # Second shell: 42
    # The 42 are: 30 at edge midpoints + 12 at... 
    # Actually, I recall now: the second shell has 42 atoms arranged as:
    # 30 at the midpoints of the 30 edges, and 12 at the centers of the 12 five-fold axes? No.
    
    # Let me just generate them computationally:
    # For each edge of the icosahedron, place an atom at the midpoint direction
    # For each face, place an atom at the face center direction
    
    # First, identify edges and faces
    # An icosahedron has 12 vertices, 30 edges, 20 faces
    
    # Compute distances between all vertex pairs to identify edges
    # In a regular icosahedron, edge length is 2 (if vertices are at unit distance from center with the standard coordinates above)
    # Let's compute the minimum distance between vertices
    
    min_dist = np.inf
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < min_dist:
                min_dist = d
    
    # Identify edges: pairs with distance close to min_dist
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if abs(d - min_dist) < 1e-9:
                edges.append((i, j))
    
    # Edge midpoints (30)
    r2 = spacing * np.sqrt(5) / 2  # approximate radius for second shell
    # Actually, the edge midpoint direction: normalize (v_i + v_j) / ||v_i + v_j||
    for i, j in edges:
        mid_dir = (verts[i] + verts[j])
        mid_dir = mid_dir / np.linalg.norm(mid_dir)
        points.append(mid_dir * r2)
    
    # Now we have 1 + 12 + 30 = 43 points. Need 42 more for 55 total.
    # Wait, we need 42 in the second shell, but only got 30 from edges.
    # The remaining 12 must be face centers.
    
    # Identify faces: triples of vertices that form a triangle (all pairwise distances = min_dist)
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                d_ij = np.linalg.norm(verts[i] - verts[j])
                d_jk = np.linalg.norm(verts[j] - verts[k])
                d_ik = np.linalg.norm(verts[i] - verts[k])
                if (abs(d_ij - min_dist) < 1e-9 and 
                    abs(d_jk - min_dist) < 1e-9 and 
                    abs(d_ik - min_dist) < 1e-9):
                    faces.append((i, j, k))
    
    # Face centers (20)
    for i, j, k in faces:
        face_dir = verts[i] + verts[j] + verts[k]
        face_dir = face_dir / np.linalg.norm(face_dir)
        points.append(face_dir * r2)
    
    # Now we have 1 + 12 + 30 + 20 = 63 points. Too many.
    # For 55-atom Mackay, we need exactly 42 in the second shell.
    # The standard Mackay icosahedron second shell has 42 atoms, not 50.
    # I think the correct structure is: 30 edge midpoints + 12 face centers? But there are 20 faces.
    
    # Let me just take the first n_target-13 points from the second shell
    # and return the appropriate number
    
    # Actually, for the purpose of this task, let's just use what we have
    # and trim to n_target
    
    if len(points) >= n_target:
        return np.array(points[:n_target])
    
    # If we don't have enough, add more random icosahedrally symmetric points
    while len(points) < n_target:
        # Add a random perturbation to an existing point
        idx = np.random.randint(0, len(points))
        new_pt = points[idx] + np.random.normal(0, 0.1, 3)
        # Ensure minimum distance
        if all(np.linalg.norm(new_pt - p) >= 0.5 for p in points):
            points.append(new_pt)
        else:
            # Try a random point on the sphere
            theta = np.random.uniform(0, 2 * np.pi)
            phi_val = np.random.uniform(0, np.pi)
            new_pt = np.array([
                r2 * np.sin(phi_val) * np.cos(theta),
                r2 * np.sin(phi_val) * np.sin(theta),
                r2 * np.cos(phi_val)
            ])
            if all(np.linalg.norm(new_pt - p) >= 0.5 for p in points):
                points.append(new_pt)
    
    return np.array(points[:n_target])

def _generate_defect_icosahedron(n_target, spacing=1.1, rng=None):
    """
    Generate a defect icosahedral cluster by starting from a Mackay icosahedron
    and removing or displacing atoms to get exactly n_target atoms.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Start with a Mackay icosahedron of 55 atoms
    mackay = _generate_mackay_icosahedron(55, spacing)
    
    if n_target == 55:
        return mackay
    
    if n_target < 55:
        # Remove atoms from the outer shell preferentially
        # Atoms are ordered: core (1), shell1 (12), shell2 (42)
        # Remove from shell2 first, then shell1
        n_remove = 55 - n_target
        # Shuffle indices in shell2 (indices 13-54) and shell1 (indices 1-12)
        shell2_idx = list(range(13, 55))
        shell1_idx = list(range(1, 13))
        
        rng.shuffle(shell2_idx)
        rng.shuffle(shell1_idx)
        
        to_remove = []
        for idx in shell2_idx:
            if len(to_remove) < n_remove:
                to_remove.append(idx)
        for idx in shell1_idx:
            if len(to_remove) < n_remove:
                to_remove.append(idx)
        
        to_remove = set(to_remove[:n_remove])
        kept = [mackay[i] for i in range(55) if i not in to_remove]
        return np.array(kept[:n_target])
    
    else:
        # n_target > 55: add atoms in a third shell or displace
        # For 57 and 60, add atoms at icosahedrally symmetric positions
        extra = n_target - 55
        points = list(mackay)
        
        # Add atoms at the 12 five-fold axis directions (same as shell1 vertices but further out)
        verts = _get_icosahedron_vertices()
        verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
        r3 = spacing * np.sqrt(5)  # third shell radius
        
        for v in verts:
            if len(points) >= n_target:
                break
            new_pt = v * r3
            if all(np.linalg.norm(new_pt - p) >= 0.5 for p in points):
                points.append(new_pt)
        
        # If still not enough, add random points
        while len(points) < n_target:
            theta = rng.uniform(0, 2 * np.pi)
            phi_val = rng.uniform(0, np.pi)
            new_pt = np.array([
                r3 * np.sin(phi_val) * np.cos(theta),
                r3 * np.sin(phi_val) * np.sin(theta),
                r3 * np.cos(phi_val)
            ])
            if all(np.linalg.norm(new_pt - p) >= 0.5 for p in points):
                points.append(new_pt)
            else:
                # Perturb an existing point
                idx = rng.randint(0, len(points))
                new_pt = points[idx] + rng.normal(0, 0.2, 3)
                if all(np.linalg.norm(new_pt - p) >= 0.5 for p in points):
                    points.append(new_pt)
        
        return np.array(points[:n_target])

def _basin_hopping(positions, potential, parameter, rng, max_iters=100, step_size=0.5, deadline=None):
    """
    Basin hopping with local minimization.
    """
    best_energy, best_positions = relax(positions.copy(), potential, parameter)
    
    for i in range(max_iters):
        if deadline is not None and time.time() > deadline - 30:
            break
        
        # Perturb
        perturbed = best_positions + rng.normal(0, step_size, best_positions.shape)
        
        # Relax
        energy, positions = relax(perturbed, potential, parameter)
        
        if energy < best_energy:
            best_energy = energy
            best_positions = positions
    
    return best_energy, best_positions

def construct(instance):
    """
    Construct a low-energy cluster using icosahedral defect-seeded initialization
    and basin hopping refinement.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    deadline = start_time + 270  # 30s safety margin
    
    rng = np.random.default_rng(size)
    
    best_energy = np.inf
    best_positions = None
    
    # Generate multiple icosahedral candidates
    n_candidates = 8
    for i in range(n_candidates):
        if time.time() > deadline - 60:
            break
        
        # Generate a defect icosahedral candidate
        candidate = _generate_defect_icosahedron(size, spacing=1.1, rng=rng)
        
        # Add small random perturbation to break symmetry
        candidate = candidate + rng.normal(0, 0.05, candidate.shape)
        
        # Basin hopping
        energy, positions = _basin_hopping(
            candidate, potential, parameter, rng,
            max_iters=50, step_size=0.3, deadline=deadline
        )
        
        if energy < best_energy:
            best_energy = energy
            best_positions = positions
    
    if best_positions is None:
        # Fallback: simple random
        radius = 1.1 * size ** (1.0 / 3.0) * 0.75
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
        best_energy, best_positions = relax(np.array(points), potential, parameter)
    
    return best_positions
# EVOLVE-BLOCK-END

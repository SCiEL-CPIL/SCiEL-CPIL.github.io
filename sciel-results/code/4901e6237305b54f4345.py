"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Constants for time management
TIME_LIMIT = 300.0
SAFETY_MARGIN = 5.0

# Dzugutov parameters
A_PARAM = 5.82
B_PARAM = 1.28
A_CUTOFF = 1.87
B_CUTOFF = 1.94
C_PARAM = 1.10
D_PARAM = 0.27
M_PARAM = 16

def _dzugutov_energy(pos):
    """Compute Dzugutov energy."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    
    energy = 0.0
    # Vectorized calculation for speed
    # V(r) = A(r^-m - B)exp(c/(r - a)) for r < a
    # V(r) = B exp(d/(r - b)) for a <= r < b
    
    mask_a = dist < A_CUTOFF
    mask_b = (dist >= A_CUTOFF) & (dist < B_CUTOFF)
    
    # Only compute for valid pairs (i < j) to avoid double counting and self-interaction
    # Using upper triangle
    iu = np.triu_indices(n, k=1)
    
    # Part 1: r < a
    r_a = dist[iu][mask_a[iu]]
    if len(r_a) > 0:
        # Avoid division by zero or negative if r < a is very small, though constraint is 0.5
        # r^-m can be large
        term_a = (r_a ** (-M_PARAM) - B_PARAM) * np.exp(C_PARAM / (r_a - A_CUTOFF))
        energy += np.sum(term_a) * A_PARAM
        
    # Part 2: a <= r < b
    r_b = dist[iu][mask_b[iu]]
    if len(r_b) > 0:
        term_b = np.exp(D_PARAM / (r_b - B_CUTOFF))
        energy += np.sum(term_b) * B_PARAM
        
    return energy

def _generate_mackay_icosahedron(n, rng):
    """
    Generate a Mackay icosahedral cluster.
    n=13: center + 12
    n=55: center + 12 + 42
    n=147: center + 12 + 42 + 92 (too big for our range mostly, but we can truncate)
    For other n, we generate the 55-atom structure and randomly remove outer atoms,
    or generate a random polytetrahedral structure.
    """
    phi = (1 + np.sqrt(5)) / 2
    # 12 vertices of icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1],
        [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis]
    
    # Scaling: Dzugutov equilibrium is around r ~ 1.5-1.8?
    # Let's assume the first shell is at distance ~1.0 for generation, 
    # relaxation will fix the scale.
    # Actually, for Dzugutov, the minimum is around r=1.5? 
    # The potential has a max at r~1.5? No, it's a model for glass formers.
    # Let's use a standard spacing of 1.0 for generation.
    
    if n == 13:
        center = np.array([[0.0, 0.0, 0.0]])
        return np.vstack([center, verts])
    
    if n == 55:
        center = np.array([[0.0, 0.0, 0.0]])
        shell1 = verts
        # Shell 2: 42 vertices. These are the face centers of the 20 faces of the icosahedron?
        # No, the 55-atom Mackay icosahedron consists of:
        # 1 center
        # 12 vertices of icosahedron
        # 42 atoms on the edges? No, 30 edges. 
        # Actually, the 55-atom cluster is:
        # 1 center
        # 12 vertices
        # 42 atoms: 30 on the edges (midpoints?) and 12 on the face centers?
        # Let's construct the 42 points properly.
        # The 55-atom Mackay icosahedron has shells: 1, 12, 42.
        # The 42 points are:
        # 30 points on the edges (midpoints of the 30 edges of the icosahedron)
        # 12 points on the faces (centroids of the 20 faces? No, 20 faces. 12 is wrong count for faces).
        
        # Correct construction of 55-atom Mackay Icosahedron:
        # It is a subset of the 147-atom one.
        # The 147-atom one has shells 1, 12, 42, 92.
        # The 42 atoms are:
        # 30 midpoints of edges of the icosahedron.
        # 12 centroids of the 20 faces? No.
        
        # Let's use a simpler heuristic for n != 13, 55.
        # Generate the 55-atom structure explicitly if possible, else random.
        
        # Vertices of icosahedron (12)
        v = verts
        # Edges: pairs of vertices connected in icosahedron
        # We need the connectivity of the icosahedron.
        # Distance between connected vertices in unit icosahedron is 2.
        # Let's find edges by distance.
        dist_matrix = np.linalg.norm(v[:, np.newaxis, :] - v[np.newaxis, :, :], axis=2)
        # Min distance is ~2.0
        min_dist = np.min(dist_matrix[dist_matrix > 0])
        edges = []
        for i in range(12):
            for j in range(i+1, 12):
                if np.isclose(dist_matrix[i,j], min_dist, rtol=1e-5):
                    edges.append((i, j))
        
        # 30 midpoints
        midpoints = []
        for i, j in edges:
            midpoints.append((v[i] + v[j]) / 2.0)
        midpoints = np.array(midpoints)
        
        # 12 face centroids?
        # The 55-atom cluster actually has 42 atoms in the second shell.
        # 30 edge midpoints + 12 face centroids? 
        # Wait, an icosahedron has 20 faces. 
        # The 55-atom cluster is often described as having 1+12+42.
        # The 42 atoms are: 30 on edges, 12 on... ?
        # Actually, the 55-atom cluster is not a perfect Mackay icosahedron in the sense of full shells.
        # The full Mackay shells are 1, 12, 42, 92...
        # The 42-atom shell consists of:
        # 30 points on the edges (midpoints)
        # 12 points on the faces? No, 20 faces.
        # Let's check the literature or derive.
        # Actually, the 42 atoms are:
        # 30 midpoints of edges.
        # 12 centroids of the 20 triangular faces? No, that would be 20.
        
        # Alternative: The 55-atom cluster is the 13-atom cluster plus 42 atoms.
        # Let's just use the 30 midpoints and 12 random points on the surface?
        # Or better: Use the 147-atom construction and take the first 55?
        
        # Let's try a different approach for n=55:
        # 1 center
        # 12 vertices
        # 42 atoms: 
        #   30 edge midpoints
        #   12 face centroids (of the 20 faces? Which 12?)
        #   Actually, the 55-atom cluster is symmetric.
        #   It contains the 12 vertices, 30 edge midpoints, and 12 face centroids?
        #   Wait, 12+30+12 = 54. Plus 1 center = 55.
        #   So we need 12 face centroids. Which ones?
        #   Perhaps the ones corresponding to the 12 vertices of the dual dodecahedron?
        #   The dual of an icosahedron is a dodecahedron (20 faces, 12 vertices).
        #   The centroids of the 20 faces of the icosahedron correspond to the 20 vertices of the dodecahedron.
        #   This doesn't match 12.
        
        # Let's skip the exact 55-atom construction and use a "Random Poly-Tetrahedral" generator
        # which places atoms in tetrahedral configurations.
        
        # For now, return the 13-atom structure scaled up? No.
        # Let's return a random compact structure for now if n != 13, 
        # but for n=55, we can try to add the 30 midpoints and 12 random surface points.
        
        points = [np.array([0.0, 0.0, 0.0])]
        points.extend(v)
        points.extend(midpoints)
        
        # Need 55 - (1 + 12 + 30) = 12 more points.
        # Add 12 points on the surface, roughly at face centroids.
        # We can approximate face centroids by averaging 3 vertices of a face.
        # We need to find the 20 faces.
        # A face is a set of 3 vertices that form an equilateral triangle of side min_dist.
        faces = []
        for i in range(12):
            for j in range(i+1, 12):
                if np.isclose(dist_matrix[i,j], min_dist, rtol=1e-5):
                    for k in range(j+1, 12):
                        if np.isclose(dist_matrix[i,k], min_dist, rtol=1e-5) and np.isclose(dist_matrix[j,k], min_dist, rtol=1e-5):
                            faces.append((i, j, k))
        
        # We have 20 faces. We need 12 centroids.
        # Let's just pick 12 random ones? Or all 20 and truncate?
        # Truncating breaks symmetry.
        # Let's just add 12 random points near the surface.
        for _ in range(12):
            # Pick a random face
            f = faces[rng.integers(0, len(faces))]
            centroid = (v[f[0]] + v[f[1]] + v[f[2]]) / 3.0
            # Add small noise
            noise = rng.normal(0, 0.05, 3)
            points.append(centroid + noise)
            
        return np.array(points)

    # For other n, generate random points in a sphere
    radius = 1.0 * (n / 13.0) ** (1/3.0)
    points = [np.array([0.0, 0.0, 0.0])]
    for _ in range(n - 1):
        # Sample in sphere
        while True:
            p = rng.normal(0, 1, 3)
            norm = np.linalg.norm(p)
            if norm <= radius:
                p = p / norm * (radius * (0.5 + 0.5 * rng.random()))
                # Check min distance
                if all(np.linalg.norm(p - q) > 0.5 for q in points):
                    points.append(p)
                    break
    return np.array(points)

def _basin_hopping(positions, n, rng, time_limit, potential, parameter):
    """
    Perform basin hopping starting from positions.
    Returns best energy and positions found.
    """
    best_energy = float('inf')
    best_pos = positions.copy()
    current_energy = float('inf')
    current_pos = positions.copy()
    
    start_time = time.time()
    kick_size = 0.2
    
    while time.time() - start_time < time_limit:
        # 1. Relax
        try:
            energy, relaxed_pos = relax(current_pos, potential, parameter)
        except Exception:
            energy = _dzugutov_energy(current_pos)
            relaxed_pos = current_pos
            
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed_pos.copy()
            
        # 2. Check time
        if time.time() - start_time > time_limit - 2.0:
            break
            
        # 3. Kick
        # Perturb the relaxed structure
        kick = rng.normal(0, kick_size, relaxed_pos.shape)
        new_pos = relaxed_pos + kick
        
        # Ensure min distance constraint (roughly)
        # The relax function should handle it, but if we kick too hard,
        # we might violate it. The evaluator checks it.
        # We rely on relax to fix it, or we should check.
        # For speed, we assume relax handles it or we accept the risk 
        # that a bad kick leads to a high energy local min, which is fine.
        
        current_pos = new_pos
        
        # Adaptive kick size?
        # If we improved, keep kick size. If not, increase slightly?
        # Standard BH uses Metropolis acceptance, but we just keep best.
        # To explore more, we can increase kick size if no improvement for a while.
        # But for now, fixed kick size is robust.
        
    return best_energy, best_pos

def construct(instance):
    """
    Construct the best structure using Basin Hopping with Poly-Tetrahedral seeds.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    rng = np.random.default_rng(size)
    
    best_energy = float('inf')
    best_positions = None
    
    # Generate a set of diverse starting points
    # 1. Mackay Icosahedron (if n matches or close)
    # 2. Random Poly-Tetrahedral
    # 3. Random Compact
    
    starts = []
    
    # Strategy 1: Poly-Tetrahedral / Mackay
    try:
        mackay = _generate_mackay_icosahedron(size, rng)
        if len(mackay) == size:
            starts.append(mackay)
    except Exception:
        pass
        
    # Strategy 2: Random compact with min distance
    for i in range(3):
        radius = 1.0 * (size / 13.0) ** (1/3.0)
        points = [np.array([0.0, 0.0, 0.0])]
        attempts = 0
        while len(points) < size and attempts < 1000:
            p = rng.uniform(-radius, radius, 3)
            if all(np.linalg.norm(p - q) > 0.6 for q in points):
                points.append(p)
            else:
                attempts += 1
        if len(points) < size:
            # Pad with random points far away? No, just random in sphere
            while len(points) < size:
                p = rng.uniform(-radius, radius, 3)
                points.append(p)
        starts.append(np.array(points))
        
    # Allocate time to each start
    # We have 4 starts.
    # Give each 1/4 of the time, but allow early termination if no improvement?
    # Simpler: Give each a fixed chunk of time, e.g., 60s each (total 240s).
    # Or adaptive: Run BH for 10s, if energy drops, give more time.
    
    time_per_start = (TIME_LIMIT - SAFETY_MARGIN) / len(starts)
    
    for i, start_pos in enumerate(starts):
        remaining_time = deadline - time.time()
        if remaining_time <= 0:
            break
            
        # Use at most time_per_start for this one, or remaining time if it's the last one
        alloc_time = min(time_per_start, remaining_time)
        
        try:
            e, pos = _basin_hopping(start_pos, size, rng, alloc_time, potential, parameter)
        except Exception:
            e, pos = float('inf'), start_pos
            
        if e < best_energy:
            best_energy = e
            best_positions = pos
            
    if best_positions is None:
        # Fallback
        best_positions = _generate_mackay_icosahedron(size, rng)
        if len(best_positions) != size:
             best_positions = np.random.uniform(-1, 1, (size, 3))
        try:
             _, best_positions = relax(best_positions, potential, parameter)
        except Exception:
             pass
             
    return best_positions
# EVOLVE-BLOCK-END

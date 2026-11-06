# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
_TIME_BUDGET = 320.0  # safety margin before 330s kill

def _time_left():
    return _TIME_BUDGET - (time.time() - _START_TIME)

def _dzugutov_energy(positions):
    """Vectorized Dzugutov energy."""
    n = len(positions)
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < 0.5:
                return 1e10
            if r < a:
                total += A * (r ** (-m) - B) * np.exp(c / (r - a))
            elif r < b:
                total += B * np.exp(d / (r - b))
    return total

def _icosahedron_vertices(radius=1.0):
    """Return 12 vertices of a regular icosahedron."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0,  1,  phi], [0, -1,  phi], [0,  1, -phi], [0, -1, -phi],
        [ 1,  phi, 0], [-1,  phi, 0], [ 1, -phi, 0], [-1, -phi, 0],
        [ phi, 0,  1], [ phi, 0, -1], [-phi, 0,  1], [-phi, 0, -1],
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, None] * radius
    return verts

def _build_icosahedral_cluster(n, seed=0):
    """Build an n-atom cluster from icosahedral fragments.
    
    Strategy: place icosahedral units (center + 12 neighbors) and merge them.
    For n atoms, we use k icosahedral units where k ≈ n/13, but we need exact n.
    """
    rng = np.random.default_rng(seed)
    radius = 1.1  # typical Dzugutov bond length
    
    # We'll build by placing atoms in icosahedral shells
    # Core: 1 atom
    # First shell: 12 atoms (icosahedron)
    # Second shell: 30 atoms (Mackay)
    # For arbitrary n, we use partial shells
    
    atoms = [np.zeros(3)]
    
    # First icosahedral shell: 12 atoms
    shell1 = _icosahedron_vertices(radius)
    for v in shell1:
        atoms.append(v.copy())
    
    if len(atoms) >= n:
        return np.array(atoms[:n])
    
    # Second shell: Mackay icosahedron (30 atoms)
    # These are at the face centers and edge midpoints
    # Face centers: 20 faces * 1 = 20
    # Edge midpoints: 30 edges * 1 = 10
    # Total: 30
    
    # Generate Mackay second shell positions
    # The 20 face centers of the icosahedron
    # Each face is a triangle of 3 icosahedron vertices
    # We need to find all 20 faces
    
    # Alternative: use known Mackay positions
    # Second shell radius = radius * sqrt(3)
    r2 = radius * np.sqrt(3.0)
    
    # Face centers: normalize sum of each face's 3 vertices
    # Icosahedron faces (20 triangles)
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0,  1,  phi], [0, -1,  phi], [0,  1, -phi], [0, -1, -phi],
        [ 1,  phi, 0], [-1,  phi,0], [ 1, -phi, 0], [-1, -phi, 0],
        [ phi, 0,  1], [ phi, 0, -1], [-phi, 0,  1], [-phi, 0, -1],
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, None]
    
    # Find all faces (triangles) of the icosahedron
    # For each pair of vertices, find the third that forms a face
    # Simpler: use the known fact that faces are formed by triples at distance 2/sqrt(1+phi^2)
    # Actually, let's just generate approximate second shell
    
    # Use a different approach: generate points on a sphere at the right radius
    # and keep those that are at the right distance from first shell atoms
    
    # Mackay second shell: 30 points
    # 20 face centers + 10 edge midpoints
    # Let's generate them properly
    
    # Edges of icosahedron: pairs of vertices at distance 2/sqrt(1+phi^2) * radius
    edge_dist = 2.0 / np.sqrt(1.0 + phi**2) * radius
    
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if abs(d - 2.0/np.sqrt(1.0+phi**2)) < 0.01:
                edges.append((i, j))
    
    # Edge midpoints (10 edges... actually 30 edges for icosahedron)
    # Wait, icosahedron has 30 edges
    # Let me recount: 12 vertices, 30 edges, 20 faces
    
    # For the second shell, let's use a simpler approach:
    # Place atoms at positions that are at distance ~1.1 from existing atoms
    # and form icosahedral coordination
    
    # Actually, for robustness, let's use a different strategy:
    # Build the cluster by adding atoms one at a time, each in an icosahedral
    # position relative to existing atoms
    
    # Reset and use incremental growth
    atoms = [np.zeros(3)]
    shell1 = _icosahedron_vertices(radius)
    for v in shell1:
        atoms.append(v.copy())
    
    if len(atoms) >= n:
        return np.array(atoms[:n])
    
    # Add more atoms by finding positions that are ~radius away from 12 existing atoms
    # This is complex; instead, let's use a layered approach
    
    # For n > 13, add a second icosahedral shell at radius * sqrt(3)
    # The 30 Mackay positions
    # Generate them as: for each face of the icosahedron, the face center
    # and for each edge, the edge midpoint
    
    # Let me just generate 30 points on a sphere of radius r2 that are
    # at approximately the right distance from the first shell
    
    # Simpler: use the fact that Mackay icosahedron second shell has
    # 20 points at face centers and 10 at edge midpoints
    
    # Face centers: average of 3 vertices of each face, normalized to r2
    # We need to identify the 20 faces
    
    # Let's use a computational approach: find all triples that form equilateral triangles
    # with side length = edge length
    
    edge_len = 2.0 / np.sqrt(1.0 + phi**2)
    
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(np.linalg.norm(verts[i] - verts[j]) - edge_len) > 0.01:
                continue
            for k in range(j+1, 12):
                if abs(np.linalg.norm(verts[i] - verts[k]) - edge_len) > 0.01:
                    continue
                if abs(np.linalg.norm(verts[j] - verts[k]) - edge_len) > 0.01:
                    continue
                faces.append((i, j, k))
    
    # Should have 20 faces
    face_centers = []
    for f in faces:
        center = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
        center = center / np.linalg.norm(center) * r2
        face_centers.append(center)
    
    # Edge midpoints
    edge_midpoints = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(np.linalg.norm(verts[i] - verts[j]) - edge_len) > 0.01:
                continue
            mid = (verts[i] + verts[j]) / 2.0
            mid = mid / np.linalg.norm(mid) * r2
            edge_midpoints.append(mid)
    
    # Add second shell atoms
    for p in face_centers:
        if len(atoms) >= n:
            break
        atoms.append(p)
    for p in edge_midpoints:
        if len(atoms) >= n:
            break
        atoms.append(p)
    
    # If still not enough, add more shells or random perturbations
    while len(atoms) < n:
        # Add a random position on a sphere of radius r2 + small offset
        theta = rng.uniform(0, 2*np.pi)
        z = rng.uniform(-1, 1)
        x = np.sqrt(1-z*z) * np.cos(theta)
        y = np.sqrt(1-z*z) * np.sin(theta)
        p = np.array([x, y, z]) * r2 * 1.1
        atoms.append(p)
    
    return np.array(atoms[:n])

def _build_random_icosahedral(n, seed=0):
    """Build cluster with icosahedral bias using random icosahedral units."""
    rng = np.random.default_rng(seed)
    radius = 1.1
    atoms = []
    
    # Place several icosahedral units at random positions
    n_units = max(1, n // 13)
    centers = []
    
    for _ in range(n_units):
        if len(atoms) >= n:
            break
        # Random center
        if len(centers) == 0:
            center = np.zeros(3)
        else:
            # Place near existing centers
            idx = rng.integers(0, len(centers))
            center = centers[idx] + rng.normal(0, 1.5, 3)
        
        centers.append(center)
        atoms.append(center)
        
        # Add icosahedral neighbors
        shell = _icosahedron_vertices(radius)
        # Random rotation
        # Simple: just add as-is (will be relaxed later)
        for v in shell:
            if len(atoms) >= n:
                break
            atoms.append(center + v)
    
    # Fill remaining with random positions
    while len(atoms) < n:
        p = rng.normal(0, 2.0, 3)
        atoms.append(p)
    
    return np.array(atoms[:n])

def _relax_with_deadline(positions, potential, parameter):
    """Relax with time check."""
    if _time_left() < 5.0:
        return _dzugutov_energy(positions), positions
    try:
        energy, new_pos = relax(positions, potential, parameter)
        return energy, new_pos
    except:
        return _dzugutov_energy(positions), positions

def construct(instance):
    """Construct lowest-energy Dzugutov cluster using icosahedral-based initialization."""
    n = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    best_energy = np.inf
    best_positions = None
    
    # Strategy 1: Icosahedral Mackay construction
    for seed in range(3):
        if _time_left() < 10.0:
            break
        pos = _build_icosahedral_cluster(n, seed)
        energy, pos = _relax_with_deadline(pos, potential, parameter)
        if energy < best_energy:
            best_energy = energy
            best_positions = pos
    
    # Strategy 2: Random icosahedral units
    for seed in range(3):
        if _time_left() < 10.0:
            break
        pos = _build_random_icosahedral(n, seed)
        energy, pos = _relax_with_deadline(pos, potential, parameter)
        if energy < best_energy:
            best_energy = energy
            best_positions = pos
    
    # Strategy 3: Basin hopping from best so far
    if _time_left() > 20.0 and best_positions is not None:
        rng = np.random.default_rng(42)
        current_energy = best_energy
        current_pos = best_positions.copy()
        
        # Simple basin hopping: perturb, relax, accept if better
        step_size = 0.3
        for i in range(20):
            if _time_left() < 10.0:
                break
            # Perturb
            perturbed = current_pos + rng.normal(0, step_size, current_pos.shape)
            energy, new_pos = _relax_with_deadline(perturbed, potential, parameter)
            
            # Accept if better (or always for exploration)
            if energy < current_energy:
                current_energy = energy
                current_pos = new_pos
                step_size = max(0.1, step_size * 0.95)
            else:
                step_size = min(0.8, step_size * 1.1)
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_positions = current_pos
    
    # Final relaxation of best
    if best_positions is not None and _time_left() > 5.0:
        energy, best_positions = _relax_with_deadline(best_positions, potential, parameter)
        if energy < best_energy:
            best_energy = energy
    
    if best_positions is None:
        # Fallback: random
        rng = np.random.default_rng(n)
        best_positions = rng.normal(0, 1.0, (n, 3))
    
    return best_positions
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
import itertools

def _icosahedron_vertices():
    """Return 12 vertices of a regular icosahedron (unit length from center)."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0,  1,  phi], [0, -1,  phi], [0,  1, -phi], [0, -1, -phi],
        [1,  phi, 0], [-1,  phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [ phi, 0,  1], [ phi, 0, -1], [-phi, 0,  1], [-phi, 0, -1]
    ])
    # Normalize to unit length
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    return verts / norms

def _mackay_55(r1=1.1, r2=1.9):
    """Construct a 55-atom Mackay icosahedron.
    1 center + 12 inner shell + 42 outer shell.
    Outer shell: 12 at icosahedron vertices (scaled), 20 at face centers, 10 at... 
    Actually, let's use: 12 vertices + 20 face centers + 10 edge-related = 42
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    # 12 icosahedron vertices
    v = np.array([
        [0,  1,  phi], [0, -1,  phi], [0,  1, -phi], [0, -1, -phi],
        [1,  phi, 0], [-1,  phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [ phi, 0,  1], [ phi, 0, -1], [-phi, 0,  1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    v_unit = v / norms
    
    # 20 face centers of icosahedron
    # Faces of icosahedron: each face is a triangle of 3 vertices
    # We need to identify the 20 triangular faces
    # For a standard icosahedron, the faces can be enumerated
    # Let's compute face centers by averaging triples of vertices that form faces
    
    # The 20 faces of an icosahedron with the above vertex ordering:
    # This is tricky to enumerate manually. Let's use a different approach.
    # Generate face centers by taking all combinations of 3 vertices that are mutually at the same distance (edge length)
    
    edge_len = np.linalg.norm(v_unit[0] - v_unit[1])
    face_centers = []
    for i, j, k in itertools.combinations(range(12), 3):
        d1 = np.linalg.norm(v_unit[i] - v_unit[j])
        d2 = np.linalg.norm(v_unit[j] - v_unit[k])
        d3 = np.linalg.norm(v_unit[k] - v_unit[i])
        if abs(d1 - edge_len) < 1e-6 and abs(d2 - edge_len) < 1e-6 and abs(d3 - edge_len) < 1e-6:
            fc = (v_unit[i] + v_unit[j] + v_unit[k]) / 3.0
            fc = fc / np.linalg.norm(fc)
            face_centers.append(fc)
    
    face_centers = np.array(face_centers)
    # Should have 20 unique face centers
    # Deduplicate
    unique_fcs = []
    for fc in face_centers:
        if not any(np.linalg.norm(fc - ufc) < 1e-6 for ufc in unique_fcs):
            unique_fcs.append(fc)
    face_centers = np.array(unique_fcs)
    
    # 10 edge midpoints? No, we need 42 - 12 - 20 = 10 more atoms
    # Actually, let me reconsider. The second shell of Mackay icosahedron has 42 atoms.
    # 12 + 20 = 32, so we need 10 more.
    # These could be at edge midpoints of the icosahedron, but there are 30 edges.
    # Let's just use 12 vertices + 20 face centers + 10 selected edge midpoints
    
    # Edge midpoints
    edge_mids = []
    for i, j in itertools.combinations(range(12), 2):
        d = np.linalg.norm(v_unit[i] - v_unit[j])
        if abs(d - edge_len) < 1e-6:
            em = (v_unit[i] + v_unit[j]) / 2.0
            em = em / np.linalg.norm(em)
            edge_mids.append(em)
    edge_mids = np.array(edge_mids)
    # Should have 30 edge midpoints
    unique_ems = []
    for em in edge_mids:
        if not any(np.linalg.norm(em - uem) < 1e-6 for uem in unique_ems):
            unique_ems.append(em)
    edge_mids = np.array(unique_ems)
    
    # Select 10 edge midpoints - let's just take the first 10
    selected_ems = edge_mids[:10]
    
    # Build the 55-atom cluster
    atoms = [np.zeros(3)]  # center
    atoms.extend(v_unit * r1)  # 12 inner shell
    atoms.extend(v_unit * r2)  # 12 outer shell vertices
    atoms.extend(face_centers * r2)  # 20 face centers
    atoms.extend(selected_ems * r2)  # 10 edge midpoints
    
    return np.array(atoms)

def _remove_atoms_symmetrically(positions, n_remove, rng):
    """Remove atoms from outer shell, preferring symmetric removal."""
    n = len(positions)
    if n_remove <= 0:
        return positions.copy()
    
    # Sort by distance from center (outer atoms first)
    center = np.mean(positions, axis=0)
    dists = np.linalg.norm(positions - center, axis=1)
    order = np.argsort(dists)[::-1]  # farthest first
    
    # Remove from outermost, but try to maintain some symmetry
    # Simple approach: remove n_remove atoms from the outer shell
    to_remove = set(order[:n_remove].tolist())
    
    remaining = [positions[i] for i in range(n) if i not in to_remove]
    return np.array(remaining)

def _add_atoms_near_surface(positions, n_add, rng, spacing=1.1):
    """Add atoms near the surface of the cluster."""
    n = len(positions)
    center = np.mean(positions, axis=0)
    
    new_atoms = []
    attempts = 0
    while len(new_atoms) < n_add and attempts < 1000:
        attempts += 1
        # Pick a random existing atom and place new atom nearby
        i = rng.integers(0, n)
        direction = rng.normal(size=3)
        direction = direction / np.linalg.norm(direction)
        # Place at a distance that maintains minimum separation
        candidate = positions[i] + direction * spacing * 0.8
        # Check minimum distance
        if all(np.linalg.norm(candidate - p) >= 0.5 for p in positions) and \
           all(np.linalg.norm(candidate - p) >= 0.5 for p in new_atoms):
            new_atoms.append(candidate)
    
    return np.vstack([positions, np.array(new_atoms)]) if new_atoms else positions

def construct(instance):
    """Construct low-energy cluster using icosahedral seeding and relaxation."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(42)
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse icosahedral-based seeds
    n_seeds = 12
    
    for seed_idx in range(n_seeds):
        # Vary the icosahedron parameters
        r1 = 1.05 + 0.05 * (seed_idx % 3)
        r2 = 1.85 + 0.05 * (seed_idx % 3)
        
        # Generate base 55-atom icosahedron
        base = _mackay_55(r1=r1, r2=r2)
        
        if size == 55:
            candidate = base
        elif size < 55:
            n_remove = 55 - size
            # Try different removal strategies
            if seed_idx < 6:
                candidate = _remove_atoms_symmetrically(base, n_remove, rng)
            else:
                # Random removal from outer atoms
                center = np.mean(base, axis=0)
                dists = np.linalg.norm(base - center, axis=1)
                order = np.argsort(dists)[::-1]
                to_remove = set(order[:n_remove].tolist())
                candidate = np.array([base[i] for i in range(55) if i not in to_remove])
        else:  # size > 55
            n_add = size - 55
            candidate = _add_atoms_near_surface(base, n_add, rng, spacing=1.0)
        
        if len(candidate) != size:
            continue
        
        # Relax
        try:
            energy, positions = relax(candidate, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            continue
    
    # Also try a few random compact seeds as fallback
    spacing = 1.1
    radius = spacing * size ** (1.0 / 3.0) * 0.75
    for _ in range(4):
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.7 * spacing for q in points):
                points.append(p)
        try:
            energy, positions = relax(np.array(points), potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            continue
    
    if best_positions is None:
        # Fallback: simple grid
        points = []
        for i in range(size):
            p = np.array([i * 1.0, (i // 10) * 1.0, (i // 100) * 1.0])
            points.append(p)
        best_positions = np.array(points)
    
    return best_positions
# EVOLVE-BLOCK-END

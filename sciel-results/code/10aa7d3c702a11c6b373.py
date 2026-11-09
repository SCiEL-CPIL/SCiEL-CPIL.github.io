import numpy as np
from autoresearch_suite.chem_clusters import relax

def generate_mackay_icosahedron(edge_length):
    """
    Generates the coordinates of a 55-atom Mackay icosahedron.
    Returns (positions, types) where types indicate shell membership for symmetric trimming.
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts = verts / 2.0 * edge_length
    
    center = np.array([0, 0, 0])
    
    # Edge midpoints
    edge_midpoints = []
    n_verts = len(verts)
    for i in range(n_verts):
        for j in range(i + 1, n_verts):
            dist = np.linalg.norm(verts[i] - verts[j])
            if np.isclose(dist, edge_length):
                mid = (verts[i] + verts[j]) / 2.0
                edge_midpoints.append(mid)
    edge_midpoints = np.array(edge_midpoints)
    
    # Face centers
    face_centers = []
    for i in range(n_verts):
        for j in range(i + 1, n_verts):
            if np.isclose(np.linalg.norm(verts[i] - verts[j]), edge_length):
                for k in range(j + 1, n_verts):
                    if (np.isclose(np.linalg.norm(verts[i] - verts[k]), edge_length) and
                        np.isclose(np.linalg.norm(verts[j] - verts[k]), edge_length)):
                        center_face = (verts[i] + verts[j] + verts[k]) / 3.0
                        face_centers.append(center_face)
    face_centers = np.array(face_centers)
    
    # Combine and assign types
    # Type 0: Center
    # Type 1: Vertices
    # Type 2: Edge Midpoints
    # Type 3: Face Centers
    
    positions = np.vstack([center[None, :], verts, edge_midpoints, face_centers])
    types = np.concatenate([
        np.zeros(1, dtype=int),
        np.ones(12, dtype=int),
        np.full(len(edge_midpoints), 2, dtype=int),
        np.full(len(face_centers), 3, dtype=int)
    ])
    
    return positions, types

def construct(instance):
    n = instance['n']
    potential = instance['potential']
    
    L = 1.05
    candidates = []
    rng = np.random.default_rng(42)
    
    # 1. Generate Mackay 55-atom cluster
    icos_pos, icos_types = generate_mackay_icosahedron(L)
    
    if n <= 55:
        # Symmetric Trimming
        # Priority of removal: Face Centers (3) > Edge Midpoints (2) > Vertices (1)
        # We want to keep the most "core" atoms.
        # Sort by type (ascending, so 0 is first, 3 is last)
        sort_idx = np.argsort(icos_types)
        selected = sort_idx[:n]
        candidates.append(icos_pos[selected])
        
        # Also try a random subset of the outer shell if n is close to 55
        if n > 30:
            # Keep all core (types 0,1,2) and random subset of type 3?
            # Or just random perturb the trimmed one
            noise = rng.normal(0, 0.02, icos_pos[selected].shape)
            candidates.append(icos_pos[selected] + noise)

    elif n > 55:
        # Start with 55-atom Mackay
        base = icos_pos
        num_to_add = n - 55
        # Add atoms on the surface
        # Estimate outer radius
        radii = np.linalg.norm(base, axis=1)
        R_outer = np.max(radii)
        
        added = []
        for _ in range(num_to_add):
            # Random point on sphere of radius R_outer
            u = rng.normal(0, 1, 3)
            u = u / np.linalg.norm(u)
            p = u * R_outer
            # Check distance to existing atoms
            valid = True
            for q in base:
                if np.linalg.norm(p - q) < 0.5:
                    valid = False
                    break
            if valid:
                added.append(p)
        
        if added:
            full = np.vstack([base, np.array(added)])
            candidates.append(full)
            
        # Also try a random icosahedral perturbation of the 55-atom cluster
        # by moving atoms slightly
        noise = rng.normal(0, 0.1, (55, 3))
        candidates.append(base + noise)
        
        # If n is significantly larger, a random start might be better?
        # But icosahedral bias is strong.
        # Let's add a random close packing fallback for n > 60? No, max is 60.
        # For n=60, adding 5 atoms to Mackay should work.

    # Add a generic random icosahedral candidate
    # Generate random points in a sphere and relax
    radius = L * n ** (1/3) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < 5000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
                attempts = 0
            else:
                attempts += 1
    if len(points) == n:
        candidates.append(np.array(points))

    best_energy = np.inf
    best_positions = None
    
    for cand in candidates:
        if len(cand) != n:
            continue
        try:
            energy, positions = relax(cand, potential)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            pass
            
    if best_positions is None:
        rng = np.random.default_rng(0)
        best_positions = rng.uniform(-5, 5, (n, 3))
        
    return best_positions

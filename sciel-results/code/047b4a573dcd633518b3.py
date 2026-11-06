import numpy as np
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
import time

def _get_icosahedron_vertices(radius=1.0):
    """
    Generate the 12 vertices of a regular icosahedron centered at origin.
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi],
        [0, -1, phi],
        [0, 1, -phi],
        [0, -1, -phi],
        [1, phi, 0],
        [-1, phi, 0],
        [1, -phi, 0],
        [-1, -phi, 0],
        [phi, 0, 1],
        [phi, 0, -1],
        [-phi, 0, 1],
        [-phi, 0, -1]
    ], dtype=float)
    
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis]
    return verts * radius

def _get_icosahedron_edges(verts):
    """
    Get the 30 edges of the icosahedron defined by the 12 vertices.
    Returns list of (i, j) index pairs.
    """
    min_dist = np.inf
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < min_dist:
                min_dist = d
    
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if np.isclose(d, min_dist):
                edges.append((i, j))
    return edges

def _get_icosahedron_faces(verts, edges):
    """
    Get the 20 faces of the icosahedron.
    Returns list of (i, j, k) index triples.
    """
    # Build adjacency
    adj = {i: set() for i in range(12)}
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    
    # Find all triangles (faces)
    faces = []
    for i in range(12):
        for j in adj[i]:
            if j > i:
                for k in adj[i] & adj[j]:
                    if k > j:
                        faces.append((i, j, k))
    
    return faces

def _build_mackay_55(r1=1.0, r2=1.87):
    """
    Build the perfect 55-atom Mackay icosahedron.
    Returns array of shape (55, 3).
    """
    verts = _get_icosahedron_vertices(1.0)
    edges = _get_icosahedron_edges(verts)
    
    # Center atom
    points = [np.array([0.0, 0.0, 0.0])]
    
    # Shell 1: 12 atoms at radius r1
    for v in verts:
        points.append(v * r1)
    
    # Shell 2: 42 atoms at radius r2
    # 12 vertices + 30 edge midpoints
    shell2 = []
    for v in verts:
        shell2.append(v)
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        shell2.append(mid)
    
    shell2 = np.array(shell2)
    norms = np.linalg.norm(shell2, axis=1)
    shell2 = shell2 / norms[:, np.newaxis] * r2
    
    for v in shell2:
        points.append(v)
    
    return np.array(points[:55])

def _defect_seed_55_to_n(n, r1=1.0, r2=1.87, seed=0):
    """
    Generate a defective Mackay seed for n atoms by removing atoms from the 55-atom Mackay.
    For n > 55, add atoms on a third shell.
    """
    mackay55 = _build_mackay_55(r1, r2)
    rng = np.random.default_rng(seed)
    
    if n <= 55:
        # Remove (55 - n) atoms from the Mackay icosahedron
        # Preferentially remove atoms from the outer shell (indices 13-54)
        # to preserve the icosahedral core structure
        
        # Indices: 0=center, 1-12=shell1, 13-54=shell2
        # For n=38: remove 17 atoms
        # For n=45: remove 10 atoms
        # For n=50: remove 5 atoms
        
        # Strategy: remove from shell2 first, then shell1 if needed
        # Keep the center and as many shell1 atoms as possible
        
        keep_indices = list(range(n))  # Keep first n atoms (center + shell1 + some shell2)
        
        # But this is biased. Let's be smarter:
        # For n >= 13, keep center + all 12 shell1 atoms
        # For n < 13, keep center + n-1 shell1 atoms
        
        if n >= 13:
            # Keep center (0) and all 12 shell1 atoms (1-12)
            keep = set([0]) | set(range(1, 13))
            remaining = n - 13
            # Pick 'remaining' atoms from shell2 (indices 13-54)
            shell2_indices = list(range(13, 55))
            rng.shuffle(shell2_indices)
            keep.update(shell2_indices[:remaining])
            keep_indices = sorted(keep)
        else:
            # Keep center and n-1 shell1 atoms
            keep = {0}
            shell1_indices = list(range(1, 13))
            rng.shuffle(shell1_indices)
            keep.update(shell1_indices[:n-1])
            keep_indices = sorted(keep)
        
        return mackay55[keep_indices]
    
    else:
        # n > 55: Start with full Mackay 55 and add (n - 55) atoms
        # Add atoms on a third shell (face centers of the icosahedron, scaled out)
        verts = _get_icosahedron_vertices(1.0)
        edges = _get_icosahedron_edges(verts)
        faces = _get_icosahedron_faces(verts, edges)
        
        points = [mackay55]
        
        # Add atoms at face centers, scaled to radius r3
        r3 = 2.6
        face_centers = []
        for i, j, k in faces:
            fc = (verts[i] + verts[j] + verts[k]) / 3.0
            fc = fc / np.linalg.norm(fc) * r3
            face_centers.append(fc)
        
        # Shuffle face centers for diversity
        rng.shuffle(face_centers)
        
        for fc in face_centers:
            if len(points) + len(points[0]) >= n:
                break
            points[0] = np.vstack([points[0], fc])
        
        # If still not enough, add random points on a sphere
        if len(points[0]) < n:
            current = points[0]
            for _ in range(n - len(current)):
                u, v = rng.uniform(0, 1), rng.uniform(0, 1)
                theta = 2 * np.pi * u
                phi_angle = np.arccos(2 * v - 1)
                dir_vec = np.array([
                    np.sin(phi_angle) * np.cos(theta),
                    np.sin(phi_angle) * np.sin(theta),
                    np.cos(phi_angle)
                ])
                r_new = r3 + rng.uniform(0.3, 0.7)
                p_new = dir_vec * r_new
                points[0] = np.vstack([points[0], p_new])
        
        return points[0][:n]

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    
    start_time = time.time()
    deadline = start_time + 290  # 10s margin before 300s limit
    
    best_energy = float('inf')
    best_positions = None
    
    def try_relax(init_positions, label=""):
        nonlocal best_energy, best_positions
        if time.time() > deadline:
            return
        try:
            # Use the provided relax function
            e, pos = relax(init_positions, potential, None)
            if e < best_energy and np.all(np.isfinite(pos)):
                best_energy = e
                best_positions = pos
        except Exception:
            pass

    # Generate diverse seeds using defect-seeded Mackay approach
    seeds = []
    
    # Multiple defect-seeded variants with different random removals/additions
    for seed_val in range(5):
        try:
            seeds.append(_defect_seed_55_to_n(size, seed=seed_val))
        except Exception:
            pass
    
    # Add some perturbed versions of the best defect seed
    if seeds:
        for perturb_seed in range(3):
            try:
                rng = np.random.default_rng(100 + perturb_seed)
                base = seeds[0]
                perturbed = base + rng.normal(0, 0.15, base.shape)
                seeds.append(perturbed)
            except Exception:
                pass
    
    # Relax all seeds
    for i, seed in enumerate(seeds):
        if time.time() > deadline:
            break
        if seed is not None and len(seed) == size:
            try_relax(seed, label=f"seed_{i}")

    # Basin Hopping / Perturbation on the best found
    if best_positions is not None and time.time() < deadline - 20:
        for attempt in range(8):
            if time.time() > deadline:
                break
            try:
                rng = np.random.default_rng(1000 + attempt)
                amp = 0.1 + 0.03 * attempt
                perturbed = best_positions + rng.normal(0, amp, best_positions.shape)
                try_relax(perturbed, label=f"perturb_{attempt}")
            except Exception:
                pass

    # Final fallback
    if best_positions is None:
        best_positions = _defect_seed_55_to_n(size, seed=0)

    return best_positions

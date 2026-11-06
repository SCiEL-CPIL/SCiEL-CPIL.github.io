import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

POTENTIAL = 'dzugutov'
_GLOBAL_START_TIME = None
_GLOBAL_DEADLINE = None


def _get_icosahedron_vertices():
    """Return 12 vertices of regular icosahedron, unit radius."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    return verts / norms


def _generate_mackay_55():
    """
    Generate the 55-atom Mackay icosahedron.
    Structure: 1 (center) + 12 (shell 1) + 42 (shell 2)
    Shell 2 consists of 30 edge midpoints and 12 face centers (forming a dodecahedron).
    """
    verts = _get_icosahedron_vertices()
    phi = (1 + np.sqrt(5)) / 2
    
    # Calculate radii
    # Shell 1 is at radius 1.
    # Shell 2 atoms are positioned to form a fullerene-like structure.
    # In a standard Mackay icosahedron, the outer shell atoms are at distance R2 from center.
    # R2 = R1 * sqrt(1 + 1/phi^2) is a common approximation for the geometric expansion.
    
    # Find edges and faces of the icosahedron
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            dists[i, j] = d
            dists[j, i] = d
    min_dist = np.min(dists[dists > 0])
    
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                edges.append((i, j))
                
    faces = []
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[j] - verts[k])
                d3 = np.linalg.norm(verts[i] - verts[k])
                if abs(d1 - min_dist) < 1e-6 and abs(d2 - min_dist) < 1e-6 and abs(d3 - min_dist) < 1e-6:
                    faces.append((i, j, k))

    center = np.zeros(3)
    shell1 = verts.copy() # 12 atoms
    
    # Shell 2: 30 edge midpoints + 12 face centers
    # We need to select 12 face centers that form a dodecahedron.
    # The 20 faces of the icosahedron correspond to the 20 vertices of the dual dodecahedron.
    # A dodecahedron has 12 vertices. We need to select 12 faces of the icosahedron that are non-adjacent?
    # Actually, the vertices of the dodecahedron are the centers of the faces of the icosahedron.
    # There are 20 faces, so there are 20 face centers.
    # We need to pick 12 of them that form a regular dodecahedron.
    # The 12 vertices of a dodecahedron correspond to the 12 faces of the icosahedron that are NOT adjacent to a specific set?
    # Simpler: The 42 atoms of the second shell in a Mackay icosahedron are:
    # 30 atoms at the midpoints of the 30 edges.
    # 12 atoms at the centers of 12 of the 20 faces.
    # Which 12 faces? The ones that form a dodecahedron.
    # A regular dodecahedron has 12 vertices. These vertices are the centers of 12 faces of the icosahedron.
    # The 20 faces can be partitioned into two sets of 12? No.
    # The dual of the icosahedron is the dodecahedron. The vertices of the dodecahedron are the face centers of the icosahedron.
    # Wait, a dodecahedron has 12 vertices. An icosahedron has 20 faces.
    # So only 12 of the 20 face centers are used?
    # No, the Mackay icosahedron shell 2 has 42 atoms.
    # 30 edge midpoints + 12 face centers = 42.
    # The 12 face centers must be chosen such that they form a regular dodecahedron.
    # The 12 vertices of a regular dodecahedron correspond to the 12 faces of the icosahedron that are "farthest" from each other?
    # Actually, the 20 faces of the icosahedron can be grouped into two sets of 10?
    # Let's just pick 12 face centers that are mutually non-adjacent (no shared edge).
    # In an icosahedron, each face is adjacent to 3 others.
    # We can use a greedy algorithm or a known set.
    # Known set: The 12 vertices of the dodecahedron are the centers of the faces of the icosahedron that are NOT part of a specific "belt"?
    # Let's just use the fact that the 12 atoms form a dodecahedron.
    # We can generate the dodecahedron vertices directly and check which ones are close to face centers.
    
    # Dodecahedron vertices (unit radius)
    phi_d = (1 + np.sqrt(5)) / 2
    dodeca_verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, 1/phi_d, phi_d], [0, 1/phi_d, -phi_d], [0, -1/phi_d, phi_d], [0, -1/phi_d, -phi_d],
        [1/phi_d, phi_d, 0], [1/phi_d, -phi_d, 0], [-1/phi_d, phi_d, 0], [-1/phi_d, -phi_d, 0],
        [phi_d, 0, 1/phi_d], [phi_d, 0, -1/phi_d], [-phi_d, 0, 1/phi_d], [-phi_d, 0, -1/phi_d]
    ], dtype=float)
    # Normalize to unit sphere
    dodeca_verts /= np.linalg.norm(dodeca_verts, axis=1, keepdims=True)
    
    # Find the 12 face centers of the icosahedron that are closest to the dodecahedron vertices
    face_centers = []
    for i, j, k in faces:
        fc = (verts[i] + verts[j] + verts[k]) / 3
        fc /= np.linalg.norm(fc)
        face_centers.append(fc)
    face_centers = np.array(face_centers)
    
    selected_face_centers = []
    used_faces = set()
    for dv in dodeca_verts:
        # Find closest face center
        dists_to_faces = np.linalg.norm(face_centers - dv, axis=1)
        min_idx = np.argmin(dists_to_faces)
        if min_idx not in used_faces:
            selected_face_centers.append(face_centers[min_idx])
            used_faces.add(min_idx)
            
    # If we didn't get 12, just take the first 12
    if len(selected_face_centers) < 12:
        for i in range(20):
            if i not in used_faces and len(selected_face_centers) < 12:
                selected_face_centers.append(face_centers[i])
                used_faces.add(i)
                
    selected_face_centers = np.array(selected_face_centers[:12])
    
    # Scale to correct radius
    # The radius of the outer shell in a Mackay icosahedron is R2.
    # R2 = R1 * sqrt(1 + 1/phi^2)
    r_shell2 = np.sqrt(1 + 1 / phi**2)
    
    edge_mids = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2
        mid /= np.linalg.norm(mid)
        edge_mids.append(mid)
    edge_mids = np.array(edge_mids)
    
    shell2 = np.vstack([edge_mids * r_shell2, selected_face_centers * r_shell2])
    
    return np.vstack([center, shell1, shell2])


def _generate_seeds_for_size(n, rng, mackay55):
    """Generate a list of seed structures for a given size n."""
    seeds = []
    
    if n == 55:
        seeds.append(mackay55.copy())
        # Add some perturbations
        for _ in range(3):
            seeds.append(mackay55 + rng.normal(0, 0.05, size=(55, 3)))
        return seeds

    if n < 55:
        n_remove = 55 - n
        # Remove atoms from the outer shell (indices 13 to 54)
        outer_indices = list(range(13, 55))
        
        # Generate multiple removal patterns
        for _ in range(5):
            remove_idx = rng.choice(outer_indices, size=n_remove, replace=False)
            keep_mask = np.ones(55, dtype=bool)
            keep_mask[remove_idx] = False
            seed = mackay55[keep_mask]
            seeds.append(seed)
            
        # Also try removing from the whole cluster if n is small (unlikely to be better, but safe)
        if n < 45:
            for _ in range(2):
                remove_idx = rng.choice(55, size=n_remove, replace=False)
                keep_mask = np.ones(55, dtype=bool)
                keep_mask[remove_idx] = False
                seed = mackay55[keep_mask]
                seeds.append(seed)
        
        return seeds

    else: # n > 55
        n_add = n - 55
        base = mackay55.copy()
        
        # Strategy 1: Add atoms on a sphere of radius slightly larger than outer shell
        r_outer = np.max(np.linalg.norm(base, axis=1))
        r_new = r_outer * 1.1
        
        for _ in range(5):
            new_atoms = rng.uniform(0, 2*np.pi, size=(n_add, 1))
            new_atoms_v = rng.uniform(-1, 1, size=(n_add, 1))
            # Spherical coords
            theta = new_atoms.flatten()
            phi = np.arccos(new_atoms_v.flatten())
            
            xs = r_new * np.sin(phi) * np.cos(theta)
            ys = r_new * np.sin(phi) * np.sin(theta)
            zs = r_new * np.cos(phi)
            
            new_pos = np.column_stack((xs, ys, zs))
            seed = np.vstack([base, new_pos])
            seeds.append(seed)
            
        # Strategy 2: Add atoms at specific symmetric positions (e.g., near the "holes" of the outer shell)
        # Just perturb the outer shell atoms slightly outward and duplicate some?
        # Let's just add random atoms near the surface
        for _ in range(3):
            new_atoms = []
            for _ in range(n_add):
                # Pick a random existing outer atom
                ref_idx = rng.integers(13, 55)
                ref_pos = base[ref_idx]
                # Add small offset
                offset = rng.normal(0, 0.3, 3)
                new_atom = ref_pos + offset
                new_atoms.append(new_atom)
            new_pos = np.array(new_atoms)
            seed = np.vstack([base, new_pos])
            seeds.append(seed)
            
        return seeds


def construct(instance):
    global _GLOBAL_START_TIME, _GLOBAL_DEADLINE

    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        _GLOBAL_DEADLINE = _GLOBAL_START_TIME + 290.0

    size = instance["n"]
    start_time = time.time()
    instance_deadline = min(start_time + 45.0, _GLOBAL_DEADLINE - 5.0)

    rng = np.random.default_rng(42 + size * 7)

    # Generate the reference Mackay 55
    try:
        mackay55 = _generate_mackay_55()
    except:
        mackay55 = None

    best_energy = float('inf')
    best_positions = None

    seeds = []
    if mackay55 is not None:
        seeds = _generate_seeds_for_size(size, rng, mackay55)
    else:
        # Fallback random seed
        radius = 0.8 * size ** (1 / 3) + 1.0
        points = []
        for _ in range(size):
            for _ in range(500):
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) > radius:
                    continue
                if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                    points.append(p)
                    break
            else:
                p = rng.uniform(-radius * 1.2, radius * 1.2, 3)
                points.append(p)
        seeds.append(np.array(points))

    # Relax each seed
    for i, seed in enumerate(seeds):
        if time.time() > instance_deadline - 5:
            break
            
        try:
            # Use the provided relax function
            energy, relaxed_pos = relax(seed, POTENTIAL)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos.copy()
        except Exception:
            continue

    # If we have time, do a short basin hopping on the best result
    if best_positions is not None and time.time() < instance_deadline - 10:
        # Simple basin hopping
        current_pos = best_positions.copy()
        current_energy = best_energy
        temperature = 0.1
        step_size = 0.2
        
        while time.time() < instance_deadline - 5:
            perturbation = rng.normal(0, step_size, size=(size, 3))
            new_pos = current_pos + perturbation
            
            try:
                new_energy, new_relaxed = relax(new_pos, POTENTIAL)
            except:
                continue
                
            if new_energy < best_energy:
                best_energy = new_energy
                best_positions = new_relaxed.copy()
                current_pos = new_relaxed.copy()
                step_size *= 0.95
            else:
                delta_e = new_energy - current_energy
                acceptance_prob = np.exp(-delta_e / temperature)
                if rng.random() < acceptance_prob:
                    current_pos = new_relaxed.copy()
                    current_energy = new_energy
                    step_size *= 0.98
                else:
                    step_size *= 1.05
                    
            temperature *= 0.99
            temperature = max(temperature, 0.01)

    if best_positions is None:
        # Last resort
        best_positions = np.random.randn(size, 3) * 0.3
        try:
            best_energy, best_positions = relax(best_positions, POTENTIAL)
        except:
            pass

    return best_positions

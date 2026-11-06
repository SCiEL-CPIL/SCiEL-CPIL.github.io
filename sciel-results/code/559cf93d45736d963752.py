"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.chem_clusters import relax
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

def _generate_icosahedron_fragment(center, scale=1.0, rotation=None):
    """
    Generate a 13-atom icosahedral fragment (1 core + 12 vertices).
    Returns (13, 3) array of positions.
    """
    verts = _get_icosahedron_vertices()
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    if rotation is not None:
        verts = verts @ rotation.T
        
    positions = np.vstack([
        center,
        center + scale * verts
    ])
    return positions

def _random_rotation_matrix():
    """Generate a random rotation matrix."""
    # Use scipy if available, otherwise simple random orthonormal generation
    try:
        from scipy.spatial.transform import Rotation
        rot = Rotation.random().as_matrix()
        return rot
    except ImportError:
        # Fallback: random orthogonal matrix via QR decomposition of random matrix
        A = np.random.randn(3, 3)
        Q, R = np.linalg.qr(A)
        # Ensure right-handedness
        d = np.sign(np.linalg.det(Q))
        Q = Q * np.array([d, 1, 1])
        return Q

def _assemble_cluster_fragments(n_target, n_fragments, deadline):
    """
    Assemble a cluster by placing n_fragments icosahedra and merging them.
    """
    if n_fragments == 0:
        return None
    
    fragments = []
    fragment_centers = []
    
    # Place first fragment at origin
    center0 = np.zeros(3)
    frag0 = _generate_icosahedron_fragment(center0, scale=1.0, rotation=None)
    fragments.append(frag0)
    fragment_centers.append(center0)
    
    # Place remaining fragments
    # We want to place them close to each other to form a cluster, 
    # but not overlapping completely.
    # Ideal distance between centers of touching icosahedra is roughly 2 * vertex_radius
    # Vertex radius ~ 1.0 (if normalized to unit sphere, distance is 1.0)
    # So centers should be ~ 2.0 apart for sharing faces/edges?
    # Actually, Mackay icosahedra share vertices/edges.
    # Let's try to place them at a distance that allows sharing.
    
    target_radius = n_fragments ** 0.5 * 2.5 # Rough estimate for cluster size
    
    for i in range(1, n_fragments):
        if time.time() > deadline:
            break
            
        # Try to place new fragment
        placed = False
        for attempt in range(50):
            # Random direction and distance
            # Distance between centers: slightly less than 2*radius to allow overlap
            dist = np.random.uniform(1.5, 2.5) 
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            
            # Place near existing cluster
            # Pick a random existing fragment center and offset from it
            base_idx = np.random.randint(0, len(fragment_centers))
            base_center = fragment_centers[base_idx]
            new_center = base_center + dist * direction
            
            # Check if this center is too close to other fragment centers (min dist 1.0)
            # and if the resulting fragment doesn't cause atoms to be too close (< 0.5)
            
            # Quick check: distance to other centers
            min_center_dist = min(np.linalg.norm(new_center - c) for c in fragment_centers)
            if min_center_dist < 1.2:
                continue
                
            # Generate the fragment
            rot = _random_rotation_matrix()
            frag = _generate_icosahedron_fragment(new_center, scale=1.0, rotation=rot)
            
            # Check for atom-atom clashes with ALL existing atoms
            # This is expensive but necessary for validity
            clash = False
            for existing_frag in fragments:
                # Compute pairwise distances
                # (13, 3) vs (13, 3) -> (13, 13)
                diff = frag[:, np.newaxis, :] - existing_frag[np.newaxis, :, :]
                dists = np.linalg.norm(diff, axis=2)
                if np.min(dists) < 0.5:
                    clash = True
                    break
            
            if not clash:
                fragments.append(frag)
                fragment_centers.append(new_center)
                placed = True
                break
        
        if not placed:
            # If we can't place it cleanly, just place it randomly far away? 
            # Or force it? Forcing might create bad structures.
            # Let's try a looser constraint for the last few fragments if needed
            # For now, skip if can't place, we might have fewer fragments than intended
            pass

    if not fragments:
        return None

    # Merge all fragments into a single array
    all_positions = np.vstack(fragments)
    
    # If we have more atoms than n_target, we need to trim.
    # Trim from the outside? Or just take the first n_target?
    # Taking first n_target keeps the core structure intact.
    if len(all_positions) > n_target:
        # Sort by distance from centroid to keep compact core?
        centroid = np.mean(all_positions, axis=0)
        dists = np.linalg.norm(all_positions - centroid, axis=1)
        sorted_indices = np.argsort(dists)
        all_positions = all_positions[sorted_indices[:n_target]]
        
    # If we have fewer atoms than n_target, we need to add some.
    # This is rare if n_fragments is chosen well, but possible.
    if len(all_positions) < n_target:
        # Add random atoms near the surface
        while len(all_positions) < n_target:
            # Pick a random existing atom and add a new one nearby
            base_idx = np.random.randint(0, len(all_positions))
            base_pos = all_positions[base_idx]
            offset = np.random.randn(3) * 1.2
            new_pos = base_pos + offset
            
            # Check clash
            clash = False
            for p in all_positions:
                if np.linalg.norm(new_pos - p) < 0.5:
                    clash = True
                    break
            if not clash:
                all_positions = np.vstack([all_positions, new_pos])
            else:
                # Try a few times, then just give up on this specific addition
                # and move on. If we fail to add, we might return < n_target
                pass

    return all_positions

def construct(instance):
    """
    Fragment-merger strategy:
    1. Determine number of icosahedral fragments needed to reach n.
    2. Generate multiple random assemblies of these fragments.
    3. Relax each assembly.
    4. Return the best one.
    """
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    best_energy = np.inf
    best_positions = None
    
    # Determine number of fragments
    # 13 atoms per fragment.
    # n=38 -> 3 frags (39 atoms, trim 1)
    # n=45 -> 4 frags (52 atoms, trim 7)
    # n=50 -> 4 frags (52 atoms, trim 2)
    # n=55 -> 5 frags (65 atoms, trim 10)
    # n=57 -> 5 frags (65 atoms, trim 8)
    # n=60 -> 5 frags (65 atoms, trim 5)
    
    # Using slightly fewer fragments might be better to avoid excessive trimming?
    # e.g. for n=38, 3 fragments is good.
    # Let's try a range of fragment counts to allow diversity.
    
    n_fragments_base = max(2, int(np.ceil(size / 13.0)))
    
    # We will try a few different assembly attempts
    n_attempts = 8
    
    for attempt in range(n_attempts):
        if time.time() > deadline:
            break
            
        # Vary the number of fragments slightly to explore different packing
        # E.g. for n=38, try 3 fragments. For n=50, try 4.
        # Sometimes 1 more or 1 less might yield better local minima after trimming?
        # Trimming removes outer atoms, so having "extra" atoms provides a buffer 
        # that relaxes out, potentially finding a better core.
        
        n_frags = n_fragments_base
        if attempt % 2 == 1 and n_frags > 2:
            n_frags -= 1 # Try slightly smaller core, add atoms later?
            # Actually, if n_frags * 13 < size, we add random atoms.
            # If n_frags * 13 > size, we trim.
            # Trimming is generally safer for maintaining icosahedral order.
            # Let's stick to n_frags_base or base+1.
            n_frags = n_fragments_base + 1 if attempt % 4 == 1 else n_fragments_base
        
        positions = _assemble_cluster_fragments(size, n_frags, deadline)
        
        if positions is None or len(positions) != size:
            continue
            
        try:
            energy, relaxed_pos = relax(positions, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos
        except Exception:
            continue
            
    # Fallback if fragment assembly failed completely
    if best_positions is None:
        # Generate a simple random compact cluster as fallback
        rng = np.random.default_rng(42)
        radius = 1.1 * size ** (1.0/3.0) * 0.8
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                    points.append(p)
        points = np.array(points)
        try:
            energy, relaxed_pos = relax(points, potential, parameter)
            best_positions = relaxed_pos
        except Exception:
            # Last resort: return the unrelaxed points if relax fails
            best_positions = points

    return best_positions
# EVOLVE-BLOCK-END

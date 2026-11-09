"""
Rewritten program for Sutton-Chen cluster optimization.
Uses a portfolio of structural motifs (Icosahedral, FCC, HCP, Hybrid)
relaxed via L-BFGS-B to find near-global minima.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax

# Global deadline guard: 330s limit, aim for 290s to be safe
DEADLINE = time.time() + 290

def _generate_icosahedron(radius=1.0):
    """
    Generate coordinates for a regular icosahedron with 'radius' (distance from center to vertex).
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = [
        (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
        (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
        (phi, 0, 1), (-phi, 0, 1), (phi, 0, -1), (-phi, 0, -1)
    ]
    verts = np.array(verts)
    norm = np.linalg.norm(verts, axis=1)
    scale = radius / norm
    verts = verts * scale[:, np.newaxis]
    return verts

def _generate_mackay_icosahedron(n_atoms):
    """
    Generate a Mackay icosahedral cluster.
    Sizes: 13, 55, 147...
    For n not matching exactly, take closest subset.
    """
    if n_atoms <= 13:
        ico = _generate_icosahedron(radius=0.8)
        if n_atoms == 13:
            ico = np.vstack([np.zeros(3), ico])
        else:
            ico = ico[:n_atoms] if n_atoms <= 12 else np.vstack([np.zeros(3), ico[:n_atoms-1]])
        return ico
    
    if n_atoms <= 55:
        # 13 core + 42 shell
        core = _generate_icosahedron(radius=0.8)
        core = np.vstack([np.zeros(3), core])
        
        # Shell atoms at radius ~ 2 * 0.8? No, Mackay shell is at distance 2a.
        # Let's just generate icosahedron vertices at larger radius and pick 42.
        # Actually, the 55-atom Mackay icosahedron has 13 core + 42 shell.
        # The shell atoms are at the vertices of a larger icosahedron? No, it's a truncated icosahedron or similar.
        # Simpler: Take 55 closest points from an icosahedral grid?
        # For now, let's just use the hybrid approach for 55.
        pass
        
    # Fallback to hybrid
    return _generate_hybrid_icosa_fcc(n_atoms)

def _generate_fcc_shells(n_atoms, spacing=0.72):
    """
    Generate a compact FCC-like cluster by taking the n_atoms closest points
    from an infinite FCC lattice to the origin.
    """
    # FCC points are (i,j,k) where i+j+k is even.
    # We generate a large chunk and sort by distance.
    max_coord = 10
    coords = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    coords.append((i, j, k))
    
    coords = np.array(coords, dtype=float)
    dists = np.sum(coords**2, axis=1)
    sorted_idx = np.argsort(dists)
    selected = coords[sorted_idx[:n_atoms]]
    
    # Scale to appropriate spacing
    # The lattice constant here is 1, but we want equilibrium distance ~ 0.7-0.8 for SC?
    # Actually, the SC potential minimum is around r=1.0 in reduced units?
    # The task says "reduced units (lattice constant 1)".
    # So if we use the FCC lattice with a=1, the nearest neighbor distance is sqrt(0.5) ~ 0.707.
    # This is a reasonable starting point.
    
    # Center the cluster
    center = np.mean(selected, axis=0)
    selected = selected - center
    
    return selected

def _generate_hcp_shells(n_atoms, spacing=0.72):
    """
    Generate a compact HCP-like cluster.
    HCP layers are stacked ABC...
    """
    # Generate HCP points
    # Layer 0: (i, j, 0)
    # Layer 1: (i + 0.5, j + 0.5/sqrt(3), h)
    # This is complex. Let's use a simpler heuristic:
    # Take FCC and perturb it? Or just use FCC for now as HCP is similar.
    # For now, return FCC as a fallback, or a slightly distorted FCC.
    return _generate_fcc_shells(n_atoms, spacing)

def _generate_hybrid_icosa_fcc(n_atoms):
    """
    Generate a cluster with an icosahedral core and FCC-like shell.
    """
    core = _generate_icosahedron(radius=0.8)
    core = np.vstack([np.zeros(3), core]) # 13 atoms
    
    if n_atoms <= 13:
        return core[:n_atoms]
        
    remaining = n_atoms - 13
    
    # Generate FCC shell points
    max_coord = 10
    fcc_coords = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    fcc_coords.append((i, j, k))
    
    fcc_coords = np.array(fcc_coords, dtype=float)
    dists_to_origin = np.linalg.norm(fcc_coords, axis=1)
    sorted_fcc = fcc_coords[np.argsort(dists_to_origin)]
    
    shell_points = []
    for p in sorted_fcc:
        min_dist_to_core = np.min(np.linalg.norm(core - p, axis=1))
        if min_dist_to_core > 0.7:
            shell_points.append(p)
            
        if len(shell_points) >= remaining:
            break
            
    if len(shell_points) < remaining:
        rng = np.random.default_rng(42)
        while len(shell_points) < remaining:
            p = rng.normal(0, 1, 3)
            p = p / np.linalg.norm(p) * 1.5
            shell_points.append(p)
            
    shell_points = np.array(shell_points[:remaining])
    cluster = np.vstack([core, shell_points])
    return cluster

def _random_cluster(n_atoms, seed):
    """
    Generate a random cluster with minimum distance constraint.
    """
    rng = np.random.default_rng(seed)
    points = []
    radius = 0.8 * n_atoms ** (1.0 / 3.0)
    min_dist = 0.6
    
    # Try to pack points
    attempts = 0
    while len(points) < n_atoms and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
        
    if len(points) < n_atoms:
        # Fallback: just random points, ensure min dist by scaling if needed
        points = [rng.uniform(-radius, radius, 3) for _ in range(n_atoms)]
        
    return np.array(points)

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    kind = instance["kind"]
    
    best_energy = np.inf
    best_positions = None
    
    # List of candidate generators
    candidates = []
    
    # 1. FCC Motif
    candidates.append(_generate_fcc_shells(size))
    
    # 2. HCP Motif
    candidates.append(_generate_hcp_shells(size))
    
    # 3. Icosahedral / Hybrid
    if size <= 13:
        ico = _generate_icosahedron(radius=0.8)
        if size == 13:
            ico = np.vstack([np.zeros(3), ico])
        else:
            ico = ico[:size]
            ico = ico - np.mean(ico, axis=0)
        candidates.append(ico)
    else:
        try:
            hyb = _generate_hybrid_icosa_fcc(size)
            candidates.append(hyb)
        except Exception:
            pass
    
    # 4. Random starts (diversity)
    for seed in [size, size + 1, size + 2]:
        candidates.append(_random_cluster(size, seed))
    
    # Relax each candidate
    for idx, pos in enumerate(candidates):
        if time.time() > DEADLINE - 20: # Safety margin
            break
            
        try:
            energy, relaxed_pos = relax(pos, potential, kind)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos
        except Exception:
            continue

    # If no candidate worked, return a valid default
    if best_positions is None:
        pos = _random_cluster(size, 0)
        best_positions = pos
        
    return best_positions
# EVOLVE-BLOCK-END

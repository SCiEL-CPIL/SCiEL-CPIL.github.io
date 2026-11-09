"""
Combined program for Sutton-Chen cluster optimization.
Uses a portfolio of structural motifs (Icosahedral, FCC, HCP, Hybrid)
relaxed via L-BFGS-B to find near-global minima.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax

# Global deadline guard: 330s limit, aim for 280s to be safe
DEADLINE = time.time() + 280

def _generate_icosahedron(radius=1.0):
    """
    Generate coordinates for a regular icosahedron with 'radius' (distance from center to vertex).
    The standard icosahedron vertices are (0, +/-1, +/-phi), (+/-1, +/-phi, 0), (+/-phi, 0, +/-1).
    The norm of these is sqrt(1 + phi^2). We scale to desired radius.
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

def _generate_fcc_shells(n_atoms, spacing=0.72):
    """
    Generate a compact FCC-like cluster by taking the n_atoms closest points
    from an infinite FCC lattice to the origin.
    """
    if n_atoms <= 0:
        return np.empty((0, 3))
    
    # Generate a sufficiently large chunk of FCC lattice
    # FCC basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    # Lattice constant a=1.
    # We generate points in a cube around origin, then sort by distance.
    # For n=80, a cube of side length ~6-8 is usually enough.
    max_coord = int(np.ceil(n_atoms ** (1.0/3.0))) + 3
    coords = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # FCC points are (i,j,k) where i+j+k is even
                if (i + j + k) % 2 == 0:
                    coords.append((i, j, k))
    
    if not coords:
        # Fallback for very small n
        coords = [(0,0,0)]
        
    coords = np.array(coords, dtype=float)
    dists = np.sum(coords**2, axis=1)
    # Sort by distance to origin
    sorted_idx = np.argsort(dists)
    # Take the first n_atoms
    selected = coords[sorted_idx[:n_atoms]]
    
    # Center the cluster
    center = np.mean(selected, axis=0)
    selected = selected - center
    
    # Scale to appropriate spacing
    selected = selected * spacing
    
    return selected

def _generate_hcp_shells(n_atoms, spacing=0.72):
    """
    Generate a compact HCP-like cluster by filling HCP lattice sites by distance from center.
    """
    if n_atoms <= 0:
        return np.empty((0, 3))
    
    # Generate HCP lattice points
    # HCP: two atoms per unit cell, close-packed layers
    # Use a simple approach: generate points in close-packed arrangement
    sites = []
    a = 1.0  # in-plane spacing
    c = 1.633 * a  # c/a ratio for ideal hcp
    max_shell = int(np.ceil(n_atoms ** (1.0/3.0))) + 2
    
    for layer in range(-max_shell, max_shell + 1):
        z = layer * c / 2.0
        # Each layer is a triangular lattice
        offset = (layer % 2) * 0.5
        for i in range(-max_shell, max_shell + 1):
            for j in range(-max_shell, max_shell + 1):
                x = (i + offset) * a
                y = j * a * np.sqrt(3) / 2.0
                sites.append([x, y, z])
    
    sites = np.array(sites)
    # Remove duplicates
    sites = np.unique(np.round(sites, 10), axis=0)
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    sites = sites[idx]
    sites = sites - sites[:n_atoms].mean(axis=0)
    sites = sites[:n_atoms] * spacing
    return sites

def _generate_hybrid_icosa_fcc(n_atoms):
    """
    Generate a cluster with an icosahedral core (12 atoms + center = 13)
    and an FCC-like shell around it.
    This mimics the Mackay icosahedron or similar competitive structures.
    """
    # Core: 12 icosahedron vertices + 1 center
    # Radius of core ~ 1.0 (equilibrium distance in SC is approx 0.8-1.0 depending on unit)
    # Let's use radius 0.9 for core vertices to ensure stable core.
    core = _generate_icosahedron(radius=0.9)
    core = np.vstack([np.zeros(3), core]) # 13 atoms
    
    if n_atoms <= 13:
        # If very small, just return scaled core (though logic above handles larger)
        # For n<13, we should probably just use icosahedron fragments
        # But this function is for larger n.
        return core[:n_atoms]
        
    remaining = n_atoms - 13
    
    # Generate FCC shell points
    # We need points that are further out than the core.
    # Use the FCC generator but filter out points too close to the core
    # Or just generate a larger FCC chunk and take the next closest ones not in core.
    
    max_coord = 10
    fcc_coords = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    fcc_coords.append((i, j, k))
    
    fcc_coords = np.array(fcc_coords, dtype=float)
    
    # We want to place these around the icosahedron.
    # A simple heuristic: place FCC points at vertices of a larger icosahedron?
    # Or just take the next closest FCC points to the origin, but ensure they are outside the core radius.
    # The core vertices are at r=0.9.
    # FCC nearest neighbors to origin are at r=sqrt(0.5) ~ 0.707 (too close, inside core).
    # Next shell at r=1.0 (e.g., (1,1,0)).
    
    dists_to_origin = np.linalg.norm(fcc_coords, axis=1)
    # Sort FCC coords by distance to origin
    sorted_fcc = fcc_coords[np.argsort(dists_to_origin)]
    
    shell_points = []
    for p in sorted_fcc:
        # Check if this point is "outside" the core
        # Distance from p to all core atoms should be > some threshold (e.g., 0.8)
        min_dist_to_core = np.min(np.linalg.norm(core - p, axis=1))
        if min_dist_to_core > 0.8:
            shell_points.append(p)
            
        if len(shell_points) >= remaining:
            break
            
    # If we didn't find enough distinct FCC points (unlikely for reasonable n), 
    # we might need to generate random points or scale up. 
    # For now, if short, add random points around the surface.
    if len(shell_points) < remaining:
        # This case is rare for standard n, but safety first
        rng = np.random.default_rng(42)
        while len(shell_points) < remaining:
            # Random point on sphere of radius ~ 1.5
            p = rng.normal(0, 1, 3)
            p = p / np.linalg.norm(p) * 1.5
            shell_points.append(p)
            
    shell_points = np.array(shell_points[:remaining])
    
    # The FCC coordinates generated above are based on an integer lattice centered at 0.
    # The core is centered at 0.
    # However, the FCC lattice might not align perfectly with the icosahedron symmetry.
    # Relaxation will fix this.
    
    cluster = np.vstack([core, shell_points])
    return cluster

def _random_cluster(n, seed, min_dist=0.6, spacing=0.72):
    """Generate a random cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    max_radius = spacing * n ** (1.0 / 3.0) * 1.2
    points = []
    attempts = 0
    max_attempts = n * 1000
    while len(points) < n and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-max_radius, max_radius, 3)
        if np.linalg.norm(p) <= max_radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
    # If we couldn't place all, fill remaining randomly
    while len(points) < n:
        p = rng.uniform(-max_radius, max_radius, 3)
        points.append(p)
    points = np.array(points)
    return points

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho", "12-6"))
    
    best_energy = np.inf
    best_positions = None
    
    # List of candidate generators
    # We try different strategies depending on size
    candidates = []
    
    # 1. FCC Motif
    # Good for large N
    candidates.append(_generate_fcc_shells(size))
    
    # 2. HCP Motif
    # Good for some transition metals
    candidates.append(_generate_hcp_shells(size))
    
    # 3. Icosahedral Shells (for smaller N, or as a core)
    if size <= 13:
        # Just icosahedron
        ico = _generate_icosahedron(radius=1.0)
        if size == 13:
            ico = np.vstack([np.zeros(3), ico])
        else:
            # Take first 'size' points from icosahedron? 
            # Better: use a subset. For n=12, use vertices.
            # For n<12, use a fragment.
            # Simple heuristic: take first N vertices of icosahedron
            ico_verts = _generate_icosahedron(radius=1.0)
            ico = ico_verts[:size]
            # Center it
            ico = ico - np.mean(ico, axis=0)
        candidates.append(ico)
        
    elif size <= 55:
        # Icosahedral core + shell
        # Use the hybrid generator
        try:
            hyb = _generate_hybrid_icosa_fcc(size)
            candidates.append(hyb)
        except Exception:
            pass
            
        # Also try a simple "double icosahedron" idea if size allows?
        # Or just rely on FCC and Hybrid.
        
    else:
        # For large N, FCC is dominant.
        # Maybe try a larger icosahedron core with FCC shell?
        try:
            hyb = _generate_hybrid_icosa_fcc(size)
            candidates.append(hyb)
        except Exception:
            pass

    # 4. Random starts (fallback and diversity)
    rng = np.random.default_rng(size)
    # 3 random compact clusters
    for seed in [size, size + 1000, size + 2000]:
        try:
            rand_pts = _random_cluster(size, seed, min_dist=0.6)
            if rand_pts.shape[0] == size and np.all(np.isfinite(rand_pts)):
                candidates.append(rand_pts)
        except Exception:
            pass

    # Relax each candidate
    for idx, pos in enumerate(candidates):
        if time.time() > DEADLINE - 30: # Safety margin
            break
            
        try:
            energy, relaxed_pos = relax(pos, potential, kind)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos
        except Exception:
            continue

    # If no candidate worked (should not happen), return a valid default
    if best_positions is None:
        # Fallback: linear chain? Or just random valid.
        # A chain is bad energy but valid if dist > 0.35.
        # Better: random points with min dist 0.8.
        pos = []
        rng = np.random.default_rng(0)
        while len(pos) < size:
            p = rng.uniform(-5, 5, 3)
            if all(np.linalg.norm(p - q) >= 0.8 for q in pos):
                pos.append(p)
        best_positions = np.array(pos[:size])
        
    return best_positions
# EVOLVE-BLOCK-END

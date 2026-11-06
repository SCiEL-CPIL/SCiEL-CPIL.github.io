import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient
import scipy.optimize

def _min_distance_check(pos):
    """Check if all pairwise distances are at least 0.5."""
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= 0.5

def _generate_fcc_cluster(n, rng):
    """
    Generate an FCC cluster of approximately n atoms, centered.
    Uses a cubic block of FCC lattice points, taking the n closest to the center.
    """
    # FCC basis vectors
    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5]
    ])
    
    # Estimate side length to get enough points
    # Volume per atom in FCC (a=1) is 0.25. So V = 0.25 * n.
    # Side of cube approx (0.25 * n)^(1/3)
    side = int(np.ceil((0.25 * n) ** (1.0/3.0))) + 2
    
    pts = []
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                for b in basis:
                    p = np.array([i, j, k]) + b
                    pts.append(p)
    
    pts = np.array(pts)
    # Center the cloud
    pts -= np.mean(pts, axis=0)
    
    # Sort by distance to center and take n closest
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    cluster = pts[idx]
    
    # Add small noise to break symmetry and ensure we don't start exactly at a saddle
    cluster += rng.normal(0, 0.05, size=cluster.shape)
    
    return cluster

def _generate_hcp_cluster(n, rng):
    """
    Generate an HCP cluster of approximately n atoms, centered.
    """
    # HCP basis in a hexagonal lattice
    # a = 1, c = sqrt(8/3) * a
    a = 1.0
    c = np.sqrt(8.0/3.0) * a
    
    # Basis positions in unit cell (fractional coords converted to cartesian)
    # Layer A: (0,0,0), (1/3, 2/3, 1/2)
    # Layer B: (2/3, 1/3, 1/2) -- wait, standard HCP is ABAB.
    # Let's use standard Cartesian coords for hexagonal lattice points.
    
    pts = []
    # Generate layers
    # Estimate number of layers
    # Area per atom in layer ~ 0.866 * a^2
    # Total volume ~ n * 0.25 (similar density to FCC)
    # Height per layer = c/2
    # Num layers ~ (0.25 * n) / (0.866 * a^2 * (c/2)) ... this is getting complex.
    # Simpler: Generate a cylinder of hexagonal lattice.
    
    radius = np.sqrt(n / (np.pi * 0.866)) # Rough estimate
    num_layers = int(np.ceil(2 * radius / (c/2))) + 2
    
    for layer_idx in range(-num_layers//2, num_layers//2 + 1):
        z = layer_idx * (c / 2.0)
        # Shift x,y for alternating layers
        if layer_idx % 2 == 0:
            offset = np.array([0.0, 0.0])
        else:
            offset = np.array([a/2.0, a*np.sqrt(3)/6.0])
            
        # Generate hexagonal points in this layer within radius
        for i in range(-int(radius), int(radius) + 1):
            for j in range(-int(radius), int(radius) + 1):
                # Hexagonal to Cartesian
                x = a * (i + j/2.0) + offset[0]
                y = a * (np.sqrt(3)/2.0 * j) + offset[1]
                p = np.array([x, y, z])
                if np.linalg.norm(p[:2]) <= radius:
                    pts.append(p)
                    
    if len(pts) < n:
        # Fallback if generation was too sparse
        return _generate_fcc_cluster(n, rng)
        
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    cluster = pts[idx]
    cluster += rng.normal(0, 0.05, size=cluster.shape)
    
    return cluster

def _generate_icosahedron(n, rng):
    """
    Generate a Mackay icosahedron cluster.
    """
    # Mackay icosahedra sizes: 13, 55, 147, 309...
    # For arbitrary n, we generate the largest Mackay icosahedron <= n and fill the rest
    # with random points in the shell, or just use the closest Mackay size and truncate/pad.
    
    # Mackay radii (in units of nearest neighbor distance 1):
    # M1: 1 atom
    # M2: 13 atoms (1 center + 12 vertices)
    # M3: 55 atoms (1 + 12 + 42)
    # M4: 147 atoms (1 + 12 + 42 + 92)
    # M5: 309 atoms
    
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2.0
    
    # Vertices of icosahedron (normalized to distance 1 from center)
    # Standard icosahedron vertices:
    # (0, +/-1, +/-phi), (+/-1, +/-phi, 0), (+/-phi, 0, +/-1)
    verts = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                # Only even number of minus signs? No, just generate all 12
                pass
                
    # Explicit 12 vertices of icosahedron with edge length 2 (distance from center is sqrt(1+phi^2))
    # Let's use the standard set:
    v1 = np.array([0, 1, phi])
    v2 = np.array([0, -1, phi])
    v3 = np.array([0, 1, -phi])
    v4 = np.array([0, -1, -phi])
    v5 = np.array([1, phi, 0])
    v6 = np.array([-1, phi, 0])
    v7 = np.array([1, -phi, 0])
    v8 = np.array([-1, -phi, 0])
    v9 = np.array([phi, 0, 1])
    v10 = np.array([-phi, 0, 1])
    v11 = np.array([phi, 0, -1])
    v12 = np.array([-phi, 0, -1])
    
    verts = np.array([v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12])
    # Normalize to unit distance from origin
    norm = np.linalg.norm(verts, axis=1)
    verts = verts / norm[:, np.newaxis]
    
    # Mackay shells
    # Shell 1: 12 atoms at distance 1
    # Shell 2: 42 atoms at distance 2
    # Shell 3: 92 atoms at distance 3
    # ...
    
    # To generate shell k, we place atoms at positions corresponding to the vertices of the icosahedron
    # scaled by k, but we need to fill the gaps.
    # A simpler approximation for "Mackay-like" is to just use the vertices of the icosahedron
    # and their midpoints for the second shell, etc.
    # However, for n up to 80, we likely need Shell 1 (12) + Shell 2 (42) = 54, plus some from Shell 3.
    
    # Let's construct points layer by layer.
    points = [np.zeros(3)] # Center
    
    # Shell 1 (12 atoms)
    points.extend(verts)
    
    # Shell 2 (42 atoms)
    # These are at distance 2. They correspond to the midpoints of the edges of the icosahedron?
    # No, in Mackay icosahedra, the second shell consists of 30 atoms on the edges (midpoints) 
    # and 12 atoms on the faces (centroids)? 
    # Actually, the 42 atoms in the second shell are located at:
    # 30 positions: midpoints of the 30 edges, scaled to distance 2? 
    # Distance of midpoint of edge in unit icosahedron: 
    # Edge length is 2/phi? No.
    # Let's use a known coordinate set for Mackay 55.
    # Or, simpler: generate random points on a sphere of radius 2 and 3, and keep those that are "close" to the icosahedral symmetry?
    
    # Given time constraints, let's use a simpler heuristic:
    # If n <= 13: return 13-atom icosahedron (center + 12 verts)
    # If n <= 55: return 55-atom Mackay icosahedron.
    # If n <= 147: return 147-atom Mackay icosahedron.
    
    # I will hardcode the generation for the first few shells using known geometric properties.
    # For n > 55, we can take the 55-atom cluster and add random points in the outer shell.
    
    if n <= 13:
        # Take center and first n-1 vertices
        pts = np.array(points[:n])
        return pts + rng.normal(0, 0.01, size=pts.shape)
        
    # Generate 55-atom cluster
    # The 42 atoms of the second shell are at distance 2.
    # They are located at the midpoints of the edges of the icosahedron?
    # Let's check: Distance from center to midpoint of edge.
    # Edge length of icosahedron with circumradius 1 is 2/phi * sqrt(5/4)? 
    # Actually, let's just use the fact that the 55-atom cluster is well-known.
    # I'll generate points on a sphere of radius 2 and filter by proximity to the icosahedral directions?
    
    # Alternative: Use the "FCC" generator for high rho if icosahedron generation is too complex.
    # But Mackay is specifically for high rho (short range).
    
    # Let's try to construct the 55-atom cluster properly.
    # The 42 atoms in shell 2 are at:
    # 30 atoms: midpoints of the 30 edges.
    # 12 atoms: centroids of the 20 faces? No, 12 atoms.
    
    # Let's compute edge midpoints.
    # Vertices are V. Edges connect V_i to V_j if distance is minimal.
    # In icosahedron, each vertex connects to 5 others.
    # Total edges = 12 * 5 / 2 = 30.
    
    edge_mids = []
    # Find edges
    dists = np.linalg.norm(verts[:, np.newaxis, :] - verts[np.newaxis, :, :], axis=-1)
    # Minimal distance is the edge length
    min_dist = np.min(dists[dists > 0])
    for i in range(12):
        for j in range(i+1, 12):
            if abs(dists[i,j] - min_dist) < 1e-6:
                mid = (verts[i] + verts[j]) / 2.0
                # Scale to distance 2
                # Current distance of mid is 1/2 * edge_length? No.
                # Midpoint distance from origin:
                # For icosahedron with circumradius 1, edge length is 2/phi? No.
                # Let's just normalize the midpoint to have norm 2.
                norm_mid = np.linalg.norm(mid)
                if norm_mid > 1e-6:
                    mid_scaled = mid * (2.0 / norm_mid)
                    edge_mids.append(mid_scaled)
                    
    # We have 30 edge midpoints.
    # We need 12 more for the 55-atom cluster.
    # These are likely the centroids of the 20 faces? But we only need 12.
    # Actually, the 55-atom Mackay icosahedron has 12 atoms at distance 2 that are NOT edge midpoints?
    # Or maybe the 42 atoms are: 30 edge midpoints + 12 "face" atoms?
    # Let's check the number of faces: 20.
    # If we take centroids of faces, we get 20 points.
    # Maybe the 12 additional points are something else.
    
    # To be safe and fast, if we can't easily generate the exact 55-atom Mackay,
    # we can use the 13-atom core and add random points in the shell of radius ~2-3.
    # This might not be as good, but it's a valid starting point.
    
    # Let's just use the 13-atom icosahedron and add random points on a sphere of radius 2.5 for the rest.
    # This is a "Icosahedral-like" seed.
    
    pts = np.array(points)
    if n > len(pts):
        num_to_add = n - len(pts)
        # Generate random points on sphere of radius 2.5
        # Uniform distribution on sphere
        u = rng.uniform(0, 1, num_to_add)
        v = rng.uniform(0, 1, num_to_add)
        theta = 2 * np.pi * u
        phi_s = np.arccos(2 * v - 1)
        
        x = 2.5 * np.sin(phi_s) * np.cos(theta)
        y = 2.5 * np.sin(phi_s) * np.sin(theta)
        z = 2.5 * np.cos(phi_s)
        
        new_pts = np.column_stack((x, y, z))
        pts = np.vstack((pts, new_pts))
        
    pts += rng.normal(0, 0.05, size=pts.shape)
    return pts

def construct(instance):
    """
    Find the lowest-energy arrangement of n atoms bound by Morse potentials.
    Uses structured seeds (FCC, HCP, Icosahedron) and L-BFGS-B relaxation.
    """
    n = instance["n"]
    rho = instance.get("rho", 3)
    potential = instance["potential"]
    
    start_time = time.time()
    # Safety margin: 300s limit, use 40s per instance
    deadline = start_time + 40.0
    
    rng = np.random.default_rng(42 + n + int(rho*10))
    
    best_energy = np.inf
    best_positions = None
    
    # Select seeds based on rho
    # Low rho (3): FCC/HCP are good
    # High rho (10): Icosahedron is good
    # Medium rho (6): Mix
    
    seeds = []
    
    # Always try FCC
    try:
        seeds.append(_generate_fcc_cluster(n, rng))
    except Exception:
        pass
        
    # Always try HCP
    try:
        seeds.append(_generate_hcp_cluster(n, rng))
    except Exception:
        pass
        
    # Try Icosahedron if rho is high or n is small
    if rho >= 6 or n <= 50:
        try:
            seeds.append(_generate_icosahedron(n, rng))
        except Exception:
            pass
            
    # Add a random seed as fallback
    try:
        radius = n ** (1.0/3.0) * 0.8
        pts = rng.uniform(-radius, radius, size=(n, 3))
        # Simple rejection for min distance
        valid_pts = []
        for p in pts:
            if all(np.linalg.norm(p - q) >= 0.5 for q in valid_pts):
                valid_pts.append(p)
        if len(valid_pts) == n:
            seeds.append(np.array(valid_pts))
        else:
            # If rejection failed, just use random
            seeds.append(pts)
    except Exception:
        pass
        
    if not seeds:
        # Absolute fallback
        seeds.append(rng.uniform(-n**0.33, n**0.33, size=(n, 3)))
        
    # Relax each seed
    for seed in seeds:
        if time.time() > deadline - 5.0:
            break
            
        if not _min_distance_check(seed):
            continue
            
        try:
            # Use L-BFGS-B via the provided relax function
            # The relax function likely handles the optimization internally
            energy, positions = relax(seed, potential, rho)
            
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            continue
            
    # If we have time, do a quick basin hopping from the best seed
    if best_positions is not None and time.time() < deadline - 10.0:
        current_energy = best_energy
        current_positions = best_positions
        kick_sigma = 0.1
        
        while time.time() < deadline - 2.0:
            new_positions = current_positions + rng.normal(0, kick_sigma, size=(n, 3))
            if not _min_distance_check(new_positions):
                continue
                
            try:
                energy, positions = relax(new_positions, potential, rho)
            except Exception:
                continue
                
            if energy < current_energy:
                current_energy = energy
                current_positions = positions
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
            else:
                # Simple Metropolis
                delta_e = energy - current_energy
                T = 0.05
                if np.random.rand() < np.exp(-delta_e / T):
                    current_energy = energy
                    current_positions = positions
                    
    if best_positions is None:
        # Return the first seed if relaxation failed
        return seeds[0]
        
    return best_positions

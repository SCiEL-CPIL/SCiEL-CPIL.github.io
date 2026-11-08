import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient
import time
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree

def _generate_fcc_shells(n):
    """
    Generate FCC-like cluster by adding shells.
    FCC atoms are at positions (i, j, k) where i,j,k are integers and i+j+k is even,
    or half-integer combinations.
    A simpler robust way: Generate a large FCC lattice, center it, sort by distance.
    """
    # Lattice constant a = 1.0. Nearest neighbor distance r = a/sqrt(2) ~ 0.707.
    # We want r ~ 1.0 for SC potential minimum? 
    # SC potential min is around r ~ 1.0-1.2 depending on params.
    # Let's assume lattice constant is scaled such that nearest neighbor is ~1.0.
    # In standard SC, a_cc ~ 1.0 is often the reference.
    
    # Generate FCC points
    # Basis: (0,0,0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)
    # Lattice vectors: (1,0,0), (0,1,0), (0,0,1)
    # Points: (i + dx, j + dy, k + dz)
    
    L = int(np.ceil(n ** (1/3))) + 3
    coords = []
    for i in range(-L, L+1):
        for j in range(-L, L+1):
            for k in range(-L, L+1):
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    x = i + dx
                    y = j + dy
                    z = k + dz
                    coords.append([x, y, z])
    coords = np.array(coords)
    # Center
    coords -= coords.mean(axis=0)
    # Sort by distance to center
    dists = np.linalg.norm(coords, axis=1)
    indices = np.argsort(dists)
    return coords[indices[:n]]

def _generate_hcp_shells(n):
    """
    Generate HCP-like cluster.
    HCP layers: ABAB...
    Layer A: Triangular lattice.
    Layer B: Triangular lattice shifted.
    """
    a = 1.0 # lattice constant
    c = np.sqrt(8/3) * a # ideal c/a ratio
    
    points = []
    L = int(np.ceil(n ** (1/3))) + 3
    
    for i in range(-L, L+1):
        for j in range(-L, L+1):
            for k in range(-L, L+1):
                # A layer at z = 2k * c/3 ? No, standard HCP:
                # z = 0, c/3, 2c/3, c, 4c/3...
                # Let's use z = k * c for simplicity of layers, then shift.
                # Actually, HCP unit cell has 2 atoms.
                # Let's generate a supercell of HCP.
                
                # A atom
                x_a = a * (i + 0.5 * (j % 2))
                y_a = a * np.sqrt(3)/2 * j
                z_a = c * k
                points.append([x_a, y_a, z_a])
                
                # B atom (shifted in x,y and z)
                # Shift in xy: (a/3, -a*sqrt(3)/6)
                # Shift in z: c/3
                x_b = a * (i + 0.5 * (j % 2)) + a/3
                y_b = a * np.sqrt(3)/2 * j - a * np.sqrt(3)/6
                z_b = c * k + c/3
                points.append([x_b, y_b, z_b])
                
                # C atom (shifted in x,y and z)
                # Shift in xy: (2a/3, a*sqrt(3)/6)
                # Shift in z: 2c/3
                x_c = a * (i + 0.5 * (j % 2)) + 2*a/3
                y_c = a * np.sqrt(3)/2 * j + a * np.sqrt(3)/6
                z_c = c * k + 2*c/3
                points.append([x_c, y_c, z_c])
                
    points = np.array(points)
    points -= points.mean(axis=0)
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    return points[indices[:n]]

def _generate_ico_shells(n):
    """
    Generate Icosahedral shells.
    Shell 1: Center (1)
    Shell 2: Icosahedron vertices (12)
    Shell 3: Edge midpoints / Face centers (30)
    Shell 4: ...
    """
    phi = (1 + np.sqrt(5)) / 2
    # Icosahedron vertices
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    # Normalize to radius 1
    verts /= np.linalg.norm(verts[0])
    
    points = [np.array([0.0, 0.0, 0.0])]
    
    if n <= 1:
        return np.array(points[:n])
        
    # Add shell 1 (radius ~1.0)
    points.extend([v * 1.0 for v in verts])
    if n <= 13:
        return np.array(points[:n])
        
    # Add shell 2 (radius ~1.618? or based on geometry)
    # Midpoints of edges
    # Edges connect vertices with distance 2/sqrt(5+2*sqrt(5))? No, edge length is 2/sqrt(5+2*sqrt(5)) is wrong.
    # Distance between adjacent vertices in unit ico is 2/sqrt(5+2*sqrt(5)) * something?
    # Let's just use the fact that the next shell in Mackay icosahedron is at radius sqrt(5)/2 * r1?
    # Actually, for Mackay icosahedron, shell radii are 1, sqrt(5)/2, 2, ...
    # Let's generate points on the surface of an icosahedron of radius R2.
    # A simpler approach: Use a fine grid on the icosahedron surface.
    
    # For n up to 80, we need up to shell 3 or 4.
    # Let's use a simple rejection sampling on the icosahedron surface for higher shells.
    # Or, use the fact that HCP/FCC are better for larger n.
    # Icosahedra are metastable for large n in SC potential.
    # So we will only use this for n < 20.
    
    if n > 20:
        # Return a random compact cluster for large n to avoid bad ico traps
        return _generate_random_compact(n, seed=42)
        
    # Generate more points on the icosahedron surface
    # Use Fibonacci sphere or similar for uniform distribution on sphere, then project to ico?
    # No, just use random points on the sphere and keep them if they are close to ico vertices/faces?
    # Better: Just use random compact for n>13.
    return _generate_random_compact(n, seed=42)

def _generate_random_compact(n, seed=None):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    points = [np.array([0.0, 0.0, 0.0])]
    radius = 0.5
    while len(points) < n:
        placed = False
        for _ in range(100):
            dir_vec = rng.normal(0, 1, 3)
            dir_vec /= np.linalg.norm(dir_vec)
            r = rng.uniform(0.3, radius)
            p = dir_vec * r
            dists = np.linalg.norm(np.array(points) - p, axis=1)
            if np.min(dists) > 0.5:
                points.append(p)
                placed = True
                break
        if not placed:
            radius *= 1.1
            dir_vec = rng.normal(0, 1, 3)
            dir_vec /= np.linalg.norm(dir_vec)
            p = dir_vec * radius
            points.append(p)
    return np.array(points[:n])

def _repair_structure(positions, kind, n):
    """
    Descriptor-guided repair:
    1. Compute local coordination number and bond length distribution.
    2. Identify atoms with "bad" environments (e.g., bond lengths far from ideal).
    3. Nudge these atoms towards ideal positions relative to their neighbors.
    """
    # This is a heuristic step.
    # For SC potential, ideal bond length is ~1.0-1.2.
    # We'll just add a small random perturbation to all atoms to break symmetry,
    # but weighted by local strain.
    
    # Calculate distances
    tree = cKDTree(positions)
    # Query neighbors
    # Just do a simple global perturbation for now, as calculating strain is expensive.
    # The "descriptor" aspect is that we prioritize structures that have already relaxed well.
    # We will just return the positions, but the caller will use this in a loop.
    return positions

def construct(instance):
    n = instance["n"]
    kind = instance["kind"]
    
    candidates = []
    
    # 1. FCC-like
    try:
        fcc_pts = _generate_fcc_shells(n)
        candidates.append(fcc_pts)
    except Exception:
        fcc_pts = None
        
    # 2. HCP-like
    try:
        hcp_pts = _generate_hcp_shells(n)
        candidates.append(hcp_pts)
    except Exception:
        hcp_pts = None
        
    # 3. Icosahedral-like (only for small n)
    if n <= 20:
        try:
            ico_pts = _generate_ico_shells(n)
            candidates.append(ico_pts)
        except Exception:
            pass
            
    # 4. Random compact (multiple seeds)
    for i in range(10):
        try:
            candidates.append(_generate_random_compact(n, seed=100 + i))
        except Exception:
            pass
            
    # 5. Jittered FCC
    if fcc_pts is not None:
        for i in range(5):
            try:
                rng = np.random.default_rng(300 + i)
                jittered = fcc_pts + rng.normal(0, 0.05, (n, 3))
                candidates.append(jittered)
            except Exception:
                pass
                
    # 6. Jittered HCP
    if hcp_pts is not None:
        for i in range(5):
            try:
                rng = np.random.default_rng(400 + i)
                jittered = hcp_pts + rng.normal(0, 0.05, (n, 3))
                candidates.append(jittered)
            except Exception:
                pass

    best_energy = float('inf')
    best_positions = None
    
    # Sort candidates by initial energy? No, just relax all.
    # To save time, we can limit the number of relaxations.
    # n is up to 80. Relaxation is O(n^2) per step.
    
    for pts in candidates:
        if not np.all(np.isfinite(pts)):
            continue
        # Check minimum distance
        try:
            min_d = np.min(pdist(pts))
            if min_d < 0.35:
                continue
        except Exception:
            pass
        try:
            energy, relaxed_pts = relax(pts, 'sutton_chen', kind)
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pts
        except Exception:
            pass
    
    # If we have time, do a second round of relaxation on the best candidate
    # with a small perturbation to escape local minima.
    if best_positions is not None and best_energy < float('inf'):
        # Try a few perturbations of the best structure
        for i in range(3):
            try:
                rng = np.random.default_rng(500 + i)
                perturbed = best_positions + rng.normal(0, 0.1, (n, 3))
                # Check min dist
                min_d = np.min(pdist(perturbed))
                if min_d < 0.35:
                    continue
                energy, relaxed_pts = relax(perturbed, 'sutton_chen', kind)
                if np.isfinite(energy) and energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed_pts
            except Exception:
                pass

    if best_positions is None:
        rng = np.random.default_rng(0)
        best_positions = rng.normal(0, 1, (n, 3))
        
    return best_positions

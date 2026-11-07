"""Sutton-Chen transition-metal cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Global start time for the entire evaluation run
_GLOBAL_START_TIME = time.time()
_TIME_LIMIT = 290.0  # Safety margin before 300s

def _get_icosahedron_vertices(radius):
    """Return 12 vertices of a regular icosahedron centered at origin with given radius."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    # 12 vertices
    v = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1]
    ])
    v = v / np.linalg.norm(v, axis=1, keepdims=True) * radius
    return v

def _get_fcc_cluster(n, spacing=1.0, seed=0):
    """Generate an FCC-like cluster of n atoms."""
    rng = np.random.default_rng(seed)
    # Generate points in a cubic grid
    # For n atoms, we need roughly (n)^(1/3) points per dimension
    side = int(np.ceil(n ** (1.0/3.0))) + 2
    points = []
    # Generate all FCC lattice points in a box
    # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
    for i in range(-side, side):
        for j in range(-side, side):
            for k in range(-side, side):
                for basis in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    p = np.array([i, j, k]) + np.array(basis)
                    points.append(p)
    points = np.array(points)
    # Center the cluster
    points -= points.mean(axis=0)
    # Sort by distance from center to take the most compact n
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    # Scale to appropriate size (heuristic: mean interatomic distance ~1.0)
    # Rough scaling based on volume
    if n > 1:
        target_r = (n / (4/3 * np.pi)) ** (1.0/3.0) * 0.5 # Heuristic
        current_r = np.linalg.norm(points, axis=1).max()
        if current_r > 0:
            points *= (target_r / current_r)
    # Add small random perturbation to break symmetry
    points += rng.normal(0, 0.01, points.shape)
    return points

def _get_hcp_cluster(n, spacing=1.0, seed=0):
    """Generate an HCP-like cluster of n atoms."""
    rng = np.random.default_rng(seed)
    # HCP is similar to FCC but with different stacking. 
    # For simplicity and robustness, we use a hexagonal grid in layers.
    # Layer 1: Hexagonal packing
    # Layer 2: Offset
    points = []
    a = 1.0
    h = np.sqrt(2/3) * a # ideal HCP height
    # Number of layers
    n_layers = max(1, int(np.ceil(n ** (1.0/3.0))))
    for l in range(n_layers):
        z = l * h
        # Offset for even/odd layers
        offset = 0 if l % 2 == 0 else (a/2, a*np.sqrt(3)/6)
        # Hexagonal grid in xy plane
        max_r = np.sqrt(n) * 0.5
        for i in range(-int(max_r), int(max_r)+1):
            for j in range(-int(max_r), int(max_r)+1):
                x = i * a + offset[0]
                y = j * a * np.sqrt(3) + offset[1] + (a * np.sqrt(3) / 2) * (i % 2)
                r = np.sqrt(x*x + y*y)
                if r < max_r:
                    points.append([x, y, z])
    points = np.array(points)
    if len(points) == 0:
        points = np.array([[0,0,0]])
    points -= points.mean(axis=0)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    if n > 1:
        target_r = (n / (4/3 * np.pi)) ** (1.0/3.0) * 0.5
        current_r = np.linalg.norm(points, axis=1).max()
        if current_r > 0:
            points *= (target_r / current_r)
    points += rng.normal(0, 0.01, points.shape)
    return points

def _get_icosahedral_cluster(n, seed=0):
    """Generate an icosahedral shell cluster."""
    rng = np.random.default_rng(seed)
    # Core: 1 atom at center? Or 13-atom icosahedron?
    # Let's start with a 13-atom icosahedron (1 center + 12 vertices)
    # Then add shells.
    points = [np.array([0.0, 0.0, 0.0])]
    # 12 vertices of first shell
    verts = _get_icosahedron_vertices(1.0)
    points.extend([v for v in verts])
    
    # If n > 13, add more points on the surface of the icosahedron
    if n > 13:
        # Use Fibonacci sphere to distribute remaining points on a sphere of radius ~1.5
        # Or just add to the vertices of a larger icosahedron?
        # Better: Add points to the faces/edges of the icosahedron.
        # Simpler: Generate points on a sphere of radius 1.4 and project to nearest face?
        # Let's just generate points on a sphere of radius 1.3 and keep those outside the core?
        # Actually, for n=38, 55, etc., bulk-like structures are often better.
        # So we will mix icosahedral and FCC/HCP.
        pass
    
    # If we need more, we can just add random points on a sphere
    # But this function is specifically for icosahedral.
    # If n is large, this might not be the best geometry, but it's a valid start.
    # Let's just return the first n points. If n > 13, we duplicate/expand?
    # No, let's make it flexible.
    # For n > 13, we generate points on a sphere of radius 1.5
    while len(points) < n:
        p = rng.normal(0, 1, 3)
        p = p / np.linalg.norm(p) * 1.4
        points.append(p)
    
    points = np.array(points[:n])
    # Perturb
    points += rng.normal(0, 0.01, points.shape)
    return points

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    kind = instance["kind"]
    
    start_time = time.time()
    time_left = _TIME_LIMIT - (start_time - _GLOBAL_START_TIME)
    if time_left <= 0:
        # Emergency exit: return a valid but likely high-energy structure
        # Random points
        rng = np.random.default_rng(0)
        return rng.uniform(-1, 1, (size, 3))

    # We have `time_left` seconds to find the best structure.
    # We will try multiple initializations.
    # Estimated time per relaxation: ~0.5 - 2 seconds depending on n.
    # Let's aim for 5-10 relaxations if time permits.
    
    # Define initial generators
    generators = []
    seeds = [0, 1, 2, 3, 4, 5, 6, 7]
    
    # 1. Icosahedral
    for s in seeds[:2]:
        generators.append(lambda s=s: _get_icosahedral_cluster(size, seed=s))
        
    # 2. FCC
    for s in seeds[2:4]:
        generators.append(lambda s=s: _get_fcc_cluster(size, seed=s))
        
    # 3. HCP
    for s in seeds[4:6]:
        generators.append(lambda s=s: _get_hcp_cluster(size, seed=s))
        
    # 4. Random Compact (Seed's method, improved)
    for s in seeds[6:8]:
        def _random_compact(s=s):
            rng = np.random.default_rng(s)
            radius = 0.8 * size ** (1.0/3.0)
            points = []
            while len(points) < size:
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) <= radius:
                    # Check distance
                    if all(np.linalg.norm(p - q) >= 0.8 for q in points):
                        points.append(p)
            return np.array(points)
        generators.append(_random_compact)

    best_energy = float('inf')
    best_positions = None
    
    # Sort generators? Maybe randomize order to avoid bias?
    # Let's just iterate.
    
    for i, gen in enumerate(generators):
        if time.time() - start_time > time_left - 1.0:
            break
            
        try:
            pos = gen()
            # Validate initial positions
            if not np.all(np.isfinite(pos)):
                continue
            
            # Relax
            energy, relaxed_pos = relax(pos, potential, kind)
            
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos.copy()
                
        except Exception as e:
            # Ignore relaxation errors
            continue

    if best_positions is None:
        # Fallback: Return a simple random structure
        rng = np.random.default_rng(0)
        best_positions = rng.uniform(-1, 1, (size, 3))
        
    return best_positions

# EVOLVE-BLOCK-END

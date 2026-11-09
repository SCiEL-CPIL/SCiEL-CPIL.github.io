import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Global deadline management
_START_TIME = time.time()
_DEADLINE = _START_TIME + 280.0  # Leave 50s buffer for safety and evaluation overhead

def _time_remaining():
    return _DEADLINE - time.time()

def _valid_position(pos):
    """Check if positions are finite and no atoms are too close."""
    if not np.all(np.isfinite(pos)):
        return False
    if len(pos) < 2:
        return True
    # Compute pairwise distances
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return d.min() > 0.35

def _generate_fcc(n):
    """Generate n atoms from an FCC lattice centered at origin."""
    points = []
    # FCC lattice vectors: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) scaled by a=1
    # We generate a large grid and pick the n closest to origin
    max_range = int(np.ceil(n ** (1/3) * 2)) + 2
    candidates = []
    for i in range(-max_range, max_range + 1):
        for j in range(-max_range, max_range + 1):
            for k in range(-max_range, max_range + 1):
                # Generate all 4 basis points for this cubic cell
                for offset in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    p = np.array([i + offset[0], j + offset[1], k + offset[2]])
                    candidates.append(p)
    
    candidates = np.array(candidates)
    # Sort by distance to origin
    dists = np.linalg.norm(candidates, axis=1)
    indices = np.argsort(dists)
    return candidates[indices[:n]]

def _generate_hcp(n):
    """Generate n atoms from an HCP lattice centered at origin."""
    points = []
    # HCP structure: ABAB... stacking
    # Basis: (0,0,0), (0, 1/sqrt(3), 0) for A layer? 
    # Let's use standard HCP lattice vectors.
    # a = 1, c/a = sqrt(8/3) ~ 1.633
    a = 1.0
    c = a * np.sqrt(8.0/3.0)
    # Generate layers
    max_range = int(np.ceil(n ** (1/3) * 2)) + 2
    candidates = []
    # Layer A: Triangular lattice in xy plane
    # Layer B: Shifted
    # We'll just generate a grid of HCP points
    for i in range(-max_range, max_range + 1):
        for j in range(-max_range, max_range + 1):
            for k in range(-max_range//2, max_range//2 + 1):
                # A layer at z = k * c
                # Positions in A layer: (i*a + j*a/2, j*a*sqrt(3)/2, k*c)
                x = i * a + j * a * 0.5
                y = j * a * np.sqrt(3) / 2
                z = k * c
                candidates.append([x, y, z])
                
                # B layer at z = (k + 0.5) * c
                # Shifted by (a/2, a*sqrt(3)/6, 0) relative to A?
                # Standard HCP: B is at (1/3, 2/3, 1/2) in fractional coords of the hexagonal cell
                # Let's just add a shifted version
                x_b = x + a * 0.5
                y_b = y + a * np.sqrt(3) / 6
                z_b = (k + 0.5) * c
                candidates.append([x_b, y_b, z_b])
                
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    indices = np.argsort(dists)
    return candidates[indices[:n]]

def _generate_icosahedron(n):
    """Generate n atoms in an icosahedral structure (Mackay icosahedron approximation)."""
    # Vertices of a regular icosahedron
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    # Normalize to unit radius
    verts /= np.linalg.norm(verts[0])
    
    # Mackay icosahedron shells: 1, 12, 42, 100, ...
    # We will generate points on shells of icosahedra
    points = [np.zeros(3)]
    
    # Shell 1: 12 vertices
    shell1 = verts * 1.0
    points.extend(shell1)
    
    # Shell 2: 42 points
    # These are vertices of a smaller icosahedron rotated and scaled? 
    # Actually, Mackay icosahedron 1-12-42-100...
    # The 42 points are the midpoints of the edges of the icosahedron? No.
    # Let's use a simpler heuristic: generate points on a sphere and project to icosahedral symmetry?
    # Or just use the FCC generator which is often close enough, and let relaxation fix it.
    # For now, let's just return the first n points from a dense icosahedral-like packing.
    # A better approach: Generate a large set of icosahedral points and pick n.
    
    # Let's generate points on icosahedron faces
    # This is complex. Let's stick to FCC/HCP which are robust.
    # We'll add a random "ball" seed as a fallback.
    return None # Placeholder, we will use FCC/HCP primarily

def construct(instance):
    n = instance["n"]
    kind = instance["kind"]
    
    best_energy = np.inf
    best_pos = None
    
    # List of seed generators
    generators = [_generate_fcc, _generate_hcp]
    
    # Add a random seed generator
    def _generate_random(n):
        rng = np.random.default_rng(n)
        # Generate points in a sphere of radius R
        R = 1.5 * n ** (1/3)
        points = []
        while len(points) < n:
            p = rng.uniform(-R, R, 3)
            if np.linalg.norm(p) <= R:
                # Check min distance to existing points
                if points:
                    dists = np.linalg.norm(np.array(points) - p, axis=1)
                    if dists.min() < 0.8:
                        continue
                points.append(p)
        return np.array(points)
    
    generators.append(_generate_random)
    
    # Try each generator
    for gen in generators:
        if _time_remaining() < 30.0:
            break
            
        try:
            seed = gen(n)
            if seed is None or len(seed) != n:
                continue
                
            # Relax
            energy, pos = relax(seed, 'sutton_chen', kind)
            
            if energy < best_energy and _valid_position(pos):
                best_energy = energy
                best_pos = pos
                
        except Exception:
            continue
            
    # If we still have time, try more random restarts from the best found
    if best_pos is not None and _time_remaining() > 60.0:
        rng = np.random.default_rng(42)
        for _ in range(10):
            if _time_remaining() < 30.0:
                break
            # Perturb the best position
            noise = rng.normal(0, 0.1, size=(n, 3))
            perturbed = best_pos + noise
            try:
                energy, pos = relax(perturbed, 'sutton_chen', kind)
                if energy < best_energy and _valid_position(pos):
                    best_energy = energy
                    best_pos = pos
            except Exception:
                continue
                
    if best_pos is None:
        # Fallback: return a simple linear chain or something valid
        # This should not happen, but just in case
        best_pos = np.array([[i, 0, 0] for i in range(n)], dtype=float)
        
    return best_pos

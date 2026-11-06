"""
Dzugutov cluster global minima search using icosahedral seeds and relaxation.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START = time.time()
TIME_LIMIT = 280.0  # Safety margin under 300s kill

def _elapsed():
    return time.time() - _START

def _remaining():
    return TIME_LIMIT - _elapsed()

def _generate_icosahedral_directions():
    """
    Generate unit vectors corresponding to icosahedral symmetry positions.
    Returns a list of (shell_index, vector) sorted by shell.
    Shell 0: Center (handled separately)
    Shell 1: 12 vertices
    Shell 2: 30 edge midpoints
    Shell 3: 20 face centers
    """
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of an icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    directions = []
    # Shell 1: Vertices
    for v in verts:
        directions.append((1, v))
        
    # Shell 2: Edge midpoints (12 vertices * 5 edges / 2 = 30 edges)
    # Find edges by checking dot product
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            dot = np.dot(verts[i], verts[j])
            # The minimum dot product for non-opposite vertices in icosahedron is -1/phi^2 ~ -0.382
            # The distance between connected vertices is 2*sin(pi/5)
            # Let's just use a threshold. The closest neighbors have dot ~ -0.382
            if -0.4 < dot < -0.3:
                mid = (verts[i] + verts[j]) / 2.0
                mid /= np.linalg.norm(mid)
                edges.append(mid)
                
    # Remove duplicates (numerical stability)
    unique_edges = []
    for e in edges:
        is_dup = False
        for ue in unique_edges:
            if np.dot(e, ue) > 0.999:
                is_dup = True
                break
        if not is_dup:
            unique_edges.append(e)
            
    for v in unique_edges:
        directions.append((2, v))
        
    # Shell 3: Face centers (20 faces)
    # Faces are triangles of vertices
    faces = []
    for i in range(12):
        for j in range(12):
            for k in range(12):
                if i < j and j < k:
                    # Check if i,j,k form a face (all pairwise connected)
                    d_ij = np.dot(verts[i], verts[j])
                    d_jk = np.dot(verts[j], verts[k])
                    d_ki = np.dot(verts[k], verts[i])
                    if (-0.4 < d_ij < -0.3) and (-0.4 < d_jk < -0.3) and (-0.4 < d_ki < -0.3):
                        center = (verts[i] + verts[j] + verts[k]) / 3.0
                        center /= np.linalg.norm(center)
                        faces.append(center)
                        
    unique_faces = []
    for f in faces:
        is_dup = False
        for uf in unique_faces:
            if np.dot(f, uf) > 0.999:
                is_dup = True
                break
        if not is_dup:
            unique_faces.append(f)
            
    for v in unique_faces:
        directions.append((3, v))
        
    return directions

# Precompute directions
_ICO_DIRECTIONS = _generate_icosahedral_directions()
# Sort by shell index
_ICO_DIRECTIONS.sort(key=lambda x: x[0])
# Extract just the vectors, keeping track of shell
_ICO_VECTORS = [v for s, v in _ICO_DIRECTIONS]
_ICO_SHELLS = [s for s, v in _ICO_DIRECTIONS]

def _make_icosahedral_seed(n, rng, perturb_scale=0.05):
    """
    Generate an icosahedral seed for n atoms.
    Uses the first n-1 icosahedral directions for the outer atoms,
    placing them at radii corresponding to their shells.
    """
    if n <= 1:
        return np.zeros((n, 3))
    
    # Radii for shells (approximate for Dzugutov minimum)
    # Dzugutov equilibrium distance is around 1.8-2.0
    # Shell 1 radius ~ 1.8, Shell 2 ~ 2.6, Shell 3 ~ 3.2
    radii = {1: 1.8, 2: 2.7, 3: 3.4}
    
    positions = [np.zeros(3)] # Center atom
    
    # We need n-1 more atoms
    # Select n-1 directions from _ICO_VECTORS
    # To allow for different isomers, we can shuffle the selection of atoms in the outermost shells
    # But for now, just take the first n-1 in sorted order (inner shells first)
    
    num_outer = n - 1
    if num_outer > len(_ICO_VECTORS):
        # If we need more than 62 atoms, we'd need to add more shells or perturb
        # For N up to 60, 62 is enough.
        pass
        
    # Create a list of indices to pick
    indices = list(range(min(num_outer, len(_ICO_VECTORS))))
    
    # Perturb the selection for the outer shells to create variants
    # We will do this in the main loop by passing different rngs or shuffles
    
    for idx in indices:
        vec = _ICO_VECTORS[idx]
        shell = _ICO_SHELLS[idx]
        r = radii.get(shell, 1.8)
        p = r * vec
        # Add small random perturbation to break exact symmetry
        p = p + rng.normal(0, perturb_scale, 3)
        positions.append(p)
        
    return np.array(positions)

def construct(instance):
    """
    Find the lowest-energy structure for the given cluster size.
    """
    size = instance["n"]
    potential = instance["potential"]
    
    best_energy = np.inf
    best_positions = None
    
    # Time check
    if _remaining() < 10:
        # Not enough time, return a quick guess
        positions = _make_icosahedral_seed(size, np.random.default_rng(size))
        return positions

    # Try multiple icosahedral variants
    # Variant 1: Standard filling
    # Variant 2: Slightly different radii
    # Variant 3: Random perturbation
    
    n_trials = 15
    
    for i in range(n_trials):
        if _remaining() < 5:
            break
            
        rng = np.random.default_rng(size * 1000 + i)
        
        # Generate seed
        # We can vary the perturb_scale or the specific atoms chosen
        # For simplicity, we just use random perturbations on the standard icosahedral seed
        # But to get different isomers, we might need to swap atoms between shells.
        # Let's just rely on relaxation from perturbed icosahedra.
        
        positions = _make_icosahedral_seed(size, rng, perturb_scale=0.1)
        
        # Relax
        try:
            energy, relaxed_positions = relax(positions, potential)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed_positions
        except Exception:
            pass
            
    return best_positions
# EVOLVE-BLOCK-END

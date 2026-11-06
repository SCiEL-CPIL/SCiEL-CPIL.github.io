"""Dzugutov cluster global minima: Mackay icosahedral seeds + local relaxation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import dzugutov_energy_gradient

# Global deadline tracking
_START_TIME = None
_DEADLINE = 300.0  # seconds, with safety margin
_THREAD_LOCK = threading.Lock()

def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME

def _remaining():
    return max(0.0, _DEADLINE - _elapsed())

def _get_icosahedron_vertices():
    """Return 12 vertices of a regular icosahedron centered at origin."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1,  phi,  0],
        [ 1,  phi,  0],
        [-1, -phi,  0],
        [ 1, -phi,  0],
        [ 0, -1,  phi],
        [ 0,  1,  phi],
        [ 0, -1, -phi],
        [ 0,  1, -phi],
        [ phi,  0, -1],
        [ phi,  0,  1],
        [-phi,  0, -1],
        [-phi,  0,  1]
    ], dtype=float)
    # Normalize to unit length
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    return verts

def _generate_mackay_55():
    """Generate a 55-atom Mackay icosahedron."""
    # 1 atom at center
    atoms = [np.zeros(3)]
    
    # 12 atoms at first shell (vertices of icosahedron)
    r1 = 1.0
    verts = _get_icosahedron_vertices()
    for v in verts:
        atoms.append(r1 * v)
        
    # 42 atoms at second shell
    # 12 along 5-fold axes (same as verts)
    r2_5 = 1.7
    for v in verts:
        atoms.append(r2_5 * v)
        
    # 20 along 2-fold axes (face normals)
    # Face normals of icosahedron are directions to face centers
    # The 20 faces correspond to the 20 vertices of a dodecahedron
    # We can compute face normals by averaging vertices of each face
    # Or use known coordinates for dodecahedron vertices
    # Dodecahedron vertices (normalized):
    # (±1, ±1, ±1)
    # (0, ±1/φ, ±φ)
    # (±1/φ, ±φ, 0)
    # (±φ, 0, ±1/φ)
    phi = (1 + np.sqrt(5)) / 2
    inv_phi = 1.0 / phi
    dodeca_verts = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                dodeca_verts.append([sx, sy, sz])
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                dodeca_verts.append([0, sx*inv_phi, sy*phi])
                dodeca_verts.append([sx*inv_phi, sy*phi, 0])
                dodeca_verts.append([sx*phi, 0, sy*inv_phi])
    
    dodeca_verts = np.array(dodeca_verts)
    # Remove duplicates? The above generation might have duplicates or 20 unique?
    # Standard dodecahedron has 20 vertices.
    # Let's just use the 20 unique directions.
    # The 20 face centers of an icosahedron are the vertices of a dodecahedron.
    # We need to ensure we have 20 unique vectors.
    # The list above has 8 + 12 = 20?
    # 8 corners of cube.
    # 12 others.
    # Total 20.
    
    # Normalize
    norms = np.linalg.norm(dodeca_verts, axis=1, keepdims=True)
    dodeca_verts = dodeca_verts / norms
    
    r2_2 = 1.7 # Same radius for simplicity, relaxation will adjust
    for v in dodeca_verts:
        atoms.append(r2_2 * v)
        
    # 10 along 3-fold axes (edge normals)
    # Edge normals of icosahedron correspond to vertices of a pentagonal prism?
    # There are 30 edges. 10 unique directions (pairs of opposite edges).
    # The 10 directions are the vertices of a regular pentagon in 2D, but in 3D?
    # Actually, the 10 3-fold axes are the directions to the midpoints of the edges.
    # We can compute them by taking the cross product of adjacent edge vectors?
    # Or simpler: The 10 3-fold axes are perpendicular to the 10 pairs of opposite edges.
    # Let's find the 30 edge midpoints and cluster them into 10 pairs?
    # Or just generate 10 vectors that are 3-fold symmetric.
    # The 10 3-fold axes are the directions to the vertices of a regular pentagonal antiprism? No.
    # They are the 10 directions that are 3-fold symmetric.
    # Let's compute them from the icosahedron edges.
    
    # Get edges of icosahedron
    # Two vertices are connected if their dot product is -1/phi (for unit icosahedron)
    # Dot product of unit icosahedron vertices:
    # Adjacent: -1/phi
    # Opposite: -1
    # Non-adjacent, non-opposite: 1/phi^2?
    
    # Let's find edges
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            dot = np.dot(verts[i], verts[j])
            if abs(dot - (-1/phi)) < 1e-6:
                edges.append((i, j))
                
    # There are 30 edges.
    # For each edge, the midpoint direction is (v_i + v_j) / |v_i + v_j|
    # But opposite edges have opposite midpoint directions.
    # We want 10 unique directions.
    
    midpoints = []
    for i, j in edges:
        mp = verts[i] + verts[j]
        mp_norm = np.linalg.norm(mp)
        if mp_norm > 1e-6:
            midpoints.append(mp / mp_norm)
            
    # Cluster midpoints into 10 pairs of opposites
    unique_mids = []
    for mp in midpoints:
        is_dup = False
        for um in unique_mids:
            if abs(np.dot(mp, um)) > 0.999: # Same or opposite
                is_dup = True
                break
        if not is_dup:
            unique_mids.append(mp)
            
    # unique_mids should have 10 elements
    r2_3 = 1.7
    for v in unique_mids:
        atoms.append(r2_3 * v)
        
    return np.array(atoms)

def _relax_structure(positions, potential='dzugutov'):
    """Relax structure using L-BFGS-B with gradient."""
    def fun_and_grad(x):
        pos = x.reshape(-1, 3)
        try:
            e, grad = dzugutov_energy_gradient(pos, potential)
            return e, grad.ravel()
        except Exception:
            return 1e10, np.zeros_like(x)
            
    x0 = positions.ravel()
    res = minimize(fun_and_grad, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10})
    return res.fun, res.x.reshape(-1, 3)

def construct(instance):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        
    n = instance['n']
    potential = instance['potential']
    
    # Generate 55-atom Mackay icosahedron
    base_55 = _generate_mackay_55()
    
    best_energy = np.inf
    best_pos = base_55[:n] if n <= 55 else base_55
    
    # Prepare seed positions
    if n <= 55:
        # Remove atoms from the outer shell (last 42 atoms)
        # We want to remove 55-n atoms from the last 42
        num_to_remove = 55 - n
        # Indices of outer shell: 13 to 54
        outer_indices = np.arange(13, 55)
        # Randomly select indices to remove
        remove_indices = np.random.choice(outer_indices, num_to_remove, replace=False)
        keep_indices = np.setdiff1d(np.arange(55), remove_indices)
        pos = base_55[keep_indices]
    else:
        # Add atoms
        num_to_add = n - 55
        pos = base_55.copy()
        # Add atoms at random positions on a sphere of radius 2.0
        # Ensure they don't overlap
        for _ in range(num_to_add):
            # Try to find a valid position
            for _attempt in range(100):
                theta = np.random.uniform(0, 2*np.pi)
                phi = np.random.uniform(0, np.pi)
                r = np.random.uniform(1.8, 2.2)
                p = np.array([
                    r * np.sin(phi) * np.cos(theta),
                    r * np.sin(phi) * np.sin(theta),
                    r * np.cos(phi)
                ])
                # Check distance to existing atoms
                dists = np.linalg.norm(pos - p, axis=1)
                if np.min(dists) > 0.8: # Min distance 0.5, use 0.8 for safety
                    pos = np.vstack([pos, p])
                    break
            else:
                # If no valid position found, just add it (relaxation will fix)
                pos = np.vstack([pos, p])
                
    # Relax the initial seed
    try:
        e, pos_relaxed = _relax_structure(pos, potential)
        if e < best_energy:
            best_energy = e
            best_pos = pos_relaxed
    except Exception:
        pass
        
    # Basin Hopping / Random Restarts
    # Time budget per instance is roughly 50s
    # Relaxation takes ~1s. We can do ~10-20 restarts.
    
    rng = np.random.default_rng(n)
    for i in range(20):
        if _remaining() < 5.0:
            break
            
        # Perturb the best structure found so far
        # Or perturb the original seed? Perturbing the best is better.
        noise = rng.normal(0, 0.2, best_pos.shape)
        pos_perturbed = best_pos + noise
        
        # Ensure min distance? Relaxation should handle it, but if atoms overlap, gradient might be bad.
        # Simple check: if any distance < 0.5, skip this restart?
        # Or just let relaxation handle it.
        
        try:
            e, pos_relaxed = _relax_structure(pos_perturbed, potential)
            if e < best_energy:
                best_energy = e
                best_pos = pos_relaxed
        except Exception:
            pass
            
    return best_pos
# EVOLVE-BLOCK-END

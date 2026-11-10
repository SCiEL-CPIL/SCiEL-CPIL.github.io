# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with side d and angle in degrees."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _base_hex_verts():
    """Unit hexagon vertices at origin, angle 0."""
    angles = np.radians(60 * np.arange(6))
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _check_separation(positions, tol=1e-6):
    """
    Check if all pairs of unit hexagons (angle 0) at positions are separated.
    Uses SAT with pre-computed edge normals.
    """
    n = len(positions)
    base = _base_hex_verts()
    # Precompute edge normals (6 axes)
    normals = np.empty((6, 2))
    for k in range(6):
        p1 = base[k]
        p2 = base[(k+1) % 6]
        edge = p2 - p1
        normals[k] = np.array([-edge[1], edge[0]])
    
    # Precompute projections of base hex on each normal
    base_projs = base @ normals.T  # (6, 6): projection of vertex v on axis a
    
    for i in range(n):
        ci = positions[i]
        for j in range(i+1, n):
            cj = positions[j]
            found_sep = False
            for k in range(6):
                nrm = normals[k]
                proj_i = base_projs[:, k] + ci @ nrm
                proj_j = base_projs[:, k] + cj @ nrm
                if proj_i.max() < proj_j.min() - tol or proj_j.max() < proj_i.min() - tol:
                    found_sep = True
                    break
            if not found_sep:
                return False
    return True

def _min_containing_hex_side(positions, ocx, ocy, angle_deg):
    """
    Compute minimum side length s of a regular hexagon centered at (ocx, ocy)
    with given angle that contains all vertices of unit hexagons at positions (angle 0).
    """
    base = _base_hex_verts()
    all_verts = positions[:, None, :] + base[None, :, :]  # (12, 6, 2)
    all_verts = all_verts.reshape(-1, 2)  # (72, 2)
    
    a_rad = np.radians(angle_deg)
    normal_angles = a_rad + np.radians(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    shifted = all_verts - np.array([ocx, ocy])
    dots = shifted @ normals.T
    max_dot = np.max(dots)
    
    s = (2.0 / np.sqrt(3.0)) * max_dot
    return s

def _objective(params):
    """
    params: (26,) array. First 24 are (x1, y1, ..., x12, y12), last is angle_deg.
    Outer center is at centroid of positions (computed, not optimized).
    Returns s (outer hexagon side length) or penalty.
    """
    positions = params[:24].reshape(12, 2)
    angle_deg = params[24]
    
    if not _check_separation(positions):
        return 1000.0
    
    centroid = positions.mean(axis=0)
    s = _min_containing_hex_side(positions, centroid[0], centroid[1], angle_deg)
    return s

def _hex_lattice_positions():
    """
    Generate a hexagonal close-packing of 12 unit hexagons.
    Hexagons of side 1 have center-to-center distance 2*sqrt(3) for edge-to-edge touching.
    """
    d = 2.0 * np.sqrt(3.0)  # center-to-center distance for touching hexagons
    h = d * 0.5              # horizontal spacing
    v = d * np.sqrt(3.0) / 2  # vertical spacing for offset rows
    
    # 4-3-3-2 arrangement (rows of 4, 3, 3, 2)
    positions = []
    # Row 0: 4 hexagons
    for i in range(4):
        positions.append([(i - 1.5) * h, 0.0])
    # Row 1: 3 hexagons (offset)
    for i in range(3):
        positions.append([(i - 1.0) * h, v])
    # Row 2: 3 hexagons (offset opposite)
    for i in range(3):
        positions.append([(i - 1.0) * h, -v])
    # Row 3: 2 hexagons
    for i in range(2):
        positions.append([(i - 0.5) * h, 2*v])
    
    return np.array(positions)

def _hex_lattice_positions_v2():
    """Alternative arrangement: 3-4-3-2 or different pattern."""
    d = 2.0 * np.sqrt(3.0)
    h = d * 0.5
    v = d * np.sqrt(3.0) / 2
    
    positions = []
    # Row 0: 3 hexagons
    for i in range(3):
        positions.append([(i - 1.0) * h, 0.0])
    # Row 1: 4 hexagons (offset)
    for i in range(4):
        positions.append([(i - 1.5) * h, v])
    # Row 2: 3 hexagons (offset opposite)
    for i in range(3):
        positions.append([(i - 1.0) * h, -v])
    # Row 3: 2 hexagons
    for i in range(2):
        positions.append([(i - 0.5) * h, 2*v])
    
    return np.array(positions)

def _circular_positions(radius=2.5):
    """Generate positions on a circle."""
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    positions = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    return positions

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns (inner, outer, s) where:
      inner: (12, 3) array of (x, y, angle_deg) for unit hexagons
      outer: (3,) array (x, y, angle_deg) for the outer hexagon
      s: side length of outer hexagon
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill
    
    np.random.seed(42)
    
    best_s = float('inf')
    best_positions = None
    best_angle = 0.0
    
    # Generate diverse initial configurations
    inits = []
    
    base_lattice = _hex_lattice_positions()
    inits.append(base_lattice)
    
    base_lattice_v2 = _hex_lattice_positions_v2()
    inits.append(base_lattice_v2)
    
    for r in [2.0, 2.5, 3.0]:
        inits.append(_circular_positions(r))
    
    # Perturbed versions of lattice
    for _ in range(5):
        inits.append(base_lattice + np.random.randn(12, 2) * 0.3)
    for _ in range(3):
        inits.append(base_lattice_v2 + np.random.randn(12, 2) * 0.3)
    
    # Random positions
    for _ in range(5):
        inits.append(np.random.randn(12, 2) * 3.0)
    
    def _nelder_mead_optimize(init_pos, init_angle, max_iter=5000, xatol=1e-10, fatol=1e-10):
        """Run Nelder-Mead optimization with time check."""
        from scipy.optimize import minimize
        
        centroid = init_pos.mean(axis=0)
        init_params = np.zeros(25)
        init_params[:24] = init_pos.flatten()
        init_params[24] = init_angle
        
        try:
            res = minimize(
                _objective, 
                init_params, 
                method='Nelder-Mead', 
                options={'maxiter': max_iter, 'xatol': xatol, 'fatol': fatol, 'adaptive': True}
            )
            return res
        except Exception:
            return None
    
    # First pass: moderate optimization
    for i, init_pos in enumerate(inits):
        if time.time() > deadline - 60:
            break
            
        for init_angle in [0, 30, 15, -15]:
            if time.time() > deadline - 30:
                    break
                    
            res = _nelder_mead_optimize(init_pos, init_angle, max_iter=3000, xatol=1e-9, fatol=1e-9)
            if res is not None and res.fun < best_s and res.fun < 100:
                best_s = res.fun
                best_positions = res.x[:24].reshape(12, 2).copy()
                best_angle = res.x[24]
    
    # Second pass: polish the best solution with tighter tolerance
    if best_positions is not None and best_s < 100:
        if time.time() < deadline - 10:
            res = _nelder_mead_optimize(best_positions, best_angle, max_iter=10000, xatol=1e-12, fatol=1e-12)
            if res is not None and res.fun < best_s and res.fun < 100:
                best_s = res.fun
                best_positions = res.x[:24].reshape(12, 2).copy()
                best_angle = res.x[24]
    
    # Fallback if optimization failed
    if best_positions is None or best_s > 100:
        best_positions = _hex_lattice_positions()
        best_angle = 0.0
        centroid = best_positions.mean(axis=0)
        best_s = _min_containing_hex_side(best_positions, centroid[0], centroid[1], best_angle)
    
    # Final: shift so outer hexagon is at origin
    centroid = best_positions.mean(axis=0)
    inner_hex_data = np.zeros((12, 3))
    inner_hex_data[:, 0] = best_positions[:, 0] - centroid[0]
    inner_hex_data[:, 1] = best_positions[:, 1] - centroid[1]
    inner_hex_data[:, 2] = 0.0
    
    outer_hex_data = np.array([0.0, 0.0, best_angle])
    
    return inner_hex_data, outer_hex_data, best_s

# EVOLVE-BLOCK-END

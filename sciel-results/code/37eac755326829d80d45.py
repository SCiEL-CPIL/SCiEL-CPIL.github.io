# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with side d and angle in degrees."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _rigid_hex_cluster():
    """
    Generate 12 positions for a compact hexagonal cluster of unit hexagons.
    The cluster has a central hexagon surrounded by 6, then 5 more in a ring.
    Returns (12, 2) array of positions relative to cluster center.
    """
    # Hexagonal lattice: neighbors are at distance 2*sqrt(3) in directions 0, 60, 120, 180, 240, 300
    # For unit hexagons (side=1), the distance between centers of adjacent hexagons is 2*sqrt(3)
    d = 2 * np.sqrt(3)
    
    # Ring 0: center
    pos0 = np.array([[0.0, 0.0]])
    
    # Ring 1: 6 hexagons at distance d
    angles1 = np.radians(60 * np.arange(6))
    pos1 = np.column_stack([d * np.cos(angles1), d * np.sin(angles1)])
    
    # Ring 2: we need 5 more hexagons. Place them at the "corners" of the hexagonal ring
    # The ring-2 positions are at distance 2*d in directions 30, 90, 150, 210, 270, 330
    # But we only need 5. Let's pick 5 that give a symmetric-ish shape.
    # Actually, for a proper hexagonal cluster of 12, the shape is:
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons
    # Row 2: 3 hexagons
    # Row 3: 2 hexagons
    # That's 12 total. But this is a parallelogram-like shape, not hexagonal.
    
    # Let me use a different approach: start with a hexagonal number cluster.
    # Hexagonal numbers: 1, 7, 19... So 12 is not a hexagonal number.
    # Let's just use a compact arrangement:
    # Center + 6 neighbors + 5 more placed to minimize bounding box.
    
    # Actually, let me define the positions explicitly for a compact cluster:
    # Using axial coordinates (q, r) for hexagonal grid:
    # Center: (0,0)
    # Ring 1: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # Ring 2 (partial): (2,0), (2,-1), (1,-2), (-1,2), (0,2)
    # That's 1 + 6 + 5 = 12
    
    # Convert axial to pixel: x = d * (q + r/2), y = d * (sqrt(3)/2) * r
    axial = np.array([
        [0, 0],    # center
        [1, 0],    # ring 1
        [1, -1],
        [0, -1],
        [-1, 0],
        [-1, 1],
        [0, 1],
        [2, 0],    # ring 2 (5 more)
        [2, -1],
        [1, -2],
        [-1, 2],
        [0, 2],
    ])
    
    q = axial[:, 0]
    r = axial[:, 1]
    x = d * (q + r / 2.0)
    y = d * (np.sqrt(3) / 2.0) * r
    
    positions = np.column_stack([x, y])
    
    # Center the cluster
    centroid = positions.mean(axis=0)
    positions = positions - centroid
    
    return positions

def _min_containing_hex_side_rigid(positions, angle_deg):
    """
    Compute minimum side length s of a regular hexagon centered at origin
    with given angle that contains all vertices of unit hexagons at positions.
    """
    # Get all vertices
    a = np.radians(60 * np.arange(6))
    base = np.column_stack([np.cos(a), np.sin(a)])  # (6,2)
    all_verts = positions[:, None, :] + base[None, :, :]  # (12,6,2)
    all_verts = all_verts.reshape(-1, 2)  # (72,2)
    
    # Outer hexagon normals
    a_rad = np.radians(angle_deg)
    normal_angles = a_rad + np.radians(30 + 60 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # Max projection
    dots = all_verts @ normals.T
    max_dot = np.max(dots)
    
    s = (2.0 / np.sqrt(3.0)) * max_dot
    return s

def _objective_rigid(params):
    """
    Objective for rigid body optimization.
    params: (3,) array: (dx, dy, angle_deg) where (dx, dy) is offset of outer center from cluster centroid.
    """
    dx, dy, angle_deg = params
    base_pos = _rigid_hex_cluster()
    positions = base_pos + np.array([dx, dy])
    return _min_containing_hex_side_rigid(positions, angle_deg)

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon.
    Uses rigid body approach: optimize only outer center offset and rotation angle.
    """
    start_time = time.time()
    deadline = start_time + 580
    
    base_pos = _rigid_hex_cluster()
    
    best_s = float('inf')
    best_params = None
    
    # Multiple initial angles and offsets
    for init_angle in [0, 30, 60, 90, 15, 45, -15, -30, 10, -10, 20, -20]:
        if time.time() > deadline - 20:
            break
        
        for init_dx in [-1, 0, 1]:
            for init_dy in [-1, 0, 1]:
                if time.time() > deadline - 20:
                    break
                
                init_params = np.array([init_dx, init_dy, init_angle])
                
                try:
                    res = minimize(
                        _objective_rigid,
                        init_params,
                        method='Nelder-Mead',
                        options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12}
                    )
                except Exception:
                    continue
                    
                if res.fun < best_s:
                    best_s = res.fun
                    best_params = res.x.copy()
    
    # Fallback
    if best_params is None or best_s > 100:
        best_s = _min_containing_hex_side_rigid(base_pos, 0)
        best_params = np.array([0.0, 0.0, 0.0])
    
    dx, dy, final_angle = best_params
    
    # Final positions: cluster shifted by (dx, dy)
    final_positions = base_pos + np.array([dx, dy])
    
    # Shift so outer hexagon is at origin
    inner_hex_data = np.zeros((12, 3))
    inner_hex_data[:, 0] = final_positions[:, 0]
    inner_hex_data[:, 1] = final_positions[:, 1]
    inner_hex_data[:, 2] = 0.0
    
    outer_hex_data = np.array([0.0, 0.0, final_angle])
    
    return inner_hex_data, outer_hex_data, best_s

# EVOLVE-BLOCK-END

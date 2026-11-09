# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _hex_vertices(cx, cy, s, a_deg):
    a = np.radians(a_deg)
    angles = a + np.radians(60.0) * np.arange(6)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _min_gap_hex(hex1, hex2):
    """Compute minimum separation between two unit hexagons.
    Returns the minimum distance between any edge of hex1 and any edge of hex2,
    or a negative value if they overlap.
    We use the 12 edge normals approach: for each edge normal of either hexagon,
    project both hexagons and find the gap.
    """
    v1 = _hex_vertices(hex1[0], hex1[1], 1.0, hex1[2])
    v2 = _hex_vertices(hex2[0], hex2[1], 1.0, hex2[2])
    
    # Get all 12 edge normals (6 from each hexagon)
    normals = []
    for v in [v1, v2]:
        for i in range(6):
            p1 = v[i]
            p2 = v[(i+1) % 6]
            edge = p2 - p1
            # Normal pointing outward
            n = np.array([-edge[1], edge[0]])
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                continue
            n = n / n_norm
            normals.append(n)
    
    min_gap = np.inf
    for n in normals:
        # Project all vertices of both hexagons onto this normal
        proj1 = v1 @ n
        proj2 = v2 @ n
        # The gap along this normal is:
        # max(proj2) - min(proj1) if hex2 is "beyond" hex1
        # min(proj1) - max(proj2) if hex1 is "beyond" hex2
        gap_forward = proj2.min() - proj1.max()
        gap_backward = proj1.min() - proj2.max()
        # The separation is the maximum of these (most positive means most separated)
        # If both are negative, they overlap along this normal
        sep = max(gap_forward, gap_backward)
        if sep < min_gap:
            min_gap = sep
    
    return min_gap

def _outer_hex_vertices(cx, cy, s, a_deg):
    a = np.radians(a_deg)
    angles = a + np.radians(60.0) * np.arange(6)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _point_in_hexagon(px, py, cx, cy, s, a_deg):
    """Check if point (px,py) is inside hexagon with center (cx,cy), side s, angle a_deg."""
    v = _outer_hex_vertices(cx, cy, s, a_deg)
    inside = True
    for i in range(6):
        p1 = v[i]
        p2 = v[(i+1) % 6]
        edge = p2 - p1
        # Cross product to check if point is on the "inside" side
        cross = (p2[0] - p1[0]) * (py - p1[1]) - (p2[1] - p1[1]) * (px - p1[0])
        # For a CCW polygon, inside means cross >= 0
        if cross < -1e-6:
            inside = False
            break
    return inside

def _objective(params, n=11):
    """Objective: minimize outer hexagon side length given inner hexagon positions.
    params: flattened array of [x1,y1,a1, x2,y2,a2, ..., x11,y11,a11, cx, cy, ca, side]
    Actually, let's fix outer center at origin and angle at 0, and optimize only inner positions + outer side.
    """
    # params: 11*3 inner + 1 outer side = 34
    inner_params = params[:33].reshape(11, 3)
    side = params[33]
    
    # Compute minimum required side to contain all inner hexagons
    required_side = 0.0
    for i in range(11):
        cx, cy, a_deg = inner_params[i]
        verts = _hex_vertices(cx, cy, 1.0, a_deg)
        for v in verts:
            x, y = v
            c1 = abs(x * np.sqrt(3) + y) / np.sqrt(3)
            c2 = abs(x * np.sqrt(3) - y) / np.sqrt(3)
            c3 = abs(y) / (np.sqrt(3)/2)
            req = max(c1, c2, c3)
            if req > required_side:
                required_side = req
    
    # Penalty for overlapping inner hexagons
    overlap_penalty = 0.0
    for i in range(11):
        for j in range(i+1, 11):
            gap = _min_gap_hex(inner_params[i], inner_params[j])
            if gap < 1e-6:
                overlap_penalty += (1e-6 - gap) ** 2
    
    # We want to minimize side, but we also need side >= required_side
    # And we need no overlaps
    # Use a penalty method: minimize side + penalty*overlap_penalty
    # But we also need side to be at least required_side
    # Let's define: if side < required_side, add a large penalty
    
    total_penalty = overlap_penalty
    if side < required_side:
        total_penalty += (required_side - side) ** 2 * 100
    
    return side + total_penalty

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    # Initial guess: Compact cluster
    # 1 center, 6 surrounding, 4 in the "corners" of the surrounding ring
    # Distances:
    # Center (0,0)
    # Ring 1 (dist sqrt(3)):
    # (sqrt(3), 0), (sqrt(3)/2, 1.5), (-sqrt(3)/2, 1.5), (-sqrt(3), 0), (-sqrt(3)/2, -1.5), (sqrt(3)/2, -1.5)
    # Ring 2 (added to make 11):
    # Place 4 more in the gaps between ring 1 hexagons?
    # Or extend the sides.
    # A known good config is a 3x3 grid minus 2 corners? No.
    # Let's use a tighter initial guess based on a 3-4-3-3-3 grid?
    # Actually, let's just use the 7-hex flower and add 4 more at the "corners" of the bounding hexagon.
    
    sqrt3 = np.sqrt(3)
    inner_init = np.array([
        [0.0, 0.0, 0.0],
        [sqrt3, 0.0, 0.0],
        [sqrt3/2, 1.5, 0.0],
        [-sqrt3/2, 1.5, 0.0],
        [-sqrt3, 0.0, 0.0],
        [-sqrt3/2, -1.5, 0.0],
        [sqrt3/2, -1.5, 0.0],
        # 4 more:
        [sqrt3*1.5, sqrt3/2, 0.0], # Top right
        [0.0, sqrt3*1.5, 0.0], # Top
        [-sqrt3*1.5, sqrt3/2, 0.0], # Top left
        [sqrt3*1.5, -sqrt3/2, 0.0] # Bottom right
    ])
    
    # Initial outer side estimate
    # The 7-hex flower has side ~ 3.
    # With 4 more, it's larger.
    # Estimate side ~ 4.0
    outer_side_init = 4.5
    
    params_init = np.concatenate([inner_init.flatten(), [outer_side_init]])
    
    # Optimize
    best_params = params_init.copy()
    best_val = _objective(params_init)
    
    # Try a few restarts with Powell method which is often more robust for this
    np.random.seed(42)
    for restart in range(3):
        perturbation = np.random.randn(33) * 0.2
        perturbation[2::3] *= 0.05 # smaller angle perturbation
        params = params_init.copy()
        params[:33] += perturbation
        params[33] = outer_side_init
        
        try:
            result = minimize(_objective, params, method='Powell', 
                             options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x
        except:
            pass
    
    inner_hex_data = best_params[:33].reshape(11, 3)
    outer_side = best_params[33]
    
    # Ensure outer side is large enough
    required_side = 0.0
    for i in range(11):
        cx, cy, a_deg = inner_hex_data[i]
        verts = _hex_vertices(cx, cy, 1.0, a_deg)
        for v in verts:
            x, y = v
            c1 = abs(x * np.sqrt(3) + y) / np.sqrt(3)
            c2 = abs(x * np.sqrt(3) - y) / np.sqrt(3)
            c3 = abs(y) / (np.sqrt(3)/2)
            req = max(c1, c2, c3)
            if req > required_side:
                required_side = req
    
    outer_side = max(outer_side, required_side + 1e-4)
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, float(outer_side)

# EVOLVE-BLOCK-END

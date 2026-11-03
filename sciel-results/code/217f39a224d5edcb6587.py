"""Packing 12 unit hexagons in a regular hexagon using SLSQP with log-sum-exp smooth minimum."""
import numpy as np
import time
from scipy.optimize import minimize

N = 12
EPS = 1e-6

# Precompute unit hexagon vertices (flat-top orientation, side length 1)
# Vertices at angles 0, 60, 120, 180, 240, 300 degrees
_UNIT_HEX_VERTS = np.array([
    [np.cos(np.radians(a)), np.sin(np.radians(a))] for a in range(0, 360, 60)
])

def hex_vertices(center, angle_deg, side=1.0):
    """Return vertices of a regular hexagon with given center, rotation, and side length."""
    angle_rad = np.radians(angle_deg)
    R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                  [np.sin(angle_rad), np.cos(angle_rad)]])
    return center + R @ (side * _UNIT_HEX_VERTS).T

def polygon_overlap_separation(A, B):
    """
    Compute the minimum translation distance needed to separate two convex polygons A and B.
    If they don't overlap, returns 0.
    Uses the separating axis theorem: if any axis separates them, they don't overlap.
    For overlapping polygons, the minimum separation distance is the minimum over all axes
    of the overlap depth along that axis.
    """
    # Get all edges' normal directions from both polygons
    def get_axes(poly):
        axes = []
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i + 1) % len(poly)]
            edge = p2 - p1
            # Normal to edge
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > EPS:
                axes.append(n / norm)
        return axes
    
    axes = get_axes(A) + get_axes(B)
    
    if not axes:
        return 0.0
    
    min_sep = float('inf')
    
    for axis in axes:
        # Project both polygons onto this axis
        proj_A = np.dot(A, axis)
        proj_B = np.dot(B, axis)
        
        min_A, max_A = proj_A.min(), proj_A.max()
        min_B, max_B = proj_B.min(), proj_B.max()
        
        # Check if intervals overlap
        if max_A <= min_B + EPS or max_B <= min_A + EPS:
            # Separated on this axis, so no overlap
            return 0.0
        
        # Compute overlap depth on this axis
        overlap = min(max_A, max_B) - max(min_A, min_B)
        if overlap < min_sep:
            min_sep = overlap
    
    return min_sep

def compute_min_outer_side(inner_positions, outer_center, outer_angle):
    """
    Compute the minimum side length of a regular hexagon (with given center and rotation)
    that contains all inner hexagons.
    """
    outer_center = np.array(outer_center)
    
    # The support function of a regular hexagon with side length s, center c, rotation theta:
    # For a direction u, the support is s * h(u - theta) + c . u
    # where h is the support function of the unit hexagon.
    
    # For a regular hexagon with side length s (flat-top, i.e., one side horizontal),
    # the vertices are at distance s from center at angles 0, 60, ..., 300.
    # The support function in direction u (unit vector) is:
    # h(u) = s * max_{i} (u . v_i) where v_i are unit hex vertices
    # For a regular hexagon with circumradius = side length, 
    # h(u) = s * max(|u_x|, |u_y| * 2/sqrt(3), |u_x * sqrt(3)/2 + u_y/2|, ...) 
    # Actually, let's just compute it directly.
    
    # For a regular hexagon with side length s, the inradius is s * sqrt(3)/2
    # and circumradius is s.
    # The support function in direction (cos alpha, sin alpha) is:
    # s * max over the 6 vertex directions of (cos(alpha - theta_i))
    # where theta_i = i * 60 degrees for flat-top.
    
    # More simply: a regular hexagon with side s can be described as the intersection of 6 half-planes.
    # For flat-top orientation (no rotation), the constraints are:
    # |x| <= s
    # |y| <= s * sqrt(3) / 2
    # |x * sqrt(3) / 2 + y / 2| <= s
    # Actually, let me think more carefully.
    
    # A regular hexagon with side length s, centered at origin, flat-top:
    # Vertices: (s, 0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s, 0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
    # The 6 supporting lines (outward normals) are at angles 30, 90, 150, 210, 270, 330 degrees.
    # The distance from center to each edge (inradius) is s * sqrt(3) / 2.
    
    # So for a hexagon with center c, rotation phi, side s, a point p is inside iff:
    # For each of the 6 outward normals n_k (rotated by phi):
    #   (p - c) . n_k <= s * sqrt(3) / 2
    
    # Therefore, the minimum s such that all points of all inner hexagons are inside is:
    # s_min = max over all inner hex vertices p and all 6 normals n_k of:
    #   (p - c) . n_k / (sqrt(3) / 2)
    
    normals = np.array([
        [np.cos(np.radians(30 + 60 * k)), np.sin(np.radians(30 + 60 * k))] 
        for k in range(6)
    ])
    
    # Rotate normals by outer_angle
    phi = np.radians(outer_angle)
    R = np.array([[np.cos(phi), -np.sin(phi)],
                  [np.sin(phi), np.cos(phi)]])
    normals_rot = (R @ normals.T).T
    
    max_val = 0.0
    for pos, ang in inner_positions:
        verts = hex_vertices(np.array(pos), ang)
        for v in verts:
            d = v - outer_center
            for n in normals_rot:
                val = np.dot(d, n) / (np.sqrt(3) / 2)
                if val > max_val:
                    max_val = val
    
    return max(0.0, max_val)

def objective_and_constraints(params, outer_center, outer_angle):
    """
    Objective: minimize outer side length.
    Constraints: no overlap between any pair of inner hexagons.
    """
    # params: 3*N = [x1, y1, a1, x2, y2, a2, ..., xN, yN, aN]
    inner_positions = [(params[3*i:3*i+2], params[3*i+2]) for i in range(N)]
    
    side = compute_min_outer_side(inner_positions, outer_center, outer_angle)
    
    # Compute pairwise separations
    verts_list = [hex_vertices(np.array(p), a) for p, a in inner_positions]
    
    min_sep = float('inf')
    for i in range(N):
        for j in range(i + 1, N):
            sep = polygon_overlap_separation(verts_list[i], verts_list[j])
            if sep < min_sep:
                min_sep = sep
    
    return side, min_sep

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    best_side = float('inf')
    best_params = None
    
    # Initial guess: arrange hexagons in a rough grid pattern
    # A hexagonal packing of 12 hexagons: 2 rows of 6, or 3 rows of 4, etc.
    # Let's try a few initial configurations.
    
    np.random.seed(42)
    
    # Configuration 1: Linear arrangement (like the seed)
    init1 = []
    for k in range(-N // 2, N - N // 2):
        init1.extend([4.0 * k, 0.0, 0.0])
    
    # Configuration 2: 2 rows of 6
    init2 = []
    for row in range(2):
        for col in range(6):
            x = (col - 2.5) * 2.0
            y = (row - 0.5) * np.sqrt(3)
            init2.extend([x, y, 0.0])
    
    # Configuration 3: Hexagonal arrangement (center + ring of 6 + ring of 5)
    init3 = [0.0, 0.0, 0.0]  # Center
    for k in range(6):
        ang = np.radians(60 * k)
        r = 2.0
        init3.extend([r * np.cos(ang), r * np.sin(ang), 0.0])
    for k in range(5):
        ang = np.radians(60 * k + 30)
        r = 4.0
        init3.extend([r * np.cos(ang), r * np.sin(ang), 0.0])
    
    inits = [init1, init2, init3]
    
    for init in inits:
        if time.time() > deadline:
            break
        
        init_arr = np.array(init, dtype=float)
        
        # Use SLSQP with a penalty for overlap
        def obj_with_penalty(x):
            side, min_sep = objective_and_constraints(x, (0.0, 0.0), 0.0)
            # Penalty for negative separation (overlap)
            penalty = 0.0
            if min_sep < 0:
                penalty = 100.0 * (-min_sep) ** 2
            return side + penalty
        
        try:
            result = minimize(obj_with_penalty, init_arr, method='SLSQP',
                             options={'maxiter': 200, 'ftol': 1e-8})
            
            if result.success or result.fun < best_side:
                side, min_sep = objective_and_constraints(result.x, (0.0, 0.0), 0.0)
                if min_sep >= -1e-6:  # Allow tiny numerical errors
                    if side < best_side:
                        best_side = side
                        best_params = result.x.copy()
        except Exception:
            pass
    
    # If we found a good solution, use it
    if best_params is not None:
        inner = [[best_params[3*i], best_params[3*i+1], best_params[3*i+2]] for i in range(N)]
        return inner, (0.0, 0.0), best_side, 0.0
    
    # Fallback to seed solution
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0

# EVOLVE-BLOCK-START
# (The construct function is defined above, but we need to keep the block structure)
# Actually, let me restructure to keep the EVOLVE-BLOCK markers.
# EVOLVE-BLOCK-END

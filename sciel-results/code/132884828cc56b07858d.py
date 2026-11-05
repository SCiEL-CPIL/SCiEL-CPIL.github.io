"""Packing 11 unit hexagons in a regular hexagon using hexagonal lattice + rotation optimization."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side, rot_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy), side length, rotation in degrees."""
    r = side  # circumradius = side length for regular hexagon
    angles = np.radians(rot_deg + np.arange(6) * 60.0)
    return np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)])

def _point_in_hex(px, py, cx, cy, side, rot_deg):
    """Check if point (px,py) is inside hexagon. Returns min distance to boundary (positive if inside)."""
    # Transform to hexagon's local frame
    dx, dy = px - cx, py - cy
    rot_rad = np.radians(rot_deg)
    cos_r, sin_r = np.cos(rot_rad), np.sin(rot_rad)
    lx = dx * cos_r + dy * sin_r
    ly = -dx * sin_r + dy * cos_r
    # In local frame, hexagon has vertices at angles 0,60,...,300 with radius=side
    # The hexagon can be described by 6 half-planes
    # For a hexagon with flat top (vertex at angle 0), the constraints are:
    # |x| <= side, |y| <= side * sqrt(3)/2, and |x|/side + |y|/side <= 1 for the slanted sides
    # More precisely, for a regular hexagon with circumradius R=side:
    # The 6 supporting lines are at distance apothem = R*sqrt(3)/2 from center
    # in directions 30, 90, 150, 210, 270, 330 degrees
    apothem = side * np.sqrt(3) / 2
    # The 6 normal directions (pointing inward)
    normals = np.array([
        [np.cos(np.radians(30 + 60*i)), np.sin(np.radians(30 + 60*i))]
        for i in range(6)
    ])
    # Distance from point to each supporting line (positive means inside)
    dists = apothem - np.abs(normals @ np.array([lx, ly]))
    return np.min(dists)

def _hexagon_vertices_local(side, rot_deg):
    """Vertices of unit hexagon at origin with given rotation."""
    r = side
    angles = np.radians(rot_deg + np.arange(6) * 60.0)
    return np.column_stack([r * np.cos(angles), r * np.sin(angles)])

def _min_enclosing_hex_side(positions, outer_rot_deg):
    """
    Find minimum side length of a regular hexagon (centered at origin, rotated by outer_rot_deg)
    that contains all points in positions.
    """
    # For a hexagon centered at origin with rotation theta, a point p is inside iff
    # for all 6 face normals n_i (in the rotated frame), n_i . p <= apothem
    # The 6 face normals in local frame are at angles 30, 90, 150, 210, 270, 330
    # After rotation by theta, they are at angles 30+theta, 90+theta, etc.
    
    theta = np.radians(outer_rot_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    
    # Local face normals (6 of them)
    local_angles = np.radians(30 + np.arange(6) * 60.0)
    local_normals = np.column_stack([np.cos(local_angles), np.sin(local_angles)])
    
    # Rotate normals to global frame
    rot_mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    global_normals = (rot_mat @ local_normals.T).T  # shape (6, 2)
    
    # For each point, compute max over all normals of (n . p)
    # The required apothem is the max of all (n_i . p_j) over all i, j
    # Then side = apothem / (sqrt(3)/2) = 2*apothem/sqrt(3)
    
    max_proj = 0.0
    for p in positions:
        projs = global_normals @ p
        max_proj = max(max_proj, np.max(projs))
    
    apothem = max_proj
    side = 2.0 * apothem / np.sqrt(3.0)
    return side

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Hexagonal lattice positions for 11 unit hexagons
    # Centers at distance sqrt(3) for edge-sharing
    d = np.sqrt(3.0)
    
    # Base positions in hexagonal lattice
    base_positions = np.array([
        [0.0, 0.0],           # center
        [d, 0.0],             # ring 1
        [d/2, d*np.sqrt(3)/2],
        [-d/2, d*np.sqrt(3)/2],
        [-d, 0.0],
        [-d/2, -d*np.sqrt(3)/2],
        [d/2, -d*np.sqrt(3)/2],
        # ring 2: pick 4 positions (to make 11 total)
        [2*d, 0.0],
        [d, d*np.sqrt(3)],
        [-d, d*np.sqrt(3)],
        [-2*d, 0.0],
    ])
    
    # Each inner hexagon has its own rotation (initially 0)
    # Parameterize: 11*2 positions + 11 rotations + 1 outer rotation + 1 outer center offset
    # But by symmetry, we can fix outer center at origin and optimize outer rotation
    
    # Actually, let's think about this differently.
    # The inner hexagons are unit hexagons. Their vertices need to be inside the outer hexagon.
    # For a unit hexagon centered at (x,y) with rotation phi, its 6 vertices are:
    # (x + cos(phi + 60*k), y + sin(phi + 60*k)) for k=0..5
    
    # The outer hexagon has center (cx, cy), side length S, rotation theta.
    # A point (px, py) is inside the outer hexagon iff for all 6 face normals:
    # n_i . ((px-cx), (py-cy)) <= apothem_outer = S * sqrt(3) / 2
    
    # To minimize S, we want to minimize the max over all inner hexagon vertices of the
    # projection onto each face normal direction.
    
    # Strategy: Fix the inner hexagons in a hexagonal lattice pattern.
    # Optimize: positions (small perturbations), inner rotations, outer rotation.
    # The outer center is at the centroid of the inner hexagons (by symmetry).
    
    # Let's use a simpler approach: assume all inner hexagons have the same rotation as each other
    # (they can be rotated independently, but for now let's try uniform rotation).
    # Actually, for a hexagonal lattice of hexagons, they all have the same orientation.
    
    # Key insight: If all inner hexagons are aligned (same rotation), the cluster has 6-fold symmetry
    # (approximately, depending on which ring-2 positions we pick).
    
    # Let me try a different topology: use all 7 in the first two rings (1+6=7), 
    # then add 4 more. The 4 extra can be at ring 2 positions.
    
    # For now, let's just optimize the outer rotation angle.
    # The inner positions are fixed in the hexagonal lattice.
    
    # Compute all vertices of all 11 inner hexagons (assuming rotation = 0 for inner hexagons)
    inner_rot = 0.0  # all inner hexagons aligned
    
    all_vertices = []
    for pos in base_positions:
        angles = np.radians(inner_rot + np.arange(6) * 60.0)
        verts = np.column_stack([pos[0] + np.cos(angles), pos[1] + np.sin(angles)])
        all_vertices.append(verts)
    
    all_vertices = np.vstack(all_vertices)  # 66 vertices total
    
    # Now optimize outer rotation to minimize the enclosing hexagon side
    def objective(theta_rad):
        theta = theta_rad
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        local_angles = np.radians(30 + np.arange(6) * 60.0)
        local_normals = np.column_stack([np.cos(local_angles), np.sin(local_angles)])
        rot_mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        global_normals = (rot_mat @ local_normals.T).T
        # Max projection of any vertex onto any normal
        max_proj = 0.0
        for vn in global_normals:
            projs = all_vertices @ vn
            max_proj = max(max_proj, np.max(projs))
        return 2.0 * max_proj / np.sqrt(3.0)
    
    # Golden section search over theta in [0, 60) degrees (60-degree periodicity)
    best_side = float('inf')
    best_theta = 0.0
    
    # Coarse search
    thetas = np.linspace(0, np.radians(60), 361)
    for t in thetas:
        s = objective(t)
        if s < best_side:
            best_side = s
            best_theta = t
    
    # Fine search around best
    if time.time() < deadline - 5.0:
        # Golden section refinement
        phi = (1 + np.sqrt(5)) / 2
        a, b = best_theta - np.radians(1), best_theta + np.radians(1)
        if a < 0: a = 0
        if b > np.radians(60): b = np.radians(60)
        c = b - (b - a) / phi
        d = a + (b - a) / phi
        fc = objective(c)
        fd = objective(d)
        for _ in range(50):
            if time.time() > deadline - 2.0:
                break
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - (b - a) / phi
                fc = objective(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + (b - a) / phi
                fd = objective(d)
        mid = (a + b) / 2
        sm = objective(mid)
        if sm < best_side:
            best_side = sm
            best_theta = mid
    
    # Build the result
    inner = []
    for pos in base_positions:
        inner.append([pos[0], pos[1], inner_rot])
    
    return inner, (0.0, 0.0), best_side, np.degrees(best_theta)
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_vertices(cx, cy, d, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with side d and angle angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _min_outer_side_for_rotation(inner_centers, rot_deg):
    """
    Given inner hex centers (unit hexagons, all axis-aligned internally),
    find minimum side length s of a regular hexagon (centered at origin, rotated by rot_deg)
    that contains all 12 inner hexagons.
    Returns (s, best_center_offset) or (inf, None) if infeasible.
    """
    # For a regular hexagon with side s centered at (ox, oy) with rotation rot_deg,
    # a point (x, y) is inside iff for each of the 6 edges, the point is on the correct side.
    # The hexagon has vertices at: (ox + s*cos(rot + 60i), oy + s*sin(rot + 60i))
    # Edge i goes from vertex i to vertex i+1.
    # Normal pointing inward for edge i.
    
    rot = np.radians(rot_deg)
    
    # Vertices of unit hexagon (side 1) centered at origin
    unit_verts = np.array([[np.cos(np.radians(60*i)), np.sin(np.radians(60*i))] for i in range(6)])
    
    # For each inner hexagon center, its vertices are center + unit_verts
    # We need: for each inner vertex v, and for each outer edge, cross(edge_vec, v - edge_start) >= -eps
    
    # The outer hexagon vertices (side s, center (ox,oy), rotation rot):
    # V_i = (ox, oy) + s * [cos(rot + 60i), sin(rot + 60i)]
    
    # For containment, we need all inner vertices to be inside.
    # An inner vertex is at: c_j + u_k where c_j is center of inner hex j, u_k is unit hex vertex k.
    
    # The condition for a point p to be inside a regular hexagon with center (ox,oy), side s, rotation rot:
    # For each edge i (from V_i to V_{i+1}), the inward normal n_i must satisfy: n_i · (p - V_i) >= 0
    # Actually, let's use: the distance from p to each edge (along the inward normal) must be >= 0.
    
    # Edge i: from V_i to V_{i+1}. 
    # V_i = (ox + s*cos(rot+60i), oy + s*sin(rot+60i))
    # The edge direction is V_{i+1} - V_i.
    # The inward normal is perpendicular to the edge, pointing toward center.
    
    # For a regular hexagon, the inward normal for edge i points toward the center.
    # The edge i is between vertices at angles (rot + 60i) and (rot + 60(i+1)).
    # The midpoint of edge i is at angle (rot + 60i + 30) from center, at distance s*cos(30°) = s*sqrt(3)/2.
    # The inward normal direction is from the midpoint toward the center, i.e., angle (rot + 60i + 30 + 180).
    
    # So inward normal for edge i: direction angle = rot + 60(i+0.5) + 180 = rot + 60i + 210
    # n_i = [cos(rot + 60i + 210), sin(rot + 60i + 210)]
    
    # The distance from point p to edge i (along inward normal) is:
    # d_i = n_i · (p - M_i) where M_i is midpoint of edge i
    # M_i = (ox, oy) + s*cos(30°) * [cos(rot + 60i + 30), sin(rot + 60i + 30)]
    # Wait, let me reconsider.
    
    # Actually, for a regular hexagon, the apothem (distance from center to each edge) is s * sqrt(3)/2.
    # The edge i is at distance s*sqrt(3)/2 from the center, in the direction of angle (rot + 60i + 30).
    # So the line containing edge i is: {p : [cos(rot+60i+30), sin(rot+60i+30)] · (p - center) = s*sqrt(3)/2}
    # And the interior is where [cos(rot+60i+30), sin(rot+60i+30)] · (p - center) <= s*sqrt(3)/2
    
    # So for point p to be inside: for all i, cos(rot+60i+30)*(p_x - ox) + sin(rot+60i+30)*(p_y - oy) <= s*sqrt(3)/2
    
    # We need this for all inner vertices p = c_j + u_k.
    
    # So: for all j, k, i:
    # cos(rot+60i+30)*(c_jx + u_kx - ox) + sin(rot+60i+30)*(c_jy + u_ky - oy) <= s*sqrt(3)/2
    
    # This is: s >= [cos(rot+60i+30)*(c_jx + u_kx - ox) + sin(rot+60i+30)*(c_jy + u_ky - oy)] / (sqrt(3)/2)
    # for all j, k, i where the numerator is positive.
    
    # To minimize s, we also want to choose ox, oy optimally.
    # This is a linear programming problem: minimize s subject to linear constraints.
    # Variables: ox, oy, s.
    # Constraints: for each (j,k,i): 
    #   -s*sqrt(3)/2 + cos_i*ox + sin_i*oy <= -(cos_i*(c_jx+u_kx) + sin_i*(c_jy+u_ky))
    
    # Let's use scipy to solve this LP.
    
    from scipy.optimize import linprog
    
    # All inner vertices
    all_verts = []
    for c in inner_centers:
        for u in unit_verts:
            all_verts.append(c + u)
    all_verts = np.array(all_verts)  # shape (72, 2)
    
    # For each edge direction i (0..5) and each vertex (72), we have a constraint.
    # cos_i = cos(rot + 60*i + 30), sin_i = sin(rot + 60*i + 30)
    
    n_constraints = 6 * len(all_verts)
    A = np.zeros((n_constraints, 3))  # variables: [ox, oy, s]
    b = np.zeros(n_constraints)
    
    idx = 0
    for i in range(6):
        angle = rot + np.radians(60*i + 30)
        ci = np.cos(angle)
        si = np.sin(angle)
        for j in range(len(all_verts)):
            # ci*(vx - ox) + si*(vy - oy) <= s*sqrt(3)/2
            # -ci*ox - si*oy - sqrt(3)/2 * s <= -(ci*vx + si*vy)
            A[idx, 0] = -ci
            A[idx, 1] = -si
            A[idx, 2] = -np.sqrt(3)/2
            b[idx] = -(ci * all_verts[j, 0] + si * all_verts[j, 1])
            idx += 1
    
    # Objective: minimize s
    c_obj = np.array([0, 0, 1])
    
    # Bounds: ox, oy free, s >= 0
    bounds = [(None, None), (None, None), (0, None)]
    
    res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    
    if res.status == 0:
        s = res.x[2]
        ox = res.x[0]
        oy = res.x[1]
        return s, (ox, oy)
    else:
        return np.inf, None


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    
    # Define 12 hexagon centers in a hexagonal lattice pattern
    # Hexagonal lattice with spacing 2 (so hexagons of side 1 don't overlap)
    # Pattern: a hexagon with 5 around it, plus 6 more in a ring
    # Actually, let's use the standard hexagonal arrangement
    
    # For 12 hexagons in a hexagonal lattice, one good arrangement:
    # Row 0: 4 hexagons
    # Row 1: 4 hexagons (offset)
    # Row 2: 4 hexagons
    # But that's a rectangular-ish pattern.
    
    # Better: use the "flower" pattern with some extensions.
    # Center at (0,0)
    # 6 neighbors at distance 2: (2,0), (1, sqrt(3)), (-1, sqrt(3)), (-2,0), (-1,-sqrt(3)), (1,-sqrt(3))
    # That's 7. Need 5 more.
    # Add 5 more at the next ring positions.
    
    # Let me think of a compact arrangement.
    # A hexagonal number pattern: 1, 7, 19... 
    # For 12, we can do: center + 6 neighbors + 5 more from the next ring.
    
    sqrt3 = np.sqrt(3)
    
    # Centers for 12 hexagons in a hexagonal lattice
    # Add a small epsilon to ensure strict separation (gap > 1e-6)
    eps = 1e-3
    scale = 1.0 + eps
    
    centers = np.array([
        [0, 0],
        [2, 0],
        [1, sqrt3],
        [-1, sqrt3],
        [-2, 0],
        [-1, -sqrt3],
        [1, -sqrt3],
        # Next ring - pick 5 that make it compact
        [3, sqrt3],
        [0, 2*sqrt3],
        [-3, sqrt3],
        [-3, -sqrt3],
        [0, -2*sqrt3],
    ]) * scale
    
    # All inner hexagons are axis-aligned (angle = 0)
    inner_hex_data = np.column_stack([centers, np.zeros(12)])
    
    # Now optimize the outer hexagon rotation to minimize its side length
    # Try a range of rotations
    best_s = np.inf
    best_rot = 0
    best_center = (0, 0)
    
    # Try rotations from 0 to 60 degrees with fine steps
    for rot_deg in np.linspace(0, 60, 121):
        if time.time() > deadline - 30:
            break
        s, center = _min_outer_side_for_rotation(centers, rot_deg)
        if s < best_s:
            best_s = s
            best_rot = rot_deg
            best_center = center
    
    # Refine with finer search around best_rot
    if time.time() < deadline - 60:
        for rot_deg in np.linspace(best_rot - 5, best_rot + 5, 101):
            if time.time() > deadline - 30:
                break
            s, center = _min_outer_side_for_rotation(centers, rot_deg)
            if s < best_s:
                best_s = s
                best_rot = rot_deg
                best_center = center
    
    outer_hex_data = np.array([best_center[0], best_center[1], best_rot])
    
    return inner_hex_data, outer_hex_data, best_s


# EVOLVE-BLOCK-END

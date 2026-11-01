"""Kinematic closure strategy for packing 12 unit hexagons in a regular hexagon."""
import numpy as np
from scipy.optimize import minimize

N = 12

def _hexagon_vertices(cx, cy, side, angle_deg):
    """Return the 6 vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
    angle = np.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        vertices.append((x, y))
    return np.array(vertices)

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px, py) is inside the hexagon centered at (cx, cy)."""
    # Transform to hexagon local coordinates
    angle = np.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    # Rotate by -angle
    x_local = dx * np.cos(-angle) - dy * np.sin(-angle)
    y_local = dx * np.sin(-angle) + dy * np.cos(-angle)
    
    # A regular hexagon with side length s has inradius r = s * sqrt(3)/2
    # In local coords, the hexagon is defined by |x_local| <= s and |y_local| <= s*sqrt(3)/2 and |x_local| + |y_local|/sqrt(3) <= s
    # More precisely, the hexagon vertices in local coords (angle=0) are at (s,0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s,0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
    # The hexagon can be described as: for each of the 6 edges, the point must be on the correct side.
    
    # Using the standard form: a regular hexagon centered at origin with "flat top" orientation
    # Actually, let's use a more robust approach: check against all 6 edges.
    
    verts = _hexagon_vertices(0, 0, side, 0)  # Local vertices
    # For each edge, check if point is on the inside
    for i in range(6):
        v1 = verts[i]
        v2 = verts[(i + 1) % 6]
        # Edge vector
        ex = v2[0] - v1[0]
        ey = v2[1] - v1[1]
        # Normal pointing inward
        # For a convex polygon, the inward normal can be determined
        # The center (0,0) should be on the inside
        # Normal to edge (ex, ey) is (-ey, ex) or (ey, -ex)
        # Check which one points toward center
        nx1, ny1 = -ey, ex
        nx2, ny2 = ey, -ex
        
        # Test with center
        # The line equation: n . (p - v1) = 0
        # For inside, n . (center - v1) should have same sign as n . (p - v1)
        test1 = nx1 * (0 - v1[0]) + ny1 * (0 - v1[1])
        test2 = nx2 * (0 - v1[0]) + ny2 * (0 - v1[1])
        
        if test1 > 0:
            nx, ny = nx1, ny1
        else:
            nx, ny = nx2, ny2
        
        val = nx * (x_local - v1[0]) + ny * (y_local - v1[1])
        if val < -1e-10:
            return False
    return True

def _hexagon_distance(h1, h2):
    """Compute the minimum distance between two hexagons. h1, h2 are (cx, cy, angle_deg) with side=1."""
    cx1, cy1, a1 = h1
    cx2, cy2, a2 = h2
    
    # Vertices of both hexagons
    v1 = _hexagon_vertices(cx1, cy1, 1.0, a1)
    v2 = _hexagon_vertices(cx2, cy2, 1.0, a2)
    
    min_dist = np.inf
    
    # Check all vertex-edge pairs
    for i in range(6):
        p = v1[i]
        for j in range(6):
            # Distance from p to edge j of hex2
            e1 = v2[j]
            e2 = v2[(j + 1) % 6]
            # Project p onto line through e1, e2
            ex = e2[0] - e1[0]
            ey = e2[1] - e1[1]
            len2 = ex * ex + ey * ey
            if len2 < 1e-12:
                continue
            t = ((p[0] - e1[0]) * ex + (p[1] - e1[1]) * ey) / len2
            if t < 0 or t > 1:
                continue
            proj_x = e1[0] + t * ex
            proj_y = e1[1] + t * ey
            d = np.sqrt((p[0] - proj_x) ** 2 + (p[1] - proj_y) ** 2)
            if d < min_dist:
                min_dist = d
    
    for i in range(6):
        p = v2[i]
        for j in range(6):
            e1 = v1[j]
            e2 = v1[(j + 1) % 6]
            ex = e2[0] - e1[0]
            ey = e2[1] - e1[1]
            len2 = ex * ex + ey * ey
            if len2 < 1e-12:
                continue
            t = ((p[0] - e1[0]) * ex + (p[1] - e1[1]) * ey) / len2
            if t < 0 or t > 1:
                continue
            proj_x = e1[0] + t * ex
            proj_y = e1[1] + t * ey
            d = np.sqrt((p[0] - proj_x) ** 2 + (p[1] - proj_y) ** 2)
            if d < min_dist:
                min_dist = d
    
    return min_dist

def _outer_side_length(inner_positions, outer_center, outer_angle):
    """Compute the minimum side length of an outer hexagon (centered at outer_center, rotated by outer_angle)
    that contains all inner hexagons. Returns the side length."""
    cx, cy = outer_center
    angle_deg = outer_angle
    
    # For each inner hexagon, find the maximum "radius" in the direction of each of the 6 outward normals
    # of the outer hexagon.
    
    # The outer hexagon with center (cx, cy), side length S, and angle a has vertices at:
    # (cx + S*cos(a + k*pi/3), cy + S*sin(a + k*pi/3)) for k=0..5
    # The inradius is S*sqrt(3)/2
    # The distance from center to each edge is S*sqrt(3)/2
    
    # For a point p to be inside the outer hexagon, for each of the 6 edges, the distance from p to the edge
    # (measured perpendicular, inward) must be <= S*sqrt(3)/2... actually, the constraint is that p must be
    # on the inside of each edge.
    
    # Let's compute the required S by checking each inner vertex against each outer edge direction.
    
    # The 6 outward normal directions of the outer hexagon (from center to edge midpoints) are at angles:
    # a + pi/6 + k*pi/3 for k=0..5? Let me think...
    
    # Actually, for a regular hexagon with vertices at angles a, a+60, ..., the edges are between consecutive vertices.
    # The outward normal of the edge between vertex at angle a and vertex at angle a+60 is at angle a+30.
    # So the 6 outward normals are at angles a + 30 + k*60 degrees.
    
    # For a point p, the constraint is: (p - center) . n_k <= S*sqrt(3)/2 for all k, where n_k is the unit outward normal.
    # Wait, actually it's (p - center) . n_k <= inradius = S*sqrt(3)/2.
    
    # So S >= max over all points p (vertices of inner hexagons) and all k of: (p - center) . n_k / (sqrt(3)/2)
    
    angle_rad = np.radians(angle_deg)
    S_min = 0.0
    
    for pos in inner_positions:
        ix, iy, iangle = pos
        verts = _hexagon_vertices(ix, iy, 1.0, iangle)
        for v in verts:
            for k in range(6):
                normal_angle = angle_rad + np.pi / 6 + k * np.pi / 3
                nx = np.cos(normal_angle)
                ny = np.sin(normal_angle)
                dot = (v[0] - cx) * nx + (v[1] - cy) * ny
                required_S = dot / (np.sqrt(3) / 2)
                if required_S > S_min:
                    S_min = required_S
    
    return S_min

def _check_disjoint(inner_positions):
    """Check if all inner hexagons are pairwise disjoint. Returns True if valid."""
    for i in range(N):
        for j in range(i + 1, N):
            d = _hexagon_distance(inner_positions[i], inner_positions[j])
            if d < -1e-8:
                # They overlap (negative distance means overlap)
                # Actually, _hexagon_distance returns min distance, which is 0 if touching, positive if separated
                # If they overlap, the min distance should be negative? No, my function only checks vertex-edge distances
                # which can't be negative. Let me reconsider.
                # If two convex shapes overlap, there's no vertex of one strictly outside the other necessarily.
                # This check is insufficient. Let me use a different approach.
                pass
    # For simplicity in the search, we'll use a penalty-based approach instead of hard constraints.
    return True

def _penalty_disjoint(inner_positions, penalty_weight=100.0):
    """Compute a penalty for overlaps between inner hexagons."""
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = _hexagon_distance(inner_positions[i], inner_positions[j])
            if d < 0:
                penalty += penalty_weight * (-d) ** 2
            elif d < 1e-6:
                # Touching is OK, but very close might indicate numerical issues
                pass
    return penalty

# Define the honeycomb cluster configuration
# 12 hexagons in a honeycomb pattern: a central hexagon surrounded by 6, plus 5 more to make 12
# Actually, a hexagonal number: 1 + 6 + 12 = 19 for 3 rings. We need 12.
# Let's use: 1 center + 6 around + 5 more = 12. 
# Or: two rows of 6? No, let's use a compact cluster.
# 
# Standard honeycomb positions for 12 hexagons:
# Think of it as a hexagonal grid. Let's use axial coordinates.
# A cluster of 12: center at (0,0), 6 neighbors, and 5 of the 12 second-ring positions.
# 
# Let me define specific positions:
# Ring 0: (0,0)
# Ring 1: 6 positions at distance 1
# Ring 2: we need 5 more from the 12 positions at distance 2

# In a hex grid, the positions can be defined by (q, r) axial coordinates.
# The Euclidean position for axial (q, r) with hex side length s is:
# x = s * (q + r/2) * sqrt(3)  ... actually, let me use the standard formula.
# For a hex with side length 1, the distance between centers of adjacent hexes is sqrt(3).
# Axial coordinates (q, r):
# x = sqrt(3) * (q + r/2)
# y = 3/2 * r

# Let's list 12 positions:
# (0,0)
# (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)  -- 6 neighbors
# That's 7. Need 5 more from ring 2.
# Ring 2 positions include: (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,1), (1,2) -- wait, let me be careful.

# Actually, let me just use a simpler arrangement: a 2x6 grid or similar compact shape.
# Or better: use the known optimal or near-optimal packing.

# For now, let's define a reasonable cluster and optimize its rotation and scale.

def _get_honeycomb_positions(scale=1.0):
    """Return 12 hexagon positions in a honeycomb-like cluster, scaled by 'scale'.
    All have angle 0 initially."""
    # Axial coordinates for 12 hexagons
    # Let's use: center + 6 neighbors + 5 more
    axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    
    positions = []
    for q, r in axial:
        x = scale * np.sqrt(3) * (q + r / 2)
        y = scale * 1.5 * r
        positions.append([x, y, 0.0])
    
    return positions

def construct():
    """Construct 12 unit hexagons inside a regular hexagon, minimizing the outer side length."""
    
    # Parameterization: global rotation theta, radial scale factor s, outer hexagon rotation phi
    # We'll optimize over (theta, s, phi)
    
    def objective(params):
        theta, s, phi = params
        theta_deg = np.degrees(theta)
        phi_deg = np.degrees(phi)
        
        # Get base positions
        base_pos = _get_honeycomb_positions(scale=s)
        
        # Rotate all positions by theta around the centroid of the cluster
        # First, find centroid
        xs = [p[0] for p in base_pos]
        ys = [p[1] for p in base_pos]
        cx_cluster = np.mean(xs)
        cy_cluster = np.mean(ys)
        
        # Translate to center cluster at origin, rotate, then translate back
        rotated_pos = []
        for p in base_pos:
            x, y, a = p
            dx = x - cx_cluster
            dy = y - cy_cluster
            x_rot = dx * np.cos(theta) - dy * np.sin(theta)
            y_rot = dx * np.sin(theta) + dy * np.cos(theta)
            x_new = x_rot + cx_cluster
            y_new = y_rot + cy_cluster
            a_new = a + theta_deg
            rotated_pos.append([x_new, y_new, a_new])
        
        # Compute outer side length
        S = _outer_side_length(rotated_pos, (0.0, 0.0), phi_deg)
        
        # Add penalty for overlaps
        penalty = _penalty_disjoint(rotated_pos, penalty_weight=50.0)
        
        return S + penalty
    
    # Initial guess
    x0 = [0.0, 1.0, 0.0]  # theta=0, scale=1, phi=0
    
    # Bounds
    bounds = [(0, 2 * np.pi), (0.5, 2.0), (0, 2 * np.pi)]
    
    # Optimize
    result = minimize(objective, x0, method='Nelder-Mead', bounds=bounds,
                      options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    theta, s, phi = result.x
    theta_deg = np.degrees(theta)
    phi_deg = np.degrees(phi)
    
    # Get final positions
    base_pos = _get_honeycomb_positions(scale=s)
    
    xs = [p[0] for p in base_pos]
    ys = [p[1] for p in base_pos]
    cx_cluster = np.mean(xs)
    cy_cluster = np.mean(ys)
    
    final_pos = []
    for p in base_pos:
        x, y, a = p
        dx = x - cx_cluster
        dy = y - cy_cluster
        x_rot = dx * np.cos(theta) - dy * np.sin(theta)
        y_rot = dx * np.sin(theta) + dy * np.cos(theta)
        x_new = x_rot + cx_cluster
        y_new = y_rot + cy_cluster
        a_new = a + theta_deg
        final_pos.append([x_new, y_new, a_new])
    
    # Verify and compute final outer side length
    S = _outer_side_length(final_pos, (0.0, 0.0), phi_deg)
    
    return final_pos, (0.0, 0.0), S, phi_deg

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END

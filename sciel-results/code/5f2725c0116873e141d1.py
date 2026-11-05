"""Packing 11 unit hexagons in a minimal regular hexagon.

Approach:
- Model the problem as finding a rigid configuration of 11 unit hexagons
  that minimizes the enclosing regular hexagon side length.
- Use a 1D shape-matching approach: pre-compute the convex hull of a
  candidate cluster, then analytically derive the minimal enclosing regular
  hexagon side length as a function of global rotation angle.
- Minimize this function using high-precision search.

For 11 hexagons, a good packing is a "flower" arrangement: one central hexagon
surrounded by 6, plus 4 more arranged optimally. However, the known optimal
for small n often involves a linear or near-linear arrangement for very small n,
but for n=11, a 2D cluster is better.

Actually, let's think about the geometry:
- A unit regular hexagon has width (distance between parallel sides) = sqrt(3)
  and height (distance between opposite vertices) = 2.
- The "flat-to-flat" distance is sqrt(3), "point-to-point" is 2.

For a regular hexagon of side length S, the distance between parallel sides is S*sqrt(3),
and the distance between opposite vertices is 2S.

If we place hexagons in a grid pattern aligned with the outer hexagon:
- Adjacent hexagons in a row (horizontal) need center-to-center distance of 2*sqrt(3)
  if they share a flat side... wait, no.
  
Let me reconsider. Two unit regular hexagons sharing a full edge: center-to-center
distance is sqrt(3) (the apothem * 2 = 2 * (sqrt(3)/2) = sqrt(3)).

Actually, the apothem of a unit hexagon is sqrt(3)/2. So two hexagons sharing an edge
have centers sqrt(3) apart.

For a "honeycomb" arrangement:
- Hexagons at positions (i, j) in a hex grid have centers at:
  x = i * 3/2 + j * 3/4  (if using 30-degree rotated coordinates)
  
Let me use a simpler coordinate system. Let's define the hex grid with:
- "flat-top" orientation: the hexagons have flat top and bottom edges.
- Center-to-center distance for horizontally adjacent hexagons: 3/2 (in units of side length... no)

Wait, let me be precise. A unit regular hexagon (side length 1):
- Width (horizontal, flat-top): 2 (from leftmost to rightmost vertex)
- Height (vertical, flat-top): sqrt(3) (from top edge to bottom edge)

No wait. For a flat-top hexagon:
- The vertices are at angles 0, 60, 120, 180, 240, 300 degrees from center.
- Vertex at 0 deg: (1, 0)
- Vertex at 60 deg: (1/2, sqrt(3)/2)
- etc.
- So width (x-direction) = 2, height (y-direction) = sqrt(3).

Two flat-top hexagons sharing a vertical edge: centers are sqrt(3) apart horizontally? No.

Let me just use the standard hex grid. For pointy-top hexagons:
- Actually, let's just try a simple linear arrangement first and see, then improve.

For n=11, a 1x11 row of hexagons:
- Each hexagon is 2 units wide (point-to-point in x if oriented with a vertex pointing right).
- If we orient them with a flat side facing up/down, and place them in a row sharing edges:
  The centers would be sqrt(3) apart in the horizontal direction if they share vertical edges... 

This is getting complicated. Let me just implement a reasonable heuristic:
1. Try several candidate configurations (linear, 2D grid, etc.)
2. For each, compute the minimal enclosing hexagon
3. Return the best one

For computing the minimal enclosing regular hexagon of a set of points (the vertices of all inner hexagons), given the outer hexagon's orientation, we can use the fact that a regular hexagon can be described by 6 half-planes. The side length is determined by the maximum projection onto the 6 normal directions.

Actually, the key insight: a regular hexagon of side length S centered at origin with one vertex at angle theta has support function h(u) = S * max over the 6 directions. The minimal S such that all points are inside is the maximum over all points p and all 6 face normals n_i of (p · n_i) / (S_factor).

Let me implement this properly.
"""

import numpy as np
from scipy.optimize import minimize_scalar

N = 11

def _hexagon_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with given center, side length, and rotation."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        verts.append((cx + side * np.cos(theta), cy + side * np.sin(theta)))
    return np.array(verts)

def _all_inner_vertices(inner):
    """Get all vertices of all inner hexagons."""
    all_verts = []
    for (x, y, ang) in inner:
        verts = _hexagon_vertices(x, y, 1.0, ang)
        all_verts.append(verts)
    return np.vstack(all_verts)

def _min_enclosing_hex_side(vertices, center, outer_angle_deg):
    """
    Given a set of vertices (points), find the minimum side length S of a regular hexagon
    centered at `center` with rotation `outer_angle_deg` that contains all points.
    
    A regular hexagon of side length S can be described by 6 half-planes.
    The support function in direction u (unit vector) for a regular hexagon of side S
    centered at origin is: h(u) = S * max_i (u · v_i) where v_i are the 6 vertices of
    the unit hexagon.
    
    Actually, for a regular hexagon with circumradius R = S (since for a regular hexagon,
    side length = circumradius), the support function in direction making angle phi with
    a vertex direction is:
    h(phi) = S * cos(phi - nearest_vertex_angle) for phi within 30 degrees of a vertex,
    but more generally, h(phi) = S * cos(min_k |phi - k*pi/3|) where the min is over
    the 6 vertex angles, and the angle difference is taken mod pi/3 and clamped to [0, pi/6].
    
    Wait, for a regular hexagon with vertices at angles 0, pi/3, 2pi/3, ..., the support
    function in direction theta is:
    h(theta) = S * cos(theta - k*pi/3) where k = round(3*theta/pi) mod 6, but we need
    to ensure the angle difference is in [-pi/6, pi/6].
    
    Actually simpler: h(theta) = S * max_k cos(theta - k*pi/3) for k=0..5.
    Since cos is symmetric and periodic, this equals S * cos(d) where d = min_k |theta - k*pi/3| mod (2pi/3), clamped to [0, pi/6].
    
    So for a point p relative to center c, the constraint is:
    (p - c) · u <= h(u) for all directions u.
    The tightest constraint for a given direction u is when u is the direction from center to p.
    So S >= |p - c| * cos(angle_diff) / cos(angle_diff) ... 
    
    Let me think differently. For direction u (unit vector), the constraint is:
    (p - c) · u <= S * h_unit(u)
    where h_unit(u) is the support function of the unit hexagon.
    
    So S >= (p - c) · u / h_unit(u) for all p and all u.
    The maximum over u of ((p-c)·u / h_unit(u)) gives the required S for that point.
    The overall S is the max over all points.
    
    h_unit(u) for unit regular hexagon: if u makes angle theta, h_unit = cos(d) where
    d = min_k |theta - k*pi/3| (mod 2pi, wrapped to [0, pi/6]).
    """
    center = np.array(center)
    outer_angle = np.radians(outer_angle_deg)
    
    # Relative positions
    rel = vertices - center
    
    # For each point, compute the required S
    max_s = 0.0
    for p in rel:
        if np.allclose(p, 0):
            continue
        theta = np.arctan2(p[1], p[0])
        # Angle relative to outer hexagon orientation
        theta_rel = theta - outer_angle
        # Find nearest vertex angle (multiples of pi/3)
        k = np.round(theta_rel / (np.pi / 3))
        d = np.abs(theta_rel - k * np.pi / 3)
        # Wrap to [0, pi/6]
        d = min(d, np.pi - d)
        d = min(d, np.pi / 6) if d > np.pi / 6 else d
        # Actually, due to symmetry, d should be in [0, pi/6]
        d = np.abs(d % (np.pi / 3))
        d = min(d, np.pi / 3 - d)
        
        h_unit = np.cos(d)
        if h_unit < 1e-12:
            h_unit = 1e-12
        
        r = np.linalg.norm(p)
        s_needed = r / h_unit
        max_s = max(max_s, s_needed)
    
    return max_s

def _score_config(inner, outer_center, outer_angle_deg):
    """Compute the side length of the minimal enclosing hexagon."""
    verts = _all_inner_vertices(inner)
    return _min_enclosing_hex_side(verts, outer_center, outer_angle_deg)

def construct():
    # Try several candidate configurations
    
    best_side = float('inf')
    best_result = None
    
    # Candidate 1: Linear arrangement along x-axis
    # Hexagons sharing edges in a row. For flat-top hexagons, 
    # adjacent centers are sqrt(3) apart in x direction if sharing vertical edges.
    # But actually for a row of hexagons sharing edges, the offset depends on orientation.
    
    # Let's try: all hexagons with angle 0 (vertex pointing right), placed in a line
    # with center spacing of 2 (point to point) - but that's not sharing edges.
    # For sharing an edge, center distance is sqrt(3) in the direction perpendicular to the shared edge.
    
    # Let's try a honeycomb-like 2D arrangement.
    # Hex grid coordinates: for "pointy-top" hexagons:
    # x = size * sqrt(3) * (col + row/2)
    # y = size * 3/2 * row
    
    # For unit hexagons (size=1):
    # Let's place them in a 4-3-4 pattern or similar.
    
    # Actually, let's try multiple configurations and pick the best.
    
    configs = []
    
    # Config A: 11 in a row, each sharing an edge with the next
    # If oriented with angle 0 (vertex at 0 degrees, so flat sides at 30, 90, 150, etc.)
    # The "width" in x is 2, "height" in y is sqrt(3).
    # To share a vertical-ish edge, offset by sqrt(3) in x? No.
    # For two hexagons with angle 0, sharing the edge between vertices at 60 and 120 degrees:
    # That edge is vertical, at x = 1/2 from center. So centers are 1 apart in x? 
    # No, the edge midpoint is at distance apothem = sqrt(3)/2 from center.
    # So two hexagons sharing that edge have centers 2*(sqrt(3)/2) = sqrt(3) apart in x.
    
    # Let's just try: centers at (k*sqrt(3), 0) for k in range, all angle 0.
    sqrt3 = np.sqrt(3)
    n_half = N // 2
    inner_a = [(k * sqrt3, 0.0, 0.0) for k in range(-n_half, N - n_half)]
    configs.append(('A', inner_a, (0.0, 0.0), 0.0))
    
    # Config B: Honeycomb 2D arrangement
    # Rows of hexagons. Row 0: 4 hexagons, Row 1: 3 hexagons offset, Row 2: 4 hexagons
    # Or 3-5-3, etc.
    
    # Let's try a compact 2D cluster.
    # Hex grid positions (axial coordinates q, r):
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    
    # Pick 11 positions that form a compact cluster.
    # Center at (0,0), neighbors at 6 positions, then next ring.
    # Ring 0: (0,0) - 1 hex
    # Ring 1: 6 hexagons at (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # Ring 2: 12 hexagons, pick 4 more.
    
    hex_positions = [(0, 0)]
    ring1 = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    hex_positions.extend(ring1)
    # Need 4 more from ring 2
    ring2 = [(2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-2), (-2,-1), (-2,0), (-2,1), (-1,2), (0,2), (1,2)]
    # Pick 4 that are most central
    hex_positions.extend(ring2[:4])
    
    inner_b = []
    for (q, r) in hex_positions[:N]:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        inner_b.append((x, y, 0.0))
    configs.append(('B', inner_b, (0.0, 0.0), 0.0))
    
    # Config C: Try rotating the outer hexagon to optimize
    # For each config, try a range of outer angles
    
    for name, inner, center, base_angle in configs:
        # Try different outer angles
        for outer_ang in range(0, 30, 5):
            side = _score_config(inner, center, outer_ang)
            if side < best_side:
                best_side = side
                best_result = (inner, center, side, outer_ang)
        
        # Fine-tune with local search on angle
        def obj(angle):
            return _score_config(inner, center, angle)
        
        result = minimize_scalar(obj, bounds=(base_angle - 15, base_angle + 15), method='bounded', options={'xatol': 1e-8})
        if result.success:
            side = result.fun
            if side < best_side:
                best_side = side
                best_result = (inner, center, side, result.x)
    
    if best_result is None:
        # Fallback
        inner = [(k * sqrt3, 0.0, 0.0) for k in range(-n_half, N - n_half)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    inner, center, side, angle = best_result
    return inner, center, side, angle

# EVOLVE-BLOCK-START
# (The construct function is defined above; this block is kept for compatibility)
# EVOLVE-BLOCK-END

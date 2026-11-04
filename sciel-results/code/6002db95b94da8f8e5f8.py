"""
Pack 11 unit regular hexagons into a minimal outer regular hexagon.

Strategy:
  1. Use a hexagonal lattice arrangement (honeycomb) which is optimal for
     hexagonal packing. For 11 hexagons, the minimal lattice arrangement is
     a central hexagon with 6 neighbors and 4 more on the next ring (or
     equivalent compact shape).
  2. The outer hexagon is axis-aligned (or we can rotate the whole config).
  3. Compute the minimal enclosing regular hexagon for the given inner
     hexagon centers + their vertices.
  4. Try several rotations of the inner configuration relative to the outer
     hexagon to find the minimum side length.
  5. Additionally, try small perturbations of positions to potentially
     improve beyond the lattice optimum.

A unit regular hexagon has circumradius = 1 (distance from center to vertex).
Its inradius (apothem) = sqrt(3)/2.

For a regular hexagon of side s, the distance from center to a vertex is s,
and the apothem is s*sqrt(3)/2.

To check if a point (px, py) is inside a regular hexagon centered at (cx, cy)
with side s and rotation theta:
  The hexagon has 6 sides. The half-width in each of the 3 axis directions
  (at angles theta + k*60 degrees for k=0..2, measuring to the midpoints of
  opposite sides) is s*sqrt(3)/2.

Actually, let's use the support function approach:
A regular hexagon centered at origin with side s, "flat-top" orientation, has
the property that a point (x,y) is inside iff for all 6 normal directions
n_k (unit vectors pointing to side midpoints), the dot product x·n_k <= s*sqrt(3)/2.

The 6 normal directions for a flat-top hexagon (one flat side on top) are:
  angles: 90, 30, -30, -90, -150, 150 degrees (i.e., 30 + k*60 for k=0..5)

Wait, let me be more careful. A regular hexagon with side length s:
- Circumradius R = s
- Inradius r = s * sqrt(3)/2

For a "pointy-top" hexagon (vertex at top), the vertices are at angles
90, 150, 210, 270, 330, 30 degrees from center, at distance s.

For a "flat-top" hexagon, vertices are at 0, 60, 120, 180, 240, 300 degrees.

The support function (maximum dot product with a direction) of a regular
hexagon with inradius r in direction making angle alpha with the normal to
a side is: r / cos(alpha mod 60 - 30) ... this is getting complicated.

Simpler approach: The minimum side length S of an outer regular hexagon
(centered at origin, with some rotation) that contains all inner hexagon
vertices is:

For each vertex v of each inner hexagon, compute the "hexagonal norm" in the
direction of the outer hexagon. The outer hexagon with side S and rotation
phi contains point p iff for all 6 directions n_k = (cos(phi + k*60°), sin(phi + k*60°)),
we have p · n_k <= S * sqrt(3)/2.

So S >= max over all vertices p and all k of (p · n_k) / (sqrt(3)/2).
And also S >= max over all vertices p and all k of (-p · n_k) / (sqrt(3)/2)
(because the hexagon is symmetric, both +n_k and -n_k are normals).

So S = (2/sqrt(3)) * max_{p, k} |p · n_k|

where n_k = (cos(phi + k*pi/3), sin(phi + k*pi/3)) for k=0..5.

This gives us a clean formula. Let's implement this.
"""

import math
import time
import numpy as np
from itertools import product

N = 11

def _hexagon_vertices(cx, cy, angle_deg, side=1.0):
    """Return the 6 vertices of a unit regular hexagon centered at (cx,cy)
    with rotation angle_deg (degrees). Vertex 0 is at angle angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts

def _min_outer_side(vertices, outer_center=(0.0, 0.0), outer_rotation=0.0):
    """Given a list of vertices (x,y), find the minimum side length S of a
    regular hexagon centered at outer_center with rotation outer_rotation
    (degrees) that contains all vertices.
    
    S = (2/sqrt(3)) * max_{v, k} |(v - c) · n_k|
    where n_k = (cos(rot + k*pi/3), sin(rot + k*pi/3)) for k=0..5.
    """
    rot = math.radians(outer_rotation)
    cx, cy = outer_center
    max_proj = 0.0
    for k in range(6):
        nx = math.cos(rot + k * math.pi / 3)
        ny = math.sin(rot + k * math.pi / 3)
        for (vx, vy) in vertices:
            dx = vx - cx
            dy = vy - cy
            proj = abs(dx * nx + dy * ny)
            if proj > max_proj:
                max_proj = proj
    return max_proj * 2.0 / math.sqrt(3.0)

def _all_vertices(inner_config):
    """inner_config: list of (x, y, angle_deg). Return all 6*N vertices."""
    verts = []
    for (x, y, ang) in inner_config:
        verts.extend(_hexagon_vertices(x, y, ang))
    return verts

def construct():
    t0 = time.time()
    DEADLINE = 120.0  # seconds
    
    # --- Build initial hexagonal lattice configuration for 11 hexagons ---
    # In a hexagonal lattice with spacing 2 (so unit hexagons touch),
    # positions are:
    #   Row 0: 7 hexagons at x = -6, -4, -2, 0, 2, 4, 6  (y=0)
    #   Row 1: 4 hexagons at x = -5, -1, 3, 7  (y = sqrt(3))
    # Total: 11
    # But this is a line + partial second row. Let's use a more compact shape.
    #
    # Better: Use the "flower" + extras pattern.
    # Center at (0,0). 6 neighbors at distance 2 in 6 directions.
    # That's 7. Need 4 more. Place them on the next ring at distance 2*sqrt(3)
    # in 4 of the 6 directions, or at distance 4 in 4 of the 6 directions.
    #
    # Actually for hexagonal packing, the optimal 11-point cluster minimizes
    # the maximum distance from center. The hexagonal lattice points within
    # hexagonal distance 2 from center form a hexagonal number pattern:
    # distance 0: 1 point
    # distance 1: 6 points  
    # distance 2: 12 points
    # Total within distance 2: 19. Too many.
    #
    # Let's just use the linear+row arrangement which is simple and known.
    # 7 in a row, 4 in the next row offset.
    
    sqrt3 = math.sqrt(3)
    
    # Configuration 1: Linear row of 7 + row of 4 offset
    # All hexagons at angle 0 (flat-top or pointy-top, doesn't matter for lattice)
    # Spacing between centers = 2 (they touch)
    
    # Row 0: 7 hexagons
    row0 = [(2.0 * i, 0.0, 0.0) for i in range(-3, 4)]  # x = -6,-4,-2,0,2,4,6
    # Row 1: 4 hexagons, offset by 1 in x, up by sqrt(3)
    row1 = [(2.0 * i + 1.0, sqrt3, 0.0) for i in range(-2, 2)]  # x=-3,-1,1,3
    
    config1 = row0 + row1  # 11 hexagons
    
    # Configuration 2: More compact "hexagonal blob"
    # Center + 6 neighbors + 4 on next ring
    config2 = [(0.0, 0.0, 0.0)]
    for i in range(6):
        ang = i * 60.0
        x = 2.0 * math.cos(math.radians(ang))
        y = 2.0 * math.sin(math.radians(ang))
        config2.append((x, y, 0.0))
    # 4 more: place at next ring, 4 of 6 positions
    for i in [0, 1, 2, 3]:
        ang = i * 60.0
        x = 4.0 * math.cos(math.radians(ang))
        y = 4.0 * math.sin(math.radians(ang))
        config2.append((x, y, 0.0))
    
    # Configuration 3: Another compact arrangement
    # 3-4-4-3 pattern? No, that's 14.
    # Let's try: center, 6 ring, then 4 at distance 2*sqrt(3) in diagonal directions
    config3 = [(0.0, 0.0, 0.0)]
    for i in range(6):
        ang = i * 60.0
        x = 2.0 * math.cos(math.radians(ang))
        y = 2.0 * math.sin(math.radians(ang))
        config3.append((x, y, 0.0))
    # 4 more at the "corner" positions of the hexagonal lattice
    # These are at distance 2*sqrt(3) from center, at angles 30, 90, 150, 210 (or similar)
    for ang in [30, 90, 150, 270]:
        x = 2.0 * sqrt3 * math.cos(math.radians(ang))
        y = 2.0 * sqrt3 * math.sin(math.radians(ang))
        config3.append((x, y, 0.0))
    
    best_score = float('inf')
    best_result = None
    
    # Evaluate each configuration with various outer rotations
    configs = [config1, config2, config3]
    
    for config in configs:
        verts = _all_vertices(config)
        
        # Try outer rotations from 0 to 60 degrees in fine steps
        for outer_rot in np.linspace(0, 60, 121):
            s = _min_outer_side(verts, (0.0, 0.0), outer_rot)
            if s < best_score:
                best_score = s
                best_result = (config, (0.0, 0.0), s, outer_rot)
        
        if time.time() - t0 > DEADLINE - 5:
            break
    
    # If we have time, try small perturbations of the best config
    if best_result is not None and time.time() - t0 < DEADLINE - 30:
        best_config, best_center, best_s, best_rot = best_result
        verts = _all_vertices(best_config)
        
        # Try translating the whole config
        for dx in np.linspace(-0.5, 0.5, 21):
            for dy in np.linspace(-0.5, 0.5, 21):
                shifted_verts = [(x + dx, y + dy) for (x, y) in verts]
                s = _min_outer_side(shifted_verts, (0.0, 0.0), best_rot)
                if s < best_score:
                    best_score = s
                    new_config = [(x + dx, y + dy, a) for (x, y, a) in best_config]
                    best_result = (new_config, (0.0, 0.0), s, best_rot)
        
        if time.time() - t0 > DEADLINE - 10:
            break
    
    # Try rotating individual hexagons slightly to reduce bounding hexagon
    if best_result is not None and time.time() - t0 < DEADLINE - 20:
        best_config, best_center, best_s, best_rot = best_result
        # Try rotating all inner hexagons by small angles
        for inner_rot in np.linspace(-30, 30, 13):
            new_config = [(x, y, inner_rot) for (x, y, _) in best_config]
            verts = _all_vertices(new_config)
            for outer_rot in np.linspace(best_rot - 10, best_rot + 10, 21):
                s = _min_outer_side(verts, (0.0, 0.0), outer_rot)
                if s < best_score:
                    best_score = s
                    best_result = (new_config, best_center, s, outer_rot)
        
        if time.time() - t0 > DEADLINE - 5:
            break
    
    if best_result is None:
        # Fallback
        return config1, (0.0, 0.0), 40.0, 0.0
    
    inner, center, side, angle = best_result
    return inner, center, side, angle

# EVOLVE-BLOCK-START
# (The construct function is defined above; this block is kept for compatibility)
# EVOLVE-BLOCK-END

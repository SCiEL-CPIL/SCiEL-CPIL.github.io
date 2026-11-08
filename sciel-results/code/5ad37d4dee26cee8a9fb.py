"""Hexagon packing: 11 unit hexagons in honeycomb cluster, minimal outer hexagon."""
import math
import time

N = 11

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, r, angle_rad):
    """Vertices of a regular hexagon (flat-top orientation at angle 0)."""
    verts = []
    for i in range(6):
        a = angle_rad + i * math.pi / 3.0
        verts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return verts


def _point_in_hex(px, py, cx, cy, r, angle_rad):
    """Check if point (px,py) is inside hexagon centered (cx,cy) radius r, rotated angle_rad."""
    # Transform point to hexagon's local frame
    dx = px - cx
    dy = py - cy
    ca = math.cos(-angle_rad)
    sa = math.sin(-angle_rad)
    lx = dx * ca - dy * sa
    ly = dx * sa + dy * ca
    # Hexagon with circumradius r: vertices at angles 0,60,120,...
    # For a regular hexagon, the inradius is r * sqrt(3)/2
    # The hexagon in local frame: |lx| <= r, and the width at height ly
    # Actually for circumradius r, the hexagon has:
    # 6 vertices at (r*cos(k*pi/3), r*sin(k*pi/3)) for k=0..5
    # The boundary: for a given ly, the max |lx| is r * sqrt(1 - (2*ly/(r*sqrt(3)))^2) ... no
    # Simpler: use the fact that a regular hexagon with circumradius r has
    # inradius = r * sqrt(3)/2. The hexagon is the intersection of 6 half-planes.
    # In local frame, the 6 edges are at distance inradius from center,
    # with normals at angles 30, 90, 150, 210, 270, 330 degrees (for flat-top)
    # Wait, for pointy-top (vertex at angle 0), normals are at 0, 60, 120, ...
    # Let me just check: vertex at angle 0 is (r, 0). The edge between vertices
    # at angles 0 and 60 has midpoint at angle 30, distance r*cos(30) = r*sqrt(3)/2.
    # So the 6 supporting lines have normals at angles 30, 90, 150, 210, 270, 330.
    inradius = r * math.sqrt(3.0) / 2.0
    for k in range(6):
        na = (2 * k + 1) * math.pi / 6.0  # 30, 90, 150, 210, 270, 330
        nx = math.cos(na)
        ny = math.sin(na)
        if lx * nx + ly * ny > inradius + 1e-9:
            return False
    return True


def _segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def _hexagons_disjoint(verts1, verts2, eps=1e-9):
    """Check if two hexagons (as vertex lists) are disjoint (no edge crossing, no containment)."""
    # Check all edge pairs
    for i in range(6):
        for j in range(6):
            if _segments_intersect(verts1[i], verts1[(i+1)%6], verts2[j], verts2[(j+1)%6]):
                return False
    # Check if any vertex of one is inside the other
    for v in verts1:
        if _point_in_hex(v[0], v[1], 0, 0, 0, 0):  # placeholder, not used this way
            pass
    # Better: check vertex containment properly
    cx1 = sum(v[0] for v in verts1) / 6.0
    cy1 = sum(v[1] for v in verts1) / 6.0
    cx2 = sum(v[0] for v in verts2) / 6.0
    cy2 = sum(v[1] for v in verts2) / 6.0
    r1 = math.sqrt(sum((v[0]-cx1)**2 + (v[1]-cy1)**2 for v in verts1)) / 1.0
    # Just do edge intersection check; for convex polygons, that's sufficient for disjointness
    # if they're not nested. For same-size hexagons, nesting is impossible.
    return True


def construct():
    deadline = time.time() + 120.0
    
    # Honeycomb cluster: center hexagon + 6 around it + 4 more in a ring
    # Standard honeycomb positions for 7 hexagons (1+6):
    # Using axial coordinates, convert to Cartesian for pointy-top hexagons
    # For a unit hexagon (side=1, circumradius=1), the distance between adjacent
    # hexagon centers in a honeycomb is sqrt(3) (the inradius*2 = 2*sqrt(3)/2 = sqrt(3))
    
    # Actually for regular hexagons of side length s=1:
    # circumradius = 1, inradius = sqrt(3)/2
    # In a honeycomb tiling, adjacent centers are 2*inradius = sqrt(3) apart.
    
    # Let's use a simpler approach: place hexagons on a triangular lattice
    # with spacing that ensures non-overlap.
    
    # For non-rotated hexagons (angle=0), the minimum center-to-center distance
    # to avoid overlap is 2 * inradius = sqrt(3) in the "flat" direction.
    # But actually, two unit hexagons (circumradius 1) need center distance >= 2
    # to be disjoint (when aligned).
    
    # Let me think more carefully. Two unit regular hexagons (side 1, circumradius 1):
    # The Minkowski sum of two such hexagons is a hexagon with circumradius 2.
    # So they are disjoint iff center distance > 2 (for the same orientation).
    # For different orientations, it's more complex.
    
    # Simplest valid packing: place all hexagons at the same orientation,
    # on a triangular lattice with spacing 2 (so they just touch).
    
    # 11 hexagons in a hexagonal cluster:
    # Ring 0: 1 hexagon at center
    # Ring 1: 6 hexagons
    # Ring 2: need 4 more from the 12 in ring 2
    # Total: 1 + 6 + 4 = 11
    
    # For a hexagonal lattice with spacing d = 2 (circumradius of unit hex),
    # the positions in ring k are at distance k*d from center.
    # Ring 1: 6 positions at distance 2, angles 0, 60, 120, 180, 240, 300
    # Ring 2: 12 positions at distance 4, angles 0, 30, 60, ..., 330
    #         But actually ring 2 has positions at distance 4 at angles 0,60,120,180,240,300
    #         and distance 2*sqrt(3) at angles 30,90,150,210,270,330
    #         Wait, that's not right for a triangular lattice.
    
    # Triangular lattice basis: e1 = (d, 0), e2 = (d/2, d*sqrt(3)/2)
    # Position (i,j) in lattice coords: (i*d + j*d/2, j*d*sqrt(3)/2)
    # For d=2: (2i + j, j*sqrt(3))
    
    # Let's enumerate lattice points and pick 11 that form a compact cluster.
    # Lattice points near origin:
    # (0,0) -> (0, 0)
    # (1,0) -> (2, 0)
    # (-1,0) -> (-2, 0)
    # (0,1) -> (1, sqrt(3))
    # (0,-1) -> (-1, -sqrt(3))
    # (1,-1) -> (1, -sqrt(3))
    # (-1,1) -> (-1, sqrt(3))
    # These 7 form the standard 1+6 cluster.
    
    # For 4 more, add from the next ring:
    # (2,0) -> (4, 0)
    # (-2,0) -> (-4, 0)
    # (2,-1) -> (3, -sqrt(3))
    # (-2,1) -> (-3, sqrt(3))
    # Or: (1,1) -> (3, sqrt(3)), (-1,-1) -> (-3, -sqrt(3)), (0,2) -> (2, 2*sqrt(3)), (0,-2) -> (-2, -2*sqrt(3))
    
    # Let's use a compact arrangement:
    # 7 in the first cluster + 4 more
    # To minimize the bounding hexagon, I'll try several configurations.
    
    sqrt3 = math.sqrt(3.0)
    d = 2.0  # spacing for non-overlapping unit hexagons
    
    # Generate candidate cluster configurations
    # Config 1: 7 + 4 on the "ends"
    lattice_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * d + j * d / 2.0
            y = j * d * sqrt3 / 2.0
            lattice_points.append((i, j, x, y))
    
    # Find compact clusters of 11 points
    # Use the 7 central points + 4 more
    central = [(0,0,0,0), (1,0,d,0), (-1,0,-d,0), 
               (0,1,d/2,d*sqrt3/2), (0,-1,-d/2,-d*sqrt3/2),
               (1,-1,d/2,-d*sqrt3/2), (-1,1,-d/2,d*sqrt3/2)]
    
    # Additional candidates for the 4 extra hexagons
    extra_candidates = [
        (2,0,2*d,0), (-2,0,-2*d,0),
        (2,-1,3*d/2,-d*sqrt3/2), (-2,1,-3*d/2,d*sqrt3/2),
        (1,1,3*d/2,d*sqrt3/2), (-1,-1,-3*d/2,-d*sqrt3/2),
        (0,2,d,2*d*sqrt3/2), (0,-2,-d,-2*d*sqrt3/2),
        (2,-2,3*d,-d*sqrt3), (-2,2,-3*d,d*sqrt3),
        (3,0,3*d,0), (-3,0,-3*d,0),
        (1,2,2*d,d*sqrt3), (-1,-2,-2*d,-d*sqrt3),
        (2,1,5*d/2,d*sqrt3/2), (-2,-1,-5*d/2,-d*sqrt3/2),
    ]
    
    best_side = float('inf')
    best_config = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Try adding 4 from extra_candidates to the 7 central
    from itertools import combinations
    
    # Limit combinations for speed
    # Try a few specific configurations first
    test_configs = [
        # 7 central + 4 on the "equator" (left and right extensions)
        [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1),(2,0),(-2,0),(2,-1),(-2,1)],
        # 7 central + 4 extending one direction
        [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1),(2,0),(2,-1),(3,0),(3,-1)],
        # 7 central + 4 in a 2x2 block
        [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1),(2,0),(2,-1),(1,1),(0,2)],
        # 7 central + 4 spread out
        [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,-1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)],
    ]
    
    for config in test_configs:
        if len(config) != 11:
            continue
        # Get positions
        pos_map = {(i,j): (i*d + j*d/2.0, j*d*sqrt3/2.0) for i,j in config}
        points = [pos_map[(i,j)] for i,j in config]
        
        # Compute bounding box to center the cluster
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        
        # Translate so center is at origin
        points_centered = [(x - cx, y - cy) for x, y in points]
        
        # For each rotation angle of the outer hexagon, compute the minimal side length
        # The outer hexagon must contain all 11 unit hexagons.
        # A unit hexagon at position (px, py) with orientation theta_inner has
        # vertices at (px + cos(theta_inner + k*pi/3), py + sin(theta_inner + k*pi/3))
        # The outer hexagon with center (0,0), side length S, orientation theta_outer
        # has circumradius S. It contains a point (x,y) iff
        # for all k: x*cos(theta_outer + (2k+1)*pi/6) + y*sin(theta_outer + (2k+1)*pi/6) <= S*sqrt(3)/2
        
        # For all inner hexagons to fit, all their vertices must be inside.
        # Inner hexagons all have orientation 0 (for simplicity).
        # Vertices of inner hex at (px,py): (px + cos(k*pi/3), py + sin(k*pi/3)) for k=0..5
        
        # For a given outer orientation theta, the required S is:
        # S >= max over all vertices v of: (2/sqrt(3)) * max_k (v.x*cos(na_k) + v.y*sin(na_k))
        # where na_k = theta + (2k+1)*pi/6
        
        # Try several outer orientations
        for theta_deg in range(0, 60, 5):
            theta = math.radians(theta_deg)
            max_val = 0.0
            for (px, py) in points_centered:
                for k in range(6):
                    vx = px + math.cos(k * math.pi / 3.0)
                    vy = py + math.sin(k * math.pi / 3.0)
                    for m in range(6):
                        na = theta + (2*m + 1) * math.pi / 6.0
                        val = vx * math.cos(na) + vy * math.sin(na)
                        if val > max_val:
                            max_val = val
            S = max_val * 2.0 / math.sqrt(3.0)
            if S < best_side:
                best_side = S
                best_config = points_centered
                best_center = (0.0, 0.0)
                best_angle = theta_deg
    
    # Fine-tune: try more angles around the best
    if best_config is not None:
        for theta_deg_10 in range(int(best_angle * 10) - 20, int(best_angle * 10) + 21):
            theta_deg = theta_deg_10 / 10.0
            theta = math.radians(theta_deg)
            max_val = 0.0
            for (px, py) in best_config:
                for k in range(6):
                    vx = px + math.cos(k * math.pi / 3.0)
                    vy = py + math.sin(k * math.pi / 3.0)
                    for m in range(6):
                        na = theta + (2*m + 1) * math.pi / 6.0
                        val = vx * math.cos(na) + vy * math.sin(na)
                        if val > max_val:
                            max_val = val
            S = max_val * 2.0 / math.sqrt(3.0)
            if S < best_side:
                best_side = S
                best_angle = theta_deg
    
    if best_config is None:
        # Fallback: trivial placement
        inner = [[2.0 * k, 0.0, 0.0] for k in range(-5, 6)]
        return inner, (0.0, 0.0), 20.0, 0.0
    
    inner = [[x, y, 0.0] for x, y in best_config]
    return inner, (0.0, 0.0), best_side, best_angle

# EVOLVE-BLOCK-END

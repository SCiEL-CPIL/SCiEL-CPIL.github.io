"""Hexagon packing: 11 unit hexagons in minimal outer hexagon."""
import math
import time
import numpy as np

N = 11
DEADLINE = time.time() + 130.0  # safety margin before 150s kill

def hex_vertices(cx, cy, s, angle_deg=0.0):
    """Vertices of regular hexagon side s, center (cx,cy), rotation in degrees.
    Vertex k at angle (60*k + angle) from center, radius = s (circumradius)."""
    a = math.radians(angle_deg)
    verts = []
    for k in range(6):
        th = math.radians(60.0 * k) + a
        verts.append((cx + s * math.cos(th), cy + s * math.sin(th)))
    return verts

def point_in_polygon(px, py, poly):
    """Ray casting."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def segments_intersect(p1, p2, p3, p4):
    """True if segments p1-p2 and p3-p4 properly intersect."""
    def ccw(a, b, c):
        return (c[1]-a[1])*(b[0]-a[0]) - (b[1]-a[1])*(c[0]-a[0])
    d1 = ccw(p3, p4, p1)
    d2 = ccw(p3, p4, p2)
    d3 = ccw(p1, p2, p3)
    d4 = ccw(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def hexagons_disjoint(v1, v2, gap=0.0):
    """Check two hexagons (vertex lists) are disjoint with gap.
    v1, v2 are lists of 6 (x,y)."""
    # Check if any vertex of v1 is strictly inside v2 (expanded by gap)
    # Simple check: vertex of v1 inside v2, or vertex of v2 inside v1
    for p in v1:
        if point_in_polygon(p[0], p[1], v2):
            return False
    for p in v2:
        if point_in_polygon(p[0], p[1], v1):
            return False
    # Check edge crossings
    n1, n2 = len(v1), len(v2)
    for i in range(n1):
        a1 = v1[i]
        b1 = v1[(i+1) % n1]
        for j in range(n2):
            a2 = v2[j]
            b2 = v2[(j+1) % n2]
            if segments_intersect(a1, b1, a2, b2):
                return False
    return True

def enclosing_hex_side_axis_aligned(points):
    """Compute the minimum side length of an axis-aligned regular hexagon
    (flat-top, vertices at 0,60,120,180,240,300 degrees) that contains all points.
    For a hexagon with circumradius R (side length R), the support in direction theta
    is R * max_k cos(theta - 60k). We need R such that for all points, point is inside.
    
    A point (x,y) is inside an axis-aligned hexagon of side R iff:
    |x| <= R and |x*sqrt(3)/2 + y/2| <= R*sqrt(3)/2 ... 
    Actually, the hexagon with vertices at angles 0,60,...,300 and radius R:
    The 6 edge normals are at angles 30, 90, 150, 210, 270, 330 degrees.
    The inradius (apothem) is R*sqrt(3)/2.
    Point (x,y) is inside iff for all 6 normals n_k: dot((x,y), n_k) <= R*sqrt(3)/2.
    
    So R >= max over points, over k: dot(p, n_k) / (sqrt(3)/2)
    where n_k are unit normals at 30+60k degrees.
    """
    # Normals at 30, 90, 150, 210, 270, 330 degrees
    normals = []
    for k in range(6):
        th = math.radians(30.0 + 60.0 * k)
        normals.append((math.cos(th), math.sin(th)))
    
    max_proj = 0.0
    for (px, py) in points:
        for (nx, ny) in normals:
            proj = px * nx + py * ny
            if proj > max_proj:
                max_proj = proj
    # R * sqrt(3)/2 >= max_proj  =>  R >= max_proj * 2/sqrt(3)
    # Also need R >= max |x|, |y|? No, the hexagon constraint handles it.
    # But R must also be at least the max distance from center in the vertex directions.
    # Actually the support function formulation: for the hexagon to contain point p,
    # we need support(p) <= apothem for all 6 face normals.
    # support in direction n is dot(p, n).
    # apothem = R * sqrt(3)/2
    # So R >= max_proj * 2 / sqrt(3)
    R = max_proj * 2.0 / math.sqrt(3.0)
    # Also ensure R >= 0
    return max(R, 0.0)

def generate_honeycomb_topologies():
    """Generate candidate topologies for 11 hexagons on honeycomb lattice.
    Honeycomb lattice: axial coordinates (q, r) with q + r + s = 0.
    Position: x = (q + r/2) * sqrt(3), y = r * 1.5  (for unit hexagons, flat-top)
    Wait, let me use the standard: for a hexagon of side 1, the distance between
    centers of adjacent hexagons in a honeycomb is 2 * inradius = sqrt(3).
    
    Axial coordinate (q, r):
    x = sqrt(3) * (q + r/2)
    y = 3/2 * r
    """
    topologies = []
    
    # Topology 1: Center + 6 around + 4 more (hexagonal number 7 + 4)
    # Hex number 7: center (0,0) + 6 neighbors
    # Add 4 more to make 11
    
    # Hex 7: (0,0), and 6 at distance 1
    base7 = [(0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    
    # Hex 12: center + 6 + 6 at distance 2 (but we only need 11, so take 11 from 12)
    # Distance 2 neighbors: (2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-2), (-2,-1), (-2,0), (-2,1), (-1,1)... 
    # Let me enumerate all hexes within distance 2 from center
    all_hexes = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if abs(q) <= 2 and abs(r) <= 2 and abs(s) <= 2:
                all_hexes.append((q, r))
    # This gives 19 hexes. We need to pick 11 that form a connected cluster.
    
    # Simpler: just use specific compact topologies
    
    # Topology A: 7-hex center + 4 extending in one direction
    # (0,0) + 6 ring + 4 more
    # Ring 1: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # Add: (2,0), (2,-1), (1,-2), (0,-2) -- extends down-right
    topoA = base7 + [(2,0), (2,-1), (1,-2), (0,-2)]
    topologies.append(topoA)
    
    # Topology B: 7-hex + 4 extending in another direction
    topoB = base7 + [(2,0), (2,1), (1,2), (0,2)]
    topologies.append(topoB)
    
    # Topology C: 7-hex + 4 in a line extending right
    topoC = base7 + [(2,0), (3,0), (4,0), (5,0)]
    topologies.append(topoC)
    
    # Topology D: 7-hex + 3 right + 1 up
    topoD = base7 + [(2,0), (2,1), (3,0), (3,1)]
    topologies.append(topoD)
    
    # Topology E: 7-hex + 4 in a diagonal
    topoE = base7 + [(2,0), (2,-1), (3,-1), (3,-2)]
    topologies.append(topoE)
    
    # Topology F: 7-hex + 4 forming a 2x2 block
    topoF = base7 + [(2,0), (2,1), (3,0), (3,1)]
    # Same as D, skip
    # Let me try: extend in two directions
    topoF = base7 + [(2,0), (2,-1), (-2,0), (-2,1)]
    topologies.append(topoF)
    
    # Topology G: 7-hex + 4 in a ring segment
    topoG = base7 + [(2,0), (2,-1), (1,-2), (0,-2)]
    # Same as A, skip
    
    # Topology H: asymmetric
    topoH = base7 + [(2,0), (3,0), (2,1), (3,1)]
    topologies.append(topoH)
    
    # Topology I: 7-hex + 4 forming a T
    topoI = base7 + [(2,0), (2,1), (2,-1), (3,0)]
    topologies.append(topoI)
    
    # Topology J: 7-hex + 4 in a cross
    topoJ = base7 + [(2,0), (-2,0), (0,2), (0,-2)]
    topologies.append(topoJ)
    
    return topologies

def axial_to_xy(q, r):
    """Convert axial coordinates to Cartesian for unit hexagons (side 1).
    Distance between adjacent centers = sqrt(3)."""
    x = math.sqrt(3.0) * (q + r / 2.0)
    y = 1.5 * r
    return (x, y)

def compute_best_for_topology(topology):
    """Given a topology (list of axial coords), compute the minimal enclosing
    hexagon side length and the inner hexagon positions."""
    # Convert to Cartesian
    points = [axial_to_xy(q, r) for (q, r) in topology]
    
    # Compute centroid
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    
    # Translate so centroid is at origin
    points_centered = [(p[0] - cx, p[1] - cy) for p in points]
    
    # Compute enclosing hexagon side (axis-aligned)
    side = enclosing_hex_side_axis_aligned(points_centered)
    
    # The inner hexagons are at the centered positions, all axis-aligned (angle 0)
    inner = [[p[0], p[1], 0.0] for p in points_centered]
    
    # Outer hexagon: center at origin, side length = side, angle 0
    # But we need to verify all inner hexagons fit inside
    # The enclosing computation already ensures the centers' support is within
    # But we need the hexagons themselves (with extent) to be inside.
    # A unit hexagon at position (px, py) with axis-alignment has extent:
    # its vertices are at distance 1 from (px, py) in the 6 vertex directions.
    # So the effective "point" for enclosing is actually the set of all vertices.
    
    # Let me recompute: for each inner hexagon, compute its 6 vertices,
    # then find the enclosing hexagon of all 66 vertices.
    all_vertices = []
    for (px, py) in points_centered:
        verts = hex_vertices(px, py, 1.0, 0.0)
        all_vertices.extend(verts)
    
    side2 = enclosing_hex_side_axis_aligned(all_vertices)
    
    return side2, inner, (0.0, 0.0)

def verify_packing(inner, center, side, angle, gap=0.001):
    """Verify all inner hexagons are inside the outer hexagon and disjoint."""
    cx, cy = center
    outer_verts = hex_vertices(cx, cy, side, angle)
    
    # Check each inner hexagon is inside outer
    for (px, py, ang) in inner:
        v = hex_vertices(px, py, 1.0, ang)
        for (vx, vy) in v:
            if not point_in_polygon(vx, vy, outer_verts):
                return False
    
    # Check disjointness
    n = len(inner)
    for i in range(n):
        vi = hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
        for j in range(i+1, n):
            vj = hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
            if not hexagons_disjoint(vi, vj, gap):
                return False
    
    return True

def construct():
    """Main entry point. Returns (inner, center, side, angle)."""
    start_time = time.time()
    
    topologies = generate_honeycomb_topologies()
    
    best_side = float('inf')
    best_result = None
    
    for topo in topologies:
        if time.time() > DEADLINE:
            break
        
        side, inner, center = compute_best_for_topology(topo)
        
        # Add a small gap to ensure disjointness is recognized
        # The gap means we scale up the outer hexagon slightly
        # Actually, the honeycomb already has gap between hexagons (they touch at edges)
        # To make them strictly disjoint, we need a tiny separation.
        # Scale the inner positions slightly away from center.
        # Or just use a slightly larger outer hexagon.
        
        # For safety, add a small margin to the side
        side_with_gap = side * 1.001
        
        # Verify
        if verify_packing(inner, center, side_with_gap, 0.0, gap=0.0005):
            if side_with_gap < best_side:
                best_side = side_with_gap
                best_result = (inner, center, side_with_gap, 0.0)
    
    if best_result is None:
        # Fallback: use the first topology with larger gap
        side, inner, center = compute_best_for_topology(topologies[0])
        best_result = (inner, center, side * 1.005, 0.0)
    
    return best_result

# Test at module level (optional, for debugging)
if __name__ == "__main__":
    result = construct()
    inner, center, side, angle = result
    print(f"Best side: {side:.6f}")
    print(f"Inner hexagons: {len(inner)}")
    for i, (x, y, a) in enumerate(inner):
        print(f"  Hex {i}: ({x:.4f}, {y:.4f}, {a}°)")

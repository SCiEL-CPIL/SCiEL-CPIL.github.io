"""
Packing 12 unit regular hexagons inside a regular hexagon, minimizing outer side length.

Strategy:
- Model each inner hexagon by (x, y, angle).
- Non-overlap: for two unit hexagons with centers c1, c2 and angles a1, a2,
  they are disjoint iff |c1 - c2| >= minimum separation distance.
  The minimum separation between two unit hexagons with relative angle delta is
  the inradius of the Minkowski sum, which equals 2 * inradius = 2 * sqrt(3)/2 = sqrt(3)
  when aligned, but changes with relative rotation.
  
  Actually, for two regular hexagons of side 1, the minimum distance between centers
  for them to be non-overlapping depends on their relative orientation.
  
  A regular hexagon of side 1 has inradius r = sqrt(3)/2.
  The Minkowski sum of two regular hexagons (one rotated by delta) is a convex polygon.
  The minimum center-to-center distance for non-overlap is the minimum distance from origin
  to the boundary of the Minkowski sum of one hexagon and the reflection of the other.
  
  For simplicity and efficiency, I'll use a conservative but tight bound:
  The minimum distance is at least sqrt(3) (when aligned) and at most 2 (when one is a point).
  Actually, the maximum minimum-separation is 2 (if we treat hexagons as disks of radius 1, but they're smaller).
  
  Let me compute this properly. A unit regular hexagon has vertices at distance 1 from center.
  The inradius is sqrt(3)/2 ≈ 0.866.
  
  For two identical convex bodies A and B, they are disjoint iff |c1-c2| >= w(A, B)
  where w is the minimum width in the direction of c1-c2... no, that's not quite right.
  
  The correct condition: A and B are disjoint iff the distance between c1 and c2 is at least
  the minimum, over all directions u, of (support_A(u) + support_B(-u)) ... no.
  
  Actually: A + (-B) is the Minkowski sum. A and B are disjoint (with interiors not touching)
  iff c1 - c2 is NOT in the interior of A + (-B). So they are disjoint iff
  c1 - c2 is outside or on the boundary of A + (-B).
  
  So the minimum |c1-c2| for non-overlap is the minimum distance from origin to the boundary
  of A + (-B), which is the minimum over unit vectors u of the support function of A + (-B) in direction u.
  
  For a regular hexagon with side 1, the support function in direction at angle phi is:
  h(phi) = r * max over vertices of cos(phi - vertex_angle)
  where r = 1 (circumradius) and vertices are at angles 0, 60, 120, 180, 240, 300 degrees.
  
  Actually for a regular hexagon centered at origin with circumradius 1:
  h(phi) = max_k cos(phi - k*pi/3) for k=0..5
  
  This equals cos(phi mod pi/3 - pi/6) * ... let me just compute numerically.
  
  For the Minkowski sum A + (-B) where A and B are both unit hexagons but B is rotated by delta:
  The support function is h_A(phi) + h_B(-phi) = h_A(phi) + h_A(delta - phi) (by symmetry of hexagon).
  
  This is getting complex. Let me use a simpler approach:
  
  Precompute the minimum separation for a grid of relative angles, then interpolate.
  Or better: use the fact that for regular hexagons, the minimum center distance for non-overlap
  as a function of relative angle delta can be computed.
  
  Actually, let me use a different approach entirely. I'll use scipy.optimize with a penalty-based
  objective that uses actual geometric overlap detection via linear programming or shapely-like
  checks. But we don't have shapely.
  
  Alternative: Use the fact that two convex polygons are disjoint iff no edge of one intersects
  any edge of the other AND neither contains the other. For hexagons, we can check all 6x6 edge
  pairs plus containment.
  
  But for optimization, I need a smooth objective. Let me use a soft penalty.
  
  Let me try a different structural approach:
  
  1. Use a hexagonal lattice arrangement as the starting point.
  2. Use scipy.optimize.minimize with L-BFGS-B on a penalty objective.
  3. The penalty is based on the minimum distance between hexagon boundaries.
  
  For computing minimum distance between two convex polygons, I can use the separating axis theorem
  to get the minimum penetration depth, which gives a measure of overlap.
  
  Let me implement this:
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return vertices of a regular hexagon."""
    angle = np.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = angle + k * np.pi / 3
        verts.append([cx + side * np.cos(theta), cy + side * np.sin(theta)])
    return np.array(verts)

def point_in_convex_polygon(p, poly):
    """Check if point p is inside convex polygon poly using cross products."""
    n = len(poly)
    sign = None
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if abs(cross) < 1e-12:
            continue
        s = 1 if cross > 0 else -1
        if sign is None:
            sign = s
        elif s != sign:
            return False
    return True

def segments_intersect(p1, p2, p3, p4):
    """Check if line segment p1-p2 intersects p3-p4."""
    def orient(a, b, c):
        val = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(val) < 1e-12:
            return 0
        return 1 if val > 0 else -1
    
    o1 = orient(p1, p2, p3)
    o2 = orient(p1, p2, p4)
    o3 = orient(p3, p4, p1)
    o4 = orient(p3, p4, p2)
    
    if o1 != o2 and o3 != o4:
        return True
    
    if o1 == 0 and min(p1[0], p2[0]) <= p3[0] <= max(p1[0], p2[0]) and min(p1[1], p2[1]) <= p3[1] <= max(p1[1], p2[1]):
        return True
    if o2 == 0 and min(p1[0], p2[0]) <= p4[0] <= max(p1[0], p2[0]) and min(p1[1], p2[1]) <= p4[1] <= max(p1[1], p2[1]):
        return True
    if o3 == 0 and min(p3[0], p4[0]) <= p1[0] <= max(p3[0], p4[0]) and min(p3[1], p4[1]) <= p1[1] <= max(p3[1], p4[1]):
        return True
    if o4 == 0 and min(p3[0], p4[0]) <= p2[0] <= max(p3[0], p4[0]) and min(p3[1], p4[1]) <= p2[1] <= max(p3[1], p4[1]):
        return True
    
    return False

def hexagons_overlap(v1, v2):
    """Check if two convex hexagons overlap."""
    # Check if any vertex of v1 is inside v2 or vice versa
    for i in range(6):
        if point_in_convex_polygon(v1[i], v2):
            return True
    for i in range(6):
        if point_in_convex_polygon(v2[i], v1):
            return True
    
    # Check edge intersections
    for i in range(6):
        for j in range(6):
            if segments_intersect(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6]):
                return True
    
    return False

def min_separation(v1, v2):
    """Compute minimum distance between two convex polygons. Positive if disjoint, negative if overlapping."""
    # First check if they overlap
    if hexagons_overlap(v1, v2):
        # Estimate penetration depth using support functions
        # Use the minimum distance from each vertex of one to the other polygon
        min_pen = float('inf')
        for i in range(6):
            # Distance from v1[i] to polygon v2
            d = _point_to_convex_polygon_dist(v1[i], v2)
            min_pen = min(min_pen, d)
        for i in range(6):
            d = _point_to_convex_polygon_dist(v2[i], v1)
            min_pen = min(min_pen, d)
        return -min_pen  # negative means overlap
    
    # Disjoint: compute minimum distance between edges
    min_d = float('inf')
    for i in range(6):
        for j in range(6):
            d = _seg_seg_dist(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6])
            min_d = min(min_d, d)
    return min_d

def _point_to_convex_polygon_dist(p, poly):
    """Minimum distance from point p to convex polygon poly (0 if inside)."""
    if point_in_convex_polygon(p, poly):
        return 0.0
    min_d = float('inf')
    for i in range(len(poly)):
        d = _point_to_seg_dist(p, poly[i], poly[(i+1)%len(poly)])
        min_d = min(min_d, d)
    return min_d

def _point_to_seg_dist(p, a, b):
    """Distance from point p to segment a-b."""
    ab = b - a
    ap = p - a
    t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-15)
    t = max(0, min(1, t))
    closest = a + t * ab
    return np.linalg.norm(p - closest)

def _seg_seg_dist(p1, p2, p3, p4):
    """Minimum distance between segments p1-p2 and p3-p4."""
    if segments_intersect(p1, p2, p3, p4):
        return 0.0
    d1 = _point_to_seg_dist(p1, p3, p4)
    d2 = _point_to_seg_dist(p2, p3, p4)
    d3 = _point_to_seg_dist(p3, p1, p2)
    d4 = _point_to_seg_dist(p4, p1, p2)
    return min(d1, d2, d3, d4)

def outer_hex_contains(inner_hex, outer_center, outer_side, outer_angle):
    """Check if inner hexagon is contained in outer hexagon."""
    outer_verts = hex_vertices(outer_center[0], outer_center[1], outer_angle, outer_side)
    for v in inner_hex:
        if not point_in_convex_polygon(v, outer_verts):
            return False
    return True

def objective(params, outer_center, outer_side, outer_angle):
    """
    Objective: minimize outer_side subject to constraints.
    params: flattened array of [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11]
    Returns penalty-based objective.
    """
    # Unpack
    centers = []
    angles = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        centers.append((x, y))
        angles.append(a)
    
    penalty = 0.0
    
    # Compute all inner hexagons
    hexes = []
    for i in range(N):
        hexes.append(hex_vertices(centers[i][0], centers[i][1], angles[i]))
    
    # Non-overlap constraint between inner hexagons
    for i in range(N):
        for j in range(i+1, N):
            sep = min_separation(hexes[i], hexes[j])
            if sep < 0:
                penalty += sep * sep  # quadratic penalty for overlap
    
    # Containment constraint: each inner hex must be inside outer hex
    outer_verts = hex_vertices(outer_center[0], outer_center[1], outer_angle, outer_side)
    for i in range(N):
        for v in hexes[i]:
            if not point_in_convex_polygon(v, outer_verts):
                # Estimate how far outside
                d = _point_to_convex_polygon_dist(v, outer_verts)
                penalty += d * d
    
    return penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial configuration: hexagonal lattice
    # For 12 hexagons, a good initial arrangement is a 3-4-3 or similar pattern
    # Let's try a hexagonal close-packed arrangement
    
    best_score = float('inf')
    best_params = None
    
    # Try multiple initial configurations
    initial_configs = []
    
    # Config 1: Hexagonal lattice
    # Place hexagons in a hexagonal pattern
    r = np.sqrt(3)  # distance between centers in hexagonal packing
    config1 = []
    # 3 rows: 4, 4, 4 or 3, 4, 5 etc.
    # Let's do a 3x4 grid with hexagonal offset
    for row in range(3):
        for col in range(4):
            x = col * 2.0 + (row % 2) * 1.0
            y = row * r
            config1.append([x, y, 0.0])
    # Center it
    cx = np.mean([c[0] for c in config1])
    cy = np.mean([c[1] for c in config1])
    config1 = [[c[0]-cx, c[1]-cy, c[2]] for c in config1]
    initial_configs.append(config1)
    
    # Config 2: Two rows of 6
    config2 = []
    for row in range(2):
        for col in range(6):
            x = col * 2.0
            y = row * r
            config2.append([x, y, 0.0])
    cx = np.mean([c[0] for c in config2])
    cy = np.mean([c[1] for c in config2])
    config2 = [[c[0]-cx, c[1]-cy, c[2]] for c in config2]
    initial_configs.append(config2)
    
    # Config 3: Ring + center
    config3 = []
    config3.append([0.0, 0.0, 0.0])
    for k in range(6):
        angle = k * np.pi / 3
        config3.append([2.0*np.cos(angle), 2.0*np.sin(angle), 0.0])
    for k in range(5):
        angle = k * np.pi / 3 + np.pi/6
        config3.append([4.0*np.cos(angle), 4.0*np.sin(angle), 0.0])
    initial_configs.append(config3)
    
    # Config 4: Random
    np.random.seed(42)
    config4 = []
    for i in range(N):
        config4.append([np.random.uniform(-3, 3), np.random.uniform(-3, 3), np.random.uniform(0, 60)])
    initial_configs.append(config4)
    
    # For each initial config, try to optimize
    for init_config in initial_configs:
        if time.time() > deadline:
            break
        
        # Flatten initial params
        x0 = []
        for c in init_config:
            x0.extend([c[0], c[1], c[2]])
        x0 = np.array(x0)
        
        # Estimate required outer side length
        # Find bounding box of initial config
        max_extent = 0
        for c in init_config:
            # Hexagon extends to radius 1 from center
            max_extent = max(max_extent, np.sqrt(c[0]**2 + c[1]**2) + 1.0)
        initial_outer_side = max_extent * 2.0 + 1.0
        
        # Optimize: minimize outer_side with penalty for constraint violations
        # Use a simple iterative approach: start with large outer, shrink it
        
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        
        # Binary search on outer side length
        lo = max_extent * 1.5
        hi = max_extent * 3.0 + 5.0
        
        # First, find a feasible solution with large outer
        outer_side = hi
        
        # Optimize positions to be feasible
        for iteration in range(5):
            if time.time() > deadline:
                break
            
            res = minimize(
                objective,
                x0,
                args=(outer_center, outer_side, outer_angle),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            x0 = res.x
            
            # Check if feasible
            penalty = objective(x0, outer_center, outer_side, outer_angle)
            if penalty < 1e-6:
                break
        
        # Now binary search for minimum outer side
        best_local = hi
        for _ in range(15):
            if time.time() > deadline:
                break
            
            mid = (lo + hi) / 2
            # Try to find feasible solution with outer_side = mid
            feasible = False
            for restart in range(3):
                if time.time() > deadline:
                    break
                
                # Perturb x0 slightly
                x_try = x0 + np.random.normal(0, 0.1, size=x0.shape)
                
                res = minimize(
                    objective,
                    x_try,
                    args=(outer_center, mid, outer_angle),
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                penalty = objective(res.x, outer_center, mid, outer_angle)
                if penalty < 1e-5:
                    feasible = True
                    x0 = res.x
                    break
            
            if feasible:
                hi = mid
                best_local = mid
            else:
                lo = mid
        
        if best_local < best_score:
            best_score = best_local
            best_params = x0
    
    # If no valid solution found, return a safe default
    if best_params is None or best_score == float('inf'):
        # Fallback: place hexagons in a line with large outer
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Extract inner hexagons
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        a = best_params[3*i+2]
        inner.append([float(x), float(y), float(a)])
    
    return inner, (0.0, 0.0), float(best_score), 0.0

# EVOLVE-BLOCK-END

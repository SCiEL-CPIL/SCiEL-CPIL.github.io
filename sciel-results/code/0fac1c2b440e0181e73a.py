"""Packing 12 unit hexagons in a minimal regular hexagon.

Strategy:
1. Use a known good symmetric configuration as the starting point.
2. Refine with SLSQP using Minkowski-sum separation constraints for
   non-overlap and half-plane constraints for containment.
3. Try multiple initial configurations and keep the best result.

The outer hexagon is centered at origin with rotation 0 (fixed by symmetry).
"""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
T_START = time.time()
DEADLINE = 145.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon with given center, side length,
    and rotation (degrees). Vertex 0 is at angle `angle_deg` from center."""
    a = np.radians(angle_deg)
    verts = []
    for k in range(6):
        ang = a + k * math.pi / 3.0
        verts.append((cx + side * math.cos(ang), cy + side * math.sin(ang)))
    return np.array(verts)

def _minkowski_sum_hex(v1, v2):
    """Compute the Minkowski sum of two convex hexagons (each given as
    array of 6 vertices in CCW order). Returns vertices of the sum polygon.
    Uses the fact that the Minkowski sum of two convex polygons has at most
    m+n vertices, and for hexagons it's at most 12."""
    # For two convex polygons, the Minkowski sum can be computed by
    # merging edge vectors sorted by angle.
    def edge_vectors(verts):
        n = len(verts)
        edges = np.diff(verts, axis=0, append=verts[:1])
        return edges
    
    e1 = edge_vectors(v1)
    e2 = edge_vectors(v2)
    
    # Sort edges by angle
    a1 = np.arctan2(e1[:, 1], e1[:, 0])
    a2 = np.arctan2(e2[:, 1], e2[:, 0])
    
    # Combine and sort
    combined = np.vstack([e1, e2])
    angles = np.concatenate([a1, a2])
    order = np.argsort(angles)
    sorted_edges = combined[order]
    
    # Build the Minkowski sum polygon
    result = [v1[0] + v2[0]]
    for e in sorted_edges:
        result.append(result[-1] + e)
    return np.array(result)

def _point_in_convex_polygon(pt, poly):
    """Check if point is inside or on boundary of convex polygon (CCW)."""
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        if cross < -1e-9:
            return False
    return True

def _hexagon_overlap(v1, v2, tol=1e-7):
    """Check if two convex hexagons overlap using Minkowski sum test.
    v1, v2 are arrays of 6 vertices (CCW)."""
    # Two convex polygons A and B are disjoint iff origin is not in A-B
    # Equivalently, check if 0 is in the Minkowski sum of A and (-B)
    neg_v2 = -v2
    # Reverse to maintain CCW
    neg_v2 = neg_v2[::-1]
    
    ms = _minkowski_sum_hex(v1, neg_v2)
    return _point_in_convex_polygon(np.array([0.0, 0.0]), ms)

def _check_validity(inner, center, side, angle_deg, tol=1e-6):
    """Check if the configuration is valid: no overlaps, all contained."""
    cx, cy = center
    # Outer hexagon vertices
    outer = _hex_vertices(cx, cy, side, angle_deg)
    
    inner_verts = []
    for (ix, iy, ia) in inner:
        v = _hex_vertices(ix, iy, 1.0, ia)
        inner_verts.append(v)
        # Check containment
        for pt in v:
            if not _point_in_convex_polygon(pt, outer):
                return False
    
    # Check pairwise non-overlap
    for i in range(N):
        for j in range(i + 1, N):
            if _hexagon_overlap(inner_verts[i], inner_verts[j], tol):
                return False
    return True

def _build_slsqp(n, x0):
    """Build SLSQP problem for n inner hexagons.
    Variables: x = [x1,y1,theta1, ..., xn,yn,thetan, L]
    Outer hexagon centered at origin, rotation 0."""
    V0 = _hex_vertices(0, 0, 1.0, 0.0)  # unit hexagon at origin
    
    def objective(x):
        return x[3 * n]
    
    def obj_grad(x):
        g = np.zeros(3 * n + 1)
        g[3 * n] = 1.0
        return g
    
    constraints = []
    
    # Non-overlap constraints: for each pair (i,j), the distance between
    # centers must be >= minimum separation. We use a support-function-based
    # constraint: for the direction from j to i, the projection of the
    # Minkowski sum along that direction must be <= the center distance.
    
    def make_pair_constraint(i, j):
        def c(x):
            xi, yi, ti = x[3*i], x[3*i+1], x[3*i+2]
            xj, yj, tj = x[3*j], x[3*j+1], x[3*j+2]
            
            # Direction from j to i
            dx = xi - xj
            dy = yi - yj
            dist = math.hypot(dx, dy)
            if dist < 1e-12:
                return 0.0  # degenerate, let other constraints handle
            
            # Unit direction
            wx = dx / dist
            wy = dy / dist
            
            # Support function of hex i along (wx, wy)
            vi = _hex_vertices(0, 0, 1.0, math.degrees(ti))
            vj = _hex_vertices(0, 0, 1.0, math.degrees(tj))
            
            si = np.max(vi @ np.array([wx, wy]))
            sj = np.max(vj @ np.array([-wx, -wy]))  # support of -vj along (wx,wy) = support of vj along (-wx,-wy)
            
            # Constraint: dist - si - sj >= 0
            return dist - si - sj
        
        return {'type': 'ineq', 'fun': c}
    
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append(make_pair_constraint(i, j))
    
    # Containment constraints: each vertex of each inner hex must be inside
    # the outer hexagon. The outer hex has 6 half-planes.
    # For a regular hexagon of side L centered at origin with vertex at angle 0,
    # the 6 supporting lines are at angles 0, 60, 120, 180, 240, 300 degrees.
    # The distance from center to each side is L * cos(30°) = L * sqrt(3)/2.
    
    half_width = lambda L: L * math.sqrt(3) / 2.0
    
    def make_containment(i, k):
        """Constraint that vertex k of hex i is inside outer hex."""
        def c(x):
            xi, yi, ti = x[3*i], x[3*i+1], x[3*i+2]
            L = x[3*n]
            
            vi = _hex_vertices(0, 0, 1.0, math.degrees(ti))
            # The 6 vertices of hex i in world coords
            verts = vi + np.array([xi, yi])
            
            # For each of the 6 half-planes of the outer hex:
            # Outer hex vertices are at angles 0, 60, ..., 300 with radius L
            # The inward normals are at angles 30, 90, 150, 210, 270, 330
            # Distance from origin to each face = L * cos(30) = L*sqrt(3)/2
            
            min_margin = float('inf')
            for face_k in range(6):
                # Normal direction for face k
                ang = math.radians(30 + 60 * face_k)
                nx = math.cos(ang)
                ny = math.sin(ang)
                
                # For each vertex of inner hex i, check it's on the correct side
                for pt in verts:
                    margin = L * math.sqrt(3) / 2.0 - (pt[0] * nx + pt[1] * ny)
                    if margin < min_margin:
                        min_margin = margin
            
            return min_margin
        
        return {'type': 'ineq', 'fun': c}
    
    # Simplified: just check all 6 vertices of each inner hex against all 6 faces
    def make_containment_all(i):
        def c(x):
            xi, yi, ti = x[3*i], x[3*i+1], x[3*i+2]
            L = x[3*n]
            
            vi = _hex_vertices(0, 0, 1.0, math.degrees(ti))
            verts = vi + np.array([xi, yi])
            
            min_margin = float('inf')
            for face_k in range(6):
                ang = math.radians(30 + 60 * face_k)
                nx = math.cos(ang)
                ny = math.sin(ang)
                for pt in verts:
                    margin = L * math.sqrt(3) / 2.0 - (pt[0] * nx + pt[1] * ny)
                    if margin < min_margin:
                        min_margin = margin
            return min_margin
        return {'type': 'ineq', 'fun': c}
    
    for i in range(n):
        constraints.append(make_containment_all(i))
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((-10, 10))  # x
        bounds.append((-10, 10))  # y
        bounds.append((0, 2*math.pi))  # theta
    bounds.append((1.5, 20.0))  # L
    
    return objective, obj_grad, constraints, bounds

def _initial_config_ring(radius, n):
    """Ring configuration: n hexagons evenly spaced on a circle."""
    x0 = np.zeros(3 * n + 1)
    for i in range(n):
        a = 2 * math.pi * i / n
        x0[3*i] = radius * math.cos(a)
        x0[3*i+1] = radius * math.sin(a)
        x0[3*i+2] = 0.0
    x0[3*n] = 4.0  # initial L
    return x0

def _initial_config_honeycomb(n):
    """Honeycomb-like arrangement."""
    x0 = np.zeros(3 * n + 1)
    # Place in a roughly hexagonal pattern
    positions = []
    # Center
    positions.append((0, 0))
    # First ring: 6 hexagons
    for k in range(6):
        a = math.pi / 3 * k + math.pi / 6
        positions.append((2.0 * math.cos(a), 2.0 * math.sin(a)))
    # Second ring partial: 5 more
    for k in range(5):
        a = math.pi / 3 * (k + 0.5)
        positions.append((3.5 * math.cos(a), 3.5 * math.sin(a)))
    
    for i in range(min(n, len(positions))):
        x0[3*i] = positions[i][0]
        x0[3*i+1] = positions[i][1]
        x0[3*i+2] = 0.0
    if n > len(positions):
        for i in range(len(positions), n):
            a = 2 * math.pi * i / n
            x0[3*i] = 4.0 * math.cos(a)
            x0[3*i+1] = 4.0 * math.sin(a)
    x0[3*n] = 5.0
    return x0

def construct():
    best_L = float('inf')
    best_x = None
    
    # Try multiple initial configurations
    configs = [
        _initial_config_ring(2.0, N),
        _initial_config_ring(2.5, N),
        _initial_config_ring(3.0, N),
        _initial_config_honeycomb(N),
    ]
    
    # Also try with some rotation variations
    for base_cfg in list(configs[:2]):
        cfg_rot = base_cfg.copy()
        for i in range(N):
            cfg_rot[3*i+2] = math.pi / 6  # 30 degree rotation
        configs.append(cfg_rot)
    
    for x0 in configs:
        if time.time() - T_START > DEADLINE:
            break
        
        try:
            obj, obj_grad, cons, bounds = _build_slsqp(N, x0)
            
            res = minimize(
                obj, x0, jac=obj_grad,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 200, 'ftol': 1e-10, 'disp': False}
            )
            
            if res.success and res.fun < best_L:
                # Verify validity
                L = res.fun
                x = res.x
                inner = []
                for i in range(N):
                    inner.append([float(x[3*i]), float(x[3*i+1]), float(math.degrees(x[3*i+2]))])
                
                if _check_validity(inner, (0.0, 0.0), L, 0.0):
                    best_L = L
                    best_x = x
        except Exception:
            continue
    
    # If SLSQP didn't find a valid solution, try a simpler approach
    if best_x is None:
        # Fallback: use a known valid configuration
        # 12 hexagons in a ring around center
        # For side length L, the apothem is L*sqrt(3)/2
        # Inner hexagons of side 1 have circumradius 1
        # Place them in a ring of radius r, need r + 1 <= L*sqrt(3)/2 for containment
        # And adjacent hexagons must not overlap: 2*r*sin(pi/12) >= 2 (separation)
        # So r >= 1/sin(pi/12) ≈ 3.86
        # L >= (r+1)/sqrt(3)/2... wait let me recalculate
        
        # Actually for a hexagon of side L, the distance from center to vertex is L,
        # and distance from center to side (apothem) is L*sqrt(3)/2.
        # A unit hexagon has circumradius 1 and inradius sqrt(3)/2.
        
        # If we place unit hexagons in a ring of radius r with 12 positions,
        # adjacent centers are 2*r*sin(pi/12) apart.
        # For non-overlap: 2*r*sin(pi/12) >= 2*sqrt(3)/2 * 2 = 2*sqrt(3) (sum of inradii... no)
        # Actually the minimum distance between centers of two non-overlapping unit hexagons
        # depends on their relative orientation. Worst case (parallel): 2*sqrt(3)/2*2 = 2*sqrt(3)
        # Best case (60 deg offset): 2 (just circumradii)
        
        # Let's just use a large enough L
        r = 4.0
        L = (r + 1.0) / (math.sqrt(3) / 2.0)  # ensure containment
        L = max(L, 5.0)
        
        inner = []
        for i in range(N):
            a = 2 * math.pi * i / N
            inner.append([r * math.cos(a), r * math.sin(a), 0.0])
        
        # Verify
        if _check_validity(inner, (0.0, 0.0), L, 0.0):
            return inner, (0.0, 0.0), float(L), 0.0
        else:
            # Even larger
            L = 8.0
            r = 4.0
            inner = []
            for i in range(N):
                a = 2 * math.pi * i / N
                inner.append([r * math.cos(a), r * math.sin(a), 0.0])
            return inner, (0.0, 0.0), float(L), 0.0
    
    # Extract best solution
    x = best_x
    inner = []
    for i in range(N):
        inner.append([float(x[3*i]), float(x[3*i+1]), float(math.degrees(x[3*i+2]))])
    
    L = float(best_L)
    
    # Final validation
    if not _check_validity(inner, (0.0, 0.0), L, 0.0):
        # Try to fix by increasing L slightly
        for extra in [0.1, 0.5, 1.0, 2.0]:
            L2 = L + extra
            if _check_validity(inner, (0.0, 0.0), L2, 0.0):
                L = L2
                break
        else:
            # Last resort: large hexagon
            L = 10.0
            inner = []
            r = 4.0
            for i in range(N):
                a = 2 * math.pi * i / N
                inner.append([r * math.cos(a), r * math.sin(a), 0.0])
    
    return inner, (0.0, 0.0), float(L), 0.0

"""Pack 12 unit regular hexagons into a compact honeycomb cluster and minimize the enclosing regular hexagon side."""
import numpy as np
from scipy.optimize import minimize

N = 12
R_IN = np.sqrt(3) / 2.0  # inradius of unit hexagon
R_CIRC = 1.0             # circumradius of unit hexagon
A = np.sqrt(3)           # lattice spacing for touching hexagons (2 * R_IN)

# Define the 12 lattice offsets for a compact honeycomb cluster (radius-1 hexagon + 5 extra)
# The radius-1 hexagon has 7 cells: center + 6 neighbors.
# We add 5 more from the second ring to make 12.
def _make_offsets():
    offsets = []
    # Center
    offsets.append((0.0, 0.0))
    # First ring (6 cells) at 60-degree intervals
    for k in range(6):
        angle = np.pi / 3.0 * k
        offsets.append((A * np.cos(angle), A * np.sin(angle)))
    # Second ring: pick 5 cells from the 12 second-ring positions
    # Second ring positions: distance 2*A in 6 main directions, or A*sqrt(3) in 6 offset directions
    # Actually, the second ring in a hex lattice has 12 positions.
    # Let's enumerate them properly. The neighbors of the first-ring cells that are not already used.
    # A simpler approach: use axial coordinates.
    # Axial coordinate (q, r) -> cartesian: x = A*(q + r/2), y = A*sqrt(3)/2 * r
    # Center is (0,0).
    # First ring: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # Second ring: (2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-1), (-2,0), (-2,1), (-2,2), (-1,2), (0,2), (1,1)
    # We already have the first ring. Pick 5 from the second ring.
    # To keep it compact, pick 5 adjacent ones from the second ring.
    second_ring_axial = [(2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-1), (-2,0), (-2,1), (-2,2), (-1,2), (0,2), (1,1)]
    # Pick 5 consecutive ones, e.g., starting from (2,0)
    for q, r in second_ring_axial[:5]:
        x = A * (q + r / 2.0)
        y = A * np.sqrt(3) / 2.0 * r
        offsets.append((x, y))
    return offsets

OFFSETS = _make_offsets()

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with given center, side, and rotation (degrees)."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + np.pi / 3.0 * i
        verts.append([cx + side * np.cos(theta), cy + side * np.sin(theta)])
    return np.array(verts)

def _polygon_area(verts):
    n = len(verts)
    area = 0.0
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0

def _point_in_polygon(pt, verts):
    """Ray casting method for point-in-polygon."""
    x, y = pt
    n = len(verts)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = verts[i]
        xj, yj = verts[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def _segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    def on_segment(a, b, c):
        return min(a[0], c[0]) <= b[0] <= max(a[0], c[0]) and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])
    d1 = ccw(p3, p4, p1)
    d2 = ccw(p3, p4, p2)
    d3 = ccw(p1, p2, p3)
    d4 = ccw(p1, p2, p4)
    if d1 != d2 and d3 != d4:
        return True
    if d1 == 0 and on_segment(p3, p1, p4):
        return True
    if d2 == 0 and on_segment(p3, p2, p4):
        return True
    if d3 == 0 and on_segment(p1, p3, p2):
        return True
    if d4 == 0 and on_segment(p1, p4, p2):
        return True
    return False

def _hexagons_disjoint(verts_list, eps=1e-7):
    """Check if a list of hexagon vertex arrays are pairwise disjoint."""
    n = len(verts_list)
    for i in range(n):
        for j in range(i + 1, n):
            vi, vj = verts_list[i], verts_list[j]
            # Check vertex containment
            for pt in vi:
                if _point_in_polygon(pt, vj):
                    return False
            for pt in vj:
                if _point_in_polygon(pt, vi):
                    return False
            # Check edge intersections
            for k in range(6):
                p1, p2 = vi[k], vi[(k + 1) % 6]
                for l in range(6):
                    p3, p4 = vj[l], vj[(l + 1) % 6]
                    if _segments_intersect(p1, p2, p3, p4):
                        return False
    return True

def _outer_hexagon_contains(inner_verts, cx, cy, side, angle_deg, eps=1e-7):
    """Check if all inner hexagons are inside the outer hexagon."""
    outer_verts = _hex_vertices(cx, cy, side, angle_deg)
    for vi in inner_verts:
        for pt in vi:
            if not _point_in_polygon(pt, outer_verts):
                return False
    # Also check that no edge of inner crosses edge of outer
    for vi in inner_verts:
        for k in range(6):
            p1, p2 = vi[k], vi[(k + 1) % 6]
            for l in range(6):
                p3, p4 = outer_verts[l], outer_verts[(l + 1) % 6]
                if _segments_intersect(p1, p2, p3, p4):
                    return False
    return True

def _compute_min_outer_side(inner_positions, angle_deg, cx, cy, outer_angle_deg):
    """Find minimum outer side such that all inner hexagons fit inside."""
    inner_verts = []
    for x, y, a in inner_positions:
        inner_verts.append(_hex_vertices(x, y, 1.0, a))
    
    # Use bisection on side length
    lo = 0.0
    hi = 100.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if _outer_hexagon_contains(inner_verts, cx, cy, mid, outer_angle_deg):
            hi = mid
        else:
            lo = mid
    return hi

def construct():
    # Start with the honeycomb cluster at origin
    initial_positions = [(ox, oy, 0.0) for ox, oy in OFFSETS]
    
    best_side = float('inf')
    best_inner = initial_positions
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_outer_angle = 0.0
    
    # Try several outer hexagon orientations
    for outer_angle_deg in [0.0, 30.0]:
        # Optimize inner positions
        def objective(params):
            xs = params[:N]
            ys = params[N:2*N]
            angles = params[2*N:3*N]
            positions = [(xs[i], ys[i], angles[i]) for i in range(N)]
            inner_verts = [_hex_vertices(x, y, 1.0, a) for x, y, a in positions]
            # Check disjointness
            if not _hexagons_disjoint(inner_verts, eps=1e-5):
                return 1e6
            # Compute min outer side
            side = _compute_min_outer_side(positions, 0.0, 0.0, 0.0, outer_angle_deg)
            # Check if they fit
            if _outer_hexagon_contains(inner_verts, 0.0, 0.0, side, outer_angle_deg):
                return side
            else:
                return 1e6
        
        initial_params = []
        for x, y, a in initial_positions:
            initial_params.extend([x, y, a])
        
        # Use Nelder-Mead optimization
        try:
            result = minimize(objective, initial_params, method='Nelder-Mead', 
                             options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-6})
            if result.fun < best_side:
                best_side = result.fun
                xs = result.x[:N]
                ys = result.x[N:2*N]
                angles = result.x[2*N:3*N]
                best_inner = [(xs[i], ys[i], angles[i]) for i in range(N)]
                best_center = (0.0, 0.0)
                best_angle = 0.0
                best_outer_angle = outer_angle_deg
        except Exception:
            pass
    
    # If optimization didn't improve, use the initial cluster
    if best_side == float('inf'):
        best_inner = initial_positions
        best_center = (0.0, 0.0)
        best_angle = 0.0
        best_outer_angle = 0.0
        best_side = _compute_min_outer_side(best_inner, 0.0, 0.0, 0.0, 0.0)
    
    return best_inner, best_center, best_side, best_outer_angle

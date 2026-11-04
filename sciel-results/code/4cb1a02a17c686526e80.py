"""Pack 12 unit regular hexagons in a regular hexagon, minimizing outer side length."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12
DEADLINE_MARGIN = 10.0  # seconds to leave before the 150s limit

def _hex_vertices(cx, cy, side, angle_deg):
    """Return the 6 vertices of a regular hexagon (flat-top convention)."""
    a = np.deg2rad(angle_deg)
    # For a regular hexagon with side length s, the vertices are at distance s from center
    # at angles 0, 60, 120, 180, 240, 300 degrees (flat-top: first vertex at angle 0)
    angles = a + np.deg2rad(np.array([0, 60, 120, 180, 240, 300]))
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _outer_hex_vertices(cx, cy, side, angle_deg):
    """Return the 6 vertices of the outer regular hexagon."""
    return _hex_vertices(cx, cy, side, angle_deg)

def _segment_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4 (proper intersection)."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-12:
        return False
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / denom
    eps = 1e-9
    return eps < t < 1.0 - eps and eps < u < 1.0 - eps

def _hexes_disjoint(v1, v2, tol=1e-7):
    """Check if two hexagons (given as arrays of 6 vertices) are disjoint (no interior overlap)."""
    # Check if any vertex of v1 is strictly inside v2 or vice versa
    # Use winding number / point-in-convex-polygon
    def point_in_convex(pt, poly):
        # poly is 6 vertices of a convex hexagon
        n = len(poly)
        signs = []
        for i in range(n):
            a = poly[i]
            b = poly[(i + 1) % n]
            cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
            if abs(cross) < 1e-10:
                continue
            signs.append(1 if cross > 0 else -1)
        if not signs:
            return True  # degenerate
        return all(s == signs[0] for s in signs)
    
    # Check vertices of v1 inside v2
    for pt in v1:
        if point_in_convex(pt, v2):
            return False
    # Check vertices of v2 inside v1
    for pt in v2:
        if point_in_convex(pt, v1):
            return False
    # Check edge intersections
    for i in range(6):
        for j in range(6):
            if _segment_intersect(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6]):
                return False
    return True

def _inner_in_outer(v_inner, v_outer, tol=1e-7):
    """Check if inner hexagon is fully inside outer hexagon."""
    for pt in v_inner:
        if not point_in_convex(pt, v_outer):
            return False
    return True

def point_in_convex(pt, poly):
    """Check if point is inside or on boundary of convex polygon."""
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        if cross < -1e-9:
            return False
    return True

def _is_valid(inner, outer_center, outer_side, outer_angle):
    """Check full validity: all inner hexagons inside outer, and pairwise disjoint."""
    v_outer = _outer_hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    inner_verts = []
    for (x, y, ang) in inner:
        v = _hex_vertices(x, y, 1.0, ang)
        inner_verts.append(v)
        if not _inner_in_outer(v, v_outer):
            return False
    for i in range(len(inner_verts)):
        for j in range(i + 1, len(inner_verts)):
            if not _hexes_disjoint(inner_verts[i], inner_verts[j]):
                return False
    return True

def _objective(params, outer_center, outer_side, outer_angle):
    """Objective: maximize minimum pairwise distance minus containment penalty.
    We want to MINIMIZE outer_side, so we use a surrogate: penalize violations.
    Actually, let's reformulate: given outer_side, check if packing is feasible.
    For optimization, we'll minimize a violation-based objective."""
    n = N
    # params: [x1, y1, a1, x2, y2, a2, ..., xn, yn, an]
    positions = []
    for i in range(n):
        x, y, a = params[3*i], params[3*i+1], params[3*i+2]
        positions.append((x, y, a))
    
    v_outer = _outer_hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    penalty = 0.0
    
    inner_verts = []
    for i in range(n):
        x, y, a = positions[i]
        v = _hex_vertices(x, y, 1.0, a)
        inner_verts.append(v)
        # Containment penalty: how far outside the outer hexagon
        for pt in v:
            if not point_in_convex(pt, v_outer):
                # Compute distance to outer boundary (approximate)
                min_dist = float('inf')
                for k in range(6):
                    a_pt = v_outer[k]
                    b_pt = v_outer[(k+1)%6]
                    # Distance from pt to segment a_pt-b_pt
                    ax, ay = a_pt
                    bx, by = b_pt
                    px, py = pt
                    dx, dy = bx - ax, by - ay
                    if dx == 0 and dy == 0:
                        d = np.sqrt((px-ax)**2 + (py-ay)**2)
                    else:
                        t = max(0, min(1, ((px-ax)*dx + (py-ay)*dy) / (dx*dx + dy*dy)))
                        cx, cy = ax + t*dx, ay + t*dy
                        d = np.sqrt((px-cx)**2 + (py-cy)**2)
                    min_dist = min(min_dist, d)
                penalty += min_dist * 10.0
    
    # Pairwise separation penalty
    for i in range(n):
        for j in range(i+1, n):
            vi, vj = inner_verts[i], inner_verts[j]
            # Check if they overlap: if any vertex of vi inside vj or vice versa, or edges cross
            overlap = False
            for pt in vi:
                if point_in_convex(pt, vj):
                    overlap = True
                    break
            if not overlap:
                for pt in vj:
                    if point_in_convex(pt, vi):
                        overlap = True
                        break
            if not overlap:
                for k in range(6):
                    for l in range(6):
                        if _segment_intersect(vi[k], vi[(k+1)%6], vj[l], vj[(l+1)%6]):
                            overlap = True
                            break
                    if overlap:
                        break
            if overlap:
                # Compute approximate overlap depth: min distance between any vertex of one to the other
                # Use a simpler penalty: just a fixed large penalty
                penalty += 5.0
    
    return penalty

def construct():
    start_time = time.time()
    deadline = 150.0 - DEADLINE_MARGIN
    
    # Strategy: 
    # 1. Use a grid of initial configurations
    # 2. For each, try to find a valid packing with binary search on outer_side
    # 3. Use SLSQP to minimize violation
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Symmetry: fix outer hexagon center at origin, angle 0
    # The outer hexagon has 6-fold symmetry, so we can fix its orientation
    
    # Initial configurations: various symmetric arrangements
    # For 12 hexagons, a natural arrangement is 2 rings around a center, or a hexagonal lattice
    
    # Hexagonal lattice: centers at distance 2*sqrt(3) apart for non-overlapping unit hexagons
    # Actually, two unit hexagons are non-overlapping if center distance >= 2*sqrt(3) * (1/something)
    # Let me compute: two unit regular hexagons, the minimum distance between centers for no overlap
    # The "radius" (center to vertex) of a unit hexagon is 1. The "apothem" (center to edge) is sqrt(3)/2.
    # Two hexagons with parallel sides: min center distance = 2 * apothem = sqrt(3) ≈ 1.732? No...
    # Actually for two regular hexagons of side 1, the minimum center-to-center distance for them to just touch
    # depends on orientation. For parallel orientations, it's 2 * (sqrt(3)/2) = sqrt(3) ≈ 1.732 if they touch edge-to-edge.
    # Wait, let me think again. A regular hexagon with side length s has:
    # - Circumradius (center to vertex): s
    # - Inradius (center to edge midpoint): s * sqrt(3)/2
    # Two such hexagons with parallel sides, touching edge to edge: center distance = 2 * inradius = s * sqrt(3)
    # So for s=1, min center distance = sqrt(3) ≈ 1.732
    
    # Let's set up initial guesses in a hexagonal pattern
    # 12 hexagons: could be arranged as a hexagonal number pattern
    # Hexagonal numbers: 1, 7, 19, ... 
    # 12 is not a hexagonal number. Let's try a rectangular-ish grid.
    
    # For a good initial guess, let's place them in a 3x4 grid with hexagonal spacing
    # or in a more symmetric pattern.
    
    # Let's try multiple initial configurations and optimize each
    
    configs = []
    
    # Config 1: 4x3 grid
    dx = np.sqrt(3) * 1.01  # slight spacing to avoid initial overlap
    dy = 1.5 * 1.01
    for i in range(3):
        for j in range(4):
            x = (j - 1.5) * dx
            y = (i - 1) * dy
            configs.append([(x, y, 0.0) for _ in [None]])
    
    # Actually, let me build full configs properly
    configs = []
    
    # Config A: 4x3 grid
    dx = np.sqrt(3) * 1.05
    dy = 1.5 * 1.05
    inner_a = []
    for i in range(3):
        for j in range(4):
            x = (j - 1.5) * dx
            y = (i - 1) * dy
            inner_a.append([x, y, 0.0])
    configs.append(inner_a)
    
    # Config B: 2x6 grid
    inner_b = []
    for i in range(2):
        for j in range(6):
            x = (j - 2.5) * dx
            y = (i - 0.5) * dy
            inner_b.append([x, y, 0.0])
    configs.append(inner_b)
    
    # Config C: hexagonal arrangement (center + ring of 6 + ring of 5 more)
    # Actually, let's do a proper hexagonal lattice subset
    # Hex lattice points: (i + j/2) * sqrt(3), j * 1.5
    inner_c = []
    # Take a 3x3 hex lattice block, that's 9, plus 3 more
    for i in range(-1, 2):
        for j in range(-1, 2):
            x = (i + j/2.0) * np.sqrt(3) * 1.05
            y = j * 1.5 * 1.05
            inner_c.append([x, y, 0.0])
    # Add 3 more
    inner_c.append([2*np.sqrt(3)*1.05, 0, 0.0])
    inner_c.append([-2*np.sqrt(3)*1.05, 0, 0.0])
    inner_c.append([0, 3*1.5*1.05, 0.0])
    configs.append(inner_c)
    
    # Config D: Two rings (center + 6 + 5)
    inner_d = [[0, 0, 0.0]]
    for k in range(6):
        ang = k * 60
        r = 2 * np.sqrt(3) * 0.95  # slightly less than touching
        x = r * np.cos(np.deg2rad(ang))
        y = r * np.sin(np.deg2rad(ang))
        inner_d.append([x, y, 0.0])
    # Add 5 more in a second partial ring
    for k in range(5):
        ang = k * 72 + 36
        r = 4 * np.sqrt(3) * 0.85
        x = r * np.cos(np.deg2rad(ang))
        y = r * np.sin(np.deg2rad(ang))
        inner_d.append([x, y, 0.0])
    configs.append(inner_d)
    
    # For each config, try to find minimal outer side via binary search + optimization
    for cfg_idx, inner_init in enumerate(configs):
        if time.time() > deadline:
            break
        
        # Binary search on outer_side
        # Lower bound: must contain all 12 hexagons
        # Upper bound: something large
        lo = 2.0
        hi = 15.0
        
        # First check if hi is feasible
        def is_feasible(side, inner_guess):
            """Check if we can fit with given outer side, starting from inner_guess."""
            center = (0.0, 0.0)
            angle = 0.0
            
            # Try direct check
            if _is_valid(inner_guess, center, side, angle):
                return True
            
            # Try SLSQP optimization to reduce violations
            params = np.array([v for tri in inner_guess for v in tri])
            
            def obj(p):
                return _objective(p, center, side, angle)
            
            try:
                res = minimize(obj, params, method='SLSQP', 
                              options={'maxiter': 200, 'ftol': 1e-8})
                if res.fun < 1e-6:
                    # Check validity
                    new_inner = [[res.x[3*i], res.x[3*i+1], res.x[3*i+2]] for i in range(N)]
                    if _is_valid(new_inner, center, side, angle):
                        return True
            except:
                pass
            
            return False
        
        # Check if hi is feasible
        if not is_feasible(hi, inner_init):
            continue
        
        # Binary search
        for _ in range(8):  # ~8 iterations for reasonable precision
            if time.time() > deadline:
                break
            mid = (lo + hi) / 2
            if is_feasible(mid, inner_init):
                hi = mid
                # Update best
                if hi < best_side:
                    # Get the actual inner configuration
                    center = (0.0, 0.0)
                    angle = 0.0
                    params = np.array([v for tri in inner_init for v in tri])
                    
                    def obj(p):
                        return _objective(p, center, hi, angle)
                    
                    try:
                        res = minimize(obj, params, method='SLSQP', 
                                      options={'maxiter': 500, 'ftol': 1e-10})
                        new_inner = [[res.x[3*i], res.x[3*i+1], res.x[3*i+2]] for i in range(N)]
                        if _is_valid(new_inner, center, hi, angle):
                            best_side = hi
                            best_inner = new_inner
                            best_center = center
                            best_angle = angle
                    except:
                        if _is_valid(inner_init, center, hi, angle):
                            best_side = hi
                            best_inner = [list(t) for t in inner_init]
                            best_center = center
                            best_angle = angle
            else:
                lo = mid
        
        # After binary search, do a final optimization at best_side
        if best_inner is not None and time.time() < deadline:
            center = best_center
            angle = best_angle
            side = best_side
            params = np.array([v for tri in best_inner for v in tri])
            
            def obj(p):
                return _objective(p, center, side, angle)
            
            try:
                res = minimize(obj, params, method='SLSQP', 
                              options={'maxiter': 1000, 'ftol': 1e-12})
                if res.fun < 1e-8:
                    new_inner = [[res.x[3*i], res.x[3*i+1], res.x[3*i+2]] for i in range(N)]
                    if _is_valid(new_inner, center, side, angle):
                        best_inner = new_inner
            except:
                pass
    
    # If no valid solution found, return a fallback
    if best_inner is None:
        # Fallback: return the seed-like solution
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, best_side, best_angle

import numpy as np
import time

N = 12

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    angle = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    verts = []
    for i in range(6):
        theta = np.radians(60 * i) + angle
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append((x, y))
    return np.array(verts)

def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with given side and angle."""
    # Transform to hexagon's local frame
    angle = np.radians(angle_deg)
    dx, dy = px - cx, py - cy
    # Rotate by -angle
    lx = dx * np.cos(angle) + dy * np.sin(angle)
    ly = -dx * np.sin(angle) + dy * np.cos(angle)
    # Check if inside regular hexagon with flat-top orientation (vertices at angles 0,60,...)
    # A regular hexagon with side s has vertices at distance s from center
    # In local coords, the hexagon boundary: for each edge, check half-plane
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees, distance = side
    # Edges are perpendicular to directions 30, 90, 150, 210, 270, 330
    # The hexagon is intersection of 6 half-planes:
    # For each k, the supporting line is at distance s*cos(30) = s*sqrt(3)/2 from center
    # in direction 30+60k degrees
    inr = side * np.sqrt(3) / 2
    for k in range(6):
        theta = np.radians(30 + 60 * k)
        nx, ny = np.cos(theta), np.sin(theta)
        if lx * nx + ly * ny > inr + 1e-9:
            return False
    return True

def _hexagons_overlap(c1, c2, a1, a2, eps=1e-9):
    """Check if two unit hexagons overlap (disjoint means no interior intersection)."""
    # Two convex polygons overlap if one vertex of each is inside the other,
    # or if edges intersect. For hexagons, a simpler check:
    # Compute minimum distance between centers; if > 2*inradius they're definitely apart
    # But with rotation, need more careful check.
    # Use SAT (Separating Axis Theorem) for two convex polygons
    v1 = _hex_vertices(c1[0], c1[1], 1.0, a1)
    v2 = _hex_vertices(c2[0], c2[1], 1.0, a2)
    
    def project(verts, axis):
        dots = verts @ axis
        return dots.min(), dots.max()
    
    def get_axes(verts):
        axes = []
        for i in range(6):
            edge = verts[(i+1)%6] - verts[i]
            # Normal to edge
            ax = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(ax)
            if norm > 1e-12:
                axes.append(ax / norm)
        return axes
    
    axes = get_axes(v1) + get_axes(v2)
    for ax in axes:
        min1, max1 = project(v1, ax)
        min2, max2 = project(v2, ax)
        if max1 < min2 + eps or max2 < min1 + eps:
            return False  # Separating axis found, no overlap
    return True  # Overlap

def _all_inside_outer(inner, outer_center, outer_side, outer_angle, eps=1e-9):
    """Check all inner hexagons are inside the outer hexagon."""
    ocx, ocy = outer_center
    for (ix, iy, ia) in inner:
        # Check all 6 vertices of inner hexagon are inside outer
        v = _hex_vertices(ix, iy, 1.0, ia)
        for (vx, vy) in v:
            if not _point_in_hex(vx, vy, ocx, ocy, outer_side, outer_angle):
                return False
    return True

def _all_disjoint(inner, eps=1e-9):
    """Check all pairs of inner hexagons are disjoint."""
    n = len(inner)
    for i in range(n):
        for j in range(i+1, n):
            ci = (inner[i][0], inner[i][1])
            cj = (inner[j][0], inner[j][1])
            if _hexagons_overlap(ci, cj, inner[i][2], inner[j][2], eps):
                return False
    return True

def _is_valid(inner, outer_center, outer_side, outer_angle):
    return _all_inside_outer(inner, outer_center, outer_side, outer_angle) and _all_disjoint(inner)

def _make_hexagonal_arrangement():
    """Create a hexagonal lattice arrangement of 12 unit hexagons."""
    inner = []
    d = np.sqrt(3)  # distance between adjacent hex centers in hex lattice
    
    # Center
    inner.append([0.0, 0.0, 0.0])
    
    # First ring: 6 hexagons at distance d, angles 0, 60, 120, 180, 240, 300
    for k in range(6):
        theta = np.radians(60 * k)
        x = d * np.cos(theta)
        y = d * np.sin(theta)
        inner.append([x, y, 0.0])
    
    # Second ring: 5 more hexagons. Place them at distance 2d at angles 30, 90, 150, 210, 270
    # But we only need 5 more. Let's place them at the "gap" positions.
    # Actually, in a hex lattice, the next ring has 12 positions at distance 2d.
    # Let's pick 5 that are well-spread.
    for k in range(5):
        theta = np.radians(60 * k + 30)
        x = 2 * d * np.cos(theta)
        y = 2 * d * np.sin(theta)
        inner.append([x, y, 0.0])
    
    return inner[:12]

def _compute_min_outer_side(inner, outer_center, outer_angle):
    """Compute the minimum outer hexagon side length needed to contain all inner hexagons."""
    ocx, ocy = outer_center
    max_required = 0.0
    for (ix, iy, ia) in inner:
        v = _hex_vertices(ix, iy, 1.0, ia)
        for (vx, vy) in v:
            # Find minimum hexagon side that contains this point
            # Transform point to outer hex frame
            angle = np.radians(outer_angle)
            dx, dy = vx - ocx, vy - ocy
            lx = dx * np.cos(angle) + dy * np.sin(angle)
            ly = -dx * np.sin(angle) + dy * np.cos(angle)
            # For a hexagon with inradius r = s*sqrt(3)/2, the point (lx,ly) is inside
            # if for all k: lx*cos(30+60k) + ly*sin(30+60k) <= r
            # So r >= max over k of (lx*cos(30+60k) + ly*sin(30+60k))
            for k in range(6):
                theta = np.radians(30 + 60 * k)
                val = lx * np.cos(theta) + ly * np.sin(theta)
                if val > 0:
                    s_req = val / (np.sqrt(3) / 2)
                    if s_req > max_required:
                        max_required = s_req
    return max_required

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin
    
    # Start with hexagonal arrangement
    inner = _make_hexagonal_arrangement()
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Compute initial outer side
    outer_side = _compute_min_outer_side(inner, outer_center, outer_angle)
    
    best_inner = [list(x) for x in inner]
    best_center = outer_center
    best_side = outer_side
    best_angle = outer_angle
    
    if not _is_valid(best_inner, best_center, best_side, best_angle):
        # Fallback: use seed-like arrangement
        best_inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        best_center = (0.0, 0.0)
        best_side = 40.0
        best_angle = 0.0
        return best_inner, best_center, best_side, best_angle
    
    # Simulated annealing on inner positions and angles
    import random
    random.seed(42)
    
    temp = 2.0
    cooling = 0.999
    
    for iteration in range(50000):
        if time.time() > deadline:
            break
        
        # Try a random perturbation
        idx = random.randint(0, N - 1)
        dx = random.gauss(0, 0.1)
        dy = random.gauss(0, 0.1)
        da = random.gauss(0, 5.0)
        
        new_inner = [list(x) for x in inner]
        new_inner[idx][0] += dx
        new_inner[idx][1] += dy
        new_inner[idx][2] += da
        
        # Check disjointness (expensive, but necessary)
        if not _all_disjoint(new_inner):
            # Reject
            temp *= cooling
            continue
        
        # Compute new outer side with current center and angle
        new_side = _compute_min_outer_side(new_inner, outer_center, outer_angle)
        
        # Also try adjusting center
        # For now, just optimize positions
        delta = new_side - outer_side
        
        if delta < 0 or random.random() < np.exp(-delta / max(temp, 1e-9)):
            inner = new_inner
            outer_side = new_side
            if outer_side < best_side:
                best_side = outer_side
                best_inner = [list(x) for x in inner]
        
        temp *= cooling
    
    # Try a few different outer angles and centers
    for angle_try in [0, 30, 60]:
        if time.time() > deadline:
            break
        outer_angle_try = float(angle_try)
        # Try to optimize center
        for cx_try in [-1.0, 0.0, 1.0]:
            for cy_try in [-1.0, 0.0, 1.0]:
                if time.time() > deadline:
                    break
                oc = (cx_try, cy_try)
                side_try = _compute_min_outer_side(best_inner, oc, outer_angle_try)
                if side_try < best_side:
                    # Verify validity
                    if _is_valid(best_inner, oc, side_try, outer_angle_try):
                        best_side = side_try
                        best_center = oc
                        best_angle = outer_angle_try
    
    return best_inner, best_center, best_side, best_angle

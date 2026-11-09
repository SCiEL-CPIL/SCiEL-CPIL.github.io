"""Hexagon packing: place 11 unit hexagons in a minimal regular hexagon."""
import random
import math
import time

N = 11
DEADLINE = 145.0  # safety margin before 150s kill

def _hex_verts(cx, cy, angle_deg, r=1.0):
    """Vertices of a regular hexagon (circumradius r) centered at (cx,cy)."""
    a = math.radians(angle_deg)
    return [(cx + r * math.cos(a + k * math.pi / 3.0),
             cy + r * math.sin(a + k * math.pi / 3.0)) for k in range(6)]

def _seg_intersect(p1, p2, p3, p4):
    """Return True if segment p1-p2 properly intersects segment p3-p4."""
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def _point_in_hex(px, py, verts):
    """Ray-casting point in polygon."""
    inside = False
    n = len(verts)
    j = n - 1
    for i in range(n):
        xi, yi = verts[i]
        xj, yj = verts[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def _overlap_dist(v1, v2):
    """
    Minimum distance between two hexagons (as polygons).
    Returns (distance, push_x, push_y) where distance < 0 means overlap.
    Uses GJK-like approach: check all vertex-edge and edge-edge pairs.
    """
    n1, n2 = len(v1), len(v2)
    
    # Check if any vertex of v1 is inside v2
    for i in range(n1):
        if _point_in_hex(v1[i][0], v1[i][1], v2):
            # Find closest point on v2 boundary to this vertex
            min_d = float('inf')
            px, py = 0.0, 0.0
            for j in range(n2):
                a, b = v2[j], v2[(j+1)%n2]
                d = _pt_seg_dist(v1[i][0], v1[i][1], a, b)
                if d < min_d:
                    min_d = d
                    # Get closest point
                    dx, dy = b[0]-a[0], b[1]-a[1]
                    d2 = dx*dx + dy*dy
                    if d2 < 1e-12:
                        cx, cy = a[0], a[1]
                    else:
                        t = max(0.0, min(1.0, ((v1[i][0]-a[0])*dx + (v1[i][1]-a[1])*dy) / d2))
                        cx, cy = a[0]+t*dx, a[1]+t*dy
                    px, py = cx, cy
            dx, dy = v1[i][0]-px, v1[i][1]-py
            ln = math.sqrt(dx*dx + dy*dy)
            if ln > 1e-12:
                return (-min_d, dx/ln, dy/ln)
            return (-min_d, 0.0, 0.0)
    
    # Check if any vertex of v2 is inside v1
    for j in range(n2):
        if _point_in_hex(v2[j][0], v2[j][1], v1):
            min_d = float('inf')
            px, py = 0.0, 0.0
            for i in range(n1):
                a, b = v1[i], v1[(i+1)%n1]
                d = _pt_seg_dist(v2[j][0], v2[j][1], a, b)
                if d < min_d:
                    min_d = d
                    dx, dy = b[0]-a[0], b[1]-a[1]
                    d2 = dx*dx + dy*dy
                    if d2 < 1e-12:
                        cx, cy = a[0], a[1]
                    else:
                        t = max(0.0, min(1.0, ((v2[j][0]-a[0])*dx + (v2[j][1]-a[1])*dy) / d2))
                        cx, cy = a[0]+t*dx, a[1]+t*dy
                    px, py = cx, cy
            dx, dy = v2[j][0]-px, v2[j][1]-py
            ln = math.sqrt(dx*dx + dy*dy)
            if ln > 1e-12:
                return (-min_d, -dx/ln, -dy/ln)
            return (-min_d, 0.0, 0.0)
    
    # Check edge-edge intersections
    for i in range(n1):
        for j in range(n2):
            if _seg_intersect(v1[i], v1[(i+1)%n1], v2[j], v2[(j+1)%n2]):
                return (-1.0, 0.0, 0.0)
    
    # Find minimum distance between all edge pairs
    min_d = float('inf')
    min_px, min_py = 0.0, 0.0
    for i in range(n1):
        a1, a2 = v1[i], v1[(i+1)%n1]
        for j in range(n2):
            b1, b2 = v2[j], v2[(j+1)%n2]
            d = _seg_seg_dist(a1, a2, b1, b2)
            if d < min_d:
                min_d = d
                m1x, m1y = (a1[0]+a2[0])/2, (a1[1]+a2[1])/2
                m2x, m2y = (b1[0]+b2[0])/2, (b1[1]+b2[1])/2
                dx, dy = m2x - m1x, m2y - m1y
                ln = math.sqrt(dx*dx + dy*dy)
                if ln > 1e-12:
                    min_px, min_py = dx/ln, dy/ln
    return (min_d, min_px, min_py)

def _seg_seg_dist(p1, p2, p3, p4):
    """Minimum distance between two segments."""
    d12 = (p2[0]-p1[0])**2 + (p2[1]-p1[1])**2
    d34 = (p4[0]-p3[0])**2 + (p4[1]-p3[1])**2
    d13 = (p3[0]-p1[0])**2 + (p3[1]-p1[1])**2
    d14 = (p4[0]-p1[0])**2 + (p4[1]-p1[1])**2
    d23 = (p3[0]-p2[0])**2 + (p3[1]-p2[1])**2
    d24 = (p4[0]-p2[0])**2 + (p4[1]-p2[1])**2
    if d12 < 1e-12 or d34 < 1e-12:
        return min(d13, d14, d23, d24) ** 0.5
    # Project
    def clamp(t): return max(0.0, min(1.0, t))
    # Parametric
    dx12 = p2[0]-p1[0]; dy12 = p2[1]-p1[1]
    dx34 = p4[0]-p3[0]; dy34 = p4[1]-p3[1]
    # Solve for closest points
    t = ((p3[0]-p1[0])*dx12 + (p3[1]-p1[1])*dy12) / d12
    t = clamp(t)
    cx1 = p1[0] + t*dx12; cy1 = p1[1] + t*dy12
    s = ((p1[0]-p3[0])*dx34 + (p1[1]-p3[1])*dy34) / d34
    s = clamp(s)
    cx2 = p3[0] + s*dx34; cy2 = p3[1] + s*dy34
    # Refine
    t = ((p3[0]-p1[0])*dx12 + (p3[1]-p1[1])*dy12) / d12
    t = clamp(t)
    cx1 = p1[0] + t*dx12; cy1 = p1[1] + t*dy12
    s = ((cx1-p3[0])*dx34 + (cy1-p3[1])*dy34) / d34
    s = clamp(s)
    cx2 = p3[0] + s*dx34; cy2 = p3[1] + s*dy34
    return ((cx1-cx2)**2 + (cy1-cy2)**2) ** 0.5

def _hex_boundary_dist(cx, cy, angle, side):
    """
    Check if a unit hexagon at (cx,cy,angle) fits inside a regular hexagon
    of side `side` centered at origin with 0 rotation.
    Returns (min_clearance, push_x, push_y) where clearance < 0 means violation.
    """
    hverts = _hex_verts(cx, cy, angle, 1.0)
    # Outer hexagon vertices (flat-top, centered at origin)
    outer = [(side * math.cos(k * math.pi/3.0), side * math.sin(k * math.pi/3.0)) for k in range(6)]
    
    # Check each vertex of inner hex is inside outer
    for vx, vy in hverts:
        if not _point_in_hex(vx, vy, outer):
            # Find closest point on outer boundary
            min_d = float('inf')
            px, py = 0.0, 0.0
            for i in range(6):
                a, b = outer[i], outer[(i+1)%6]
                d = _pt_seg_dist(vx, vy, a, b)
                if d < min_d:
                    min_d = d
                    px, py = a[0], a[1]
            # Push direction: from boundary point toward hex center
            dx, dy = cx - px, cy - py
            ln = math.sqrt(dx*dx + dy*dy)
            if ln > 1e-12:
                return (-min_d, dx/ln, dy/ln)
            return (-min_d, 0.0, 0.0)
    return (1.0, 0.0, 0.0)  # fully inside, positive clearance

def _pt_seg_dist(px, py, a, b):
    """Distance from point to segment, and closest point."""
    dx, dy = b[0]-a[0], b[1]-a[1]
    d2 = dx*dx + dy*dy
    if d2 < 1e-12:
        return math.sqrt((px-a[0])**2 + (py-a[1])**2)
    t = max(0.0, min(1.0, ((px-a[0])*dx + (py-a[1])*dy) / d2))
    cx, cy = a[0]+t*dx, a[1]+t*dy
    return math.sqrt((px-cx)**2 + (py-cy)**2)

def _initial_grid():
    """
    Initial configuration: 11 unit hexagons in a compact arrangement.
    Uses pointy-top orientation for better packing.
    Layout: 3-4-4 rows with proper hexagonal spacing.
    """
    hexes = []
    # For pointy-top hexagons:
    # Horizontal spacing = sqrt(3) * side = sqrt(3)
    # Vertical spacing = 1.5 * side = 1.5
    # Odd rows offset by sqrt(3)/2
    
    sx = math.sqrt(3)  # horizontal spacing
    sy = 1.5           # vertical spacing
    offset = sx / 2.0  # offset for odd rows
    
    rows = [3, 4, 4]
    y_positions = [-sy, 0.0, sy]
    
    for ri, (count, y) in enumerate(zip(rows, y_positions)):
        x_start = -(count-1) * sx / 2.0
        for ci in range(count):
            x = x_start + ci * sx
            if ri % 2 == 1:  # offset middle row
                x += offset
            hexes.append([x, y, 0.0])
    
    return hexes

def _check_no_overlap(hexes):
    """Check that no two hexagons overlap. Returns True if valid."""
    for i in range(N):
        for j in range(i+1, N):
            v1 = _hex_verts(hexes[i][0], hexes[i][1], hexes[i][2], 1.0)
            v2 = _hex_verts(hexes[j][0], hexes[j][1], hexes[j][2], 1.0)
            dist, _, _ = _overlap_dist(v1, v2)
            if dist < -1e-9:  # overlap with tolerance
                return False
    return True

def _check_fits(hexes, side):
    """Check that all hexagons fit inside outer hexagon of given side."""
    for i in range(N):
        clearance, _, _ = _hex_boundary_dist(hexes[i][0], hexes[i][1], hexes[i][2], side)
        if clearance < -1e-9:
            return False
    return True

def _required_side(hexes):
    """Find minimum side length for outer hexagon to contain all hexagons."""
    # Binary search
    lo = 3.0
    hi = 10.0
    for _ in range(30):
        mid = (lo + hi) / 2
        if _check_fits(hexes, mid):
            hi = mid
        else:
            lo = mid
    return hi

def construct():
    t0 = time.time()
    
    # Start from a valid initial configuration
    hexes = _initial_grid()
    
    # Verify initial configuration is valid
    if not _check_no_overlap(hexes):
        # If initial grid has overlaps, space them out more
        sx = math.sqrt(3) * 1.1
        sy = 1.5 * 1.1
        offset = sx / 2.0
        hexes = []
        rows = [3, 4, 4]
        y_positions = [-sy, 0.0, sy]
        for ri, (count, y) in enumerate(zip(rows, y_positions)):
            x_start = -(count-1) * sx / 2.0
            for ci in range(count):
                x = x_start + ci * sx
                if ri % 2 == 1:
                    x += offset
                hexes.append([x, y, 0.0])
    
    best_side = _required_side(hexes)
    best_hexes = [h[:] for h in hexes]
    
    # Local optimization: perturb and improve
    random.seed(42)
    num_restarts = 5
    max_time_per_restart = (DEADLINE - (time.time() - t0)) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() - t0 > DEADLINE - 10:
            break
        
        # Create a perturbed version
        current = [h[:] for h in best_hexes]
        
        # Add small random perturbations
        for i in range(N):
            current[i][0] += random.uniform(-0.3, 0.3)
            current[i][1] += random.uniform(-0.3, 0.3)
        
        # Local optimization using gradient-free approach
        step_size = 0.1
        for step in range(200):
            if time.time() - t0 > DEADLINE - 10:
                break
            
            # Compute objective: sum of overlaps + boundary violations
            def objective(config):
                penalty = 0.0
                # Overlap penalties
                for i in range(N):
                    for j in range(i+1, N):
                        v1 = _hex_verts(config[i][0], config[i][1], config[i][2], 1.0)
                        v2 = _hex_verts(config[j][0], config[j][1], config[j][2], 1.0)
                        d, _, _ = _overlap_dist(v1, v2)
                        if d < 0:
                            penalty += d * d * 100
                # Boundary violations
                for i in range(N):
                    c, _, _ = _hex_boundary_dist(config[i][0], config[i][1], config[i][2], best_side)
                    if c < 0:
                        penalty += c * c * 100
                return penalty
            
            current_obj = objective(current)
            
            # Try small perturbations
            improved = False
            for _ in range(20):
                trial = [h[:] for h in current]
                idx = random.randint(0, N-1)
                trial[idx][0] += random.uniform(-step_size, step_size)
                trial[idx][1] += random.uniform(-step_size, step_size)
                
                trial_obj = objective(trial)
                if trial_obj < current_obj:
                    current = trial
                    current_obj = trial_obj
                    improved = True
            
            if not improved:
                step_size *= 0.9
                if step_size < 0.001:
                    break
        
        # Check if this configuration is valid and better
        if _check_no_overlap(current):
            side = _required_side(current)
            if side < best_side:
                best_side = side
                best_hexes = [h[:] for h in current]
    
    # Final validation: ensure no overlaps
    if not _check_no_overlap(best_hexes):
        # Fallback: use a known valid configuration
        best_hexes = _initial_grid()
        best_side = _required_side(best_hexes)
    
    # Final boundary check
    if not _check_fits(best_hexes, best_side):
        best_side = _required_side(best_hexes)
    
    inner = [[best_hexes[i][0], best_hexes[i][1], best_hexes[i][2]] for i in range(N)]
    return inner, (0.0, 0.0), best_side, 0.0

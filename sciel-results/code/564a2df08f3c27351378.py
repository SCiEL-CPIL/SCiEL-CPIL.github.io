"""Improved packing of 12 unit hexagons inside a regular hexagon."""
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0  # seconds, safety margin before 150s kill


def _hex_verts(cx, cy, s, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy), side s, rotated angle_deg."""
    a = np.radians(angle_deg)
    # Regular hexagon vertices at angles 0, 60, 120, 180, 240, 300 degrees from center
    # For a flat-top hexagon, vertices are at angles 0, 60, ...
    # Actually for a regular hexagon with side s, the circumradius = s
    # Vertices at angles: 0, 60, 120, 180, 240, 300 (flat-top if we start at 0)
    # Let's use pointy-top: vertices at 30, 90, 150, 210, 270, 330
    # Actually let's just use: vertex i at angle = angle_deg + i*60
    angles = a + np.arange(6) * np.pi / 3.0
    xs = cx + s * np.cos(angles)
    ys = cy + s * np.sin(angles)
    return np.column_stack([xs, ys])


def _point_in_hexagon(px, py, cx, cy, s, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with side s, rotated angle_deg.
    Uses the fact that a regular hexagon is the intersection of 6 half-planes."""
    a = np.radians(angle_deg)
    # The 6 normals point outward from center to midpoints of edges
    # For a regular hexagon, edge midpoints are at angles 30, 90, 150, 210, 270, 330 from center
    # The apothem (distance from center to edge midpoint) = s * sqrt(3)/2
    apothem = s * np.sqrt(3) / 2.0
    # Normal directions (outward): 30, 90, 150, 210, 270, 330 degrees + rotation
    normal_angles = a + (np.pi / 6.0) + np.arange(6) * np.pi / 3.0
    dx = px - cx
    dy = py - cy
    for na in normal_angles:
        # Project (dx,dy) onto normal direction
        proj = dx * np.cos(na) + dy * np.sin(na)
        if proj > apothem:
            return False
    return True


def _hex_overlap_depth(h1, h2):
    """Compute minimum translation distance to separate two hexagons h1, h2.
    Each hex is (cx, cy, s, angle_deg). Returns 0 if no overlap, negative if separated."""
    cx1, cy1, s1, a1 = h1
    cx2, cy2, s2, a2 = h2
    
    # Quick rejection: if centers are far apart, no overlap
    dist = np.sqrt((cx2-cx1)**2 + (cy2-cy1)**2)
    max_radius = s1 + s2  # circumradius = side length for regular hexagon
    if dist > 2 * max_radius:
        return 0.0
    
    # For two convex polygons, we can use SAT (Separating Axis Theorem)
    # Get vertices of both hexagons
    v1 = _hex_verts(cx1, cy1, s1, a1)
    v2 = _hex_verts(cx2, cy2, s2, a2)
    
    # All edge normals from both polygons
    def get_normals(verts):
        normals = []
        n = len(verts)
        for i in range(n):
            p1 = verts[i]
            p2 = verts[(i+1) % n]
            edge = p2 - p1
            # Normal (perpendicular, outward)
            normal = np.array([-edge[1], edge[0]])
            normal = normal / (np.linalg.norm(normal) + 1e-15)
            # Ensure normal points outward: check against center
            center = np.mean(verts, axis=0)
            if np.dot(normal, center - p1) < 0:
                normal = -normal
            normals.append(normal)
        return np.array(normals)
    
    n1 = get_normals(v1)
    n2 = get_normals(v2)
    all_normals = np.vstack([n1, n2])
    
    min_overlap = float('inf')
    
    for normal in all_normals:
        # Project both polygons onto this normal
        proj1 = v1 @ normal
        proj2 = v2 @ normal
        
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        
        # Check for separation
        if max1 <= min2 or max2 <= min1:
            return 0.0  # Separated along this axis
        
        # Overlap depth along this axis
        overlap = min(max1, max2) - max(min1, min2)
        min_overlap = min(min_overlap, overlap)
    
    if min_overlap == float('inf'):
        return 0.0
    
    return -min_overlap  # Negative means overlap depth


def _containment_penalty(inner, outer_center, outer_side, outer_angle):
    """Penalty for inner hexagons not being contained in outer hexagon."""
    cx, cy = outer_center
    penalty = 0.0
    for (ix, iy, ia) in inner:
        # Check all 6 vertices of inner hexagon
        verts = _hex_verts(ix, iy, 1.0, ia)
        for v in verts:
            if not _point_in_hexagon(v[0], v[1], cx, cy, outer_side, outer_angle):
                # Compute how far outside
                # Find the violating edge
                a = np.radians(outer_angle)
                apothem = outer_side * np.sqrt(3) / 2.0
                normal_angles = a + (np.pi / 6.0) + np.arange(6) * np.pi / 3.0
                dx = v[0] - cx
                dy = v[1] - cy
                for na in normal_angles:
                    proj = dx * np.cos(na) + dy * np.sin(na)
                    if proj > apothem:
                        penalty += (proj - apothem) ** 2
                        break
    return penalty


def _disjointness_penalty(inner):
    """Penalty for overlapping inner hexagons."""
    penalty = 0.0
    n = len(inner)
    for i in range(n):
        for j in range(i+1, n):
            h1 = (inner[i][0], inner[i][1], 1.0, inner[i][2])
            h2 = (inner[j][0], inner[j][1], 1.0, inner[j][2])
            depth = _hex_overlap_depth(h1, h2)
            if depth < 0:  # Overlapping
                penalty += depth ** 2
    return penalty


def _total_penalty(params, outer_side):
    """Total penalty: containment + disjointness."""
    # params: [x1, y1, a1, x2, y2, a2, ..., x12, y12, a12]
    inner = []
    for i in range(N):
        inner.append([params[3*i], params[3*i+1], params[3*i+2]])
    
    # Outer hexagon centered at origin for simplicity during optimization
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    cp = _containment_penalty(inner, outer_center, outer_side, outer_angle)
    dp = _disjointness_penalty(inner)
    
    return cp + dp, inner


def _init_hexagonal_grid(rows, cols, spacing_x, spacing_y, start_x=0, start_y=0):
    """Initialize inner hexagons in a hexagonal grid pattern."""
    inner = []
    count = 0
    for r in range(rows):
        for c in range(cols):
            if count >= N:
                break
            x = start_x + c * spacing_x + (r % 2) * spacing_x / 2
            y = start_y + r * spacing_y
            inner.append([x, y, 0.0])
            count += 1
    return inner


def construct():
    start_time = time.time()
    
    best_score = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_side = float('inf')
    best_angle = 0.0
    
    # Initial configurations to try
    configs = []
    
    # 3x4 grid
    configs.append(_init_hexagonal_grid(3, 4, 2.2, 2.0, -2.2, -2.0))
    # 4x3 grid
    configs.append(_init_hexagonal_grid(4, 3, 2.2, 2.0, -2.2, -3.0))
    # 2x6 grid
    configs.append(_init_hexagonal_grid(2, 6, 2.2, 2.0, -5.5, -1.0))
    # 6x2 grid
    configs.append(_init_hexagonal_grid(6, 2, 2.2, 2.0, -2.2, -5.5))
    # Random-ish
    np.random.seed(42)
    for _ in range(3):
        inner = []
        for i in range(N):
            x = np.random.uniform(-3, 3)
            y = np.random.uniform(-3, 3)
            a = np.random.uniform(0, 60)
            inner.append([x, y, a])
        configs.append(inner)
    
    for init_inner in configs:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Initial outer side estimate
        # Find bounding box of inner hexagons
        all_x = [p[0] for p in init_inner]
        all_y = [p[1] for p in init_inner]
        # For hexagons, the extent is roughly the range plus 1 (circumradius)
        span_x = max(all_x) - min(all_x) + 2.0
        span_y = max(all_y) - min(all_y) + 2.0
        # For a regular hexagon containing these, side length needs to be at least
        # the max of the projections
        init_side = max(span_x, span_y) / np.sqrt(3) + 1.0
        init_side = max(init_side, 3.0)
        
        # Optimize: minimize outer_side subject to constraints
        # We'll optimize positions and angles, then compute minimal enclosing hexagon
        
        def objective_and_check(x):
            # x: [x1, y1, a1, ..., x12, y12, a12, outer_side]
            inner = []
            for i in range(N):
                inner.append([x[3*i], x[3*i+1], x[3*i+2]])
            outer_side = x[3*N]
            
            if outer_side < 2.0:
                return 1e6, None
            
            # Compute minimal enclosing regular hexagon centered at origin
            # Actually, let's just use the given outer_side and check containment
            # with center at origin
            
            cp = _containment_penalty(inner, (0.0, 0.0), outer_side, 0.0)
            dp = _disjointness_penalty(inner)
            
            total = outer_side + 100.0 * (cp + dp)
            return total, inner
        
        # Initial parameters
        x0 = []
        for p in init_inner:
            x0.extend([p[0], p[1], p[2]])
        x0.append(init_side)
        
        # Bounds
        bounds = []
        for i in range(N):
            bounds.append((-10, 10))  # x
            bounds.append((-10, 10))  # y
            bounds.append((0, 60))    # angle (hexagon has 60-degree symmetry)
        bounds.append((2.5, 20))      # outer side
        
        # Optimize
        try:
            result = minimize(objective_and_check, x0, method='L-BFGS-B',
                             bounds=bounds,
                             options={'maxiter': 100, 'ftol': 1e-8})
        except Exception:
            continue
        
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Extract result
        inner = []
        for i in range(N):
            inner.append([result.x[3*i], result.x[3*i+1], result.x[3*i+2]])
        outer_side = result.x[3*N]
        
        # Verify disjointness and containment
        dp = _disjointness_penalty(inner)
        cp = _containment_penalty(inner, (0.0, 0.0), outer_side, 0.0)
        
        if dp < 1e-6 and cp < 1e-6:
            if outer_side < best_score:
                best_score = outer_side
                best_inner = inner
                best_center = (0.0, 0.0)
                best_side = outer_side
                best_angle = 0.0
    
    # If no valid solution found, return a conservative valid packing
    if best_inner is None:
        # Linear arrangement with guaranteed separation
        inner = []
        for i in range(N):
            x = (i - (N-1)/2.0) * 2.1
            y = 0.0
            inner.append([x, 0.0, 0.0])
        # Compute bounding hexagon
        span = (N-1) * 2.1 + 2.0
        side = span / np.sqrt(3) + 0.5
        return inner, (0.0, 0.0), side, 0.0
    
    return best_inner, best_center, best_side, best_angle

import time
import math
import numpy as np
from scipy.optimize import minimize

N = 12

def hex_vertices(cx, cy, s, angle_deg):
    """Vertices of regular hexagon (flat-top at angle 0), side s, centered at (cx,cy)."""
    a = math.radians(angle_deg)
    v = np.empty((6, 2))
    for i in range(6):
        t = a + i * math.pi / 3.0
        v[i] = [cx + s * math.cos(t), cy + s * math.sin(t)]
    return v

def point_in_hex(px, py, cx, cy, s, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with side s and angle angle_deg."""
    # Translate to center
    dx = px - cx
    dy = py - cy
    # Rotate by -angle to align hexagon with axes
    a = math.radians(-angle_deg)
    rx = dx * math.cos(a) - dy * math.sin(a)
    ry = dx * math.sin(a) + dy * math.cos(a)
    
    # In the aligned frame, the hexagon is defined by |x| <= s, |y| <= s*sqrt(3)/2 ? No.
    # Vertices of aligned flat-top hex (angle 0):
    # (s, 0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s, 0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
    # The boundaries are:
    # 1. x = s
    # 2. x = -s
    # 3. y = s*sqrt(3)/2
    # 4. y = -s*sqrt(3)/2
    # 5. y = sqrt(3)(x - s/2) ? No.
    # Let's use the distance to edges.
    # The apothem is h = s * sqrt(3) / 2.
    # The 3 pairs of edges are at angles 0, 60, 120 degrees to the x-axis?
    # For flat-top (angle 0), the vertical edges are at x = +/- s.
    # The slanted edges:
    # Top-right: connects (s, 0) and (s/2, s*sqrt(3)/2).
    # Equation: y - 0 = (s*sqrt(3)/2 - 0) / (s/2 - s) * (x - s)
    # y = (s*sqrt(3)/2) / (-s/2) * (x - s) = -sqrt(3) (x - s)
    # y = -sqrt(3)x + s*sqrt(3)
    # sqrt(3)x + y = s*sqrt(3)
    # Check (s, 0): s*sqrt(3) + 0 = s*sqrt(3). OK.
    # Check (s/2, s*sqrt(3)/2): s*sqrt(3)/2 + s*sqrt(3)/2 = s*sqrt(3). OK.
    # So constraint: sqrt(3)x + y <= s*sqrt(3)
    
    # Top-left: connects (s/2, s*sqrt(3)/2) and (-s/2, s*sqrt(3)/2).
    # This is the top horizontal edge? No, flat-top has vertical left/right.
    # Wait, "flat-top" usually means a flat side on top.
    # My vertices: (s,0) is right. (s/2, s*sqrt(3)/2) is top-right.
    # So the top edge is between (s/2, s*sqrt(3)/2) and (-s/2, s*sqrt(3)/2).
    # This is a horizontal line y = s*sqrt(3)/2.
    # So constraints:
    # 1. x <= s
    # 2. x >= -s
    # 3. y <= s*sqrt(3)/2
    # 4. y >= -s*sqrt(3)/2
    # 5. sqrt(3)x + y <= s*sqrt(3)  (Top-Right slant)
    # 6. -sqrt(3)x + y <= s*sqrt(3) (Top-Left slant) -> Check: (-s/2, s*sqrt(3)/2) -> s*sqrt(3)/2 + s*sqrt(3)/2 = s*sqrt(3). OK.
    # 7. -sqrt(3)x - y <= s*sqrt(3) (Bottom-Left slant)
    # 8. sqrt(3)x - y <= s*sqrt(3) (Bottom-Right slant)
    
    # Actually, due to symmetry, we just need to check the 3 unique directions.
    # The hexagon is the intersection of 3 strips.
    # Strip 1: -s <= x <= s
    # Strip 2: -s*sqrt(3)/2 <= y <= s*sqrt(3)/2
    # Strip 3: |sqrt(3)x + y| <= s*sqrt(3) ? No.
    # Let's just check all 6 vertices or use the 6 inequalities.
    
    h = s * math.sqrt(3) / 2
    if rx > s or rx < -s: return False
    if ry > h or ry < -h: return False
    if math.sqrt(3)*rx + ry > s*math.sqrt(3): return False
    if -math.sqrt(3)*rx + ry > s*math.sqrt(3): return False
    if -math.sqrt(3)*rx - ry > s*math.sqrt(3): return False
    if math.sqrt(3)*rx - ry > s*math.sqrt(3): return False
    return True

def enclosing_hex_penalty(points, cx, cy, s, angle_deg):
    """Penalty if any point is outside the hexagon."""
    pen = 0.0
    for p in points:
        if not point_in_hex(p[0], p[1], cx, cy, s, angle_deg):
            # Calculate distance to outside for gradient? No, just large penalty.
            # For better optimization, calculate violation amount.
            # Translate and rotate
            dx = p[0] - cx
            dy = p[1] - cy
            a = math.radians(-angle_deg)
            rx = dx * math.cos(a) - dy * math.sin(a)
            ry = dx * math.sin(a) + dy * math.cos(a)
            h = s * math.sqrt(3) / 2
            # Max violation of the 6 constraints
            v1 = max(0, rx - s)
            v2 = max(0, -rx - s)
            v3 = max(0, ry - h)
            v4 = max(0, -ry - h)
            v5 = max(0, math.sqrt(3)*rx + ry - s*math.sqrt(3))
            v6 = max(0, -math.sqrt(3)*rx + ry - s*math.sqrt(3))
            # Note: v5 and v6 might need scaling to be comparable to v1?
            # v1 is in units of x. v5 is in units of y-like.
            # Let's just sum the squares of violations.
            pen += v1**2 + v2**2 + v3**2 + v4**2 + v5**2 + v6**2
    return pen

def solve_enclosing_hex(points):
    """Find min enclosing hexagon for a set of points."""
    # Initial guess
    cx0 = np.mean([p[0] for p in points])
    cy0 = np.mean([p[1] for p in points])
    # Bounding box diagonal
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    s0 = max(w, h) / 1.5 + 1.0
    
    def obj(x):
        cx, cy, s, ang = x
        if s < 0.1: s = 0.1
        return s + 10.0 * enclosing_hex_penalty(points, cx, cy, s, ang)
    
    x0 = [cx0, cy0, s0, 0.0]
    res = minimize(obj, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
    return res.x

def axial_to_cartesian(q, r):
    x = 1.5 * q
    y = math.sqrt(3) * (r + q / 2.0)
    return x, y

def generate_polyhexes():
    """Generate a few candidate 12-hexagon shapes."""
    shapes = []
    
    # Shape 1: 2x6 rectangle
    s1 = []
    for i in range(6):
        s1.append((i, 0))
        s1.append((i, 1))
    shapes.append(s1)
    
    # Shape 2: 3x4 rectangle
    s2 = []
    for i in range(4):
        for j in range(3):
            s2.append((i, j))
    shapes.append(s2)
    
    # Shape 3: Hexagon of side 2 (7 hexes) + 5 more
    # Center (0,0). Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Add 5 more to make it compact.
    # (2,0), (1,1), (0,2), (-1,2), (-2,1) ?
    s3 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    # Add neighbors to fill gaps
    s3.extend([(2,0), (1,1), (0,2), (-1,2), (-2,1)])
    shapes.append(s3)
    
    # Shape 4: 1x12 line (bad, but baseline)
    s4 = [(i, 0) for i in range(12)]
    shapes.append(s4)
    
    return shapes

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    best_score = float('inf')
    best_inner = None
    best_outer = None
    
    shapes = generate_polyhexes()
    
    for shape in shapes:
        if time.time() > deadline:
            break
            
        # Convert to Cartesian
        centers = [axial_to_cartesian(q, r) for q, r in shape]
        
        # Generate vertices for all 12 hexagons
        points = []
        for cx, cy in centers:
            v = hex_vertices(cx, cy, 1.0, 0.0)
            points.extend(v.tolist())
        
        # Solve for enclosing hex
        res_x = solve_enclosing_hex(points)
        cx, cy, s, ang = res_x
        
        # Verify feasibility
        pen = enclosing_hex_penalty(points, cx, cy, s, ang)
        if pen < 1e-6:
            if s < best_score:
                best_score = s
                best_inner = [[cx, cy, 0.0] for cx, cy in centers]
                best_outer = (cx, cy, s, ang)
        
    if best_inner is None:
        # Fallback: return a valid configuration even if not optimal
        # Use the first shape and a large enclosing hex
        shape = shapes[0]
        centers = [axial_to_cartesian(q, r) for q, r in shape]
        points = []
        for cx, cy in centers:
            v = hex_vertices(cx, cy, 1.0, 0.0)
            points.extend(v.tolist())
        
        cx0 = np.mean([p[0] for p in points])
        cy0 = np.mean([p[1] for p in points])
        s0 = 10.0
        best_inner = [[cx, cy, 0.0] for cx, cy in centers]
        best_outer = (cx0, cy0, s0, 0.0)
        best_score = s0

    return best_inner, (float(best_outer[0]), float(best_outer[1])), float(best_outer[2]), float(best_outer[3])

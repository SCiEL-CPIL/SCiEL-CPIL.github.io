import numpy as np
from scipy.optimize import differential_evolution

N = 11

def get_hex_vertices(x, y, angle):
    # Unit hexagon vertices
    # Side length 1
    # Vertices at distance 1 from center
    # Angles: 0, 60, 120, 180, 240, 300 degrees + angle
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False) + angle
    vx = x + np.cos(angles)
    vy = y + np.sin(angles)
    return np.column_stack((vx, vy))

def point_in_polygon(p, poly):
    # Ray casting algorithm
    x, y = p
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(1, n + 1):
        p2x, p2y = poly[i % n]
        if (y > p1y) != (y > p2y):
            x_int = (p2x - p1x) * (y - p1y) / (p2y - p1y) + p1x
            if x < x_int:
                inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def segments_intersect(p1, p2, p3, p4):
    # Check if segment p1-p2 intersects p3-p4
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
        
    # Handle collinear cases (optional, might not be needed for generic positions)
    return False

def check_overlap(poly1, poly2):
    # Check if any vertex of poly1 is in poly2
    for p in poly1:
        if point_in_polygon(p, poly2):
            return True
    # Check if any vertex of poly2 is in poly1
    for p in poly2:
        if point_in_polygon(p, poly1):
            return True
    # Check if any edges intersect
    n1 = len(poly1)
    n2 = len(poly2)
    for i in range(n1):
        p1 = poly1[i]
        p2 = poly1[(i + 1) % n1]
        for j in range(n2):
            p3 = poly2[j]
            p4 = poly2[(j + 1) % n2]
            if segments_intersect(p1, p2, p3, p4):
                return True
    return False

def cost(params):
    # params: 11 * (x, y, angle)
    # Fixed outer rotation 0
    polys = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i + 1]
        angle = params[3*i + 2]
        poly = get_hex_vertices(x, y, angle)
        polys.append(poly)
        
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            if check_overlap(polys[i], polys[j]):
                return 1000.0 # Large penalty
                
    # Calculate required side length S
    # Outer hexagon centered at (0,0), rotation 0
    # Vertices at angles 0, 60, ...
    # Supporting lines are perpendicular to axes at 30, 90, 150 deg?
    # For a flat-top hexagon (vertices at 0, 60...), the normals are at 30, 90, 150...
    # Wait, if vertices are at 0, 60, 120, 180, 240, 300:
    # The side between 0 and 60 is at angle 30. Normal is 30+90=120?
    # Let's use the property: max projection on 3 axes.
    # Axes of symmetry for a regular hexagon are at 0, 60, 120 degrees?
    # If vertices are at 0, 60..., the axes of symmetry are at 0, 60, 120.
    # The distance from center to side is $S \cos(30) = S \sqrt{3}/2$.
    # So for any point (x,y), the constraint is:
    # |x cos(0) + y sin(0)| <= S sqrt(3)/2  => |x| <= S sqrt(3)/2
    # |x cos(60) + y sin(60)| <= S sqrt(3)/2
    # |x cos(120) + y sin(120)| <= S sqrt(3)/2
    
    max_proj = 0
    axes = [0, np.pi/3, 2*np.pi/3]
    for poly in polys:
        for p in poly:
            for a in axes:
                proj = abs(p[0] * np.cos(a) + p[1] * np.sin(a))
                if proj > max_proj:
                    max_proj = proj
                    
    S = max_proj * 2 / np.sqrt(3)
    return S

def construct():
    # Initial guess: A grid-like structure
    # Let's use a 3x3 grid centered at 0, plus 2 extra
    # Or just use random initialization in DE
    bounds = [(-3, 3), (-3, 3), (0, np.pi)] * N
    
    # Run differential evolution
    result = differential_evolution(cost, bounds, maxiter=100, popsize=10, tol=1e-4, seed=42)
    
    if result.success:
        params = result.x
    else:
        # Fallback to a simple valid configuration if DE fails
        # This is a risk. Let's try to construct a valid one manually if DE fails.
        # But DE usually finds *something*.
        params = result.x
        
    # Extract positions and angles
    inner = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i + 1]
        angle = params[3*i + 2]
        inner.append([x, y, np.degrees(angle)])
        
    # Calculate final S
    S = cost(params)
    
    # If S is still 1000, it means there was an overlap.
    # We should try to fix it or return a known valid config.
    if S >= 1000:
        # Fallback: Simple line packing (inefficient but valid)
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        S = 40.0
        
    return inner, (0.0, 0.0), S, 0.0

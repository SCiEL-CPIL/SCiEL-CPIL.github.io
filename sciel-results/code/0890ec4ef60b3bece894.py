import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
EPS = 1e-9

def hex_vertices(cx, cy, angle):
    """Return the 6 vertices of a unit hexagon centered at (cx, cy) with rotation angle."""
    # Vertices of a unit hexagon (side 1) centered at origin, flat-top orientation (angle=0)
    # Actually, let's define angle=0 as pointing right (vertex at (1,0)) or flat top?
    # Standard hexagon with side 1:
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees from center?
    # Distance from center to vertex (circumradius) is 1.
    # Distance from center to edge (apothem) is sqrt(3)/2.
    # Let's use circumradius = 1.
    angles = np.arange(6) * (math.pi / 3.0) + angle
    x = np.cos(angles)
    y = np.sin(angles)
    return np.column_stack((x, y))

def get_all_vertices(inner_params, outer_params):
    """
    inner_params: (N, 3) array [x, y, angle]
    outer_params: (4,) array [cx, cy, side, angle]
    Returns:
        inner_verts: (N, 6, 2) array
        outer_verts: (6, 2) array
    """
    inner_verts = np.zeros((N, 6, 2))
    for i in range(N):
        x, y, ang = inner_params[i]
        v = hex_vertices(x, y, ang)
        inner_verts[i] = v
    
    cx, cy, side, ang = outer_params
    # Scale the unit hexagon vertices by 'side'
    v = hex_vertices(cx, cy, ang) * side
    return inner_verts, v

def check_separation(v1, v2):
    """
    Check if two convex polygons (v1, v2) are separated by a margin.
    v1: (6, 2), v2: (6, 2)
    Returns the minimum distance between the two polygons (negative if overlapping).
    We use the Separating Axis Theorem (SAT).
    """
    # Edges of v1
    edges1 = np.roll(v1, -1, axis=0) - v1
    # Edges of v2
    edges2 = np.roll(v2, -1, axis=0) - v2
    
    min_dist = float('inf')
    
    # Check normals of v1
    for i in range(6):
        edge = edges1[i]
        # Normal
        nx = -edge[1]
        ny = edge[0]
        norm = np.sqrt(nx*nx + ny*ny)
        if norm < 1e-10: continue
        nx /= norm
        ny /= norm
        
        # Project v1 and v2 onto this normal
        p1 = np.dot(v1, [nx, ny])
        p2 = np.dot(v2, [nx, ny])
        
        min1, max1 = np.min(p1), np.max(p1)
        min2, max2 = np.min(p2), np.max(p2)
        
        # Distance between intervals
        if max1 < min2:
            dist = min2 - max1
        elif max2 < min1:
            dist = min1 - max2
        else:
            dist = 0.0 # Overlapping or touching
            
        if dist < min_dist:
            min_dist = dist
            
    # Check normals of v2
    for i in range(6):
        edge = edges2[i]
        nx = -edge[1]
        ny = edge[0]
        norm = np.sqrt(nx*nx + ny*ny)
        if norm < 1e-10: continue
        nx /= norm
        ny /= norm
        
        p1 = np.dot(v1, [nx, ny])
        p2 = np.dot(v2, [nx, ny])
        
        min1, max1 = np.min(p1), np.max(p1)
        min2, max2 = np.min(p2), np.max(p2)
        
        if max1 < min2:
            dist = min2 - max1
        elif max2 < min1:
            dist = min1 - max2
        else:
            dist = 0.0
            
        if dist < min_dist:
            min_dist = dist
            
    return min_dist

def objective(params):
    """
    params: (N*3 + 4,) vector
    First N*3: inner hexagons (x, y, angle)
    Last 4: outer hexagon (cx, cy, side, angle)
    """
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return float('inf')
        
    inner_params = params[:N*3].reshape(N, 3)
    outer_params = params[N*3:]
    
    # Side length must be positive
    side = outer_params[2]
    if side < 1.0: # Minimum possible side is roughly 1 (if 1 hex) but we have 11
        side = 1.0 + EPS
        # Actually, if side is too small, penalty will be huge.
        # Let's just compute with the given side, but ensure no division by zero etc.
        
    inner_verts, outer_verts = get_all_vertices(inner_params, outer_params)
    
    penalty = 0.0
    margin = 0.01 # Target separation between inner hexagons
    
    # 1. Inner-Inner collisions
    for i in range(N):
        for j in range(i+1, N):
            d = check_separation(inner_verts[i], inner_verts[j])
            # We want d >= margin
            # If d < margin, penalty = (margin - d)^2
            if d < margin:
                penalty += (margin - d) ** 2
                
    # 2. Inner-Outer collisions (Inner must be inside Outer)
    # This is trickier with SAT distance. 
    # If an inner vertex is outside the outer hexagon, distance is negative? 
    # SAT distance is positive if separated, 0 if intersecting.
    # We need to ensure inner is strictly inside outer.
    # Let's check if any vertex of inner is outside outer.
    # Or better: use the same separation logic but inverted?
    # Actually, if inner is inside outer, they are not "separated" in the SAT sense of external distance.
    # We need a different check for containment.
    
    # Containment Check:
    # For each inner hexagon, check if all its vertices are inside the outer hexagon.
    # And check that the inner hexagon doesn't cross the boundary.
    # A robust way: For each edge of the outer hexagon, check if the inner hexagon is on the "inside".
    # Let's use a signed distance approach or simple vertex checking + edge crossing.
    # Given time, let's use a simpler penalty: 
    # Project inner vertices onto outer normals.
    
    cx, cy, side, ang = outer_params
    # Outer normals (pointing inward)
    # Vertices of outer are at angles ang + k*pi/3
    # Edges are perpendicular to angles ang + pi/6 + k*pi/3
    # Inward normal for edge k:
    outer_normals = []
    for k in range(6):
        theta = ang + math.pi / 6.0 + k * math.pi / 3.0
        # Normal pointing OUT
        nx_out = math.cos(theta)
        ny_out = math.sin(theta)
        # Normal pointing IN
        nx_in = -nx_out
        ny_in = -ny_out
        outer_normals.append((nx_in, ny_in))
        
    # Apothem of outer hexagon
    apothem_outer = side * math.sqrt(3) / 2.0
    
    # For each inner hexagon
    for i in range(N):
        # Check vertices
        v_inner = inner_verts[i]
        for v in v_inner:
            vx, vy = v
            # Distance from center of outer
            dx = vx - cx
            dy = vy - cy
            # Check against each outer edge
            for nx, ny in outer_normals:
                # Projection of vertex onto inward normal
                proj = dx * nx + dy * ny
                # The vertex must be at a distance <= apothem - inner_apothem? 
                # No, the vertex must be strictly inside.
                # The "depth" of the vertex into the outer hexagon is proj.
                # It must be > 0.
                # To ensure a margin, we want proj > margin_dist.
                # Let's say margin_dist = 0.01
                if proj < margin:
                    penalty += (margin - proj) ** 2
                    
    # Return side length + penalty
    return side + 100.0 * penalty

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    best_S = float('inf')
    best_params = None
    
    # Initial Candidates
    # 1. Hexagonal Cluster (Center + 6 neighbors + 4 more)
    # Axial to Cartesian
    def axial_to_cartesian(axial_list, scale=1.0):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0) * scale
            y = 1.5 * r * scale
            pts.append((x, y))
        return np.array(pts)
    
    # Candidate 1: 11 hexagons in a hexagonal pattern
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    pts_1 = axial_to_cartesian(axial_1)
    
    # Candidate 2: 3x3 + 2
    axial_2 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3)
    ]
    pts_2 = axial_to_cartesian(axial_2)
    
    # Candidate 3: 4x3 + 1
    axial_3 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    pts_3 = axial_to_cartesian(axial_3)
    
    # Generate initial parameter vectors
    def make_initial_params(pts, outer_side_est, outer_angle=0.0):
        # Estimate outer center and side
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        
        # Estimate side length by projecting points
        max_proj = 0
        for k in range(6):
            ang = outer_angle + math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
            
        # Side length S relates to max projection D by D = S * apothem = S * sqrt(3)/2
        # So S = 2 * D / sqrt(3)
        S_est = 2.0 * max_proj / math.sqrt(3)
        # Add some slack
        S_est *= 1.1
        
        params = np.zeros(N * 3 + 4)
        for i in range(N):
            params[i*3] = pts[i, 0]
            params[i*3+1] = pts[i, 1]
            params[i*3+2] = 0.0 # Angle
            
        params[N*3] = cx
        params[N*3+1] = cy
        params[N*3+2] = S_est
        params[N*3+3] = outer_angle
        
        return params

    initial_sets = [
        make_initial_params(pts_1, 5.0),
        make_initial_params(pts_2, 5.0),
        make_initial_params(pts_3, 5.0),
    ]
    
    # Add some random perturbations to break symmetry
    for i in range(3):
        p = initial_sets[i].copy()
        # Jitter positions
        p[0:N*3:3] += np.random.normal(0, 0.1, N)
        p[1:N*3:3] += np.random.normal(0, 0.1, N)
        # Jitter angles
        p[2:N*3:3] += np.random.normal(0, 0.1, N)
        initial_sets.append(p)
    
    # Optimization Loop
    for init in initial_sets:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        try:
            res = minimize(objective, init, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
            
            if res.fun < best_S and res.fun < 100.0: # Sanity check
                best_S = res.fun
                best_params = res.x
        except Exception:
            pass

    if best_params is None:
        # Fallback to a simple valid configuration
        pts = axial_to_cartesian(axial_1)
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        S = 2.0 * max_proj / math.sqrt(3) * 1.05
        
        inner = []
        for x, y in pts:
            inner.append([float(x), float(y), 0.0])
        return inner, [float(cx), float(cy)], float(S), 0.0

    # Extract results
    inner_params = best_params[:N*3].reshape(N, 3)
    outer_params = best_params[N*3:]
    
    inner = []
    for i in range(N):
        x, y, ang = inner_params[i]
        inner.append([float(x), float(y), float(math.degrees(ang))])
        
    cx, cy, side, ang = outer_params
    center = [float(cx), float(cy)]
    side = float(side)
    rotation = float(math.degrees(ang))
    
    return inner, center, side, rotation


import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 140.0  # Leave 10s safety margin for 150s limit
SAFETY_MARGIN = 5.0

def hexagon_vertices(cx, cy, angle_deg, side=1.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    angle = math.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = angle + k * math.pi / 3.0
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append((x, y))
    return np.array(verts)

def hexagon_contains_point(hex_center, hex_angle, point):
    """Check if point is inside or on boundary of hexagon using 6 half-plane tests."""
    cx, cy = hex_center
    angle = math.radians(hex_angle)
    # 6 inward normals
    for k in range(6):
        theta = angle + k * math.pi / 3.0
        nx = math.cos(theta)
        ny = math.sin(theta)
        # Distance from center to edge along normal
        d = (point[0] - cx) * nx + (point[1] - cy) * ny
        if d > APOTHEM + 1e-9:
            return False
    return True

def hexagons_overlap(c1, a1, c2, a2, margin=1e-7):
    """Fast check for overlap using bounding boxes and vertex containment."""
    # Bounding box check
    # Approximate bounding box for unit hexagon is 2x2
    if abs(c1[0] - c2[0]) > 2.0 or abs(c1[1] - c2[1]) > 2.0:
        return False
    
    # If aligned, simple distance check
    da = abs(a1 - a2) % 60.0
    if da > 30.0:
        da = 60.0 - da
    
    # For small relative angles, we can use a tighter distance threshold
    # Max distance between centers for non-overlap is 2*APOTHEM if aligned
    # If rotated, the "width" increases slightly in some directions? No, width is constant for aligned, but the shape changes relative to each other.
    # Conservative: if dist > 2.0, no overlap.
    dx = c2[0] - c1[0]
    dy = c2[1] - c1[1]
    dist_sq = dx*dx + dy*dy
    if dist_sq > 4.0: # 2^2
        return False
        
    # If very close, likely overlap
    if dist_sq < (2*APOTHEM - 0.1)**2:
        return True

    # Vertex containment check (fast path)
    v1 = hexagon_vertices(c1[0], c1[1], a1)
    v2 = hexagon_vertices(c2[0], c2[1], a2)
    
    # Check if any vertex of hex1 is inside hex2
    for v in v1:
        if hexagon_contains_point(c2, a2, v):
            return True
    # Check if any vertex of hex2 is inside hex1
    for v in v2:
        if hexagon_contains_point(c1, a1, v):
            return True
            
    return False

def check_all_non_overlap(positions, angles):
    """Check that no two hexagons overlap."""
    n = len(positions)
    for i in range(n):
        for j in range(i+1, n):
            if hexagons_overlap(positions[i], angles[i], positions[j], angles[j]):
                return False
    return True

def compute_outer_hexagon_side(positions, angles, outer_center, outer_angle):
    """Compute the minimum outer hexagon side length that contains all inner hexagons."""
    ocx, ocy = outer_center
    outer_angle_rad = math.radians(outer_angle)
    
    max_proj = 0.0
    for k in range(6):
        theta = outer_angle_rad + k * math.pi / 3.0
        nx = math.cos(theta)
        ny = math.sin(theta)
        
        for i in range(len(positions)):
            cx, cy = positions[i]
            a_i = angles[i]
            # Max projection of hexagon i onto direction (nx, ny)
            # For a unit hexagon, the support function h(u) is:
            # Let delta be the angle between u and the nearest edge normal of the inner hex.
            # h = APOTHEM / cos(delta) if |delta| <= pi/6
            # This is faster than vertex iteration
            
            inner_angle_rad = math.radians(a_i)
            # Find the angle of the nearest normal to the inner hexagon
            # Normals are at inner_angle + k*pi/3
            # We want the projection of the inner hexagon onto the outer normal (nx, ny)
            # This is equivalent to the support function of the inner hex in direction (nx, ny)
            
            # Calculate the angle of the outer normal
            outer_norm_angle = theta
            # The inner hexagon normals are at inner_angle + m*pi/3
            # The angle difference to the nearest normal determines the support
            diff = (outer_norm_angle - inner_angle_rad) % (math.pi/3.0)
            # Normalize diff to [-pi/6, pi/6]
            if diff > math.pi/6.0:
                diff = diff - math.pi/3.0
            if diff < -math.pi/6.0:
                diff = diff + math.pi/3.0
            
            support = APOTHEM / math.cos(diff)
            
            # Projection of center
            center_proj = (cx - ocx) * nx + (cy - ocy) * ny
            max_proj = max(max_proj, center_proj + support, -(center_proj + support))
    
    S = (max_proj + APOTHEM) / APOTHEM
    return S

def objective(params):
    """Objective function for optimization: outer hexagon side length + penalty for overlaps."""
    # Time guard inside objective
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return float('inf')

    n = N
    # params layout: [x1,y1,a1, x2,y2,a2, ..., xn,yn,an, cx,cy, outer_angle]
    positions = []
    angles = []
    for i in range(n):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        positions.append((x, y))
        angles.append(a)
    
    ocx = params[3*n]
    ocy = params[3*n+1]
    outer_angle = params[3*n+2]
    
    # Compute outer side length
    S = compute_outer_hexagon_side(positions, angles, (ocx, ocy), outer_angle)
    
    # Penalty for overlaps
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            if hexagons_overlap(positions[i], angles[i], positions[j], angles[j]):
                penalty += 100.0
                dx = positions[j][0] - positions[i][0]
                dy = positions[j][1] - positions[i][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < 1e-6:
                    penalty += 1000.0
                else:
                    penalty += 10.0 * (2*APOTHEM - dist)
    
    return S + penalty

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    # Initial guess: hexagonal lattice pattern with 11 hexagons
    # Use a 3-4-3-1 pattern in axial coords, converted to Cartesian
    # Axial to Cartesian: x = sqrt(3)*(q + r/2), y = 1.5*r
    sqrt3 = math.sqrt(3)
    
    # Pattern: 3-4-3-1
    axial = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3)
    ]
    
    init_positions = []
    for q, r in axial:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        init_positions.append((x, y))
    
    # Center the pattern
    xs = [p[0] for p in init_positions]
    ys = [p[1] for p in init_positions]
    cx0 = (min(xs) + max(xs)) / 2.0
    cy0 = (min(ys) + max(ys)) / 2.0
    
    # Initial parameters
    init_params = []
    for x, y in init_positions:
        init_params.extend([x - cx0, y - cy0, 0.0])
    init_params.extend([0.0, 0.0, 0.0])  # outer center and angle
    
    # Try multiple initial configurations
    best_result = None
    best_S = float('inf')
    
    # Configuration 1: aligned hexagons, hexagonal lattice
    configs = [
        ("hex_lattice", init_params),
    ]
    
    # Configuration 2: slightly rotated inner hexagons
    init_params2 = list(init_params)
    for i in range(N):
        init_params2[3*i+2] = 30.0  # 30 degree rotation
    configs.append(("rotated_30", init_params2))
    
    # Configuration 3: different lattice pattern
    axial2 = [
        (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (0, 3), (1, 3), (2, 3)
    ]
    init_positions2 = []
    for q, r in axial2:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        init_positions2.append((x, y))
    xs2 = [p[0] for p in init_positions2]
    ys2 = [p[1] for p in init_positions2]
    cx2 = (min(xs2) + max(xs2)) / 2.0
    cy2 = (min(ys2) + max(ys2)) / 2.0
    
    init_params3 = []
    for x, y in init_positions2:
        init_params3.extend([x - cx2, y - cy2, 0.0])
    init_params3.extend([0.0, 0.0, 0.0])
    configs.append(("hex_lattice2", init_params3))
    
    for config_name, init_p in configs:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        try:
            # Reduce maxiter to ensure we finish in time
            result = minimize(
                objective,
                init_p,
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            if result.success or result.fun < best_S:
                params = result.x
                positions = []
                angles = []
                for i in range(N):
                    positions.append((params[3*i], params[3*i+1]))
                    angles.append(params[3*i+2])
                
                ocx = params[3*N]
                ocy = params[3*N+1]
                outer_angle = params[3*N+2]
                
                # Check if solution is valid (no overlaps)
                if check_all_non_overlap(positions, angles):
                    S = compute_outer_hexagon_side(positions, angles, (ocx, ocy), outer_angle)
                    if S < best_S:
                        best_S = S
                        best_result = (positions, angles, (ocx, ocy), outer_angle)
                        
        except Exception as e:
            pass
    
    # If no valid result from optimization, use a simple fallback
    if best_result is None:
        # Fallback: use the first initial configuration with computed S
        positions = [(p[0] - cx0, p[1] - cy0) for p in init_positions]
        angles = [0.0] * N
        ocx, ocy = 0.0, 0.0
        outer_angle = 0.0
        S = compute_outer_hexagon_side(positions, angles, (ocx, ocy), outer_angle)
        best_result = (positions, angles, (ocx, ocy), outer_angle)
        best_S = S
    
    positions, angles, outer_center, outer_angle = best_result
    
    # Build the inner list
    inner = []
    for i in range(N):
        x, y = positions[i]
        a = angles[i]
        inner.append([float(x), float(y), float(a)])
    
    center = [float(outer_center[0]), float(outer_center[1])]
    side = float(best_S)
    rotation = float(outer_angle)
    
    return inner, center, side, rotation

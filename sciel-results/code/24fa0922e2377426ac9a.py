import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 12

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
    angle_rad = np.radians(angle_deg)
    angles = angle_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def polygon_area(verts):
    """Compute area of a polygon given its vertices."""
    x = verts[:, 0]
    y = verts[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def polygons_disjoint(v1, v2, tol=1e-6):
    """
    Check if two convex polygons are disjoint (no interior intersection).
    Uses SAT (Separating Axis Theorem) for convex polygons.
    """
    for verts in (v1, v2):
        for i in range(len(verts)):
            p1 = verts[i]
            p2 = verts[(i + 1) % len(verts)]
            # Edge normal
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            # Project both polygons onto this normal
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            max1 = np.max(proj1)
            min1 = np.min(proj1)
            max2 = np.max(proj2)
            min2 = np.min(proj2)
            # Check for gap
            if max1 < min2 - tol or max2 < min1 - tol:
                return True  # Disjoint
    return False  # Not disjoint (intersecting)

def point_in_polygon(pt, verts):
    """Check if a point is inside or on the boundary of a convex polygon."""
    # Use cross product test for convex polygon
    n = len(verts)
    for i in range(n):
        p1 = verts[i]
        p2 = verts[(i + 1) % n]
        edge = p2 - p1
        pt_vec = pt - p1
        cross = edge[0] * pt_vec[1] - edge[1] * pt_vec[0]
        if cross < -1e-9:  # Outside
            return False
    return True

def check_configuration(inner, outer_center, outer_side, outer_angle):
    """
    Check if the configuration is valid:
    - All inner hexagons are inside the outer hexagon
    - All inner hexagons are pairwise disjoint
    Returns True if valid, False otherwise.
    Also returns the maximum violation (0 if valid).
    """
    outer_verts = hexagon_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    # Get all inner vertices
    inner_verts = []
    for x, y, ang in inner:
        v = hexagon_vertices(x, y, 1.0, ang)
        inner_verts.append(v)
    
    # Check containment
    for v in inner_verts:
        for pt in v:
            if not point_in_polygon(pt, outer_verts):
                return False
    
    # Check pairwise disjointness
    for i in range(len(inner_verts)):
        for j in range(i + 1, len(inner_verts)):
            if not polygons_disjoint(inner_verts[i], inner_verts[j]):
                return False
    
    return True

def objective(params):
    """
    Objective function to minimize outer side length.
    params: [x1, y1, a1, x2, y2, a2, ..., x12, y12, a12, cx, cy, cside]
    where a is angle in radians
    """
    inner = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i + 1]
        a = params[3*i + 2]
        inner.append([x, y, np.degrees(a)])
    
    cx = params[3*N]
    cy = params[3*N + 1]
    side = params[3*N + 2]
    
    # Enforce minimum side length
    side = max(side, 1.0)
    
    if not check_configuration(inner, (cx, cy), side, 0.0):
        return 1e6
    
    return side

def construct():
    deadline = time.time() + 140.0  # Safety margin
    
    # Initial guess: arrange hexagons in a grid pattern
    # Hexagonal grid spacing: for unit hexagons, center-to-center distance is sqrt(3) for adjacent in same row
    # For hexagonal packing, the pattern is more complex
    
    # Let's start with a simple linear arrangement and use optimization
    # Actually, for 12 hexagons, a hexagonal arrangement would be best
    # Hex numbers: 1, 7, 19, 37... So 12 is between 7 and 19
    # We can do a central hexagon with 6 around it (that's 7), then add 5 more
    
    # Start with a reasonable initial configuration
    # Place hexagons in a hexagonal pattern
    positions = []
    # Center
    positions.append([0, 0, 0.0])
    # 6 around center at distance sqrt(3)
    for i in range(6):
        angle = i * np.pi / 3
        x = np.sqrt(3) * np.cos(angle)
        y = np.sqrt(3) * np.sin(angle)
        positions.append([x, y, 0.0])
    # That's 7. Need 5 more.
    # Next layer at distance 2*sqrt(3)
    for i in range(6):
        angle = i * np.pi / 3 + np.pi / 6
        x = 2 * np.sqrt(3) * np.cos(angle)
        y = 2 * np.sqrt(3) * np.sin(angle)
        if len(positions) < 12:
            positions.append([x, y, 0.0])
    
    # Initial outer hexagon
    cx, cy = 0.0, 0.0
    # Estimate initial side length
    max_dist = 0
    for x, y, _ in positions:
        max_dist = max(max_dist, np.sqrt(x*x + y*y))
    outer_side = max_dist + 2.0  # Add margin
    
    # Parameters: [x1, y1, a1, ..., x12, y12, a12, cx, cy, side]
    x0 = []
    for x, y, a in positions:
        x0.extend([x, y, np.radians(a)])
    x0.extend([cx, cy, outer_side])
    
    best_params = x0
    best_side = outer_side
    
    # Optimization
    n_params = 3 * N + 3
    bounds = [(-10, 10)] * (3 * N) + [(-10, 10), (-10, 10), (1.0, 30.0)]
    
    # Use Nelder-Mead for local optimization from multiple starting points
    for restart in range(5):
        if time.time() > deadline:
            break
        
        if restart == 0:
            start = x0
        else:
            # Perturb initial solution
            start = x0.copy()
            for i in range(3 * N):
                start[i] += np.random.normal(0, 0.5)
        
        try:
            result = minimize(objective, start, method='Nelder-Mead',
                            options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if result.fun < 1e6 and result.fun < best_side:
                best_params = result.x
                best_side = result.fun
        except:
            pass
    
    # Extract final solution
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i + 1]
        a = best_params[3*i + 2]
        inner.append([x, y, np.degrees(a)])
    
    cx = best_params[3*N]
    cy = best_params[3*N + 1]
    side = best_params[3*N + 2]
    
    # Final validation
    if not check_configuration(inner, (cx, cy), side, 0.0):
        # Fall back to a simple valid configuration
        # Linear arrangement
        inner = []
        for i in range(N):
            x = i * np.sqrt(3)
            y = 0.0
            inner.append([x, y, 0.0])
        total_len = (N - 1) * np.sqrt(3) + 2
        cx = total_len / 2
        cy = 0.0
        side = total_len / 2 + 1.5
    
    return inner, (cx, cy), side, 0.0

"""Optimized packing of 12 unit hexagons inside a regular hexagon."""
import numpy as np
from scipy.optimize import minimize
import time
import math

N = 12

# EVOLVE-BLOCK-START
def construct():
    """
    Place 12 disjoint unit regular hexagons inside a regular hexagon,
    minimizing the outer side length.
    
    Returns: (inner, center, side, angle)
      inner: list of [x, y, angle_degrees] for each inner hexagon
      center: (x, y) of outer hexagon
      side: outer hexagon side length
      angle: outer hexagon rotation (degrees)
    """
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Precompute unit hexagon vertices (flat-top orientation, angle=0)
    # A regular hexagon with side 1 has vertices at angles 0, 60, 120, 180, 240, 300
    unit_hex_verts = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
    
    def get_hex_verts(center, angle_deg, scale=1.0):
        """Get vertices of a hexagon with given center, rotation, and scale."""
        theta = math.radians(angle_deg)
        R = np.array([[math.cos(theta), -math.sin(theta)],
                      [math.sin(theta), math.cos(theta)]])
        return center + (unit_hex_verts * scale).T @ R
    
    def outer_hex_verts(cx, cy, side, angle_deg):
        """Get vertices of the outer hexagon."""
        theta = math.radians(angle_deg)
        R = np.array([[math.cos(theta), -math.sin(theta)],
                      [math.sin(theta), math.cos(theta)]])
        # Outer hexagon vertices at distance 'side' from center
        verts = np.array([[side * math.cos(math.radians(a)), side * math.sin(math.radians(a))] for a in range(0, 360, 60)])
        return np.array([cx, cy]) + verts.T @ R
    
    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        """Check if point (px,py) is inside outer hexagon centered at (cx,cy)."""
        # Transform to local coordinates
        dx, dy = px - cx, py - cy
        theta = math.radians(angle_deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        lx = cos_t * dx + sin_t * dy
        ly = -sin_t * dx + cos_t * dy
        # For a regular hexagon with side 'side', the apothem is side * sqrt(3)/2
        # In local coords, the hexagon is defined by |lx| <= side and constraints on ly
        # Actually, let's use the half-plane method
        # The six sides of a hexagon with side s centered at origin (flat-top):
        # Normal vectors at 0, 60, 120, 180, 240, 300 degrees
        # Distance from center to each side = s * sqrt(3) / 2
        apothem = side * math.sqrt(3) / 2
        for a in range(0, 360, 60):
            nx = math.cos(math.radians(a))
            ny = math.sin(math.radians(a))
            if lx * nx + ly * ny > apothem:
                return False
        return True
    
    def hexagon_overlap_penalty(v1, v2, penalty_weight=1.0):
        """
        Compute a penalty for overlap between two hexagons.
        v1, v2 are arrays of 6 vertices each.
        Uses a soft measure: sum of squared distances from vertices of one to the other hexagon's interior.
        """
        penalty = 0.0
        # Check vertices of v1 against v2 and vice versa
        for i in range(6):
            for j in range(6):
                # Distance between vertices
                pass
        # Better: use separating axis theorem approximation with smooth penalty
        # For each edge of v1, compute the minimum distance from v2's vertices to that edge (signed)
        # This is complex. Let's use a simpler approach: check if any vertex of one is inside the other
        # and add a penalty based on penetration depth.
        
        # Simple approach: for each vertex of v1, compute signed distance to v2 (negative if inside)
        # and sum the squares of negative values
        for i in range(6):
            px, py = v1[i]
            # Check if point is inside v2 using winding or half-plane
            # For convex polygon, point is inside if it's on the same side of all edges
            inside = True
            min_dist = float('inf')
            for j in range(6):
                x1, y1 = v2[j]
                x2, y2 = v2[(j+1)%6]
                # Edge from (x1,y1) to (x2,y2)
                ex, ey = x2-x1, y2-y1
                # Normal pointing inward (for CCW polygon)
                nx, ny = -ey, ex
                # Normalize
                nlen = math.sqrt(nx*nx + ny*ny)
                if nlen == 0:
                    continue
                nx, ny = nx/nlen, ny/nlen
                # Signed distance
                d = (px-x1)*nx + (py-y1)*ny
                # For a CCW polygon, the interior is where d >= 0 (if normal points inward)
                # Let's just compute the minimum absolute distance to any edge
                # Project point onto edge
                t = ((px-x1)*ex + (py-y1)*ey) / (ex*ex + ey*ey) if (ex*ex+ey*ey) != 0 else 0
                t = max(0, min(1, t))
                qx = x1 + t*ex
                qy = y1 + t*ey
                dist = math.sqrt((px-qx)**2 + (py-qy)**2)
                min_dist = min(min_dist, dist)
            
            if min_dist < 0.01:  # Close to or on the hexagon
                penalty += penalty_weight * max(0, 0.01 - min_dist)**2
        
        # Also check v2 vertices against v1
        for i in range(6):
            px, py = v2[i]
            min_dist = float('inf')
            for j in range(6):
                x1, y1 = v1[j]
                x2, y2 = v1[(j+1)%6]
                ex, ey = x2-x1, y2-y1
                nlen2 = ex*ex + ey*ey
                if nlen2 == 0:
                    continue
                t = ((px-x1)*ex + (py-y1)*ey) / nlen2
                t = max(0, min(1, t))
                qx = x1 + t*ex
                qy = y1 + t*ey
                dist = math.sqrt((px-qx)**2 + (py-qy)**2)
                min_dist = min(min_dist, dist)
            
            if min_dist < 0.01:
                penalty += penalty_weight * max(0, 0.01 - min_dist)**2
        
        return penalty
    
    def is_valid_configuration(inner_positions, outer_center, outer_side, outer_angle):
        """Check if all inner hexagons are inside the outer hexagon and don't overlap."""
        cx, cy = outer_center
        # Check containment
        for x, y, ang in inner_positions:
            verts = get_hex_verts(np.array([x, y]), ang, 1.0)
            for vx, vy in verts:
                if not point_in_hexagon(vx, vy, cx, cy, outer_side, outer_angle):
                    return False
        
        # Check non-overlap
        for i in range(N):
            for j in range(i+1, N):
                vi = get_hex_verts(np.array([inner_positions[i][0], inner_positions[i][1]]), inner_positions[i][2], 1.0)
                vj = get_hex_verts(np.array([inner_positions[j][0], inner_positions[j][1]]), inner_positions[j][2], 1.0)
                # Check if any vertex of one is inside the other
                for k in range(6):
                    if point_in_hexagon(vi[k][0], vi[k][1], inner_positions[j][0], inner_positions[j][1], 1.0, inner_positions[j][2]):
                        return False
                    if point_in_hexagon(vj[k][0], vj[k][1], inner_positions[i][0], inner_positions[i][1], 1.0, inner_positions[i][2]):
                        return False
                # Check edge intersections (simplified)
                for k in range(6):
                    for m in range(6):
                        # Check if edges vi[k]-vi[k+1] and vj[m]-vj[m+1] intersect
                        p1, p2 = vi[k], vi[(k+1)%6]
                        p3, p4 = vj[m], vj[(m+1)%6]
                        # Line intersection test
                        d1 = (p4[0]-p3[0])*(p1[1]-p3[1]) - (p4[1]-p3[1])*(p1[0]-p3[0])
                        d2 = (p4[0]-p3[0])*(p2[1]-p3[1]) - (p4[1]-p3[1])*(p2[0]-p3[0])
                        d3 = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
                        d4 = (p2[0]-p1[0])*(p4[1]-p1[1]) - (p2[1]-p1[1])*(p4[0]-p1[0])
                        if d1*d2 < 0 and d3*d4 < 0:
                            return False
        return True
    
    def objective(params):
        """
        Objective function: outer_side + penalty for violations.
        params: [x1,y1,a1, x2,y2,a2, ..., x12,y2,a12, cx,cy, side, oangle]
        """
        if len(params) != 3*N + 4:
            return 1e10
        inner = []
        for i in range(N):
            x, y, a = params[3*i], params[3*i+1], params[3*i+2]
            inner.append([x, y, a])
        cx, cy, side, oangle = params[3*N], params[3*N+1], params[3*N+2], params[3*N+3]
        
        if side < 1.5:  # Minimum possible side
            return 1e10
        
        penalty = 0.0
        # Containment penalty
        for i in range(N):
            x, y, a = inner[i]
            verts = get_hex_verts(np.array([x, y]), a, 1.0)
            for vx, vy in verts:
                if not point_in_hexagon(vx, vy, cx, cy, side, oangle):
                    # Compute violation depth
                    dx, dy = vx - cx, vy - cy
                    theta = math.radians(oangle)
                    cos_t, sin_t = math.cos(theta), math.sin(theta)
                    lx = cos_t * dx + sin_t * dy
                    ly = -sin_t * dx + cos_t * dy
                    apothem = side * math.sqrt(3) / 2
                    for ang in range(0, 360, 60):
                        nx = math.cos(math.radians(ang))
                        ny = math.sin(math.radians(ang))
                        proj = lx * nx + ly * ny
                        if proj > apothem:
                            penalty += (proj - apothem)**2
        
        # Overlap penalty
        for i in range(N):
            for j in range(i+1, N):
                vi = get_hex_verts(np.array([inner[i][0], inner[i][1]]), inner[i][2], 1.0)
                vj = get_hex_verts(np.array([inner[j][0], inner[j][1]]), inner[j][2], 1.0)
                for k in range(6):
                    # Check if vi[k] is inside vj
                    inside = True
                    for m in range(6):
                        x1, y1 = vj[m]
                        x2, y2 = vj[(m+1)%6]
                        ex, ey = x2-x1, y2-y1
                        nx, ny = -ey, ex
                        nlen = math.sqrt(nx*nx + ny*ny)
                        if nlen == 0:
                            continue
                        nx, ny = nx/nlen, ny/nlen
                        d = (vi[k][0]-x1)*nx + (vi[k][1]-y1)*ny
                        # For CCW polygon, interior has d >= 0
                        if d < 0:
                            inside = False
                            penalty += max(0, -d)**2
                            break
                    if inside:
                        penalty += 1.0  # Vertex is inside, heavy penalty
        
        return side + penalty * 10.0
    
    # Initial guess: arrange hexagons in a grid-like pattern
    # Hexagonal packing: rows offset by sqrt(3)/2
    sqrt3 = math.sqrt(3)
    initial_inner = []
    # 4 rows of 3, or 3 rows of 4, etc.
    # Let's try a 3x4 grid with hexagonal offset
    row_spacing = sqrt3  # Distance between rows in hex packing
    col_spacing = 2.0    # Distance between columns
    
    # Try multiple initial configurations
    best_score = float('inf')
    best_params = None
    
    configs = []
    
    # Config 1: 3 rows x 4 cols
    for r in range(3):
        for c in range(4):
            x = (c - 1.5) * col_spacing + (0.5 if r % 2 else 0)
            y = (r - 1) * row_spacing
            initial_inner.append([x, y, 0.0])
    configs.append(initial_inner.copy())
    
    # Config 2: 4 rows x 3 cols
    initial_inner2 = []
    for r in range(4):
        for c in range(3):
            x = (c - 1) * col_spacing + (0.5 if r % 2 else 0)
            y = (r - 1.5) * row_spacing
            initial_inner2.append([x, y, 0.0])
    configs.append(initial_inner2.copy())
    
    # Config 3: Central + ring
    initial_inner3 = [[0, 0, 0.0]]
    for k in range(6):
        ang = math.radians(60*k)
        initial_inner3.append([2*sqrt3*math.cos(ang)/2, 2*sqrt3*math.sin(ang)/2, 0.0])
    # Add 5 more around
    for k in range(5):
        ang = math.radians(60*k + 30)
        initial_inner3.append([3*sqrt3*math.cos(ang)/2, 3*sqrt3*math.sin(ang)/2, 0.0])
    configs.append(initial_inner3)
    
    for config in configs:
        if time.time() > deadline:
            break
        
        # Build initial params
        params = []
        for x, y, a in config:
            params.extend([x, y, a])
        # Outer hexagon: center at origin, side estimated
        # Estimate side from bounding box
        xs = [p[0] for p in config]
        ys = [p[1] for p in config]
        max_x = max(abs(min(xs)), abs(max(xs)))
        max_y = max(abs(min(ys)), abs(max(ys)))
        est_side = max(max_x, max_y) / (sqrt3/2) + 1.0
        params.extend([0.0, 0.0, est_side, 0.0])
        
        # Optimize
        try:
            res = minimize(objective, params, method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8})
            if res.success or res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception as e:
            pass
    
    if best_params is None:
        # Fallback: return a valid but not optimal configuration
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Extract results
    inner_result = []
    for i in range(N):
        x, y, a = best_params[3*i], best_params[3*i+1], best_params[3*i+2]
        inner_result.append([float(x), float(y), float(a)])
    cx, cy = float(best_params[3*N]), float(best_params[3*N+1])
    side = float(best_params[3*N+2])
    oangle = float(best_params[3*N+3])
    
    # Ensure side is reasonable
    if side < 2.0:
        side = 2.0
    
    return inner_result, (cx, cy), side, oangle
# EVOLVE-BLOCK-END

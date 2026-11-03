"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    sqrt3 = math.sqrt(3)

    def get_hex_vertices(cx, cy, side, angle_deg):
        """Returns 6 vertices of a regular hexagon."""
        angle_rad = math.radians(angle_deg)
        verts = []
        for k in range(6):
            a = angle_rad + k * math.pi / 3
            verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
        return verts

    def point_in_polygon(point, polygon):
        """Ray casting algorithm to determine if a point is inside a polygon."""
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        else:
                            xinters = p1x
                        if (p1x <= xinters or p2x <= xinters) and (p1x != p2x):
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def on_segment(p, q, r):
        """Check if point q lies on segment pr."""
        if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
            return True
        return False

    def orientation(p, q, r):
        """Find orientation of triangle formed by p, q, r. 0: collinear, 1: clockwise, 2: counterclockwise"""
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if abs(val) < 1e-10:
            return 0
        return 1 if val > 0 else 2

    def do_intersect(p1, q1, p2, q2):
        """Check if segment p1-q1 intersects with segment p2-q2."""
        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)

        if o1 != o2 and o3 != o4:
            return True

        if o1 == 0 and on_segment(p1, p2, q1):
            return True
        if o2 == 0 and on_segment(p1, q2, q1):
            return True
        if o3 == 0 and on_segment(p2, p1, q2):
            return True
        if o4 == 0 and on_segment(p2, q1, q2):
            return True

        return False

    def edges_intersect(v1, v2):
        """Check if any edge of v1 intersects any edge of v2."""
        for i in range(6):
            p1 = v1[i]
            q1 = v1[(i + 1) % 6]
            for j in range(6):
                p2 = v2[j]
                q2 = v2[(j + 1) % 6]
                if do_intersect(p1, q1, p2, q2):
                    return True
        return False

    def hexes_overlap(v1, v2):
        """
        Check if two sets of vertices (hexagons) overlap.
        Returns True if they overlap (interiors intersect).
        """
        for p in v1:
            if point_in_polygon(p, v2):
                return True
        for p in v2:
            if point_in_polygon(p, v1):
                return True
        if edges_intersect(v1, v2):
            return True
        return False

    def min_outer_side(positions, outer_angle_deg):
        """Calculate the minimum side length of an outer hexagon centered at origin with given angle to contain all inner hexagons (side=1, angle=0)."""
        theta_rad = math.radians(outer_angle_deg)
        normals = []
        for k in range(6):
            normal_angle = theta_rad + math.pi/6 + k * math.pi / 3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            normals.append((nx, ny))
            
        max_required_inradius = 0.0
        
        for px, py in positions:
            # Vertices of inner hexagon (side=1, angle=0)
            for k in range(6):
                vx = px + math.cos(k * math.pi / 3)
                vy = py + math.sin(k * math.pi / 3)
                
                for nx, ny in normals:
                    proj = vx * nx + vy * ny
                    if proj > max_required_inradius:
                        max_required_inradius = proj
                        
        return 2.0 * max_required_inradius / sqrt3

    def objective(params):
        """
        Objective function: minimize outer side length + penalty for overlaps.
        params: [x0, y0, ..., x11, y11, outer_angle_deg]
        """
        positions = [(params[2*i], params[2*i+1]) for i in range(N)]
        outer_angle = params[24]
        
        side = min_outer_side(positions, outer_angle)
        
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                x1, y1 = positions[i]
                x2, y2 = positions[j]
                dx = x1 - x2
                dy = y1 - y2
                dist_sq = dx*dx + dy*dy
                
                # Quick rejection
                if dist_sq < 6.25: # 2.5^2
                    v1 = get_hex_vertices(x1, y1, 1.0, 0.0)
                    v2 = get_hex_vertices(x2, y2, 1.0, 0.0)
                    if hexes_overlap(v1, v2):
                        penalty += 10.0
                        if penalty > 50.0:
                            return side + penalty

        return side + penalty

    def is_valid(params):
        """Check if configuration is valid (no overlaps)."""
        positions = [(params[2*i], params[2*i+1]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                x1, y1 = positions[i]
                x2, y2 = positions[j]
                dx = x1 - x2
                dy = y1 - y2
                dist_sq = dx*dx + dy*dy
                if dist_sq < 6.25:
                    v1 = get_hex_vertices(x1, y1, 1.0, 0.0)
                    v2 = get_hex_vertices(x2, y2, 1.0, 0.0)
                    if hexes_overlap(v1, v2):
                        return False
        return True

    # Initial guesses
    # 1. Honeycomb 4x3
    pos_honeycomb = []
    for row in range(4):
        y = row * sqrt3
        for col in range(3):
            if row % 2 == 0:
                x = (col - 1) * 2.0
            else:
                x = (col - 1) * 2.0 + 1.0
            pos_honeycomb.append((x, y))
    xs = [p[0] for p in pos_honeycomb]
    ys = [p[1] for p in pos_honeycomb]
    cx_off = (min(xs) + max(xs)) / 2.0
    cy_off = (min(ys) + max(ys)) / 2.0
    pos_honeycomb = [(x - cx_off, y - cy_off) for x, y in pos_honeycomb]
    
    init_honeycomb = []
    for x, y in pos_honeycomb:
        init_honeycomb.extend([x, y])
    init_honeycomb.append(0.0) # angle
    
    # 2. Ring 12
    pos_ring = []
    R = 3.5
    for i in range(12):
        angle = i * math.pi / 6
        x = R * math.cos(angle)
        y = R * math.sin(angle)
        pos_ring.append((x, y))
        
    init_ring = []
    for x, y in pos_ring:
        init_ring.extend([x, y])
    init_ring.append(0.0)
    
    # 3. Random
    np.random.seed(42)
    init_random = []
    for _ in range(N):
        init_random.extend([np.random.uniform(-4, 4), np.random.uniform(-4, 4)])
    init_random.append(0.0)
    
    best_side = float('inf')
    best_params = None
    
    initial_guesses = [init_honeycomb, init_ring, init_random]
    
    # Simulated Annealing Parameters
    T_start = 10.0
    T_end = 0.01
    alpha = 0.9995 # Cooling rate
    
    for guess in initial_guesses:
        if time.time() > deadline:
            break
            
        # Evaluate initial guess
        val = objective(guess)
        if val < best_side:
            best_side = val
            best_params = guess[:]
            
        # Simulated Annealing
        current_params = guess[:]
        current_val = val
        T = T_start
        
        while T > T_end and time.time() < deadline:
            # Perturb
            new_params = current_params[:]
            # Perturb positions
            for i in range(N):
                new_params[2*i] += np.random.normal(0, T * 0.5)
                new_params[2*i+1] += np.random.normal(0, T * 0.5)
            # Perturb angle
            new_params[24] += np.random.normal(0, T * 5.0)
            
            new_val = objective(new_params)
            
            # Metropolis criterion
            delta_E = new_val - current_val
            if delta_E < 0 or np.random.rand() < math.exp(-delta_E / T):
                current_params = new_params
                current_val = new_val
                if current_val < best_side:
                    best_side = current_val
                    best_params = current_params[:]
            
            T *= alpha
            
        # After SA, try a local polish with Nelder-Mead if time permits
        if time.time() < deadline - 10:
            try:
                from scipy.optimize import minimize
                res = minimize(
                    objective,
                    best_params,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5}
                )
                if res.fun < best_side:
                    best_side = res.fun
                    best_params = res.x[:]
            except Exception:
                pass

    if best_params is None:
        best_params = init_honeycomb

    # Extract final answer
    inner = []
    for i in range(N):
        ix = best_params[2*i]
        iy = best_params[2*i+1]
        inner.append([ix, iy, 0.0]) # Fixed angle 0
        
    outer_cx = 0.0
    outer_cy = 0.0
    positions = [(best_params[2*i], best_params[2*i+1]) for i in range(N)]
    outer_angle = best_params[24]
    outer_side = min_outer_side(positions, outer_angle)
    
    # Validate
    if not is_valid(best_params):
        # Fallback to honeycomb
        best_params = init_honeycomb
        positions = [(best_params[2*i], best_params[2*i+1]) for i in range(N)]
        outer_angle = best_params[24]
        outer_side = min_outer_side(positions, outer_angle)
        inner = [[best_params[2*i], best_params[2*i+1], 0.0] for i in range(N)]
        return inner, (0.0, 0.0), outer_side, outer_angle

    return inner, (outer_cx, outer_cy), outer_side, outer_angle
# EVOLVE-BLOCK-END

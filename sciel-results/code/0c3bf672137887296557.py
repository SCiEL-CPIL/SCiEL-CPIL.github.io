"""Packing 12 unit regular hexagons inside a regular hexagon, minimizing outer side length."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12
DEADLINE = 110.0  # safety margin before 120s limit

def hex_vertices(cx, cy, s, theta):
    """Vertices of a regular hexagon with center (cx,cy), side s, rotation theta (radians)."""
    angles = theta + np.pi * np.arange(6) / 3.0
    return np.column_stack([cx + s * np.cos(angles),
                            cy + s * np.sin(angles)])

def point_in_hexagon(px, py, cx, cy, s, theta):
    """Check if point (px,py) is inside hexagon with center (cx,cy), side s, rotation theta."""
    # Transform point into hexagon's local frame
    dx = px - cx
    dy = py - cy
    # Rotate by -theta
    ct = np.cos(-theta)
    st = np.sin(-theta)
    lx = ct * dx - st * dy
    ly = st * dx + ct * dy
    # In local frame, hexagon has vertices at angles 0, 60, ..., 300 degrees
    # The hexagon is defined by 6 half-planes. For a regular hexagon with side s,
    # the apothem (distance from center to edge midpoint) is s * sqrt(3)/2
    apothem = s * np.sqrt(3.0) / 2.0
    # The 6 edge normals are at angles 30, 90, 150, 210, 270, 330 degrees
    for k in range(6):
        ang = np.pi / 6.0 + k * np.pi / 3.0
        nx = np.cos(ang)
        ny = np.sin(ang)
        if lx * nx + ly * ny > apothem:
            return False
    return True

def min_distance_to_hexagon_edge(px, py, cx, cy, s, theta):
    """Return signed distance: positive if inside, negative if outside.
    For a regular hexagon, the distance to the nearest edge."""
    dx = px - cx
    dy = py - cy
    ct = np.cos(-theta)
    st = np.sin(-theta)
    lx = ct * dx - st * dy
    ly = st * dx + ct * dy
    apothem = s * np.sqrt(3.0) / 2.0
    min_dist = np.inf
    for k in range(6):
        ang = np.pi / 6.0 + k * np.pi / 3.0
        nx = np.cos(ang)
        ny = np.sin(ang)
        d = lx * nx + ly * ny - apothem
        if d < min_dist:
            min_dist = d
    return -min_dist  # positive means inside

def check_inner_disjoint(inner, s_inner=1.0, tol=1e-6):
    """Check that all pairs of inner hexagons are disjoint (no overlap).
    inner: list of [x, y, angle_degrees]
    Returns True if disjoint."""
    n = len(inner)
    for i in range(n):
        xi, yi, ai = inner[i][0], inner[i][1], np.radians(inner[i][2])
        for j in range(i + 1, n):
            xj, yj, aj = inner[j][0], inner[j][1], np.radians(inner[j][2])
            # Check if any vertex of i is inside j, or any vertex of j is inside i
            vi = hex_vertices(xi, yi, s_inner, ai)
            vj = hex_vertices(xj, yj, s_inner, aj)
            for vx, vy in vi:
                if point_in_hexagon(vx, vy, xj, yj, s_inner, aj):
                    return False
            for vx, vy in vj:
                if point_in_hexagon(vx, vy, xi, yi, s_inner, ai):
                    return False
    return True

def check_all_inside(inner, outer_center, outer_side, outer_angle, s_inner=1.0):
    """Check that all vertices of all inner hexagons are inside the outer hexagon."""
    ocx, ocy = outer_center
    oang = np.radians(outer_angle)
    for i in range(len(inner)):
        xi, yi, ai = inner[i][0], inner[i][1], np.radians(inner[i][2])
        vi = hex_vertices(xi, yi, s_inner, ai)
        for vx, vy in vi:
            if not point_in_hexagon(vx, vy, ocx, ocy, outer_side, oang):
                return False
    return True

def is_valid(inner, outer_center, outer_side, outer_angle, s_inner=1.0):
    """Full validity check."""
    if not check_inner_disjoint(inner, s_inner):
        return False
    if not check_all_inside(inner, outer_center, outer_side, outer_angle, s_inner):
        return False
    return True

def construct():
    start_time = time.time()
    best_side = 40.0
    best_result = None
    
    # Initial best: the seed program's result
    seed_inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    seed_outer = (0.0, 0.0, 40.0, 0.0)
    if is_valid(seed_inner, (0.0, 0.0), 40.0, 0.0):
        best_side = 40.0
        best_result = (seed_inner, (0.0, 0.0), 40.0, 0.0)
    
    # Strategy: Use a grid-based initial configuration and optimize
    # The optimal packing likely has some symmetry. Let's try to find a good
    # initial configuration and then use SLSQP to minimize the outer side.
    
    # Key insight: For 12 hexagons in a hexagonal container, a natural arrangement
    # is a hexagonal number pattern. 12 = 1 + 6 + 5 (not quite a perfect hex number)
    # Actually, hex numbers are 1, 7, 19, 37... 
    # 12 hexagons: perhaps 3 rows of 4, or some other arrangement.
    
    # Let's try a systematic approach:
    # 1. Generate several candidate initial configurations
    # 2. For each, use SLSQP to minimize the outer hexagon side length
    # 3. Keep the best valid result
    
    # Candidate 1: Linear arrangement (seed)
    # Candidate 2: 3x4 grid
    # Candidate 3: Hexagonal cluster
    # Candidate 4: Various symmetric arrangements
    
    def try_configuration(inner_init, outer_side_init, outer_angle_init=0.0):
        """Try to optimize from a given initial configuration."""
        nonlocal best_side, best_result, start_time
        
        if time.time() - start_time > DEADLINE - 30:
            return
        
        n = len(inner_init)
        
        # Parameters: x, y, angle for each inner hexagon (3n params)
        # Plus outer center (2), outer side (1), outer angle (1) = 3n + 4 params
        # But we can fix outer center at origin and outer angle at 0 by symmetry
        # So: 3n + 1 params (just outer side)
        
        # Actually, let's allow the outer to move and rotate for generality
        # But fixing center at (0,0) and angle at 0 saves 3 params
        
        # Let's fix outer center at (0,0) and outer angle at 0
        # Params: for each inner hex: x, y, angle_deg -> 3n params
        # Plus outer_side -> 1 param
        # Total: 3n + 1 = 37 params for n=12
        
        def objective(params):
            if time.time() - start_time > DEADLINE - 5:
                return 1e10
            outer_side = params[-1]
            if outer_side <= 0:
                return 1e10
            inner_x = params[0:n]
            inner_y = params[n:2*n]
            inner_angle = params[2*n:3*n]
            
            # Build inner list
            inner = []
            for i in range(n):
                inner.append([inner_x[i], inner_y[i], inner_angle[i]])
            
            # Check if all inner hexagons fit inside outer
            # And are disjoint
            # For the objective, we need to handle infeasibility
            # Use a penalty approach
            
            # Check containment: all vertices of inner must be inside outer
            penalty = 0.0
            oang = 0.0
            for i in range(n):
                xi, yi, ai = inner_x[i], inner_y[i], np.radians(inner_angle[i])
                vi = hex_vertices(xi, yi, 1.0, ai)
                for vx, vy in vi:
                    if not point_in_hexagon(vx, vy, 0.0, 0.0, outer_side, oang):
                        # Penalty for being outside
                        dist = -min_distance_to_hexagon_edge(vx, vy, 0.0, 0.0, outer_side, oang)
                        penalty += max(0, dist) ** 2
            
            # Check disjointness: no two inner hexagons overlap
            # Approximate: check distance between centers vs minimum separation
            # For unit hexagons, minimum center distance to be disjoint is 2 * apothem = sqrt(3)
            # But this depends on relative angles. Use a simpler check.
            for i in range(n):
                for j in range(i + 1, n):
                    xi, yi, ai = inner_x[i], inner_y[i], np.radians(inner_angle[i])
                    xj, yj, aj = inner_x[j], inner_y[j], np.radians(inner_angle[j])
                    # Check if any vertex of i is inside j
                    vi = hex_vertices(xi, yi, 1.0, ai)
                    for vx, vy in vi:
                        if point_in_hexagon(vx, vy, xj, yj, 1.0, aj):
                            dist = -min_distance_to_hexagon_edge(vx, vy, xj, yj, 1.0, aj)
                            penalty += max(0, dist) ** 2
                    vj = hex_vertices(xj, yj, 1.0, aj)
                    for vx, vy in vj:
                        if point_in_hexagon(vx, vy, xi, yi, 1.0, ai):
                            dist = -min_distance_to_hexagon_edge(vx, vy, xi, yi, 1.0, ai)
                            penalty += max(0, dist) ** 2
            
            return outer_side + 100.0 * penalty
        
        # Initial params
        inner_x = np.array([inner_init[i][0] for i in range(n)])
        inner_y = np.array([inner_init[i][1] for i in range(n)])
        inner_angle = np.array([inner_init[i][2] for i in range(n)])
        params0 = np.concatenate([inner_x, inner_y, inner_angle, [outer_side_init]])
        
        # Bounds
        # Inner positions: within some reasonable range
        # Angles: -180 to 180
        # Outer side: 2 to 50
        bounds = []
        for i in range(n):
            bounds.append((-20, 20))  # x
        for i in range(n):
            bounds.append((-20, 20))  # y
        for i in range(n):
            bounds.append((-180, 180))  # angle
        bounds.append((2.0, 50.0))  # outer side
        
        try:
            result = minimize(objective, params0, method='SLSQP', bounds=bounds,
                              options={'maxiter': 200, 'ftol': 1e-8})
            
            if time.time() - start_time > DEADLINE - 10:
                return
            
            if result.success or result.fun < best_side:
                params_opt = result.x
                outer_side_opt = params_opt[-1]
                inner_opt_x = params_opt[0:n]
                inner_opt_y = params_opt[n:2*n]
                inner_opt_angle = params_opt[2*n:3*n]
                
                inner_opt = []
                for i in range(n):
                    inner_opt.append([float(inner_opt_x[i]), float(inner_opt_y[i]), float(inner_opt_angle[i])])
                
                # Verify validity
                if is_valid(inner_opt, (0.0, 0.0), float(outer_side_opt), 0.0):
                    if float(outer_side_opt) < best_side:
                        best_side = float(outer_side_opt)
                        best_result = (inner_opt, (0.0, 0.0), float(outer_side_opt), 0.0)
        except Exception:
            pass
    
    # Generate initial configurations
    
    # Config 1: Seed (linear)
    inner1 = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    try_configuration(inner1, 40.0)
    
    # Config 2: 3 rows of 4, hexagonal lattice
    # In a hexagonal lattice, hexagons of side 1 have center distance sqrt(3)
    # Let's place them in a 3x4 grid with hexagonal spacing
    inner2 = []
    for row in range(3):
        for col in range(4):
            x = col * np.sqrt(3) + (row % 2) * np.sqrt(3) / 2
            y = row * 1.5
            inner2.append([x, y, 0.0])
    # Center the configuration
    cx = np.mean([p[0] for p in inner2])
    cy = np.mean([p[1] for p in inner2])
    inner2 = [[p[0] - cx, p[1] - cy, p[2]] for p in inner2]
    try_configuration(inner2, 10.0)
    
    # Config 3: Hexagonal cluster (center + ring of 6 + another 5)
    inner3 = []
    # Center
    inner3.append([0.0, 0.0, 0.0])
    # Ring of 6 at distance 2*apothem = sqrt(3)
    for k in range(6):
        ang = np.pi / 3 * k
        x = np.sqrt(3) * np.cos(ang)
        y = np.sqrt(3) * np.sin(ang)
        inner3.append([x, y, 0.0])
    # 5 more: place them around the outer ring
    for k in range(5):
        ang = np.pi / 3 * k + np.pi / 6
        r = 2 * np.sqrt(3)
        x = r * np.cos(ang)
        y = r * np.sin(ang)
        inner3.append([x, y, 0.0])
    try_configuration(inner3, 12.0)
    
    # Config 4: Two rows of 6
    inner4 = []
    for row in range(2):
        for col in range(6):
            x = col * np.sqrt(3)
            y = row * 1.5
            inner4.append([x, y, 0.0])
    cx = np.mean([p[0] for p in inner4])
    cy = np.mean([p[1] for p in inner4])
    inner4 = [[p[0] - cx, p[1] - cy, p[2]] for p in inner4]
    try_configuration(inner4, 10.0)
    
    # Config 5: 4 rows of 3
    inner5 = []
    for row in range(4):
        for col in range(3):
            x = col * np.sqrt(3) + (row % 2) * np.sqrt(3) / 2
            y = row * 1.5
            inner5.append([x, y, 0.0])
    cx = np.mean([p[0] for p in inner5])
    cy = np.mean([p[1] for p in inner5])
    inner5 = [[p[0] - cx, p[1] - cy, p[2]] for p in inner5]
    try_configuration(inner5, 10.0)
    
    # Config 6: Random perturbations of config 3
    np.random.seed(42)
    for trial in range(5):
        if time.time() - start_time > DEADLINE - 20:
            break
        inner6 = []
        for p in inner3:
            x = p[0] + np.random.normal(0, 0.5)
            y = p[1] + np.random.normal(0, 0.5)
            a = p[2] + np.random.normal(0, 10)
            inner6.append([x, y, a])
        try_configuration(inner6, 12.0)
    
    if best_result is not None:
        return best_result
    else:
        # Fallback to seed
        return (seed_inner, (0.0, 0.0), 40.0, 0.0)

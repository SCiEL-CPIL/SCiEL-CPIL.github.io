import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Generate a set of diverse initial configurations using random hexagonal lattice perturbations
    # and some hand-crafted topologies.
    # We will represent configurations as a list of (x, y, angle) for each of the N hexagons.
    # The outer hexagon is defined by center (cx, cy), side length S, and rotation angle.
    # We optimize S by finding the minimum enclosing regular hexagon for the given set of inner hexagons.

    def hex_vertices(cx, cy, side, angle_deg):
        """Return the 6 vertices of a regular hexagon."""
        rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = rad + i * math.pi / 3.0
            # The apothem is side * sqrt(3)/2. The distance from center to vertex is side.
            # Wait, for a regular hexagon, side length = radius.
            # So vertices are at distance 'side' from center.
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            verts.append((x, y))
        return verts

    def is_point_in_hex(px, py, hx, hy, side, angle_deg):
        """Check if point (px, py) is inside hexagon centered at (hx, hy)."""
        # Transform to hexagon local coordinates
        rad = math.radians(angle_deg)
        cos_a = math.cos(-rad)
        sin_a = math.sin(-rad)
        dx = px - hx
        dy = py - hy
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        
        # In local coords, hexagon is symmetric.
        # Check against the 6 half-planes defined by the sides.
        # The sides are at distance 'apothem' from the center.
        apothem = side * math.sqrt(3) / 2.0
        for i in range(6):
            theta = i * math.pi / 3.0 + math.pi / 6.0 # Normal angle
            nx = math.cos(theta)
            ny = math.sin(theta)
            proj = lx * nx + ly * ny
            if proj > apothem + 1e-9:
                return False
        return True

    def do_hexes_overlap(h1, h2):
        """Check if two hexagons overlap. h = (cx, cy, angle). Side is 1.0."""
        x1, y1, a1 = h1
        x2, y2, a2 = h2
        
        # Quick distance check
        dx = x1 - x2
        dy = y1 - y2
        dist2 = dx*dx + dy*dy
        max_dist = 2.0 # Max distance between centers for overlap is 2.0 (touching)
        if dist2 > (2.0 + 1e-6)**2:
            return False
            
        # Check if any vertex of h1 is inside h2 or vice versa
        v1 = hex_vertices(x1, y1, 1.0, a1)
        for px, py in v1:
            if is_point_in_hex(px, py, x2, y2, 1.0, a2):
                return True
        v2 = hex_vertices(x2, y2, 1.0, a2)
        for px, py in v2:
            if is_point_in_hex(px, py, x1, y1, 1.0, a1):
                return True
        return False

    def check_validity(config):
        """Check if the configuration has no overlapping inner hexagons."""
        for i in range(N):
            for j in range(i+1, N):
                if do_hexes_overlap(config[i], config[j]):
                    return False
        return True

    def min_enclosing_hexagon_size(config):
        """
        Find the minimum side length S of a regular hexagon that contains all inner hexagons.
        The outer hexagon can rotate and translate.
        We use a grid search over the outer hexagon's rotation angle and center, 
        then compute the required size.
        """
        # The configuration is fixed. We need to find min S such that there exists (cx, cy, theta)
        # where all inner hexagons are inside the outer hexagon of side S.
        
        # This is a convex optimization problem if we fix the orientation.
        # For a fixed orientation theta, the problem reduces to finding the smallest S and center (cx, cy)
        # such that for all inner hexagons, their vertices are inside the outer hexagon.
        
        # Let's iterate over possible outer hexagon orientations.
        # Due to symmetry, we only need to check theta in [0, pi/3).
        
        best_S = float('inf')
        
        # Coarse grid for outer rotation
        for theta_deg in np.linspace(0, 60, 37):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            theta = math.radians(theta_deg)
            
            # For this orientation, we need to find min S and (cx, cy).
            # This is a linear programming problem:
            # Minimize S
            # Subject to: For each inner hexagon vertex v, and each of the 6 normals n_k of the outer hex:
            #   (v_x - cx) * n_kx + (v_y - cy) * n_ky <= S * apothem_unit
            #   -(v_x - cx) * n_kx - (v_y - cy) * n_ky <= S * apothem_unit
            # Where apothem_unit = sqrt(3)/2 * S.
            # Let A = S * sqrt(3)/2.
            # Minimize A / (sqrt(3)/2) = A * 2/sqrt(3)
            # Subject to:
            #   n_k . v - n_k . c <= A
            #   -n_k . v + n_k . c <= A
            #   A >= 0
            
            # Variables: cx, cy, A
            # c_obj = [0, 0, 1] (minimize A)
            
            from scipy.optimize import linprog
            
            normals = []
            for k in range(6):
                ang = theta + k * math.pi / 3.0 + math.pi / 6.0
                normals.append((math.cos(ang), math.sin(ang)))
            
            A_ub = []
            b_ub = []
            
            # Collect all vertices of all inner hexagons
            all_verts = []
            for hx, hy, ha in config:
                verts = hex_vertices(hx, hy, 1.0, ha)
                all_verts.extend(verts)
            
            for vx, vy in all_verts:
                for nx, ny in normals:
                    # (vx - cx)nx + (vy - cy)ny <= A  => -nx*cx - ny*cy - A <= -vx*nx - vy*ny
                    A_ub.append([-nx, -ny, -1.0])
                    b_ub.append(-(vx * nx + vy * ny))
                    # -(vx - cx)nx - (vy - cy)ny <= A => nx*cx + ny*cy - A <= vx*nx + vy*ny
                    A_ub.append([nx, ny, -1.0])
                    b_ub.append(vx * nx + vy * ny)
            
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            bounds = [(None, None), (None, None), (0, None)]
            
            try:
                res = linprog(c_obj=[0, 0, 1], A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                if res.status == 0:
                    A_val = res.x[2]
                    S_val = A_val * 2.0 / math.sqrt(3)
                    if S_val < best_S:
                        best_S = S_val
            except Exception:
                pass
        
        return best_S

    def total_score(config):
        if not check_validity(config):
            return float('inf')
        return min_enclosing_hexagon_size(config)

    # Generate initial candidates
    candidates = []
    
    # 1. Hexagonal lattice packing (standard)
    # Center at (0,0)
    # Neighbors at distance 2*apothem = sqrt(3)
    # In axial coordinates, neighbors are at (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Cartesian: x = sqrt(3)*(q + r/2), y = 1.5*r
    
    def axial_to_cartesian(q, r):
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return (x, y, 0.0)
    
    # Standard hexagonal cluster of 19 (radius 2) -> take 11
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
    
    lattice_pts = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (1,1), (0,2), (-1,2)
    ]
    config1 = [axial_to_cartesian(q, r) for q, r in lattice_pts]
    candidates.append(config1)
    
    # 2. Linear chain
    config2 = []
    for i in range(N):
        x = i * math.sqrt(3)
        y = 0.0
        config2.append((x, y, 0.0))
    candidates.append(config2)
    
    # 3. Two rows
    config3 = []
    for i in range(6):
        x = i * math.sqrt(3)
        y = 0.0
        config3.append((x, y, 0.0))
    for i in range(5):
        x = i * math.sqrt(3) + math.sqrt(3)/2
        y = 1.5
        config3.append((x, y, 0.0))
    candidates.append(config3)
    
    # 4. Random perturbations of lattice
    np.random.seed(42)
    for _ in range(20):
        config = []
        for q, r in lattice_pts:
            x, y, a = axial_to_cartesian(q, r)
            # Add small noise
            x += np.random.uniform(-0.1, 0.1)
            y += np.random.uniform(-0.1, 0.1)
            a = np.random.uniform(0, 60)
            config.append((x, y, a))
        candidates.append(config)
        
    # 5. Random configurations
    for _ in range(50):
        config = []
        # Start with a random center
        cx = np.random.uniform(-5, 5)
        cy = np.random.uniform(-5, 5)
        config.append((cx, cy, np.random.uniform(0, 60)))
        for _ in range(N-1):
            # Place next hexagon near existing ones
            base = np.random.choice(config)
            angle = np.random.uniform(0, 2*math.pi)
            dist = np.random.uniform(1.8, 2.2)
            nx = base[0] + dist * math.cos(angle)
            ny = base[1] + dist * math.sin(angle)
            na = np.random.uniform(0, 60)
            config.append((nx, ny, na))
        candidates.append(config)
    
    best_config = None
    best_score = float('inf')
    
    # Evaluate all candidates
    for config in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        score = total_score(config)
        if score < best_score:
            best_score = score
            best_config = config[:]
            
    # Local optimization using Nelder-Mead
    # We optimize the positions and angles of the inner hexagons.
    # The objective is the min enclosing hexagon size, with a large penalty if overlap occurs.
    
    def objective(params):
        # Unpack params
        config = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            a = params[i*3+2]
            config.append((x, y, a))
        
        # Check overlap
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if do_hexes_overlap(config[i], config[j]):
                    # Calculate overlap depth approximation
                    dx = config[i][0] - config[j][0]
                    dy = config[i][1] - config[j][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < 2.0:
                        penalty += (2.0 - dist) ** 2 * 1000
                    else:
                        penalty += 1000
        
        if penalty > 0:
            return penalty + min_enclosing_hexagon_size(config)
        
        return min_enclosing_hexagon_size(config)
    
    if best_config is not None:
        # Flatten best_config
        init_params = []
        for x, y, a in best_config:
            init_params.extend([x, y, a])
            
        try:
            res = minimize(objective, init_params, method='Nelder-Mead',
                         options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_config = []
                for i in range(N):
                    best_config.append((res.x[i*3], res.x[i*3+1], res.x[i*3+2]))
        except Exception:
            pass
    
    # If still no good config, fallback to simple lattice
    if best_config is None or best_score == float('inf'):
        best_config = candidates[0]
        best_score = total_score(best_config)
        
    # Final calculation for output
    # We need to return:
    # inner: list of [x, y, angle_degrees]
    # center: [cx, cy] of outer hexagon
    # side: side length of outer hexagon
    # angle: rotation of outer hexagon (degrees)
    
    # We need to find the optimal outer hexagon parameters for the best_config
    # Re-run the min_enclosing_hexagon_size logic but return the parameters
    
    def get_outer_params(config):
        best_S = float('inf')
        best_cx, best_cy, best_theta = 0, 0, 0
        
        for theta_deg in np.linspace(0, 60, 37):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            theta = math.radians(theta_deg)
            
            normals = []
            for k in range(6):
                ang = theta + k * math.pi / 3.0 + math.pi / 6.0
                normals.append((math.cos(ang), math.sin(ang)))
            
            A_ub = []
            b_ub = []
            
            all_verts = []
            for hx, hy, ha in config:
                verts = hex_vertices(hx, hy, 1.0, ha)
                all_verts.extend(verts)
            
            for vx, vy in all_verts:
                for nx, ny in normals:
                    A_ub.append([-nx, -ny, -1.0])
                    b_ub.append(-(vx * nx + vy * ny))
                    A_ub.append([nx, ny, -1.0])
                    b_ub.append(vx * nx + vy * ny)
            
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            bounds = [(None, None), (None, None), (0, None)]
            
            try:
                res = linprog(c_obj=[0, 0, 1], A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                if res.status == 0:
                    A_val = res.x[2]
                    S_val = A_val * 2.0 / math.sqrt(3)
                    if S_val < best_S:
                        best_S = S_val
                        best_cx = res.x[0]
                        best_cy = res.x[1]
                        best_theta = theta_deg
            except Exception:
                pass
        
        return best_cx, best_cy, best_S, best_theta
    
    cx, cy, side, angle = get_outer_params(best_config)
    
    inner = []
    for x, y, a in best_config:
        inner.append([float(x), float(y), float(a)])
        
    return inner, [float(cx), float(cy)], float(side), float(angle)

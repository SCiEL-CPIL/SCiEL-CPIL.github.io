import math
import time
import numpy as np
from scipy.optimize import linprog

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0
_SQRT3 = math.sqrt(3)
_INVSQRT3 = 1.0 / _SQRT3

def _hex_vertices(cx, cy, angle_deg):
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _min_enclosing_hex_side_lp(verts, outer_angle_deg):
    """
    Find the minimum side length of a regular hexagon with given angle 
    that encloses all points in 'verts'.
    Returns (center_x, center_y, side_length).
    """
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    M = len(verts)
    A_rows = []
    b_vals = []
    
    # Constraints: 
    # For each vertex v and each normal n_k:
    # -n_k . (c - v) <= r  => -n_k . c + n_k . v <= r
    #  n_k . (c - v) <= r  =>  n_k . c - n_k . v <= r
    # This can be rewritten as:
    # -n_k . c - r <= -n_k . v
    #  n_k . c - r <=  n_k . v
    
    for i in range(M):
        vi = verts[i]
        for k in range(6):
            nk = normals[k]
            # -nk[0]*cx - nk[1]*cy - r <= -nk[0]*vi[0] - nk[1]*vi[1]
            A_rows.append([-nk[0], -nk[1], -1.0])
            b_vals.append(-nk[0]*vi[0] - nk[1]*vi[1])
            
            # nk[0]*cx + nk[1]*cy - r <= nk[0]*vi[0] + nk[1]*vi[1]
            A_rows.append([nk[0], nk[1], -1.0])
            b_vals.append(nk[0]*vi[0] + nk[1]*vi[1])
    
    A = np.array(A_rows)
    b = np.array(b_vals)
    c_obj = np.array([0.0, 0.0, 1.0])
    bounds = [(None, None), (None, None), (0, None)]
    
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if res.status == 0:
            cx, cy, r = res.x
            side = r * 2.0 / _SQRT3
            return cx, cy, side
    except Exception:
        pass
    
    # Fallback: Centroid based
    cx, cy = np.mean(verts, axis=0)
    rel = verts - np.array([cx, cy])
    projections = rel @ normals.T
    max_proj = np.max(projections, axis=0)
    min_proj = np.min(projections, axis=0)
    r = np.max(np.maximum(max_proj, -min_proj))
    if r < 0: r = 0
    side = r * 2.0 / _SQRT3
    return cx, cy, side

def _hex_distance_exact(p1, a1, p2, a2):
    """
    Calculate the minimum distance between two regular hexagons of side 1.
    p1, p2: centers (np.array)
    a1, a2: angles in radians
    Returns distance (positive if disjoint, negative if overlapping).
    """
    d = np.linalg.norm(p1 - p2)
    if d < 1e-10:
        return 0.0
    
    dir_vec = (p2 - p1) / d
    theta = math.atan2(dir_vec[1], dir_vec[0])
    
    def support(a, theta):
        # Support function of a unit regular hexagon with angle a
        # in direction theta
        # Vertices are at angles a + k*pi/3
        # Distance from center to edge in direction theta is 1/cos(theta - vertex_angle) ? No.
        # The hexagon is the intersection of 6 half-planes.
        # Distance from center to boundary in direction theta is 1 / max_k(cos(theta - (a + k*pi/3))) ?
        # Actually, for a regular hexagon with side 1, the distance from center to vertex is 1.
        # The distance from center to edge midpoint is sqrt(3)/2.
        # The support function h(u) = max_{v in P} u . v.
        # For a regular hexagon centered at origin with one vertex at angle a:
        # The vertices are v_k = (cos(a + k*pi/3), sin(a + k*pi/3)).
        # h(theta) = max_k cos(theta - (a + k*pi/3)).
        vals = np.cos(a + np.arange(6) * math.pi / 3.0 - theta)
        return np.max(vals)
    
    h1 = support(a1, theta)
    h2 = support(a2, theta + math.pi)
    
    return d - h1 - h2

def _check_validity(inner, center, side, outer_angle):
    """
    Verify that the configuration is valid (no overlaps, inside outer hex).
    """
    # Check overlaps
    for i in range(N):
        for j in range(i+1, N):
            p1 = np.array([inner[i][0], inner[i][1]])
            a1 = math.radians(inner[i][2])
            p2 = np.array([inner[j][0], inner[j][1]])
            a2 = math.radians(inner[j][2])
            d = _hex_distance_exact(p1, a1, p2, a2)
            if d < -1e-6:  # Overlap tolerance
                return False
    
    # Check inside outer hexagon
    outer_verts = _hex_vertices(center[0], center[1], outer_angle)
    # Use LP to check if all inner vertices are inside the outer hex defined by center, side, angle
    # Or simpler: check if the LP returns a side >= the current side? 
    # Actually, the LP finds the MINIMUM enclosing hex. 
    # If we construct the outer hex with the given side and center, we need to ensure 
    # all inner vertices are inside it.
    
    # Let's just re-run the LP to get the minimum side. If min_side > side + eps, it's invalid.
    all_vertices = np.vstack([_hex_vertices(px, py, a) for px, py, a in inner])
    min_cx, min_cy, min_side = _min_enclosing_hex_side_lp(all_vertices, outer_angle)
    if min_side > side + 1e-4:
        return False
        
    return True

def _generate_initial_layouts():
    """
    Generate a set of diverse initial layouts.
    """
    layouts = []
    
    # Layout 0: Hexagonal cluster (Center + 6 + 4)
    # Center at (0,0)
    # Ring 1: distance sqrt(3)
    # Ring 2: distance 2*sqrt(3)
    pos = [[0.0, 0.0, 0.0]]
    for k in range(6):
        ang = k * math.pi / 3
        x = _SQRT3 * math.cos(ang)
        y = _SQRT3 * math.sin(ang)
        pos.append([x, y, 0.0])
    # Add 4 more in the gaps of Ring 2? 
    # The vertices of Ring 1 are at distance sqrt(3).
    # The "corners" of the hexagon formed by Ring 1 are at distance 2*sqrt(3) * cos(30)? No.
    # Let's just place them at distance 2*sqrt(3) at angles 0, 60, 120, 180.
    for k in range(4):
        ang = k * math.pi / 2 # 0, 90, 180, 270
        # Wait, hex symmetry is 60 deg. 
        # Let's try angles 0, 60, 120, 180 (4 points)
        ang = k * math.pi / 3
        x = 2.0 * _SQRT3 * math.cos(ang)
        y = 2.0 * _SQRT3 * math.sin(ang)
        pos.append([x, y, 0.0])
    layouts.append(pos[:N])
    
    # Layout 1: Linear-ish
    pos = []
    for i in range(N):
        pos.append([i * _SQRT3, 0.0, 0.0])
    layouts.append(pos)
    
    # Layout 2: Two rows
    pos = []
    # Row 1: 6 hexes
    for i in range(6):
        pos.append([i * _SQRT3, 0.0, 0.0])
    # Row 2: 5 hexes, offset
    for i in range(5):
        pos.append([i * _SQRT3 + _SQRT3/2, 1.5, 0.0])
    layouts.append(pos[:N])
    
    # Layout 3: Random compact
    np.random.seed(42)
    pos = []
    for i in range(N):
        ang = np.random.uniform(0, 2*math.pi)
        r = np.random.uniform(0, 3)
        x = r * math.cos(ang)
        y = r * math.sin(ang)
        pos.append([x, y, np.random.uniform(0, 60)])
    layouts.append(pos)
    
    # Layout 4: Random compact (different seed)
    np.random.seed(123)
    pos = []
    for i in range(N):
        ang = np.random.uniform(0, 2*math.pi)
        r = np.random.uniform(0, 3)
        x = r * math.cos(ang)
        y = r * math.sin(ang)
        pos.append([x, y, np.random.uniform(0, 60)])
    layouts.append(pos)
        
    return layouts

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        # Emergency fallback
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    layouts = _generate_initial_layouts()
    
    # For each layout, run a local search
    for layout_idx, init_pos in enumerate(layouts):
        if time.time() > _DEADLINE - 10.0:
            break
            
        current_inner = [list(x) for x in init_pos]
        
        # Initial evaluation: find best outer angle
        # We sweep the outer angle from 0 to 60
        best_local_side = float('inf')
        best_local_center = (0.0, 0.0)
        best_local_angle = 0.0
        
        # Coarse sweep
        for angle_deg in np.arange(0, 60, 5.0):
            all_vertices = np.vstack([_hex_vertices(px, py, a) for px, py, a in current_inner])
            cx, cy, side = _min_enclosing_hex_side_lp(all_vertices, angle_deg)
            if side < best_local_side:
                best_local_side = side
                best_local_center = (cx, cy)
                best_local_angle = angle_deg
        
        # Fine sweep around best
        for angle_deg in np.arange(best_local_angle - 5, best_local_angle + 5, 0.5):
            all_vertices = np.vstack([_hex_vertices(px, py, a) for px, py, a in current_inner])
            cx, cy, side = _min_enclosing_hex_side_lp(all_vertices, angle_deg)
            if side < best_local_side:
                best_local_side = side
                best_local_center = (cx, cy)
                best_local_angle = angle_deg
                
        current_side = best_local_side
        current_center = best_local_center
        current_outer_angle = best_local_angle
        
        # Local Search (Hill Climbing / Simulated Annealing)
        # We will try to improve the configuration
        temperature = 0.05
        min_temperature = 0.001
        cooling_rate = 0.99
        
        max_iters = 300
        for iter in range(max_iters):
            if time.time() > _DEADLINE - 5.0:
                break
                
            idx = np.random.randint(N)
            old_x, old_y, old_a = current_inner[idx]
            
            # Perturbation
            dx = np.random.normal(0, temperature)
            dy = np.random.normal(0, temperature)
            da = np.random.normal(0, temperature * 0.1) # Angle perturbation smaller
            
            new_x = old_x + dx
            new_y = old_y + dy
            new_a_deg = old_a + math.degrees(da)
            
            # Check collisions
            p1_new = np.array([new_x, new_y])
            a1_new = math.radians(new_a_deg)
            
            collision = False
            for j in range(N):
                if j == idx:
                    continue
                p2 = np.array([current_inner[j][0], current_inner[j][1]])
                a2 = math.radians(current_inner[j][2])
                
                dist = _hex_distance_exact(p1_new, a1_new, p2, a2)
                if dist < -1e-6:
                    collision = True
                    break
            
            if collision:
                continue
                
            current_inner[idx] = [new_x, new_y, new_a_deg]
            
            # Re-evaluate best outer angle (coarse + fine)
            # To save time, we can just check a few angles around the current best
            # or do a quick sweep
            new_side = float('inf')
            new_center = current_center
            
            # Quick sweep of outer angles
            for angle_deg in np.arange(current_outer_angle - 5, current_outer_angle + 5, 1.0):
                all_vertices = np.vstack([_hex_vertices(px, py, a) for px, py, a in current_inner])
                cx, cy, side = _min_enclosing_hex_side_lp(all_vertices, angle_deg)
                if side < new_side:
                    new_side = side
                    new_center = (cx, cy)
                    new_angle = angle_deg
            
            # Acceptance criterion
            delta = new_side - current_side
            if delta < 0:
                # Improvement
                current_side = new_side
                current_center = new_center
                current_outer_angle = new_angle
            else:
                # Worse, accept with probability exp(-delta/T)
                prob = math.exp(-delta / temperature)
                if np.random.random() < prob:
                    current_side = new_side
                    current_center = new_center
                    current_outer_angle = new_angle
                else:
                    # Reject, revert
                    current_inner[idx] = [old_x, old_y, old_a]
                    
            temperature *= cooling_rate
            if temperature < min_temperature:
                temperature = min_temperature
        
        # Update global best
        if current_side < best_side:
            best_side = current_side
            best_inner = current_inner
            best_center = current_center
            best_outer_angle = current_outer_angle

    if best_inner is None:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Final polish: Try to optimize outer angle more finely for the best solution
    all_vertices = np.vstack([_hex_vertices(px, py, a) for px, py, a in best_inner])
    for angle_deg in np.arange(best_outer_angle - 1, best_outer_angle + 1, 0.1):
        cx, cy, side = _min_enclosing_hex_side_lp(all_vertices, angle_deg)
        if side < best_side:
            best_side = side
            best_center = (cx, cy)
            best_outer_angle = angle_deg

    return best_inner, best_center, best_side, best_outer_angle

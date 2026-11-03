import numpy as np
from scipy.optimize import linprog, minimize
import time

N = 11

def hexagon_vertices(cx, cy, side, angle_deg=0.0):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        a = angle + i * np.pi / 3
        verts.append((cx + side * np.cos(a), cy + side * np.sin(a)))
    return np.array(verts)

def point_in_polygon(pt, poly):
    """Ray casting point in polygon test."""
    x, y = pt
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def polygon_overlap(v1, v2, eps=1e-9):
    """Check if two convex polygons overlap using separating axis theorem."""
    axes = []
    for poly in [v1, v2]:
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i + 1) % len(poly)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                axes.append(normal / norm)
    
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 - eps or max2 < min1 - eps:
            return False
    return True

def min_enclosing_hexagon(inner_hexes, time_limit=5.0):
    """Find minimum side length regular hexagon enclosing all inner hexagons using LP.
    
    For a regular hexagon with center (cx, cy), side s, and rotation theta,
    the constraint that a point (px, py) is inside is:
    For each of 6 directions k=0..5:
    dot((px-cx, py-cy), n_k) <= s * cos(pi/6)
    where n_k are the 6 outward normals at angles theta + k*pi/3.
    
    We can linearize by fixing theta and optimizing over cx, cy, s.
    Then we try multiple theta values.
    """
    start_time = time.time()
    best_s = float('inf')
    best_cx, best_cy, best_theta = 0, 0, 0
    
    # Collect all vertices of inner hexagons
    all_points = []
    for hx, hy, hangle in inner_hexes:
        verts = hexagon_vertices(hx, hy, 1.0, hangle)
        all_points.extend(verts.tolist())
    all_points = np.array(all_points)
    
    # Try multiple rotation angles for the outer hexagon
    theta_candidates = np.linspace(0, np.pi/6, 13)  # 0 to 30 degrees in 2.5 degree steps
    
    for theta in theta_candidates:
        if time.time() - start_time > time_limit:
            break
        
        # For this fixed theta, the 6 normal directions are:
        normals = []
        for k in range(6):
            a = theta + k * np.pi / 3
            normals.append(np.array([np.cos(a), np.sin(a)]))
        
        # We want to minimize s such that for all points p and all normals n_k:
        # n_k . (p - c) <= s * cos(pi/6)
        # i.e., n_k . p - n_k . c <= s * cos(pi/6)
        # Let c = (cx, cy), s is the variable.
        # Variables: [cx, cy, s]
        # Constraints: for each point p and each normal n:
        # n . p - n[0]*cx - n[1]*cy - s*cos(pi/6) <= 0
        # i.e., -n[0]*cx - n[1]*cy - cos(pi/6)*s <= -n . p
        
        n_points = len(all_points)
        n_normals = 6
        n_constraints = n_points * n_normals
        n_vars = 3  # cx, cy, s
        
        A_ub = np.zeros((n_constraints, n_vars))
        b_ub = np.zeros(n_constraints)
        
        idx = 0
        for p in all_points:
            for n in normals:
                A_ub[idx, 0] = -n[0]
                A_ub[idx, 1] = -n[1]
                A_ub[idx, 2] = -np.cos(np.pi / 6)
                b_ub[idx] = -np.dot(n, p)
                idx += 1
        
        c_obj = np.array([0, 0, 1])  # minimize s
        
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None), (0, None)], method='highs')
        
        if res.success:
            s_val = res.x[2]
            if s_val < best_s:
                best_s = s_val
                best_cx = res.x[0]
                best_cy = res.x[1]
                best_theta = theta
    
    return best_s, best_cx, best_cy, best_theta

def compute_objective(params, topology, time_limit=3.0):
    """Compute the objective: outer hexagon side length given parameters.
    
    params: [hinge_0, ..., hinge_k, global_rotation]
    topology: defines the base lattice positions and which pairs have hinges
    """
    start_time = time.time()
    
    n_params = len(params)
    global_rot = params[-1]
    hinges = params[:-1]
    
    # Compute positions of all hexagons based on topology and hinges
    # topology is a list of (base_x, base_y, base_angle, hinge_index or -1, parent_index or -1)
    positions = []
    angles = []
    
    for i, (bx, by, bangle, hinge_idx, parent_idx) in enumerate(topology):
        if parent_idx == -1:
            # Root hexagon
            x, y = bx, by
            a = bangle + global_rot
        else:
            # Child hexagon: position relative to parent
            px, py = positions[parent_idx]
            pa = angles[parent_idx]
            
            # The hinge angle modifies the relative orientation
            if hinge_idx >= 0:
                h = hinges[hinge_idx]
            else:
                h = 0.0
            
            # Base offset in parent's frame
            ox, oy = bx - 0, by - 0  # base offset from origin
            # Rotate offset by parent angle + hinge
            rot = pa + h
            x = px + ox * np.cos(rot) - oy * np.sin(rot)
            y = py + ox * np.sin(rot) + oy * np.cos(rot)
            a = pa + h + bangle
        
        positions.append((x, y))
        angles.append(a)
    
    inner_hexes = [(positions[i][0], positions[i][1], np.degrees(angles[i])) for i in range(len(positions))]
    
    # Check for overlaps (with small tolerance)
    for i in range(len(inner_hexes)):
        if time.time() - start_time > time_limit:
            return 100.0
        vi = hexagon_vertices(inner_hexes[i][0], inner_hexes[i][1], 1.0, inner_hexes[i][2])
        for j in range(i + 1, len(inner_hexes)):
            if polygon_overlap(vi, hexagon_vertices(inner_hexes[j][0], inner_hexes[j][1], 1.0, inner_hexes[j][2]), eps=-0.001):
                return 100.0
    
    # Find minimum enclosing hexagon
    s, cx, cy, theta = min_enclosing_hexagon(inner_hexes, time_limit=max(0.1, time_limit - (time.time() - start_time)))
    
    return s

# Define a hexagonal lattice topology for 11 hexagons
# Center at (0,0), 6 in first ring at distance 2 (side-to-side touching), 4 in second ring
# For regular hexagons of side 1, the distance between centers of touching hexagons is 2*sqrt(3)/2 * 2 = 2*sqrt(3)?
# Actually, for regular hexagons with side s, the distance between centers when touching edge-to-edge is 2*s*sqrt(3)/2 = s*sqrt(3).
# Wait, let me reconsider. A regular hexagon with side 1 has:
# - Distance from center to vertex: 1
# - Distance from center to edge midpoint (apothem): sqrt(3)/2
# - When two hexagons touch edge-to-edge, center distance = 2 * apothem = sqrt(3)

# So for a hexagonal lattice of touching hexagons:
# First ring: 6 hexagons at distance sqrt(3) from center, at angles 0, 60, 120, 180, 240, 300
# Second ring: hexagons at distance 2*sqrt(3) in certain directions

# Let me use a simpler topology: center + 6 surrounding + 4 more
# Base positions (in units of center-to-center distance d = sqrt(3)):
d = np.sqrt(3)

# Topology: (base_x, base_y, base_angle, hinge_idx, parent_idx)
# Index 0: center
# Index 1-6: first ring
# Index 7-10: second ring (4 hexagons)

topology = [
    (0, 0, 0, -1, -1),  # 0: center
    (d, 0, 0, 0, 0),    # 1: right
    (d/2, d*np.sqrt(3)/2, 0, 1, 0),   # 2: upper-right
    (-d/2, d*np.sqrt(3)/2, 0, 2, 0),  # 3: upper-left
    (-d, 0, 0, 3, 0),    # 4: left
    (-d/2, -d*np.sqrt(3)/2, 0, 4, 0), # 5: lower-left
    (d/2, -d*np.sqrt(3)/2, 0, 5, 0),  # 6: lower-right
    # Second ring: 4 hexagons at the "corners"
    (d*1.5, d*np.sqrt(3)/2, 0, 6, 1),   # 7
    (-d*1.5, d*np.sqrt(3)/2, 0, 7, 3),  # 8
    (-d*1.5, -d*np.sqrt(3)/2, 0, 8, 5), # 9
    (d*1.5, -d*np.sqrt(3)/2, 0, 9, 6),  # 10
]

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # Initial guess: all hinges zero, global rotation 0
    n_hinges = 10
    x0 = np.zeros(n_hinges + 1)
    
    best_result = None
    best_score = float('inf')
    
    # Multi-start optimization
    n_starts = 20
    rng = np.random.RandomState(42)
    
    for start in range(n_starts):
        if time.time() > deadline - 15:
            break
        
        if start == 0:
            x0 = np.zeros(n_hinges + 1)
        else:
            x0 = rng.uniform(-0.3, 0.3, n_hinges + 1)
            x0[-1] = rng.uniform(-np.pi/6, np.pi/6)  # global rotation
        
        # Use L-BFGS-B to minimize
        def obj(x):
            return compute_objective(x, topology, time_limit=2.0)
        
        try:
            res = minimize(obj, x0, method='L-BFGS-B', 
                          bounds=[(-0.5, 0.5)] * n_hinges + [(-np.pi/6, np.pi/6)],
                          options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6})
            
            if res.success or res.fun < best_score:
                if res.fun < best_score:
                    best_score = res.fun
                    best_result = res.x
        except Exception:
            pass
    
    # If no optimization succeeded, use initial configuration
    if best_result is None:
        best_result = np.zeros(n_hinges + 1)
        best_score = compute_objective(best_result, topology, time_limit=5.0)
    
    # Final polish: try to improve with tighter tolerance
    if time.time() < deadline - 5:
        try:
            def obj(x):
                return compute_objective(x, topology, time_limit=2.0)
            
            res = minimize(obj, best_result, method='L-BFGS-B',
                          bounds=[(-0.5, 0.5)] * n_hinges + [(-np.pi/6, np.pi/6)],
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.fun < best_score:
                best_result = res.x
                best_score = res.fun
        except Exception:
            pass
    
    # Compute final configuration
    params = best_result
    global_rot = params[-1]
    hinges = params[:-1]
    
    positions = []
    angles = []
    for i, (bx, by, bangle, hinge_idx, parent_idx) in enumerate(topology):
        if parent_idx == -1:
            x, y = bx, by
            a = bangle + global_rot
        else:
            px, py = positions[parent_idx]
            pa = angles[parent_idx]
            if hinge_idx >= 0:
                h = hinges[hinge_idx]
            else:
                h = 0.0
            ox, oy = bx, by
            rot = pa + h
            x = px + ox * np.cos(rot) - oy * np.sin(rot)
            y = py + ox * np.sin(rot) + oy * np.cos(rot)
            a = pa + h + bangle
        positions.append((x, y))
        angles.append(a)
    
    inner_hexes = [(positions[i][0], positions[i][1], np.degrees(angles[i])) for i in range(len(positions))]
    
    # Find minimum enclosing hexagon
    s, cx, cy, theta = min_enclosing_hexagon(inner_hexes, time_limit=5.0)
    
    return inner_hexes, (cx, cy), s, np.degrees(theta)

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END

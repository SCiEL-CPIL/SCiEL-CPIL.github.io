import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0
_SQRT3 = math.sqrt(3)
_INVSQRT3 = 1.0 / _SQRT3

def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _min_enclosing_hex_side_lp(verts, outer_angle_deg):
    """
    Find minimum side length of a regular hexagon (with flat-top orientation offset by outer_angle_deg)
    that contains all points in verts. Uses LP to find optimal center and inradius.
    """
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    M = len(verts)
    A_rows = []
    b_vals = []
    
    for i in range(M):
        vi = verts[i]
        for k in range(6):
            nk = normals[k]
            A_rows.append([-nk[0], -nk[1], -1.0])
            b_vals.append(-nk[0]*vi[0] - nk[1]*vi[1])
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
    
    # Fallback: centroid-based
    cx, cy = np.mean(verts, axis=0)
    rel = verts - np.array([cx, cy])
    projections = rel @ normals.T
    max_proj = np.max(projections, axis=0)
    min_proj = np.min(projections, axis=0)
    r = np.max(np.maximum(max_proj, -min_proj))
    if r < 0: r = 0
    side = r * 2.0 / _SQRT3
    return cx, cy, side

def _build_positions(layout_idx):
    """Generate 11 positions for given layout index."""
    sqrt3 = _SQRT3
    scale = 1.001
    u = (sqrt3*scale, 0.0)
    v = (sqrt3/2*scale, 1.5*scale)
    
    if layout_idx == 0:
        target = [(0,0)]
        ring1 = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
        target.extend(ring1)
        ring2 = [(2,0),(1,1),(0,2),(-1,2),(-2,1),(-2,0),(-1,-1),(0,-2),(1,-2),(2,-1)]
        ring2_sorted = sorted(ring2, key=lambda p: p[0]**2 + p[1]**2)
        target.extend(ring2_sorted[:4])
        positions = []
        for ti, tj in target:
            x = ti*u[0] + tj*v[0]
            y = ti*u[1] + tj*v[1]
            positions.append((x,y))
        return positions[:N]
    elif layout_idx == 1:
        positions = []
        y1 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-1, 3):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 2:
        positions = []
        y1 = -1.5*scale
        for k in range(-2, 2):
            x = k*sqrt3*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-2, 2):
            x = k*sqrt3*scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 3:
        positions = []
        y1 = -3.0*scale
        for k in range(0, 2):
            x = k*sqrt3*scale - sqrt3/2*scale
            positions.append((x, y1))
        y2 = -1.5*scale
        for k in range(-1, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 0.0
        for k in range(-1, 1):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        y4 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y4))
        return positions[:N]
    elif layout_idx == 4:
        positions = []
        y1 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 5:
        positions = []
        positions.append((0, 0))
        ring1 = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
        for ti, tj in ring1:
            x = ti*u[0] + tj*v[0]
            y = ti*u[1] + tj*v[1]
            positions.append((x, y))
        ring2 = [(2,0),(1,1),(-1,2),(-2,0),(-1,-1),(0,-2)]
        for ti, tj in ring2[:4]:
            x = ti*u[0] + tj*v[0]
            y = ti*u[1] + tj*v[1]
            positions.append((x, y))
        return positions[:N]
    elif layout_idx == 6:
        positions = []
        y1 = -3.0*scale
        for k in range(0, 2):
            x = k*sqrt3*scale - sqrt3/2*scale
            positions.append((x, y1))
        y2 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 0.0
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        y4 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y4))
        return positions[:N]
    elif layout_idx == 7:
        positions = []
        y1 = -1.5*scale
        for k in range(-2, 2):
            x = k*sqrt3*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 8:
        positions = []
        y1 = -0.75*scale
        for k in range(-2, 3):
            x = k*sqrt3*scale
            positions.append((x, y1))
        y2 = 0.75*scale
        for k in range(-3, 3):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y2))
        return positions[:N]
    elif layout_idx == 9:
        positions = []
        y1 = -0.75*scale
        for k in range(-3, 3):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y1))
        y2 = 0.75*scale
        for k in range(-2, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        return positions[:N]
    elif layout_idx == 10:
        positions = []
        y0 = -3.0*scale
        positions.append((0, y0))
        y1 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y3))
        y4 = 3.0*scale
        positions.append((0, y4))
        return positions[:N]
    elif layout_idx == 11:
        positions = []
        y1 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-2, 3):
            x = k*sqrt3*scale
            positions.append((x, y3))
        return positions[:N]
    else:
        coords = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i*u[0] + j*v[0]
                y = i*u[1] + j*v[1]
                coords.append((x, y))
        coords_sorted = sorted(coords, key=lambda t: t[0]**2 + t[1]**2)
        return coords_sorted[:N]

def _hex_distance(p1, a1, p2, a2):
    """Minimum distance between two unit hexagons at positions p1, p2 with angles a1, a2 (radians)."""
    # For two unit hexagons, the minimum distance between them is related to the 
    # distance between centers minus the maximum extent in that direction.
    # A simpler check: if centers are closer than 2*sqrt(3) they might overlap.
    # More precisely, for two regular hexagons of side 1, they are disjoint if
    # the distance between centers is at least 2*sqrt(3) when aligned, but this
    # depends on relative orientation.
    # 
    # Actually, for two unit hexagons, the minimum center-to-center distance for
    # them to be tangent (touching at one point) depends on the relative angle.
    # The support function of a unit hexagon in direction theta is:
    # h(theta) = max over vertices of dot(vertex, direction)
    # Two hexagons are disjoint if for all directions, the sum of support functions
    # is less than the projection of the center difference.
    # 
    # A simpler approach: compute the minimum distance between the two hexagons
    # by checking all pairs of edges, or use the fact that for regular hexagons,
    # the minimum distance is determined by the relative orientation.
    #
    # For now, use a conservative estimate: the inradius is sqrt(3)/2, so the
    # minimum center distance for non-overlap is at least 2 * inradius = sqrt(3)
    # when aligned, but can be up to 2 * circumradius = 2 when at 30 degrees.
    # Actually, the circumradius of a unit hexagon is 1, inradius is sqrt(3)/2.
    # Two hexagons are disjoint if the distance between centers is at least
    # the sum of their support functions in the direction of the center line.
    #
    # Let's use a simpler check: compute the distance between centers and compare
    # to a threshold that depends on relative angle.
    
    d = np.linalg.norm(p1 - p2)
    if d < 1e-10:
        return 0.0
    
    # Direction from p1 to p2
    dir_vec = (p2 - p1) / d
    theta = math.atan2(dir_vec[1], dir_vec[0])
    
    # Support function of unit hexagon in direction theta
    # h(theta) = max over vertices of dot(vertex, dir)
    # Vertices of unit hexagon with angle a: (cos(a + k*pi/3), sin(a + k*pi/3))
    # dot with dir = cos(a + k*pi/3 - theta)
    # max over k of cos(a + k*pi/3 - theta)
    
    def support(a, theta):
        vals = np.cos(a + np.arange(6) * math.pi / 3.0 - theta)
        return np.max(vals)
    
    h1 = support(a1, theta)
    h2 = support(a2, theta + math.pi)  # opposite direction for hexagon 2
    
    return d - h1 - h2

def _objective(params, positions, outer_angle_deg):
    """Objective: outer hexagon side length for given inner positions and angles."""
    # params: [x1, y1, a1, x2, y2, a2, ...] for N hexagons
    n = len(positions)
    new_positions = []
    new_angles = []
    for i in range(n):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        new_positions.append((x, y))
        new_angles.append(a)
    
    # Check for overlaps
    min_dist = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            dist = _hex_distance(new_positions[i], new_angles[i], new_positions[j], new_angles[j])
            if dist < min_dist:
                min_dist = dist
            if dist < -1e-6:
                # Overlap penalty
                return 1000.0 + min_dist * 100
    
    if min_dist < 0:
        return 1000.0 + min_dist * 100
    
    # Compute enclosing hexagon
    all_vertices = []
    for i in range(n):
        verts = _hex_vertices(new_positions[i][0], new_positions[i][1], math.degrees(new_angles[i]))
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    cx, cy, side = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
    return side

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    num_layouts = 12
    layouts = [_build_positions(i) for i in range(num_layouts)]
    
    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 10.0:
            break
        
        inner_angle = 30.0
        
        all_vertices = np.vstack([_hex_vertices(px, py, inner_angle) for px, py in positions])
        
        best_local_side = float('inf')
        best_local_center = (0.0, 0.0)
        best_local_angle = 0.0
        
        # Try outer angles in finer steps
        for outer_angle_deg in np.arange(0, 30.0, 0.25):
            if time.time() > _DEADLINE - 10.0:
                break
            
            res = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
            if res:
                cx, cy, side = res
                if side < best_local_side:
                    best_local_side = side
                    best_local_center = (cx, cy)
                    best_local_angle = outer_angle_deg
        
        if best_local_side < best_side:
            best_side = best_local_side
            best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
            best_center = best_local_center
            best_outer_angle = best_local_angle
    
    # Try to improve the best layout with local optimization
    if best_inner is not None and time.time() < _DEADLINE - 20.0:
        # Find which layout was best
        for layout_idx, positions in enumerate(layouts):
            if time.time() > _DEADLINE - 20.0:
                break
            
            # Check if this layout matches the best
            match = True
            for i in range(N):
                if abs(positions[i][0] - best_inner[i][0]) > 1e-6 or abs(positions[i][1] - best_inner[i][1]) > 1e-6:
                    match = False
                    break
            if not match:
                continue
            
            # Local optimization
            x0 = []
            for i in range(N):
                x0.append(positions[i][0])
                x0.append(positions[i][1])
                x0.append(math.radians(inner_angle))
            
            # Try different outer angles
            for outer_angle_deg in [best_outer_angle, best_outer_angle + 5, best_outer_angle - 5]:
                if time.time() > _DEADLINE - 20.0:
                    break
                
                try:
                    res = minimize(_objective, x0, args=(positions, outer_angle_deg),
                                  method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-8})
                    if res.fun < best_side and res.fun < 100:
                        new_positions = []
                        new_angles = []
                        for i in range(N):
                            new_positions.append((res.x[3*i], res.x[3*i+1]))
                            new_angles.append(math.degrees(res.x[3*i+2]))
                        
                        all_vertices = np.vstack([_hex_vertices(px, py, a) for px, py, a in zip(new_positions, new_angles)])
                        cx, cy, side = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
                        
                        if side < best_side:
                            best_side = side
                            best_inner = [[px, py, a] for px, py, a in zip(new_positions, new_angles)]
                            best_center = (cx, cy)
                            best_outer_angle = outer_angle_deg
                except Exception:
                    pass
            break
    
    if best_inner is None:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, best_side, best_outer_angle

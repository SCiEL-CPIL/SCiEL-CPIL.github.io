import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0
_SQRT3 = math.sqrt(3)

def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _min_enclosing_hex_side(verts, outer_angle_deg):
    """
    Find minimum side length of a regular hexagon (with vertex at outer_angle_deg)
    that contains all points in verts (numpy array of shape (M,2)).
    Returns (cx, cy, side) or None.
    """
    # Normals to the 6 sides of the outer hexagon
    # Vertices at outer_angle + 60k => edge normals at outer_angle + 30 + 60k
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    M = len(verts)
    # Variables: [cx, cy, R], minimize R
    # Constraints for each vertex v and normal n:
    #   (v - c).n <= R  =>  -n.c - R <= -v.n
    #   (c - v).n <= R  =>   n.c - R <=  v.n
    
    # Vectorized construction of A and b
    # verts: (M, 2), normals: (6, 2)
    # We need to form (M*6, 3) matrix A and (M*6,) vector b
    # For each pair (i, k):
    #   row1: [-nx_k, -ny_k, -1] <= - (v_i . n_k)
    #   row2: [nx_k, ny_k, -1] <=   (v_i . n_k)
    
    # v_dot_n: (M, 6)
    v_dot_n = verts @ normals.T
    
    # Reshape for broadcasting
    # We want to interleave or just stack them.
    # Let's create arrays for the 2 rows per (i,k)
    
    # A_ub shape: (2*M*6, 3)
    # b_ub shape: (2*M*6,)
    
    # Create indices
    # It's easier to just loop over normals and vectorize over vertices, or vice versa.
    # Since M is small (66), a simple loop over normals is fast enough and memory efficient.
    
    A_rows = []
    b_vals = []
    
    for k in range(6):
        nx, ny = normals[k]
        v_dots = v_dot_n[:, k]
        
        # Row 1: -nx*cx - ny*cy - R <= -v_dot
        A_rows.append(np.column_stack([np.full(M, -nx), np.full(M, -ny), np.full(M, -1.0)]))
        b_vals.append(-v_dots)
        
        # Row 2: nx*cx + ny*cy - R <= v_dot
        A_rows.append(np.column_stack([np.full(M, nx), np.full(M, ny), np.full(M, -1.0)]))
        b_vals.append(v_dots)
    
    A = np.vstack(A_rows)
    b = np.concatenate(b_vals)
    
    # R >= 0
    A = np.vstack([A, [[0.0, 0.0, -1.0]]])
    b = np.append(b, 0.0)
    
    c_obj = [0.0, 0.0, 1.0]
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b,
                      bounds=[(None, None), (None, None), (0, None)],
                      method='highs')
        if res.success:
            cx, cy, R = res.x
            side = R / (_SQRT3/2)
            return cx, cy, side
    except Exception:
        pass
    return None

def _build_positions(layout_idx):
    """Generate 11 positions for given layout index."""
    scale = 1.0000001
    u = (_SQRT3*scale, 0.0)
    v = (_SQRT3/2*scale, 1.5*scale)
    
    if layout_idx == 0:
        # Compact hexagonal: center + 6 ring + 4 from second ring
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
        # Row-based: 3-4-4
        positions = []
        y1 = -1.5*scale
        for k in range(-1, 2):
            x = k*_SQRT3*scale + _SQRT3/2*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 3):
            x = k*_SQRT3*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-1, 3):
            x = k*_SQRT3*scale + _SQRT3/2*scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 2:
        # 4-3-4 shifted
        positions = []
        y1 = -1.5*scale
        for k in range(-2, 2):
            x = k*_SQRT3*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 2):
            x = k*_SQRT3*scale + _SQRT3/2*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-2, 2):
            x = k*_SQRT3*scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 3:
        # 5-3-3 arrangement
        positions = []
        y1 = -1.5*scale
        for k in range(-2, 3):
            x = k*_SQRT3*scale + _SQRT3/2*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 2):
            x = k*_SQRT3*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-1, 2):
            x = k*_SQRT3*scale + _SQRT3/2*scale
            positions.append((x, y3))
        return positions[:N]
    else:
        # Fallback: sorted by distance
        coords = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i*u[0] + j*v[0]
                y = i*u[1] + j*v[1]
                coords.append((x, y))
        coords_sorted = sorted(coords, key=lambda t: t[0]**2 + t[1]**2)
        return coords_sorted[:N]


def _hex_min_distance(p1, p2, a1, a2):
    """Compute minimum distance between two unit hexagons (0 if overlapping)."""
    # For two regular hexagons, the minimum distance between their boundaries
    # can be approximated by checking vertex-to-edge distances
    v1 = _hex_vertices(p1[0], p1[1], a1)
    v2 = _hex_vertices(p2[0], p2[1], a2)
    
    min_dist = float('inf')
    for vi in v1:
        for vj in v2:
            d = np.sqrt((vi[0]-vj[0])**2 + (vi[1]-vj[1])**2)
            if d < min_dist:
                min_dist = d
    return max(0.0, min_dist - 0.0)


def _spring_mass_relax(positions, inner_angle, n_iter=50, tol=1e-8):
    """
    Relax positions using a spring-mass model to find a tighter packing.
    Each pair of hexagons has a 'spring' that pushes them apart if too close.
    """
    pos = np.array(positions, dtype=float)
    n = len(pos)
    min_dist = _SQRT3 * 0.999999  # required center-to-center distance for non-overlap
    
    for it in range(n_iter):
        if time.time() > _DEADLINE - 10.0:
            break
        forces = np.zeros_like(pos)
        for i in range(n):
            for j in range(i+1, n):
                d = pos[j] - pos[i]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    d = np.array([0.001, 0.001])
                    dist = np.linalg.norm(d)
                if dist < min_dist:
                    # Repulsive force
                    f = (min_dist - dist) / dist * 10.0
                    forces[i] -= d * f
                    forces[j] += d * f
        
        # Update positions
        pos += forces * 0.5
        
        # Check convergence
        if np.max(np.abs(forces)) < tol:
            break
    
    return [tuple(p) for p in pos]

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_inner_angle = 0.0
    
    layouts = [_build_positions(i) for i in range(5)]
    
    # Try both original and relaxed positions
    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 10.0:
            break
        
        # Generate candidate position sets
        candidate_positions = [positions]
        
        # Add relaxed version
        for inner_angle_try in [0.0, 30.0]:
            if time.time() > _DEADLINE - 10.0:
                break
            relaxed = _spring_mass_relax(positions, inner_angle_try, n_iter=30)
            candidate_positions.append(relaxed)
        
        for pos_set in candidate_positions:
            if time.time() > _DEADLINE - 10.0:
                break
            
            for inner_angle in [0.0, 30.0]:
                if time.time() > _DEADLINE - 10.0:
                    break
                
                all_vertices = []
                for px, py in pos_set:
                    all_vertices.extend(_hex_vertices(px, py, inner_angle))
                all_vertices = np.array(all_vertices)
                
                # Coarse search over outer angle
                best_coarse_side = float('inf')
                best_coarse_angle = 0.0
                best_coarse_center = (0.0, 0.0)
                
                for outer_angle_deg in np.arange(0, 30.1, 0.5):
                    if time.time() > _DEADLINE - 10.0:
                        break
                    res = _min_enclosing_hex_side(all_vertices, outer_angle_deg)
                    if res:
                        cx, cy, side = res
                        if side < best_coarse_side:
                            best_coarse_side = side
                            best_coarse_angle = outer_angle_deg
                            best_coarse_center = (cx, cy)
                
                # Fine search around best coarse angle
                if best_coarse_angle > 0:
                    for outer_angle_deg in np.arange(max(0, best_coarse_angle - 0.5),
                                                     min(30.1, best_coarse_angle + 0.5), 0.01):
                        if time.time() > _DEADLINE - 10.0:
                            break
                        res = _min_enclosing_hex_side(all_vertices, outer_angle_deg)
                        if res:
                            cx, cy, side = res
                            if side < best_coarse_side:
                                best_coarse_side = side
                                best_coarse_angle = outer_angle_deg
                                best_coarse_center = (cx, cy)
                
                if best_coarse_side < best_side:
                    best_side = best_coarse_side
                    best_inner = [[pos[0], pos[1], inner_angle] for pos in pos_set]
                    best_center = best_coarse_center
                    best_outer_angle = best_coarse_angle
                    best_inner_angle = inner_angle
    
    if best_inner is None:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, best_side, best_outer_angle

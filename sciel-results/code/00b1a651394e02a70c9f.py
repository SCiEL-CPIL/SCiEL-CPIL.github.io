import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0
_SQRT3 = math.sqrt(3)
_GAP = 1.0005

def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _hex_edge_distance(p, v1, v2, hex_center, hex_angle):
    """
    Compute distance from point p to the edge (v1,v2) of a hexagon, 
    returning signed distance (positive if outside).
    """
    # Direction along edge
    e = v2 - v1
    elen = np.linalg.norm(e)
    if elen < 1e-10:
        return np.linalg.norm(p - v1)
    e_hat = e / elen
    
    # Normal to edge (outward)
    # For a regular hexagon, the outward normal from center to edge midpoint
    mid = (v1 + v2) / 2.0
    n_vec = mid - hex_center
    n_len = np.linalg.norm(n_vec)
    if n_len < 1e-10:
        n_hat = np.array([0, 1])
    else:
        n_hat = n_vec / n_len
    
    # Signed distance from p to the line containing the edge
    # Positive means p is on the outside of the hexagon
    return np.dot(p - mid, n_hat)

def _min_distance_between_hexes(c1, a1, c2, a2):
    """
    Compute minimum distance between two unit regular hexagons.
    Returns 0 if they overlap, otherwise the minimum gap.
    """
    v1 = _hex_vertices(c1[0], c1[1], a1)
    v2 = _hex_vertices(c2[0], c2[1], a2)
    
    min_dist = float('inf')
    
    # Check each vertex of hex1 against each edge of hex2
    for vi in v1:
        for k in range(6):
            vj = v2[k]
            vj_next = v2[(k+1) % 6]
            dist = _hex_edge_distance(vi, vj, vj_next, c2, a2)
            # Only consider if the projection falls on the edge segment
            e = vj_next - vj
            elen2 = np.dot(e, e)
            if elen2 > 1e-10:
                t = np.dot(vi - vj, e) / elen2
                if 0 <= t <= 1:
                    if dist < min_dist:
                        min_dist = dist
    
    # Check each vertex of hex2 against each edge of hex1
    for vi in v2:
        for k in range(6):
            vj = v1[k]
            vj_next = v1[(k+1) % 6]
            dist = _hex_edge_distance(vi, vj, vj_next, c1, a1)
            e = vj_next - vj
            elen2 = np.dot(e, e)
            if elen2 > 1e-10:
                t = np.dot(vi - vj, e) / elen2
                if 0 <= t <= 1:
                    if dist < min_dist:
                        min_dist = dist
    
    return max(0.0, min_dist)

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
    v_dot_n = verts @ normals.T
    
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
    scale = _GAP
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

def _check_non_overlap(positions, inner_angle, tol=1e-9):
    """Check if all hexagons are non-overlapping with tolerance."""
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            c1 = positions[i]
            c2 = positions[j]
            dist = _min_distance_between_hexes(c1, inner_angle, c2, inner_angle)
            if dist < -tol:
                return False
    return True

def _optimize_positions(positions, inner_angle, n_iter=100):
    """
    Optimize positions to minimize the enclosing hexagon side length,
    subject to non-overlap constraints.
    Uses a penalty method with gradient descent.
    """
    pos = np.array(positions, dtype=float)
    n = len(pos)
    
    best_pos = pos.copy()
    best_side = float('inf')
    
    for it in range(n_iter):
        if time.time() > _DEADLINE - 15.0:
            break
        
        # Compute all vertices
        all_vertices = []
        for px, py in pos:
            all_vertices.extend(_hex_vertices(px, py, inner_angle))
        all_vertices = np.array(all_vertices)
        
        # Find best enclosing hexagon
        best_local_side = float('inf')
        best_local_angle = 0.0
        best_local_center = (0.0, 0.0)
        
        for outer_angle_deg in np.arange(0, 30.1, 1.0):
            res = _min_enclosing_hex_side(all_vertices, outer_angle_deg)
            if res:
                cx, cy, side = res
                if side < best_local_side:
                    best_local_side = side
                    best_local_angle = outer_angle_deg
                    best_local_center = (cx, cy)
        
        if best_local_side < best_side:
            best_side = best_local_side
            best_pos = pos.copy()
        
        # Compute gradient of side length w.r.t. positions
        # Approximate using finite differences
        grad = np.zeros_like(pos)
        eps = 1e-5
        for i in range(n):
            for dim in range(2):
                pos_plus = pos.copy()
                pos_plus[i, dim] += eps
                all_v_plus = []
                for px, py in pos_plus:
                    all_v_plus.extend(_hex_vertices(px, py, inner_angle))
                all_v_plus = np.array(all_v_plus)
                
                side_plus = float('inf')
                for outer_angle_deg in np.arange(best_local_angle - 1, best_local_angle + 2, 1.0):
                    if outer_angle_deg < 0 or outer_angle_deg > 30:
                        continue
                    res = _min_enclosing_hex_side(all_v_plus, outer_angle_deg)
                    if res:
                        cx, cy, side = res
                        if side < side_plus:
                            side_plus = side
                
                pos_minus = pos.copy()
                pos_minus[i, dim] -= eps
                all_v_minus = []
                for px, py in pos_minus:
                    all_v_minus.extend(_hex_vertices(px, py, inner_angle))
                all_v_minus = np.array(all_v_minus)
                
                side_minus = float('inf')
                for outer_angle_deg in np.arange(best_local_angle - 1, best_local_angle + 2, 1.0):
                    if outer_angle_deg < 0 or outer_angle_deg > 30:
                        continue
                    res = _min_enclosing_hex_side(all_v_minus, outer_angle_deg)
                    if res:
                        cx, cy, side = res
                        if side < side_minus:
                            side_minus = side
                
                grad[i, dim] = (side_plus - side_minus) / (2 * eps)
        
        # Compute non-overlap penalty gradient
        penalty_grad = np.zeros_like(pos)
        penalty = 0.0
        for i in range(n):
            for j in range(i+1, n):
                c1 = tuple(pos[i])
                c2 = tuple(pos[j])
                dist = _min_distance_between_hexes(c1, inner_angle, c2, inner_angle)
                if dist < 0:
                    # Overlapping - add penalty
                    penalty += dist * dist
                    # Approximate gradient
                    d = pos[j] - pos[i]
                    dist_c = np.linalg.norm(d)
                    if dist_c > 1e-10:
                        d_hat = d / dist_c
                        penalty_grad[i] += 2 * dist * d_hat
                        penalty_grad[j] -= 2 * dist * d_hat
        
        total_grad = grad + 10.0 * penalty_grad
        
        # Gradient descent step
        lr = 0.01 / (1 + it * 0.01)
        pos -= lr * total_grad
        
        # Project to maintain minimum distance
        for i in range(n):
            for j in range(i+1, n):
                d = pos[j] - pos[i]
                dist = np.linalg.norm(d)
                min_dist = _SQRT3 * 0.999
                if dist < min_dist and dist > 1e-10:
                    d_hat = d / dist
                    correction = (min_dist - dist) / 2.0
                    pos[i] -= correction * d_hat
                    pos[j] += correction * d_hat
    
    return [tuple(p) for p in best_pos], best_side

def _spring_mass_relax(positions, inner_angle, n_iter=50, tol=1e-8):
    """
    Relax positions using a spring-mass model to find a tighter packing.
    Each pair of hexagons has a 'spring' that pushes them apart if too close.
    """
    pos = np.array(positions, dtype=float)
    n = len(pos)
    min_dist = _SQRT3 * 0.999
    
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
        if time.time() > _DEADLINE - 15.0:
            break
        
        # Generate candidate position sets
        candidate_positions = [positions]
        
        # Add relaxed version
        for inner_angle_try in [0.0, 30.0]:
            if time.time() > _DEADLINE - 15.0:
                break
            relaxed = _spring_mass_relax(positions, inner_angle_try, n_iter=30)
            candidate_positions.append(relaxed)
        
        for pos_set in candidate_positions:
            if time.time() > _DEADLINE - 15.0:
                break
            
            for inner_angle in [0.0, 30.0]:
                if time.time() > _DEADLINE - 15.0:
                    break
                
                # First, check if current configuration is valid
                if not _check_non_overlap(pos_set, inner_angle, tol=1e-6):
                    continue
                
                all_vertices = []
                for px, py in pos_set:
                    all_vertices.extend(_hex_vertices(px, py, inner_angle))
                all_vertices = np.array(all_vertices)
                
                # Coarse search over outer angle
                best_coarse_side = float('inf')
                best_coarse_angle = 0.0
                best_coarse_center = (0.0, 0.0)
                
                for outer_angle_deg in np.arange(0, 30.1, 0.5):
                    if time.time() > _DEADLINE - 15.0:
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
                        if time.time() > _DEADLINE - 15.0:
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
                
                # Try to optimize further if time permits
                if time.time() < _DEADLINE - 20.0 and best_coarse_side < best_side + 0.1:
                    optimized_pos, opt_side = _optimize_positions(pos_set, inner_angle, n_iter=50)
                    if _check_non_overlap(optimized_pos, inner_angle, tol=1e-6):
                        if opt_side < best_side:
                            # Verify with fine angle search
                            all_v_opt = []
                            for px, py in optimized_pos:
                                all_v_opt.extend(_hex_vertices(px, py, inner_angle))
                            all_v_opt = np.array(all_v_opt)
                            
                            for outer_angle_deg in np.arange(max(0, best_outer_angle - 1),
                                                             min(30.1, best_outer_angle + 1), 0.01):
                                if time.time() > _DEADLINE - 15.0:
                                    break
                                res = _min_enclosing_hex_side(all_v_opt, outer_angle_deg)
                                if res:
                                    cx, cy, side = res
                                    if side < best_side:
                                        best_side = side
                                        best_inner = [[pos[0], pos[1], inner_angle] for pos in optimized_pos]
                                        best_center = (cx, cy)
                                        best_outer_angle = outer_angle_deg
                                        best_inner_angle = inner_angle
    
    if best_inner is None:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, best_side, best_outer_angle

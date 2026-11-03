"""
Hexagon packing via constrained gradient optimization with SDF repulsion.
Directly optimizes inner hexagon positions to minimize enclosing hexagon size.
"""
import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 118.0  # Safety margin before 120s kill

def _hex_vertices(cx, cy, side, angle_deg):
    verts = []
    for k in range(6):
        a = math.radians(angle_deg + 60 * k)
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return verts

def _sdf_hexagon(px, py, cx, cy, side, angle_deg):
    """
    Signed distance from point (px,py) to hexagon centered at (cx,cy)
    with side length `side`, rotated by angle_deg.
    Negative inside, positive outside.
    """
    dx = px - cx
    dy = py - cy
    a = -math.radians(angle_deg)
    lx = dx * math.cos(a) - dy * math.sin(a)
    ly = dx * math.sin(a) + dy * math.cos(a)
    
    apothem = side * math.sqrt(3) / 2
    # Hexagon in local coords: |lx| <= side, |ly| <= apothem, |lx|/side + |ly|/apothem <= 1
    # Actually for regular hexagon with flat top:
    # The hexagon is intersection of 6 half-planes
    # Distance to boundary = min over edges of (distance to edge line) clipped
    
    # For a regular hexagon with "pointy top" (vertex at angle 0):
    # Vertices at angles 0, 60, 120, 180, 240, 300
    # Edge normals at angles 30, 90, 150, 210, 270, 330
    # Inradius (apothem) = side * sqrt(3)/2
    
    # The SDF for a regular hexagon can be computed as:
    # d = min over 6 edges of max(0, dot(p, n_i) - apothem) for outside
    # But we need signed distance
    
    # Simpler: check if inside, then compute distance to closest edge
    inside = True
    for k in range(6):
        a_norm = math.radians(30 + 60 * k)
        nx = math.cos(a_norm)
        ny = math.sin(a_norm)
        if lx * nx + ly * ny > apothem + 1e-12:
            inside = False
            break
    
    if inside:
        # Inside: distance to closest edge (negative)
        min_dist = float('inf')
        for k in range(6):
            a_norm = math.radians(30 + 60 * k)
            nx = math.cos(a_norm)
            ny = math.sin(a_norm)
            d = apothem - (lx * nx + ly * ny)
            if d < min_dist:
                min_dist = d
        return -min_dist
    else:
        # Outside: distance to closest edge
        min_dist = float('inf')
        for k in range(6):
            a_norm = math.radians(30 + 60 * k)
            nx = math.cos(a_norm)
            ny = math.sin(a_norm)
            d = (lx * nx + ly * ny) - apothem
            if d > 0 and d < min_dist:
                min_dist = d
        # Also check distance to vertices for closer approach
        for k in range(6):
            a_v = math.radians(60 * k)
            vx = side * math.cos(a_v)
            vy = side * math.sin(a_v)
            dxv = lx - vx
            dyv = ly - vy
            dv = math.sqrt(dxv*dxv + dyv*dyv)
            if dv < min_dist:
                min_dist = dv
        return min_dist

def _sdf_hex_to_hex(c1, a1, c2, a2, side1=1.0, side2=1.0):
    """
    Approximate minimum distance between two hexagons.
    Returns negative if overlapping, positive if separated.
    Uses vertex-to-hexagon SDF sampling.
    """
    cx1, cy1 = c1
    cx2, cy2 = c2
    
    # Quick reject
    dx = cx2 - cx1
    dy = cy2 - cy1
    dist_centers = math.sqrt(dx*dx + dy*dy)
    if dist_centers > 2.0 * max(side1, side2) + 0.1:
        return dist_centers - 2.0 * max(side1, side2)
    
    # Sample vertices of hex1, compute SDF to hex2
    min_sdf = float('inf')
    for k in range(6):
        a = math.radians(a1 + 60 * k)
        vx = cx1 + side1 * math.cos(a)
        vy = cy1 + side1 * math.sin(a)
        sdf = _sdf_hexagon(vx, vy, cx2, cy2, side2, a2)
        if sdf < min_sdf:
            min_sdf = sdf
    
    # Sample vertices of hex2, compute SDF to hex1
    for k in range(6):
        a = math.radians(a2 + 60 * k)
        vx = cx2 + side2 * math.cos(a)
        vy = cy2 + side2 * math.sin(a)
        sdf = _sdf_hexagon(vx, vy, cx1, cy1, side1, a1)
        if sdf < min_sdf:
            min_sdf = sdf
    
    return min_sdf

def _all_vertices(positions, angle_deg, side=1.0):
    """Get all vertices of all inner hexagons."""
    verts = []
    for px, py in positions:
        hv = _hex_vertices(px, py, side, angle_deg)
        verts.extend(hv)
    return verts

def _min_outer_side_lp(vertices, angle_deg):
    """
    Find minimum side length of outer hexagon with given rotation angle
    that contains all given vertices, using LP.
    Returns (cx, cy, side) or None.
    """
    if not vertices:
        return 0.0, 0.0, 0.0
    
    pts = np.array(vertices)
    
    normals = []
    for k in range(6):
        a = math.radians(angle_deg + 30 + 60 * k)
        normals.append((math.cos(a), math.sin(a)))
    
    n_pts = len(pts)
    A_ub = []
    b_ub = []
    
    for k in range(6):
        nx, ny = normals[k]
        for i in range(n_pts):
            vx, vy = pts[i]
            v_dot_n = vx * nx + vy * ny
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-v_dot_n)
            A_ub.append([nx, ny, -1.0])
            b_ub.append(v_dot_n)
    
    A_ub.append([0.0, 0.0, -1.0])
    b_ub.append(0.0)
    
    c_vec = [0.0, 0.0, 1.0]
    
    try:
        res = linprog(c_vec, A_ub=A_ub, b_ub=b_ub,
                     bounds=[(None, None), (None, None), (0, None)],
                     method='highs')
        if res.success:
            cx, cy, t = res.x
            side = t / (math.sqrt(3) / 2)
            return cx, cy, side
    except Exception:
        pass
    
    return None

def _compute_objective(pos_flat, inner_angle, outer_angle, penalty_weight=1000.0):
    """
    Objective function: outer hexagon side length + penalty for overlaps.
    pos_flat: 2N array of (x1,y1,x2,y2,...)
    Returns (objective_value, side_length, is_valid)
    """
    positions = [(pos_flat[i], pos_flat[i+1]) for i in range(0, len(pos_flat), 2)]
    
    # Compute overlap penalty
    overlap_penalty = 0.0
    n = len(positions)
    for i in range(n):
        for j in range(i+1, n):
            sdf = _sdf_hex_to_hex(positions[i], inner_angle, positions[j], inner_angle)
            if sdf < 0:
                # Overlapping: penalize quadratically
                overlap_penalty += penalty_weight * sdf * sdf
    
    # Get all vertices
    all_verts = _all_vertices(positions, inner_angle)
    
    # Find min enclosing hexagon
    res = _min_outer_side_lp(all_verts, outer_angle)
    if res is None:
        return overlap_penalty + 100.0, 100.0, False
    
    _, _, side = res
    total = side + overlap_penalty
    return total, side, True

def _initial_hex_lattice_positions(gap=1.0001):
    """Generate initial positions on hex lattice."""
    sqrt3 = math.sqrt(3)
    positions = []
    
    # Center
    positions.append((0.0, 0.0))
    
    # Ring 1
    for k in range(6):
        a = math.radians(60 * k)
        x = gap * 2 * math.cos(a) * 0.866025  # sqrt(3)
        y = gap * 2 * math.sin(a) * 0.866025
        # Actually for hex lattice with side 1, centers are 2*apothem apart = sqrt(3)
        # Let me use proper hex lattice spacing
        pass
    
    # Proper hex lattice: neighbors at distance sqrt(3) from center
    # 6 neighbors at angles 0, 60, 120, 180, 240, 300
    for k in range(6):
        a = math.radians(60 * k)
        r = gap * math.sqrt(3)
        x = r * math.cos(a)
        y = r * math.sin(a)
        positions.append((x, y))
    
    # Ring 2: 12 positions, pick 4
    ring2 = []
    for k in range(6):
        a1 = math.radians(60 * k)
        a2 = math.radians(60 * k + 30)
        # Two positions per sector
        r1 = gap * 2 * math.sqrt(3)
        r2 = gap * math.sqrt(3)
        # Actually ring 2 in hex lattice:
        # At distance 2*sqrt(3) in directions 0,60,... and at distance sqrt(3) in directions 30,90,...
        x1 = r1 * math.cos(a1)
        y1 = r1 * math.sin(a1)
        ring2.append((x1, y1))
        x2 = r2 * math.cos(a2)
        y2 = r2 * math.sin(a2)
        # Hmm, this isn't quite right. Let me use axial coordinates.
    
    # Use axial coordinates for proper hex lattice
    def axial_to_xy(q, r, scale=1.0):
        x = scale * 1.5 * q
        y = scale * math.sqrt(3) * (r + q/2.0)
        return (x, y)
    
    positions = []
    # Core: center + ring 1
    core_axial = [(0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    for q, r in core_axial:
        positions.append(axial_to_xy(q, r, gap))
    
    # Ring 2 candidates (12 positions)
    ring2_axial = [(2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-2),
                   (-2,-2), (-2,-1), (-2,0), (-2,1), (-1,2), (0,2)]
    
    # Pick 4 from ring 2: try symmetric selections
    # For 11 total: 7 core + 4 from ring 2
    # Symmetric: pick (2,0), (2,-2), (-2,-1), (-2,1) or similar
    selected = [(2,0), (2,-2), (-2,-1), (-2,1)]
    for q, r in selected:
        positions.append(axial_to_xy(q, r, gap))
    
    return positions

def construct():
    if time.time() > _DEADLINE - 5.0:
        # Emergency fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_side = float('inf')
    best_positions = None
    best_inner_angle = 0.0
    best_outer_angle = 0.0
    best_center = (0.0, 0.0)
    
    # Try multiple initial configurations
    initial_configs = []
    
    # Config 1: hex lattice
    initial_configs.append(_initial_hex_lattice_positions(1.0001))
    
    # Config 2: slightly different lattice
    initial_configs.append(_initial_hex_lattice_positions(1.0005))
    
    # Config 3: compact arrangement
    sqrt3 = math.sqrt(3)
    config3 = [(0, 0),
               (sqrt3, 0), (-sqrt3, 0),
               (sqrt3/2, 1.5), (-sqrt3/2, 1.5),
               (sqrt3/2, -1.5), (-sqrt3/2, -1.5),
               (2*sqrt3, 0), (-2*sqrt3, 0),
               (sqrt3, 3), (-sqrt3, 3)]
    initial_configs.append(config3)
    
    # Config 4: another arrangement
    config4 = [(0, 0),
               (1.732, 0), (-1.732, 0),
               (0.866, 1.5), (-0.866, 1.5),
               (0.866, -1.5), (-0.866, -1.5),
               (3.464, 0), (-3.464, 0),
               (1.732, 3), (-1.732, 3)]
    initial_configs.append(config4)
    
    for init_pos in initial_configs:
        if time.time() > _DEADLINE - 20.0:
            break
        
        x0 = []
        for px, py in init_pos:
            x0.extend([px, py])
        
        for inner_angle in [0.0, 30.0]:
            if time.time() > _DEADLINE - 15.0:
                break
            
            for outer_angle in [0.0, 15.0, 30.0]:
                if time.time() > _DEADLINE - 10.0:
                    break
                
                # Run gradient optimization
                def objective(pos_flat):
                    val, side, valid = _compute_objective(pos_flat, inner_angle, outer_angle, penalty_weight=500.0)
                    return val
                
                try:
                    res = minimize(objective, x0, method='L-BFGS-B',
                                  options={'maxiter': 100, 'gtol': 1e-6, 'ftol': 1e-8})
                    
                    if res.fun < best_side + 50:  # Reasonable
                        new_pos = [(res.x[i], res.x[i+1]) for i in range(0, len(res.x), 2)]
                        
                        # Check validity
                        val, side, valid = _compute_objective(res.x, inner_angle, outer_angle, penalty_weight=0.0)
                        if valid and side < best_side:
                            # Verify no overlaps
                            overlap = False
                            for i in range(len(new_pos)):
                                for j in range(i+1, len(new_pos)):
                                    sdf = _sdf_hex_to_hex(new_pos[i], inner_angle, new_pos[j], inner_angle)
                                    if sdf < -1e-6:
                                        overlap = True
                                        break
                                if overlap:
                                    break
                            
                            if not overlap:
                                best_side = side
                                best_positions = new_pos
                                best_inner_angle = inner_angle
                                best_outer_angle = outer_angle
                                # Get center from LP
                                all_verts = _all_vertices(new_pos, inner_angle)
                                lp_res = _min_outer_side_lp(all_verts, outer_angle)
                                if lp_res:
                                    best_center = (lp_res[0], lp_res[1])
                except Exception:
                    pass
        
        # Also try Nelder-Mead from best found so far
        if best_positions is not None and time.time() < _DEADLINE - 10.0:
            x0 = []
            for px, py in best_positions:
                x0.extend([px, py])
            
            def nm_objective(pos_flat):
                val, side, valid = _compute_objective(pos_flat, best_inner_angle, best_outer_angle, penalty_weight=200.0)
                return val
            
            try:
                res = minimize(nm_objective, x0, method='Nelder-Mead',
                              options={'maxiter': 100, 'xatol': 1e-5, 'fatol': 1e-7})
                
                if res.fun < best_side + 10:
                    new_pos = [(res.x[i], res.x[i+1]) for i in range(0, len(res.x), 2)]
                    val, side, valid = _compute_objective(res.x, best_inner_angle, best_outer_angle, penalty_weight=0.0)
                    if valid and side < best_side:
                        # Verify
                        overlap = False
                        for i in range(len(new_pos)):
                            for j in range(i+1, len(new_pos)):
                                sdf = _sdf_hex_to_hex(new_pos[i], best_inner_angle, new_pos[j], best_inner_angle)
                                if sdf < -1e-6:
                                    overlap = True
                                    break
                            if overlap:
                                break
                        
                        if not overlap:
                            best_side = side
                            best_positions = new_pos
                            all_verts = _all_vertices(new_pos, best_inner_angle)
                            lp_res = _min_outer_side_lp(all_verts, best_outer_angle)
                            if lp_res:
                                best_center = (lp_res[0], lp_res[1])
            except Exception:
                pass
    
    # Fine-tune outer angle with best positions
    if best_positions is not None and time.time() < _DEADLINE - 5.0:
        all_verts = _all_vertices(best_positions, best_inner_angle)
        
        # Scan outer angles more finely
        for ang in np.linspace(0, 30, 61):
            if time.time() > _DEADLINE - 3.0:
                break
            res = _min_outer_side_lp(all_verts, ang)
            if res:
                _, _, side = res
                if side < best_side:
                    best_side = side
                    best_outer_angle = ang
                    best_center = (res[0], res[1])
        
        # Even finer
        if time.time() < _DEADLINE - 2.0:
            lo = max(0, best_outer_angle - 1.0)
            hi = min(30, best_outer_angle + 1.0)
            for ang in np.linspace(lo, hi, 41):
                if time.time() > _DEADLINE - 1.0:
                    break
                res = _min_outer_side_lp(all_verts, ang)
                if res:
                    _, _, side = res
                    if side < best_side:
                        best_side = side
                        best_outer_angle = ang
                        best_center = (res[0], res[1])
    
    # Fallback if nothing worked
    if best_positions is None:
        best_positions = _initial_hex_lattice_positions(1.0001)
        best_inner_angle = 0.0
        best_outer_angle = 0.0
        all_verts = _all_vertices(best_positions, best_inner_angle)
        res = _min_outer_side_lp(all_verts, best_outer_angle)
        if res:
            best_center = (res[0], res[1])
            best_side = res[2]
        else:
            # Absolute fallback
            inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            return inner, (0.0, 0.0), 40.0, 0.0
    
    # Final verification and slight expansion for safety
    all_verts = _all_vertices(best_positions, best_inner_angle)
    res = _min_outer_side_lp(all_verts, best_outer_angle)
    if res:
        best_center = (res[0], res[1])
        best_side = res[2] * 1.001  # Tiny safety margin
    
    inner = [[best_positions[i][0], best_positions[i][1], best_inner_angle] for i in range(N)]
    
    return inner, (best_center[0], best_center[1]), best_side, best_outer_angle

import math
import time
import numpy as np
from scipy.optimize import linprog, minimize, differential_evolution

N = 11

def _hex_vertices_array(cx, cy, s, angle_deg):
    """
    Returns vertices of a regular hexagon centered at (cx, cy) with side s
    and rotation angle_deg (degrees) as a numpy array.
    """
    a = math.radians(angle_deg)
    angles = a + np.arange(6) * math.pi / 3
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _point_in_hex_dist_sq(px, py, cx, cy, s, angle_deg):
    """
    Returns squared distance from point (px, py) to the hexagon if outside,
    or 0 if inside.
    """
    a = math.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    
    cos_a = math.cos(-a)
    sin_a = math.sin(-a)
    rx = dx * cos_a - dy * sin_a
    ry = dx * sin_a + dy * cos_a
    
    apothem = s * math.sqrt(3) / 2
    sum_sq_viol = 0.0
    for i in range(3):
        theta_n = math.radians(30 + i * 60)
        proj = rx * math.cos(theta_n) + ry * math.sin(theta_n)
        if proj > apothem:
            sum_sq_viol += (proj - apothem)**2
        if proj < -apothem:
            sum_sq_viol += (-apothem - proj)**2
    return sum_sq_viol

def _hexagons_disjoint(p1, p2):
    """
    p1, p2: (cx, cy, angle_deg)
    Returns True if disjoint, False otherwise.
    Uses vertex-in-hex check and segment intersection check for robustness.
    """
    c1x, c1y, a1 = p1
    c2x, c2y, a2 = p2
    
    dx = c1x - c2x
    dy = c1y - c2y
    # Quick reject
    if dx*dx + dy*dy > 4.5:
        return True
        
    v1 = _hex_vertices_array(c1x, c1y, 1.0, a1)
    v2 = _hex_vertices_array(c2x, c2y, 1.0, a2)
    
    # Check vertices of 1 in 2
    for pt in v1:
        if _point_in_hex_dist_sq(pt[0], pt[1], c2x, c2y, 1.0, a2) > 1e-8:
            return False
            
    # Check vertices of 2 in 1
    for pt in v2:
        if _point_in_hex_dist_sq(pt[0], pt[1], c1x, c1y, 1.0, a1) > 1e-8:
            return False
            
    # Check edge intersections
    # This is more robust for cases where vertices are outside but edges cross
    for i in range(6):
        p1s = v1[i]
        p1e = v1[(i+1)%6]
        for j in range(6):
            p2s = v2[j]
            p2e = v2[(j+1)%6]
            if _segments_intersect(p1s, p1e, p2s, p2e):
                return False
                
    return True

def _segments_intersect(p1, p2, p3, p4):
    """
    Check if segment p1-p2 intersects segment p3-p4.
    p1, p2, p3, p4 are (x, y) tuples.
    """
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def _all_disjoint(inner_hexes):
    """
    inner_hexes: list of (cx, cy, angle_deg)
    Returns True if all pairs are disjoint.
    """
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if not _hexagons_disjoint(inner_hexes[i], inner_hexes[j]):
                return False
    return True

def _inner_in_outer(inner_hexes, outer):
    """
    inner_hexes: list of (cx, cy, angle_deg)
    outer: (cx, cy, side, angle_deg)
    Returns True if all inner hexes are strictly inside (or on boundary) of outer.
    """
    ocx, ocy, os, oang = outer
    for h in inner_hexes:
        verts = _hex_vertices_array(h[0], h[1], 1.0, h[2])
        for v in verts:
            if _point_in_hex_dist_sq(v[0], v[1], ocx, ocy, os, oang) > 1e-6:
                return False
    return True

def _min_outer_side_lp(points, angle_deg):
    """
    points: numpy array of shape (M, 2)
    angle_deg: orientation of the enclosing hexagon
    Returns minimum side length s and center (cx, cy) such that all points are inside.
    If infeasible, returns (inf, 0, 0).
    """
    a = math.radians(angle_deg)
    # Normals of the enclosing hexagon sides
    # The hexagon is defined by 3 pairs of parallel lines.
    # The distance from center to each side is the apothem = s * sqrt(3) / 2.
    # The normal vectors for the 3 pairs are at angles a + 30, a + 90, a + 150 degrees.
    # Wait, the normals to the sides of a regular hexagon are spaced by 60 degrees.
    # If a vertex is at angle a, the side opposite to it has normal at a + 180.
    # The 6 sides have normals at a + 30, a + 90, a + 150, a + 210, a + 270, a + 330.
    # Or simply: normals at a + pi/6 + k*pi/3 for k=0..5.
    
    normals = []
    for k in range(3):
        theta_n = a + math.pi/6 + k * math.pi/3
        normals.append((math.cos(theta_n), math.sin(theta_n)))
        
    # We have 3 pairs of constraints.
    # For each normal n_k = (nx, ny):
    # n_k . (p - c) <= apothem  =>  nx*px + ny*py - nx*cx - ny*cy <= apothem
    # -n_k . (p - c) <= apothem =>  -nx*px - ny*py + nx*cx + ny*cy <= apothem
    
    # Variables: cx, cy, apothem
    # Objective: minimize apothem (since s is proportional to apothem)
    
    A_ub = []
    b_ub = []
    
    for nx, ny in normals:
        for px, py in points:
            # nx*px + ny*py - nx*cx - ny*cy <= apothem
            # -nx*cx - ny*cy - apothem <= -nx*px - ny*py
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-nx*px - ny*py)
            
            # -nx*px - ny*py + nx*cx + ny*cy <= apothem
            # nx*cx + ny*cy - apothem <= nx*px + ny*py
            A_ub.append([nx, ny, -1.0])
            b_ub.append(nx*px + ny*py)
            
    # Bounds: cx, cy unbounded (or large), apothem >= 0
    bounds = [(-100, 100), (-100, 100), (0, None)]
    
    try:
        res = linprog(c=[0, 0, 1], A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            apothem = res.x[2]
            s = apothem * 2 / math.sqrt(3)
            return s, res.x[0], res.x[1]
        else:
            return float('inf'), 0, 0
    except Exception:
        return float('inf'), 0, 0

def _compute_outer_for_positions(inner_hexes):
    """
    inner_hexes: list of (cx, cy, angle_deg)
    Returns (cx, cy, side, angle_deg) for the minimum enclosing hexagon.
    """
    points = []
    for h in inner_hexes:
        verts = _hex_vertices_array(h[0], h[1], 1.0, h[2])
        points.extend(verts)
        
    points = np.array(points)
    if len(points) == 0:
        return 0, 0, 1.0, 0.0

    best_side = float('inf')
    best_params = (0, 0, 1.0, 0.0)
    
    # Scan angles 0-60 degrees (symmetry)
    # Step 0.5 degrees
    for angle_deg in np.arange(0, 60.5, 0.5):
        s, cx, cy = _min_outer_side_lp(points, angle_deg)
        if s < best_side:
            best_side = s
            best_params = (cx, cy, s, angle_deg)
            
    # Refine around best angle
    if best_side < float('inf'):
        best_ang = best_params[3]
        for angle_deg in np.arange(best_ang - 1.0, best_ang + 1.0, 0.05):
            s, cx, cy = _min_outer_side_lp(points, angle_deg)
            if s < best_side:
                best_side = s
                best_params = (cx, cy, s, angle_deg)
                
    return best_params

def _separation_distance(p1, p2):
    """
    Returns the minimum distance between the two hexagons.
    If they overlap, returns a negative value (penalty).
    """
    # This is a heuristic. For optimization, we want a smooth penalty.
    # We can use the sum of squared violations of vertices as before, but scaled.
    c1x, c1y, a1 = p1
    c2x, c2y, a2 = p2
    
    v1 = _hex_vertices_array(c1x, c1y, 1.0, a1)
    v2 = _hex_vertices_array(c2x, c2y, 1.0, a2)
    
    penalty = 0.0
    for pt in v1:
        penalty += _point_in_hex_dist_sq(pt[0], pt[1], c2x, c2y, 1.0, a2)
    for pt in v2:
        penalty += _point_in_hex_dist_sq(pt[0], pt[1], c1x, c1y, 1.0, a1)
        
    # This penalty is 0 if disjoint (vertices outside), >0 if overlapping.
    # But it doesn't capture edge intersections without vertices inside.
    # For a smooth optimization, this is often sufficient if we start from a good initial guess.
    return penalty

def _pairwise_separation_constraint(inner_hexes):
    """
    Returns total penalty for overlaps among inner hexes.
    """
    n = len(inner_hexes)
    total_pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            total_pen += _separation_distance(inner_hexes[i], inner_hexes[j])
    return total_pen

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Strategy:
    # 1. Generate candidate layouts using a structured search (grid/radial).
    # 2. For each layout, compute the minimum enclosing hexagon.
    # 3. Keep the best valid layout.
    # 4. Optionally, polish the best layout using local search.
    
    best_side = float('inf')
    best_inner = None
    best_outer = None
    
    # Candidate 1: Hexagonal lattice (perfect packing)
    # 11 hexes: 1 in center, 6 around, 4 more?
    # A 3x3 grid of hexes (9 hexes) plus 2 extra.
    # Let's try a few standard configurations.
    
    configs = []
    
    # Config 1: 3x3 grid (9 hexes) + 2 on top
    # Hex lattice coordinates:
    # (0,0), (1,0), (2,0)
    # (0,1), (1,1), (2,1)
    # (0,2), (1,2), (2,2)
    # This is a rectangular grid in axial coordinates? No, hex grid.
    # Let's use flat-top hexagons.
    # Spacing: sqrt(3) in x, 1.5 in y for adjacent rows?
    # For flat-top:
    # Width = 2*s = 2
    # Height = sqrt(3)*s = sqrt(3)
    # Horizontal spacing = 1.5 * s = 1.5
    # Vertical spacing = sqrt(3) * s = sqrt(3)
    # Offset for odd columns = sqrt(3)/2 * s = sqrt(3)/2
    
    # Let's just use a simple grid of centers.
    # For 11 hexes, a 3x4 grid minus 1 corner?
    # Or 4x3 grid minus 1 corner?
    
    # Let's try a few random-ish but structured layouts.
    
    # Layout 1: 3 rows of 4, 4, 3
    # Row 0: 4 hexes
    # Row 1: 4 hexes (offset)
    # Row 2: 3 hexes (offset)
    # Total 11.
    
    def make_layout(rows, offset_x=0.0, offset_y=0.0):
        """
        rows: list of integers, number of hexes in each row.
        """
        hexes = []
        y = 0.0
        for i, count in enumerate(rows):
            x_start = - (count - 1) * 0.75 # 1.5/2
            for j in range(count):
                x = x_start + j * 1.5
                if i % 2 == 1:
                    x += 0.75
                hexes.append((x + offset_x, y + offset_y, 0.0))
            y += math.sqrt(3)
        return hexes
    
    layouts = []
    layouts.append(make_layout([4, 4, 3]))
    layouts.append(make_layout([3, 4, 4]))
    layouts.append(make_layout([4, 3, 4]))
    layouts.append(make_layout([5, 3, 3]))
    layouts.append(make_layout([3, 5, 3]))
    layouts.append(make_layout([2, 4, 5]))
    layouts.append(make_layout([5, 4, 2]))
    
    # Also try a more circular arrangement
    # Center + 6 + 4
    def make_circular():
        hexes = [(0, 0, 0)]
        for i in range(6):
            ang = i * math.pi / 3
            x = math.sqrt(3) * math.cos(ang)
            y = math.sqrt(3) * math.sin(ang)
            hexes.append((x, y, 0))
        # Add 4 more around
        # They need to fit in the gaps.
        # Gaps are at angles 30, 90, 150, 210, 270, 330
        # Distance from center to gap center is sqrt(3) * 2? No.
        # The gap between two adjacent hexes in the first ring is at distance sqrt(3) * sqrt(3) = 3?
        # Let's just try placing them at distance 3, angles 30, 90, 150, 210.
        for i in range(4):
            ang = (i + 0.5) * math.pi / 3 # 30, 90, 150, 210
            x = 3.0 * math.cos(ang)
            y = 3.0 * math.sin(ang)
            hexes.append((x, y, 0))
        return hexes
    
    layouts.append(make_circular())
    
    # Evaluate each layout
    for layout in layouts:
        if time.time() > deadline:
            break
            
        # Check disjointness
        if not _all_disjoint(layout):
            continue
            
        # Compute outer hexagon
        outer = _compute_outer_for_positions(layout)
        cx, cy, s, ang = outer
        
        if s < best_side:
            best_side = s
            best_inner = layout
            best_outer = outer
            
    # If no valid layout found, or we have time, do a local search on the best one.
    if best_inner is not None and time.time() < deadline - 20:
        # Local search using Nelder-Mead on the positions of the inner hexes.
        # Objective: minimize outer side length + penalty for overlaps.
        
        x0 = []
        for h in best_inner:
            x0.extend([h[0], h[1], h[2]])
            
        def objective(x):
            inner = []
            for i in range(N):
                inner.append((x[3*i], x[3*i+1], x[3*i+2]))
                
            # Check disjointness
            pen = _pairwise_separation_constraint(inner)
            if pen > 1e-4:
                return 100.0 + pen # Large penalty
            
            outer = _compute_outer_for_positions(inner)
            return outer[2]
            
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if res.fun < best_side:
                # Extract new inner positions
                new_inner = []
                for i in range(N):
                    new_inner.append((res.x[3*i], res.x[3*i+1], res.x[3*i+2]))
                    
                # Verify validity
                if _all_disjoint(new_inner):
                    new_outer = _compute_outer_for_positions(new_inner)
                    if new_outer[2] < best_side:
                        best_side = new_outer[2]
                        best_inner = new_inner
                        best_outer = new_outer
        except Exception:
            pass
            
    # Fallback if still no valid solution
    if best_inner is None:
        # Greedy placement
        placed = [(0.0, 0.0, 0.0)]
        for k in range(1, N):
            best_pos = None
            best_cl = -1
            for x in np.arange(-5, 5, 0.2):
                for y in np.arange(-5, 5, 0.2):
                    min_cl = float('inf')
                    for p in placed:
                        dx = x - p[0]
                        dy = y - p[1]
                        d = math.sqrt(dx*dx + dy*dy)
                        if d < 2.0:
                            min_cl = -1.0
                            break
                        min_cl = min(min_cl, d - 2.0)
                    if min_cl > best_cl:
                        best_cl = min_cl
                        best_pos = (x, y, 0.0)
            if best_pos:
                placed.append(best_pos)
            else:
                placed.append((0,0,0))
                
        while len(placed) < N:
            placed.append((0,0,0))
            
        best_inner = placed
        best_outer = _compute_outer_for_positions(placed)
        
    # Final check and return
    # Ensure best_inner is a list of lists [x, y, angle]
    final_inner = []
    for h in best_inner:
        final_inner.append([h[0], h[1], h[2]])
        
    cx, cy, s, ang = best_outer
    return final_inner, (cx, cy), s, ang

if __name__ == "__main__":
    r = construct()
    print(r)

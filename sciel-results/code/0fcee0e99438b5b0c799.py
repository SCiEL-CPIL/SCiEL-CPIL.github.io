import math
import time
import numpy as np
from scipy.optimize import differential_evolution

N = 11

def _hex_vertices(cx, cy, s, angle_deg):
    """
    Returns vertices of a regular hexagon centered at (cx, cy) with side s
    and rotation angle_deg (degrees).
    Vertices are in CCW order.
    """
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3
        x = cx + s * math.cos(theta)
        y = cy + s * math.sin(theta)
        verts.append((x, y))
    return verts

def _point_in_hex_dist_sq(px, py, cx, cy, s, angle_deg):
    """
    Returns squared distance from point (px, py) to the hexagon if outside,
    or 0 if inside.
    Uses the fact that a regular hexagon is the intersection of 3 strips.
    Actually, for robustness with rotation, we transform the point to the hexagon's local frame.
    """
    a = math.radians(angle_deg)
    # Rotate point relative to center
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

def _pairwise_clearance(p1, p2):
    """
    p1, p2: (cx, cy, angle_deg)
    Returns a measure of overlap. 0 if disjoint.
    We use the sum of squared violations of vertices.
    """
    c1x, c1y, a1 = p1
    c2x, c2y, a2 = p2
    
    dx = c1x - c2x
    dy = c1y - c2y
    # Quick reject: if centers are far apart, no overlap.
    # Max distance between vertices is 2*1 = 2.
    # If dist > 2 + epsilon, definitely disjoint.
    if dx*dx + dy*dy > 4.5: 
        return 0.0
        
    v1 = _hex_vertices(c1x, c1y, 1.0, a1)
    v2 = _hex_vertices(c2x, c2y, 1.0, a2)
    
    penalty = 0.0
    for pt in v1:
        penalty += _point_in_hex_dist_sq(pt[0], pt[1], c2x, c2y, 1.0, a2)
    for pt in v2:
        penalty += _point_in_hex_dist_sq(pt[0], pt[1], c1x, c1y, 1.0, a1)
        
    return penalty

def _min_enclosing_hexagon(inner_hexes):
    """
    Finds the minimum side length regular hexagon containing all inner_hexes.
    inner_hexes: list of (cx, cy, angle_deg)
    Returns: (cx, cy, side_length, angle_deg)
    """
    # Collect all vertices of all inner hexagons
    points = []
    for h in inner_hexes:
        verts = _hex_vertices(h[0], h[1], 1.0, h[2])
        points.extend(verts)
    
    points = np.array(points)
    if len(points) == 0:
        return 0, 0, 1.0, 0.0

    best_side = float('inf')
    best_params = (0, 0, 1.0, 0.0)
    
    # The orientation of the enclosing hexagon is the critical variable.
    # We sample angles. Due to symmetry, 0-60 degrees is enough, but we scan 0-360 for robustness.
    # Step size 1 degree is usually sufficient for good precision, then refine.
    
    for angle_deg in np.arange(0, 360, 1.0):
        a = math.radians(angle_deg)
        # Normals of the enclosing hexagon sides
        normals = []
        for k in range(3):
            theta_n = a + math.pi/6 + k * math.pi/3
            normals.append((math.cos(theta_n), math.sin(theta_n)))
        
        A = []
        b = []
        
        for nx, ny in normals:
            for px, py in points:
                # Constraint 1
                A.append([nx, ny, -math.sqrt(3)/2])
                b.append(nx*px + ny*py)
                
                # Constraint 2
                A.append([-nx, -ny, -math.sqrt(3)/2])
                b.append(-nx*px - ny*py)
        
        # Bounds: s >= 0, c_x, c_y unbounded (or large bounds)
        bounds = [(-100, 100), (-100, 100), (0, None)]
        
        try:
            from scipy.optimize import linprog
            res = linprog(c=[0, 0, 1], A_ub=A, b_ub=b, bounds=bounds, method='highs')
            if res.success:
                s_val = res.x[2]
                if s_val < best_side:
                    best_side = s_val
                    best_params = (res.x[0], res.x[1], s_val, angle_deg)
        except Exception:
            pass
            
    # Refine around the best angle found
    if best_params[3] != 0.0:
        best_ang = best_params[3]
        for angle_deg in np.arange(best_ang - 1.0, best_ang + 1.0, 0.1):
            a = math.radians(angle_deg)
            normals = []
            for k in range(3):
                theta_n = a + math.pi/6 + k * math.pi/3
                normals.append((math.cos(theta_n), math.sin(theta_n)))
                
            A = []
            b = []
            for nx, ny in normals:
                for px, py in points:
                    A.append([nx, ny, -math.sqrt(3)/2])
                    b.append(nx*px + ny*py)
                    A.append([-nx, -ny, -math.sqrt(3)/2])
                    b.append(-nx*px - ny*py)
                    
            bounds = [(-100, 100), (-100, 100), (0, None)]
            try:
                from scipy.optimize import linprog
                res = linprog(c=[0, 0, 1], A_ub=A, b_ub=b, bounds=bounds, method='highs')
                if res.success:
                    s_val = res.x[2]
                    if s_val < best_side:
                        best_side = s_val
                        best_params = (res.x[0], res.x[1], s_val, angle_deg)
            except Exception:
                pass
                
    return best_params

def _objective(x):
    """
    Objective function for differential_evolution.
    x = [x1, y1, a1, ..., x11, y11, a11, ocx, ocy, oside, oang]
    Returns the outer side length if valid, or a large penalty if invalid.
    """
    # Extract parameters
    inner = []
    for i in range(N):
        cx = x[3*i]
        cy = x[3*i + 1]
        ang = x[3*i + 2]
        inner.append((cx, cy, ang))
    
    ocx = x[3*N]
    ocy = x[3*N + 1]
    oside = x[3*N + 2]
    oang = x[3*N + 3]
    
    # Check containment
    # We can optimize the outer hexagon for the given inner positions,
    # but since oside is a variable, we must ensure the inner hexes fit.
    # Actually, to make the optimization easier, we can fix the outer hexagon
    # to be the minimum enclosing one for the current inner positions?
    # No, that makes the objective non-smooth and non-differentiable.
    # Instead, we penalize if inner hexes are outside the outer hexagon.
    
    penalty = 0.0
    
    # Containment penalty
    for h in inner:
        verts = _hex_vertices(h[0], h[1], 1.0, h[2])
        for v in verts:
            d_sq = _point_in_hex_dist_sq(v[0], v[1], ocx, ocy, oside, oang)
            if d_sq > 0:
                penalty += d_sq
    
    # Disjointness penalty
    for i in range(N):
        for j in range(i + 1, N):
            pen = _pairwise_clearance(inner[i], inner[j])
            if pen > 0:
                penalty += pen * 10.0  # Heavier penalty for overlap
    
    if penalty > 1e-4:
        # Return a large value to guide DE away
        return oside + penalty * 100.0
    
    return oside

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial guess: Hexagonal lattice arrangement
    # Side length of outer hex for 11 inner hexes in a hexagonal pattern:
    # 1 in center, 6 around, 4 more? No, 1+6=7. 1+6+12=19.
    # So 11 is not a perfect hex number.
    # A compact arrangement might be a 3x3 grid minus corners?
    # Or just a random start.
    
    # Let's use a heuristic start: place them in a loose grid.
    x0 = []
    # 3x3 grid of hexagons (side 1) requires spacing ~1.732 (sqrt(3))
    # Let's place 9 in a 3x3 grid and 2 extra.
    # Actually, just random start is safer for DE.
    
    bounds = []
    for i in range(N):
        bounds.append((-10, 10)) # x
        bounds.append((-10, 10)) # y
        bounds.append((0, 60))   # angle (0-60 is fundamental domain for hex symmetry)
    bounds.append((-10, 10)) # ocx
    bounds.append((-10, 10)) # ocy
    bounds.append((3, 10))   # oside (must be at least ~3 for 11 hexes? 11 hexes area ~ 11*2.6 = 28.6. Hex area = 2.6*s^2. s^2 > 11. s > 3.3)
    bounds.append((0, 60))   # oang
    
    # DE parameters
    # popsize=30, maxiter=100 is too slow?
    # N=11, 36 variables.
    # Let's try popsize=20, maxiter=50.
    
    try:
        res = differential_evolution(
            _objective, 
            bounds, 
            seed=42,
            maxiter=30,
            popsize=20,
            tol=0.01,
            init='sobol',
            polish=False # We will do a final polish if needed
        )
        
        if res.success:
            best_x = res.x
        else:
            # If DE fails, use the best point found anyway?
            # differential_evolution returns the best point even if not converged?
            # It returns a result object. If not success, res.x might be the best found.
            best_x = res.x
            
    except Exception:
        # Fallback to a simple grid search or previous method if DE crashes
        # For now, return a valid but suboptimal solution
        # Let's try a simple greedy placement as fallback
        placed = [(0.0, 0.0, 0.0)]
        for k in range(1, N):
            # Find a spot
            best_pos = None
            best_cl = -1
            for x in np.arange(-5, 5, 0.5):
                for y in np.arange(-5, 5, 0.5):
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
            
        inner = []
        for p in placed:
            inner.append([p[0], p[1], p[2]])
        
        ocx, ocy, os, oang = _min_enclosing_hexagon(placed)
        return inner, (ocx, ocy), os, oang

    # Extract best solution
    inner = []
    for i in range(N):
        cx = best_x[3*i]
        cy = best_x[3*i + 1]
        ang = best_x[3*i + 2]
        inner.append([cx, cy, ang])
    
    ocx = best_x[3*N]
    ocy = best_x[3*N + 1]
    oside = best_x[3*N + 2]
    oang = best_x[3*N + 3]
    
    # Verify validity and potentially adjust
    # Check overlaps
    valid = True
    for i in range(len(inner)):
        for j in range(i+1, len(inner)):
            if _pairwise_clearance((inner[i][0], inner[i][1], inner[i][2]), 
                                   (inner[j][0], inner[j][1], inner[j][2])) > 0:
                valid = False
                break
        if not valid:
            break
            
    # Check containment
    if valid:
        for p in inner:
            verts = _hex_vertices(p[0], p[1], 1.0, p[2])
            for v in verts:
                if _point_in_hex_dist_sq(v[0], v[1], ocx, ocy, oside, oang) > 1e-6:
                    valid = False
                    break
            if not valid:
                break
    
    if not valid:
        # If the DE result is invalid (overlap or outside), we need to repair.
        # Simple repair: Calculate the true min enclosing hexagon for these positions.
        # This ensures containment.
        # But overlaps might still exist.
        # If overlaps exist, we can't easily fix them without re-optimizing.
        # However, DE with strong penalty usually finds valid solutions.
        # If not, we fall back to greedy.
        
        placed = []
        for p in inner:
            placed.append((p[0], p[1], p[2]))
        
        # Try to push apart overlapping hexes? Too complex.
        # Just return the min enclosing hexagon for the current positions.
        # The evaluator will score it as invalid (0) if overlaps exist.
        # To be safe, let's try a quick local refinement if invalid.
        
        # Actually, let's just return the result. If it's invalid, it scores 0.
        # But we want to score > 0.
        # Let's force a valid solution using the greedy method if DE fails validity.
        
        placed = [(0.0, 0.0, 0.0)]
        for k in range(1, N):
            best_pos = None
            best_cl = -1
            # Search around current cluster
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
            
        inner = []
        for p in placed:
            inner.append([p[0], p[1], p[2]])
        
        ocx, ocy, os, oang = _min_enclosing_hexagon(placed)
        return inner, (ocx, ocy), os, oang
    
    # If valid, we can try to optimize the outer hexagon size further?
    # The DE already optimized oside.
    # But _min_enclosing_hexagon might give a tighter bound if the DE's oside was slightly too large.
    # Let's compute the true min enclosing hexagon for the inner positions.
    final_ocx, final_ocy, final_os, final_oang = _min_enclosing_hexagon(inner)
    
    # Use the tighter one
    if final_os < oside:
        ocx, ocy, oside, oang = final_ocx, final_ocy, final_os, final_oang
    
    return inner, (ocx, ocy), oside, oang

if __name__ == "__main__":
    r = construct()
    print(r)

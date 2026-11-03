from scipy.optimize import differential_evolution, minimize
import math
import time
import numpy as np
from scipy.spatial import ConvexHull

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
    """
    a = math.radians(angle_deg)
    # Rotate point relative to center
    dx = px - cx
    dy = py - cy
    # Rotate by -a to align hexagon with axes
    rx = dx * math.cos(-a) - dy * math.sin(-a)
    ry = dx * math.sin(-a) + dy * math.cos(-a)
    
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

def _pairwise_penalty(p1, p2):
    """
    p1, p2: (cx, cy, angle_deg)
    Returns penalty if they overlap.
    """
    c1x, c1y, a1 = p1
    c2x, c2y, a2 = p2
    
    dx = c1x - c2x
    dy = c1y - c2y
    if dx*dx + dy*dy > 4.0:
        return 0.0
        
    v1 = _hex_vertices(c1x, c1y, 1.0, a1)
    v2 = _hex_vertices(c2x, c2y, 1.0, a2)
    
    penalty = 0.0
    
    # Vertices of 1 in 2
    for pt in v1:
        penalty += _point_in_hex_dist_sq(pt[0], pt[1], c2x, c2y, 1.0, a2)
        
    # Vertices of 2 in 1
    for pt in v2:
        penalty += _point_in_hex_dist_sq(pt[0], pt[1], c1x, c1y, 1.0, a1)
        
    # If penalty is 0, we must check edge intersections.
    if penalty == 0.0:
        # Check if edges intersect
        # Edges of H1: (v1[i], v1[(i+1)%6])
        # Edges of H2: (v2[j], v2[(j+1)%6])
        
        def segments_intersect(p1, p2, p3, p4):
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

        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i+1)%6]
            for j in range(6):
                p3 = v2[j]
                p4 = v2[(j+1)%6]
                if segments_intersect(p1, p2, p3, p4):
                    mid1 = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
                    mid2 = ((p3[0]+p4[0])/2, (p3[1]+p4[1])/2)
                    penalty += _point_in_hex_dist_sq(mid1[0], mid1[1], c2x, c2y, 1.0, a2)
                    penalty += _point_in_hex_dist_sq(mid2[0], mid2[1], c1x, c1y, 1.0, a1)
                    if penalty == 0.0:
                        penalty += 1.0 # Hard penalty
                        
    return penalty

def _outer_penalty(inner_params, outer_params):
    """
    inner_params: list of (cx, cy, angle)
    outer_params: (cx, cy, s, angle)
    Returns penalty if any inner hexagon is not fully inside the outer hexagon.
    """
    ocx, ocy, os, oang = outer_params
    
    penalty = 0.0
    for ip in inner_params:
        icx, icy, iang = ip
        v_in = _hex_vertices(icx, icy, 1.0, iang)
        for pt in v_in:
            penalty += _point_in_hex_dist_sq(pt[0], pt[1], ocx, ocy, os, oang)
            
    return penalty

def _objective(x):
    """
    x: [cx1, cy1, a1, ..., cxN, cyN, aN, ocx, ocy, os, oang]
    """
    n_inner = N
    inner_params = []
    for i in range(n_inner):
        cx = x[i*3]
        cy = x[i*3+1]
        a = x[i*3+2]
        inner_params.append((cx, cy, a))
        
    ocx = x[n_inner*3]
    ocy = x[n_inner*3+1]
    os = x[n_inner*3+2]
    oang = x[n_inner*3+3]
    outer_params = (ocx, ocy, os, oang)
    
    pen_pair = 0.0
    for i in range(n_inner):
        for j in range(i+1, n_inner):
            pen_pair += _pairwise_penalty(inner_params[i], inner_params[j])
            
    pen_outer = _outer_penalty(inner_params, outer_params)
    
    return os + 100.0 * (pen_pair + pen_outer)

def _compute_min_enclosing_hexagon(inner_params):
    """
    Given a list of (cx, cy, angle) for inner hexagons,
    find the minimum side length and optimal center/angle for an enclosing hexagon.
    Returns (cx, cy, s, angle).
    """
    # Collect all vertices of inner hexagons
    all_pts = []
    for ip in inner_params:
        cx, cy, ang = ip
        verts = _hex_vertices(cx, cy, 1.0, ang)
        all_pts.extend(verts)
        
    if len(all_pts) < 3:
        return (0, 0, 2.0, 0.0)
        
    all_pts = np.array(all_pts)
    
    # Use scipy.spatial.ConvexHull to get hull vertices
    try:
        hull = ConvexHull(all_pts)
        hull_pts = all_pts[hull.vertices]
    except Exception:
        hull_pts = all_pts
        
    # Optimize enclosing hexagon parameters: (cx, cy, s, angle)
    # Objective: minimize s, subject to all hull_pts being inside the hexagon.
    
    def _enc_obj(params):
        cx, cy, s, ang = params
        # s must be positive
        if s < 0: s = 0
        pen = 0.0
        for pt in hull_pts:
            pen += _point_in_hex_dist_sq(pt[0], pt[1], cx, cy, s, ang)
        return s + 1000.0 * pen
        
    # Initial guess
    center = np.mean(hull_pts, axis=0)
    max_dist = np.max(np.linalg.norm(hull_pts - center, axis=1))
    init = [center[0], center[1], max_dist * 1.5, 0.0]
    
    bounds = [(center[0]-10, center[0]+10),
              (center[1]-10, center[1]+10),
              (1.0, 20.0),
              (0, 360)]
              
    # Use Nelder-Mead for local refinement of the enclosing hexagon
    # Since the landscape of "min enclosing" is convex-ish (min of convex functions is not, but here it's a single hexagon containing points),
    # Nelder-Mead is often sufficient if started well.
    # To be safer, we could run DE here too, but time budget is tight.
    # Let's try a few random starts for Nelder-Mead.
    
    best_params = None
    best_val = float('inf')
    
    # Start 1: Mean center, estimated radius
    try:
        res = minimize(_enc_obj, init, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5})
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
    except Exception:
        pass
        
    # Start 2: Bounding box center
    min_x, min_y = np.min(hull_pts, axis=0)
    max_x, max_y = np.max(hull_pts, axis=0)
    center2 = [(min_x+max_x)/2, (min_y+max_y)/2]
    init2 = [center2[0], center2[1], max_dist * 1.5, 0.0]
    try:
        res = minimize(_enc_obj, init2, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5})
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
    except Exception:
        pass
        
    if best_params is None:
        # Fallback
        return (center[0], center[1], max_dist * 2.0, 0.0)
        
    cx, cy, s, ang = best_params
    return (cx, cy, s, ang)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    # Initial guess: Lattice packing
    init_inner = []
    init_inner.append((0.0, 0.0, 0.0))
    for i in range(6):
        a = math.radians(60*i)
        x = 1.5 * math.cos(a)
        y = 1.5 * math.sin(a)
        init_inner.append((x, y, 0.0))
    # Add 4 more from ring 2
    # (3.0, 0), (1.5, 2.598), (-1.5, 2.598), (-3.0, 0)
    init_inner.append((3.0, 0.0, 0.0))
    init_inner.append((1.5, 2.598, 0.0))
    init_inner.append((-1.5, 2.598, 0.0))
    init_inner.append((-3.0, 0.0, 0.0))
    
    # x0 for DE: inner params only
    x0_inner = []
    for p in init_inner:
        x0_inner.append(p[0])
        x0_inner.append(p[1])
        x0_inner.append(p[2])
        
    bounds_inner = []
    for i in range(N):
        bounds_inner.append((-10, 10))
        bounds_inner.append((-10, 10))
        bounds_inner.append((0, 360))
        
    best_inner = None
    best_side = float('inf')
    
    # Run DE to optimize inner hexagon positions
    # Objective for DE: only pairwise penalties. We want to minimize the "spread" or just maximize packing density?
    # Actually, if we just minimize pairwise penalty, they might fly apart.
    # We need to constrain them to be compact.
    # Strategy: Minimize the sum of squared distances to the centroid?
    # Or better: Use the enclosing hexagon side length as the objective for DE?
    # That makes the objective expensive (calls _compute_min_enclosing_hexagon which does Nelder-Mead).
    # That might be too slow for DE.
    
    # Alternative Strategy:
    # 1. Run DE to minimize a proxy objective: Pairwise Penalty + Sum of squared distances to origin (to keep them centered).
    # 2. After DE, take the best configuration and compute the exact min enclosing hexagon.
    # 3. Optionally, run a second phase where we optimize the inner positions again, but now with the objective being the actual enclosing hexagon side length (computed efficiently via a fast approximation or just accepting the cost if we reduce iterations).
    
    # Let's try a two-phase approach.
    # Phase 1: DE to find a compact, non-overlapping cluster.
    # Objective: Pairwise Penalty + lambda * (Sum of squared distances to centroid)
    
    def _phase1_obj(x):
        inner_params = []
        for i in range(N):
            inner_params.append((x[i*3], x[i*3+1], x[i*3+2]))
            
        pen_pair = 0.0
        for i in range(N):
            for j in range(i+1, N):
                pen_pair += _pairwise_penalty(inner_params[i], inner_params[j])
                
        # Centroid
        cx = np.mean([p[0] for p in inner_params])
        cy = np.mean([p[1] for p in inner_params])
        dist_sq = sum((p[0]-cx)**2 + (p[1]-cy)**2 for p in inner_params)
        
        return pen_pair + 0.1 * dist_sq
        
    try:
        res = differential_evolution(_phase1_obj, bounds_inner, maxiter=30, popsize=10, seed=42, polish=False)
        if res.x is not None:
            best_inner = res.x
    except Exception:
        best_inner = x0_inner
        
    # Check time
    if time.time() > deadline - 20:
        # Not enough time for phase 2
        pass
    else:
        # Phase 2: Refine with actual enclosing hexagon objective
        # We need to be careful with time.
        # Let's run a smaller DE or just local optimization.
        
        # Actually, let's just take the best_inner from Phase 1, compute the enclosing hex,
        # and then try to improve the inner positions using Nelder-Mead on the full objective
        # (inner positions + enclosing hex side length).
        
        # To do this, we need to interleave the optimization.
        # This is hard with standard minimizers.
        
        # Alternative: Run DE again, but with the objective being the side length of the enclosing hexagon.
        # To make it fast, we can approximate the enclosing hexagon side length.
        # Approximation: Max distance from centroid to any vertex, scaled.
        # This is a convex function of the positions? No.
        
        # Let's stick to the simpler approach:
        # 1. Get a good cluster from Phase 1.
        # 2. Compute min enclosing hex.
        # 3. Try a few random perturbations of the inner positions and re-compute min enclosing hex.
        # 4. Keep the best.
        
        # This is a heuristic local search.
        
        best_config = best_inner
        best_side_val = float('inf')
        
        # Compute initial enclosing hex
        inner_params = []
        for i in range(N):
            inner_params.append((best_config[i*3], best_config[i*3+1], best_config[i*3+2]))
            
        cx, cy, s, ang = _compute_min_enclosing_hexagon(inner_params)
        best_side_val = s
        
        # Local search: perturb inner positions
        for _ in range(50): # 50 iterations
            if time.time() > deadline - 10:
                break
                
            # Perturb
            new_config = best_config.copy()
            for i in range(N):
                if np.random.random() < 0.5:
                    new_config[i*3] += np.random.uniform(-0.5, 0.5)
                if np.random.random() < 0.5:
                    new_config[i*3+1] += np.random.uniform(-0.5, 0.5)
                if np.random.random() < 0.5:
                    new_config[i*3+2] += np.random.uniform(-15, 15)
                    
            # Check pairwise penalty
            new_inner_params = []
            for i in range(N):
                new_inner_params.append((new_config[i*3], new_config[i*3+1], new_config[i*3+2]))
                
            pen = 0.0
            for i in range(N):
                for j in range(i+1, N):
                    pen += _pairwise_penalty(new_inner_params[i], new_inner_params[j])
                    
            if pen > 1e-6:
                continue
                
            # Compute enclosing hex
            cx, cy, s, ang = _compute_min_enclosing_hexagon(new_inner_params)
            
            if s < best_side_val:
                best_side_val = s
                best_config = new_config
                # Update inner_params for next iteration? No, we just keep best_config.
                
    # Final result
    # Convert best_config to inner list
    inner = []
    for i in range(N):
        inner.append([best_config[i*3], best_config[i*3+1], best_config[i*3+2]])
        
    # Compute final enclosing hex
    inner_params = []
    for p in inner:
        inner_params.append((p[0], p[1], p[2]))
        
    cx, cy, s, ang = _compute_min_enclosing_hexagon(inner_params)
    
    return inner, (cx, cy), s, ang

if __name__ == "__main__":
    r = construct()
    print(r)

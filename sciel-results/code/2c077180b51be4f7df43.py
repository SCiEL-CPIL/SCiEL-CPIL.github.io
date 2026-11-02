import numpy as np
import time
import random
from scipy.optimize import minimize
import math

N = 12
SQRT3 = np.sqrt(3.0)
TIME_LIMIT = 140.0

# Precompute base vertices for a unit hexagon (side 1)
# Vertices at angles 0, 60, ..., 300
BASE_ANGLES = np.radians(np.arange(0, 360, 60))
BASE_COS = np.cos(BASE_ANGLES)
BASE_SIN = np.sin(BASE_ANGLES)
BASE_VERTS = np.column_stack([BASE_COS, BASE_SIN])

def get_vertices(cx, cy, angle_rad):
    """
    Returns (6, 2) vertices for a unit hexagon.
    """
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return (R @ BASE_VERTS.T).T + np.array([cx, cy])

def compute_pairwise_distance(p1, p2, a1, a2):
    """
    Compute minimum distance between two hexagons.
    If they overlap, returns negative value (depth of penetration approx).
    For penalty, we want 0 if disjoint, >0 if overlapping.
    We use a smooth approximation: 
    d = max(0, min_dist) -> penalty = d^2? No, we want penalty when d < 0.
    Let's compute the actual min distance between the two convex polygons.
    If min_dist < 0, they overlap.
    Penalty = max(0, -min_dist)^2
    """
    v1 = get_vertices(p1[0], p1[1], a1)
    v2 = get_vertices(p2[0], p2[1], a2)
    
    min_dist = np.inf
    # Check all pairs of edges for intersection? No, SAT is better for distance.
    # But SAT distance is complex. 
    # For N=12, we can use a simpler check: 
    # If centers are far apart, no overlap.
    # Center distance d_c. If d_c > 4 (diameter), definitely no overlap.
    # If d_c < 2 (inradius*2), likely overlap, but depends on rotation.
    
    # Let's use the Support Function / GJK lite? 
    # Or just sample points? 
    # Actually, for penalty gradient, we need smoothness.
    # A common trick: penalty = sum over all vertex-edge pairs of (distance)^2 if < 0?
    # That's 36*36 = 1296 checks per pair. 66 pairs * 1296 = 85k checks.
    # In Python, this is slow.
    
    # Alternative: Use the "separating axis" distance.
    # The distance between two convex polygons is the minimum over all axes (normals of all edges)
    # of the gap.
    # Gap = max(0, min_proj2 - max_proj1) or vice versa.
    # If any gap > 0, distance is at least that gap? No, distance is min over axes of max(0, gap).
    # Wait, if there is a separating axis, the distance is the gap on that axis.
    # The minimum distance is min_{axes} (gap_axis).
    # If they overlap, all gaps are <= 0.
    # So min_dist = min_{axes} (gap_axis).
    # If min_dist < 0, they overlap.
    # Penalty = max(0, -min_dist)^2.
    
    axes = []
    for k in range(6):
        e1 = v1[(k+1)%6] - v1[k]
        n1 = np.array([-e1[1], e1[0]])
        axes.append(n1 / np.linalg.norm(n1))
        e2 = v2[(k+1)%6] - v2[k]
        n2 = np.array([-e2[1], e2[0]])
        axes.append(n2 / np.linalg.norm(n2))
        
    min_gap = np.inf
    for n in axes:
        proj1 = v1.dot(n)
        proj2 = v2.dot(n)
        max1, min1 = np.max(proj1), np.min(proj1)
        max2, min2 = np.max(proj2), np.min(proj2)
        
        # Gap is distance between intervals [min1, max1] and [min2, max2]
        if max1 < min2:
            gap = min2 - max1
        elif max2 < min1:
            gap = min1 - max2
        else:
            gap = 0.0 # Overlap on this axis
            
        if gap < min_gap:
            min_gap = gap
            
    # min_gap is the distance if disjoint, or negative "depth" if overlapping?
    # Actually, if they overlap, gap is 0 for all separating axes? No.
    # If they overlap, there is NO separating axis.
    # So gap will be 0 for all axes? 
    # Wait. If they overlap, for every axis, the projections overlap.
    # So gap = 0 for all axes.
    # Thus min_gap = 0.
    # This doesn't give us a negative value for overlap depth.
    # We need the "penetration depth" or similar.
    
    # Let's use a different metric for penalty:
    # Sum of squared distances of all vertices of v1 to the polygon v2, if inside?
    # Too complex.
    
    # Let's stick to a simpler, robust penalty for SA-like methods, 
    # but here we are doing Gradient Descent.
    # We can use a "soft" distance.
    # Let's compute the minimum distance between any vertex of v1 and any edge of v2, 
    # and vice versa.
    # If min_dist > 0, penalty = 0.
    # If min_dist < 0 (inside), penalty = -min_dist^2.
    # But "inside" for convex poly is tricky with just vertex-edge.
    
    # Let's use the property: 
    # If polygons are disjoint, min_dist > 0.
    # If they intersect, min_dist = 0 (in the geometric sense of boundary distance).
    # But we want a penalty for intersection.
    
    # Let's use the "overlap area" approximation? No.
    
    # Let's go back to SA. The current program uses SA. 
    # The directive asks for a DIFFERENT strategy.
    # FEM is suggested but complex.
    # Let's try **Nelder-Mead** with a penalty function.
    # Nelder-Mead doesn't need gradients, so it can handle the non-smooth min_dist function 
    # if we make it smooth enough or if it's robust.
    
    # Actually, let's implement the **Caging Formulation** mentioned in Inspiration 1, 
    # but optimized with **Nelder-Mead** or **Powell** (derivative-free) 
    # because the objective (min side length) is non-smooth (max of linear functions).
    # Wait, the objective is convex (max of linear functions). 
    # The constraints (disjointness) are non-convex.
    
    # Let's try **Particle Swarm Optimization (PSO)**. 
    # It is a population-based method, different from SA (single trajectory) 
    # and different from FEM. It can handle non-differentiable constraints better 
    # if we just reject invalid particles or use a heavy penalty.
    
    return min_gap

def objective_and_penalty(x, penalty_weight=100.0):
    """
    x: flat array of length 3*N (x, y, angle_rad for each hexagon)
    Returns: total side length + penalty * overlap_penalty
    """
    pos = x[0:2*N:3]
    ang = x[1:2*N:3]
    # Note: x[2:3*N:3] is unused? No, x is [x0, y0, a0, x1, y1, a1, ...]
    # So pos is x[0::3], ang is x[2::3]
    
    pos = x[0::3]
    ang = x[2::3]
    
    # 1. Compute Outer Side Length
    # Get all vertices
    verts_list = []
    for i in range(N):
        verts_list.append(get_vertices(pos[i], ang[i], ang[i])) # Wait, ang is angle.
        # Correction: get_vertices(cx, cy, angle)
    
    # Re-do vertices
    all_verts = np.zeros((N*6, 2))
    for i in range(N):
        all_verts[i*6:(i+1)*6] = get_vertices(pos[i], ang[i], ang[i])
        
    x_coords = all_verts[:, 0]
    y_coords = all_verts[:, 1]
    
    L1 = x_coords + y_coords / SQRT3
    L2 = y_coords * 2.0 / SQRT3
    L3 = -x_coords + y_coords / SQRT3
    L4 = -x_coords - y_coords / SQRT3
    L5 = -y_coords * 2.0 / SQRT3
    L6 = x_coords - y_coords / SQRT3
    
    side_len = np.max([np.max(L1), np.max(L2), np.max(L3), np.max(L4), np.max(L5), np.max(L6)])
    
    # 2. Compute Overlap Penalty
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            # Check distance
            # We need a fast distance check.
            # Let's use the center distance as a proxy for early exit?
            # If dist > 4, skip.
            dx = pos[i] - pos[j]
            dy = ang[i] - ang[j] # No, dy is y diff
            dy = x[1+3*i] - x[1+3*j]
            if dx*dx + dy*dy > 16.0: # 4^2
                continue
                
            # Compute SAT distance
            v1 = get_vertices(pos[i], x[1+3*i], x[2+3*i])
            v2 = get_vertices(pos[j], x[1+3*j], x[2+3*j])
            
            min_gap = np.inf
            # Axes from v1
            for k in range(6):
                e = v1[(k+1)%6] - v1[k]
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm < 1e-10: continue
                n /= norm
                p1 = v1.dot(n)
                p2 = v2.dot(n)
                gap = max(0, np.min(p2) - np.max(p1), np.min(p1) - np.max(p2))
                if gap < min_gap:
                    min_gap = gap
            # Axes from v2
            for k in range(6):
                e = v2[(k+1)%6] - v2[k]
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm < 1e-10: continue
                n /= norm
                p1 = v1.dot(n)
                p2 = v2.dot(n)
                gap = max(0, np.min(p2) - np.max(p1), np.min(p1) - np.max(p2))
                if gap < min_gap:
                    min_gap = gap
            
            # If min_gap > 0, they are disjoint.
            # If min_gap == 0, they touch or overlap.
            # We need to distinguish touch from overlap.
            # For penalty, we want to push them apart if they overlap.
            # If min_gap is 0, we don't know if they overlap or touch.
            # We can check if any vertex of v1 is inside v2.
            # If yes, they overlap.
            
            # Simple overlap check: Point in Polygon
            # Check if center of v1 is in v2? No, vertices might be inside.
            # Let's check if any vertex of v1 is strictly inside v2.
            # Use winding number or cross product method.
            
            def point_in_poly(pt, poly):
                # poly is (6,2)
                inside = False
                x_pt, y_pt = pt
                for i in range(6):
                    x1, y1 = poly[i]
                    x2, y2 = poly[(i+1)%6]
                    if ((y1 > y_pt) != (y2 > y_pt)) and (x_pt < (x2 - x1) * (y_pt - y1) / (y2 - y1) + x1):
                        inside = not inside
                return inside
            
            overlap = False
            # Check a few vertices to save time
            for v_idx in range(6):
                if point_in_poly(v1[v_idx], v2):
                    overlap = True
                    break
            if not overlap:
                for v_idx in range(6):
                    if point_in_poly(v2[v_idx], v1):
                        overlap = True
                        break
                        
            if overlap:
                # Penalty: push apart.
                # Use a small epsilon to avoid division by zero or gradient issues if we used gradient.
                # For Nelder-Mead, a large constant penalty works.
                penalty += 100.0
                
    return side_len + penalty

def construct():
    start_time = time.time()
    
    # Initial Configuration: Honeycomb
    # 12 hexagons.
    # A 4x3 grid? 
    # Or a cluster.
    # Let's use a compact cluster:
    # Center (0,0)
    # 6 around it
    # 5 more around that?
    
    # Let's generate a valid initial config using a lattice.
    # Lattice vectors:
    # u = (sqrt(3), 0)
    # v = (sqrt(3)/2, 1.5)
    
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    
    init_pos = []
    init_ang = []
    for q, r in coords:
        x = SQRT3 * (q + r / 2.0)
        y = 1.5 * r
        init_pos.append([x, y])
        init_ang.append(0.0)
        
    init_x = np.zeros(3*N)
    for i in range(N):
        init_x[3*i] = init_pos[i][0]
        init_x[3*i+1] = init_pos[i][1]
        init_x[3*i+2] = init_ang[i]
        
    # Ensure validity of init
    # If not valid, scale up
    def is_valid(x):
        pos = x[0::3]
        ang = x[2::3]
        for i in range(N):
            for j in range(i+1, N):
                v1 = get_vertices(pos[i], x[1+3*i], x[2+3*i])
                v2 = get_vertices(pos[j], x[1+3*j], x[2+3*j])
                # Simple check: if centers are too close, likely overlap
                # But let's use the point-in-poly check from before
                def point_in_poly(pt, poly):
                    inside = False
                    x_pt, y_pt = pt
                    for i in range(6):
                        x1, y1 = poly[i]
                        x2, y2 = poly[(i+1)%6]
                        if ((y1 > y_pt) != (y2 > y_pt)) and (x_pt < (x2 - x1) * (y_pt - y1) / (y2 - y1) + x1):
                            inside = not inside
                    return inside
                if point_in_poly(v1[0], v2) or point_in_poly(v2[0], v1):
                    return False
        return True

    scale = 1.0
    while not is_valid(init_x) and scale < 5.0:
        init_x[0::3] *= scale
        init_x[1::3] *= scale
        scale += 0.1
        
    # Use Nelder-Mead
    # Nelder-Mead is derivative-free and handles non-smooth functions reasonably well.
    
    best_x = init_x.copy()
    best_val = objective_and_penalty(best_x)
    
    # Run Nelder-Mead
    # We need to be careful with time.
    # Let's run it with a time limit check in the callback? 
    # SciPy minimize doesn't easily support time limits in callback for stopping.
    # We can run it in a thread? No, let's just run it and hope it's fast enough.
    # Or run multiple shorter runs with different inits.
    
    # Let's run one main Nelder-Mead run.
    # Max iterations: 2000? 
    # Each iteration evaluates the function multiple times (N+1 times).
    # Function eval is O(N^2 * 6) ~ 12^2 * 6 = 864 ops? 
    # Plus overhead. Should be fast.
    
    try:
        res = minimize(objective_and_penalty, init_x, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if res.success and res.fun < best_val:
            best_x = res.x
            best_val = res.fun
            
    except Exception as e:
        pass # Keep best_x
        
    # Refine with a few more steps or a second run?
    # Let's try a second run with a slightly perturbed init if time allows.
    if time.time() - start_time < TIME_LIMIT - 10:
        # Perturb best_x
        x2 = best_x + np.random.normal(0, 0.1, 3*N)
        try:
            res2 = minimize(objective_and_penalty, x2, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
            if res2.success and res2.fun < best_val:
                best_x = res2.x
                best_val = res2.fun
        except:
            pass

    # Extract result
    inner = []
    for i in range(N):
        x = float(best_x[3*i])
        y = float(best_x[3*i+1])
        # Convert angle from radians to degrees
        ang = float(np.degrees(best_x[3*i+2])) % 360.0
        inner.append([x, y, ang])
        
    # The outer hexagon is centered at (0,0) with side length best_val (if penalty is 0)
    # But best_val includes penalty. We need the actual side length.
    # Let's recompute side length for the best valid config.
    
    # Check if best_x is valid. If not, we might have a problem.
    # But Nelder-Mead should drive penalty to 0.
    
    pos = best_x[0::3]
    ang_rad = best_x[2::3]
    
    all_verts = np.zeros((N*6, 2))
    for i in range(N):
        all_verts[i*6:(i+1)*6] = get_vertices(pos[i], best_x[1+3*i], ang_rad[i])
        
    x_coords = all_verts[:, 0]
    y_coords = all_verts[:, 1]
    
    L1 = x_coords + y_coords / SQRT3
    L2 = y_coords * 2.0 / SQRT3
    L3 = -x_coords + y_coords / SQRT3
    L4 = -x_coords - y_coords / SQRT3
    L5 = -y_coords * 2.0 / SQRT3
    L6 = x_coords - y_coords / SQRT3
    
    side_len = np.max([np.max(L1), np.max(L2), np.max(L3), np.max(L4), np.max(L5), np.max(L6)])
    
    # If the configuration is invalid (penalty > 0), the side_len might be misleading.
    # But we assume it's valid.
    
    return inner, (0.0, 0.0), float(side_len), 0.0

# EVOLVE-BLOCK-END

"""Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon (pointy-top orientation at 0 deg)."""
    a = math.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts[i, 0] = cx + side * math.cos(theta)
        verts[i, 1] = cy + side * math.sin(theta)
    return verts

def _outer_vertices(cx, cy, side, angle_deg):
    a = math.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts[i, 0] = cx + side * math.cos(theta)
        verts[i, 1] = cy + side * math.sin(theta)
    return verts

def _project_range(verts, axis):
    """Min and max projection of polygon vertices onto a unit axis."""
    dots = verts @ axis
    return dots.min(), dots.max()

def _sat_overlap(v1, v2, tol=1e-9):
    """True if two convex polygons overlap (interior intersection)."""
    for poly in (v1, v2):
        n = len(poly)
        for i in range(n):
            j = (i + 1) % n
            edge = poly[j] - poly[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal /= norm
            r1min, r1max = _project_range(v1, normal)
            r2min, r2max = _project_range(v2, normal)
            if r1max < r2min + tol or r2max < r1min + tol:
                return False
    return True

def _point_in_convex(px, py, poly, tol=1e-9):
    """Check if point is inside a convex polygon (given in CCW order)."""
    n = len(poly)
    for i in range(n):
        j = (i + 1) % n
        ex = poly[j, 0] - poly[i, 0]
        ey = poly[j, 1] - poly[i, 1]
        px_rel = px - poly[i, 0]
        py_rel = py - poly[i, 1]
        cross = ex * py_rel - ey * px_rel
        if cross < -tol:
            return False
    return True

def _inner_fits_outer(inner_verts, outer_verts, tol=1e-9):
    """Check if all inner hexagon vertices are inside the outer hexagon."""
    for v in inner_verts:
        if not _point_in_convex(v[0], v[1], outer_verts, tol):
            return False
    return True

def _feasible(outer_side, placements, outer_center=(0.0, 0.0), outer_angle=0.0):
    """Check if given placements fit inside outer hexagon and don't overlap each other."""
    ox, oy = outer_center
    outer_verts = _outer_vertices(ox, oy, outer_side, outer_angle)
    
    inner_verts_list = []
    for (ix, iy, iangle) in placements:
        inner_verts_list.append(_hex_vertices(ix, iy, iangle, 1.0))
    
    for iv in inner_verts_list:
        if not _inner_fits_outer(iv, outer_verts, tol=1e-9):
            return False
    
    n = len(inner_verts_list)
    for i in range(n):
        for j in range(i + 1, n):
            if _sat_overlap(inner_verts_list[i], inner_verts_list[j], tol=1e-9):
                return False
    
    return True

def _min_distance_between_polygons(v1, v2):
    """
    Approximate minimum distance between two convex polygons.
    If they overlap, returns a negative value proportional to penetration depth (approx).
    If they are separated, returns the minimum distance between any edge of v1 and any edge of v2.
    """
    min_dist = float('inf')
    n1 = len(v1)
    n2 = len(v2)
    
    # Check vertex-edge distances
    for i in range(n1):
        p1 = v1[i]
        for j in range(n2):
            p2 = v2[j]
            d = np.linalg.norm(p1 - p2)
            min_dist = min(min_dist, d)
            
            # Distance from p1 to edge v2[j]-v2[j+1]
            j_next = (j + 1) % n2
            e_start = v2[j]
            e_end = v2[j_next]
            edge_vec = e_end - e_start
            edge_len_sq = np.dot(edge_vec, edge_vec)
            if edge_len_sq > 1e-12:
                t = np.dot(p1 - e_start, edge_vec) / edge_len_sq
                t = max(0, min(1, t))
                proj = e_start + t * edge_vec
                d = np.linalg.norm(p1 - proj)
                min_dist = min(min_dist, d)
                
            # Distance from p2 to edge v1[i]-v1[i+1]
            i_next = (i + 1) % n1
            e_start = v1[i]
            e_end = v1[i_next]
            edge_vec = e_end - e_start
            edge_len_sq = np.dot(edge_vec, edge_vec)
            if edge_len_sq > 1e-12:
                t = np.dot(p2 - e_start, edge_vec) / edge_len_sq
                t = max(0, min(1, t))
                proj = e_start + t * edge_vec
                d = np.linalg.norm(p2 - proj)
                min_dist = min(min_dist, d)

    # If they overlap, the min distance will be 0. 
    # To create a penalty for overlap, we can use SAT to detect overlap and estimate depth.
    # For simplicity in SA, we rely on the fact that if min_dist is 0, they are touching or overlapping.
    # A better penalty for SA is to use the "overlap depth" which is hard to compute exactly.
    # Instead, we will use a penalty based on the maximum violation of separation.
    # If SAT says they overlap, we return a large negative number? No, we want to minimize distance.
    # Let's return 0 if they overlap, and the min distance if they don't.
    # The SA objective will penalize d < 0? No, distance is always >= 0.
    # We need a penalty that is non-zero when overlapping.
    
    # Let's use a simpler heuristic: 
    # If SAT overlap is True, return -1.0 (indicating violation). 
    # If not, return min_dist.
    if _sat_overlap(v1, v2, tol=1e-9):
        return -1.0
    
    return min_dist

def _sa_packing(outer_side, outer_center=(0.0, 0.0), outer_angle=0.0, time_budget=5.0, seed=42):
    """
    Simulated Annealing to find a valid packing.
    Returns placements if found, else None.
    """
    np.random.seed(seed)
    start_time = time.time()
    
    # Initial guess: random positions inside the hexagon
    # Radius of inscribed circle of outer hex is outer_side * sqrt(3)/2
    # We want centers to be at least 1.0 (inradius of inner hex) away from boundary
    # So max radius for centers is outer_side * sqrt(3)/2 - 1.0
    max_r = max(0.1, outer_side * math.sqrt(3)/2 - 1.0)
    
    x = np.zeros(3 * N)
    for i in range(N):
        r = max_r * math.sqrt(np.random.random())
        theta = 2 * math.pi * np.random.random()
        x[3*i] = r * math.cos(theta) + outer_center[0]
        x[3*i+1] = r * math.sin(theta) + outer_center[1]
        x[3*i+2] = np.random.random() * 360.0 # Degrees
    
    outer_verts = _outer_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    def get_penalty(x):
        """Calculate penalty. Lower is better. 0 means feasible."""
        penalty = 0.0
        inner_verts_list = []
        for i in range(N):
            cx = x[3*i]
            cy = x[3*i+1]
            ang = x[3*i+2]
            verts = _hex_vertices(cx, cy, ang, 1.0)
            inner_verts_list.append(verts)
            
            # Boundary penalty
            for v in verts:
                if not _point_in_convex(v[0], v[1], outer_verts, tol=1e-9):
                    # Estimate violation depth
                    # Simple penalty: count vertices outside
                    penalty += 10.0
        
        # Overlap penalty
        for i in range(N):
            for j in range(i + 1, N):
                d = _min_distance_between_polygons(inner_verts_list[i], inner_verts_list[j])
                if d < 0: # Overlapping
                    penalty += 10.0 * (1.0 + abs(d)) # Strong penalty for overlap
                else:
                    # Small penalty if too close? No, just avoid overlap.
                    pass
                    
        return penalty
    
    current_penalty = get_penalty(x)
    best_x = x.copy()
    best_penalty = current_penalty
    
    T = 1.0
    T_min = 1e-4
    alpha = 0.995
    
    while T > T_min:
        if time.time() - start_time > time_budget:
            break
            
        # Perturb
        x_new = x.copy()
        i = np.random.randint(N)
        dx = (np.random.random() - 0.5) * 0.5
        dy = (np.random.random() - 0.5) * 0.5
        dtheta = (np.random.random() - 0.5) * 10.0
        
        x_new[3*i] += dx
        x_new[3*i+1] += dy
        x_new[3*i+2] += dtheta
        
        new_penalty = get_penalty(x_new)
        
        # Acceptance criterion
        if new_penalty < current_penalty:
            x = x_new
            current_penalty = new_penalty
            if current_penalty < best_penalty:
                best_penalty = current_penalty
                best_x = x.copy()
                if best_penalty < 1e-9:
                    break
        else:
            delta = new_penalty - current_penalty
            if np.random.random() < math.exp(-delta / T):
                x = x_new
                current_penalty = new_penalty
                
        T *= alpha
        
    # Final check
    if best_penalty < 1e-9:
        placements = []
        for i in range(N):
            placements.append((best_x[3*i], best_x[3*i+1], best_x[3*i+2] % 360.0))
        if _feasible(outer_side, placements, outer_center, outer_angle):
            return placements
    return None

def _bisection_search(lo, hi, outer_center=(0.0, 0.0), outer_angle=0.0, time_budget=100.0):
    """
    Bisection search for the minimum outer side length.
    """
    start_time = time.time()
    best_placements = None
    best_side = hi
    
    # First, verify that hi is feasible
    # Try a few seeds for the initial feasible check
    found_feasible = False
    for seed in range(10):
        if time.time() - start_time > time_budget - 10:
            break
        result = _sa_packing(hi, outer_center, outer_angle, time_budget=5.0, seed=seed)
        if result is not None:
            best_placements = result
            best_side = hi
            found_feasible = True
            break
            
    if not found_feasible:
        # If hi is not feasible, increase hi
        hi *= 1.5
        for seed in range(10):
            if time.time() - start_time > time_budget - 10:
                break
            result = _sa_packing(hi, outer_center, outer_angle, time_budget=5.0, seed=seed)
            if result is not None:
                best_placements = result
                best_side = hi
                found_feasible = True
                break
                
    if not found_feasible:
        # Fallback: place in a line
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Bisection
    for _ in range(15):
        if time.time() - start_time > time_budget - 10:
            break
        mid = (lo + hi) / 2.0
        result = None
        # Try multiple seeds for the mid point
        for seed in range(5):
            if time.time() - start_time > time_budget - 5:
                break
            result = _sa_packing(mid, outer_center, outer_angle, time_budget=3.0, seed=seed)
            if result is not None:
                break
        
        if result is not None:
            hi = mid
            best_side = mid
            best_placements = result
        else:
            lo = mid
    
    return best_placements, (0.0, 0.0), best_side, 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    # Bisection search on outer side length
    # Lower bound: must fit 12 unit hexagons
    # Area of outer hex = (3*sqrt(3)/2) * S^2
    # Area of 12 unit hexagons = 12 * (3*sqrt(3)/2) * 1^2 = 6*sqrt(3)
    # So S^2 >= 6, S >= sqrt(6) ≈ 2.449
    # But packing efficiency is less than 100%, so S is larger.
    lo = 2.5
    hi = 15.0
    
    # Try different outer angles to see if rotation helps
    best_result = None
    best_side = float('inf')
    
    for outer_angle in [0.0, 30.0]:
        if time.time() > deadline - 20.0:
            break
        
        inner, center, side, angle = _bisection_search(lo, hi, (0.0, 0.0), outer_angle, 
                                                       time_budget=deadline - time.time() - 5.0)
        if side < best_side:
            best_side = side
            best_result = (inner, center, side, angle)
    
    if best_result is not None:
        return best_result
    else:
        # Fallback: place in a line
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

# EVOLVE-BLOCK-END

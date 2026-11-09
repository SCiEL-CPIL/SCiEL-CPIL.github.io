import math
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 12
sqrt3 = math.sqrt(3.0)

# Precompute normal angles for the outer hexagon (flat-top)
# The outer hexagon is fixed at angle 0.
# Normals are at 30, 90, 150, 210, 270, 330 degrees.
normal_angles = [math.radians(30 + 60 * j) for j in range(6)]
normal_cos = np.array([math.cos(a) for a in normal_angles])
normal_sin = np.array([math.sin(a) for a in normal_angles])

def get_hex_vertices(cx, cy, theta):
    """Returns 6 vertices of a unit hexagon at (cx, cy) with rotation theta."""
    # Vertices of a unit hexagon centered at origin, flat-top (angle 0)
    # Angles: 0, 60, 120, 180, 240, 300
    angles = np.array([0, 60, 120, 180, 240, 300]) * math.pi / 180.0 + theta
    vx = cx + np.cos(angles)
    vy = cy + np.sin(angles)
    return np.column_stack((vx, vy))

def calc_L_bound(positions, angles):
    """
    Calculate the minimum side length L of the outer hexagon 
    (centered at 0,0, angle 0) to contain all inner hexagons.
    """
    max_proj = 0.0
    for i in range(N):
        cx, cy = positions[i]
        theta = angles[i]
        verts = get_hex_vertices(cx, cy, theta)
        # Project onto 6 normals
        projs = verts[:, 0] * normal_cos + verts[:, 1] * normal_sin
        # Max projection for this hexagon
        current_max = np.max(projs)
        if current_max > max_proj:
            max_proj = current_max
    # L is related to max_proj.
    # For a regular hexagon of side L, the distance from center to side (apothem) is L * sqrt(3) / 2.
    # The projection of a vertex onto the normal must be <= apothem.
    # So max_proj <= L * sqrt(3) / 2
    # L >= max_proj * 2 / sqrt(3)
    return max_proj * 2.0 / sqrt3

def min_dist_hexagon_hexagon(v1, v2):
    """
    Calculate the minimum distance between two convex polygons (hexagons).
    Returns negative distance if they overlap (penetration depth).
    Uses Separating Axis Theorem logic for penetration depth.
    v1, v2: (6, 2) arrays of vertices.
    """
    # Axes to check: normals of edges of v1 and v2
    # Since they are regular hexagons, we can just use the 12 potential axes.
    # However, calculating axes from vertices is safer.
    
    # Helper to get edge normals
    def get_normals(verts):
        normals = []
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i + 1) % 6]
            edge = p2 - p1
            # Normal is perpendicular to edge
            # For counter-clockwise vertices, normal is (edge_y, -edge_x) normalized?
            # Let's just use the edge vector rotated 90 deg.
            nx = edge[1]
            ny = -edge[0]
            norm = math.sqrt(nx*nx + ny*ny)
            if norm > 1e-9:
                normals.append(np.array([nx/norm, ny/norm]))
        return normals

    normals1 = get_normals(v1)
    normals2 = get_normals(v2)
    
    min_gap = float('inf')
    all_overlap = True
    
    for n in normals1 + normals2:
        # Project v1 and v2 onto n
        proj1 = np.dot(v1, n)
        proj2 = np.dot(v2, n)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check for gap
        if max1 < min2 or max2 < min1:
            # Disjoint on this axis
            gap = min(abs(min2 - max1), abs(min1 - max2))
            if gap < min_gap:
                min_gap = gap
            all_overlap = False
        else:
            # Overlap on this axis
            # Calculate overlap depth
            overlap = min(max1, max2) - max(min1, min2)
            if overlap < 0:
                # Should not happen if they overlap on this axis
                pass
            # We track the minimum overlap depth to find the minimal translation
            # Actually, for SAT, if they overlap on ALL axes, the minimum penetration
            # is the minimum overlap depth among all axes?
            # Yes.
            if overlap < min_gap:
                min_gap = overlap

    if all_overlap:
        return -min_gap # Penetration depth
    else:
        return min_gap

def calc_overlap_penalty(positions, angles):
    """
    Calculate the maximum overlap penetration depth among all pairs.
    Returns 0 if no overlaps, positive value if overlaps exist.
    """
    max_pen = 0.0
    verts_list = []
    for i in range(N):
        verts_list.append(get_hex_vertices(positions[i][0], positions[i][1], angles[i]))
    
    for i in range(N):
        for j in range(i + 1, N):
            d = min_dist_hexagon_hexagon(verts_list[i], verts_list[j])
            if d < 0:
                pen = -d
                if pen > max_pen:
                    max_pen = pen
    return max_pen

def cost_function(params):
    """
    Cost function for optimization.
    params: [x0, y0, a0, x1, y1, a1, ...]
    """
    positions = np.array(params[0:2*N]).reshape(N, 2)
    angles = np.array(params[2*N:3*N])
    
    L_b = calc_L_bound(positions, angles)
    pen = calc_overlap_penalty(positions, angles)
    
    # Heavily penalize overlaps
    return L_b + 1000.0 * pen

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # 1. Grid Search Phase (Fast)
    # Generate 19 centers of side-3 hexagon cluster
    all_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                all_centers.append((x, y))
    
    # Try a few cluster rotations
    best_initial_params = None
    best_initial_score = float('inf')
    
    for cluster_angle_deg in [0, 10, 20, 30, 40, 50]:
        if time.time() > deadline - 20: break
        ca = math.cos(math.radians(cluster_angle_deg))
        sa = math.sin(math.radians(cluster_angle_deg))
        rotated_centers = []
        for x, y in all_centers:
            rx = x * ca - y * sa
            ry = x * sa + y * ca
            rotated_centers.append((rx, ry))
            
        # Try all subsets of 12
        # To speed up, just pick the 12 closest to origin? No, that's not optimal.
        # But for initialization, it's good.
        # Let's sort by distance from origin and take first 12?
        # Or take the ones that form a compact shape.
        # The current code tried all combinations. That is C(19,12) = 50388.
        # 50388 * 6 angles = 300k iterations.
        # Each iteration calculates L_bound (fast).
        # Let's do it.
        
        indices = list(range(19))
        for subset in combinations(indices, 12):
            if time.time() > deadline - 20: break
            # Construct positions
            pos = [rotated_centers[i] for i in subset]
            # Initial angles: 0 for all
            angs = [0.0] * 12
            
            L_b = calc_L_bound(np.array(pos), np.array(angs))
            # Check for overlaps roughly (distance between centers)
            # If centers are too close, skip
            valid = True
            for i in range(12):
                for j in range(i+1, 12):
                    dx = pos[i][0] - pos[j][0]
                    dy = pos[i][1] - pos[j][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < 1.7: # Roughly 2 * apothem = 1.73
                         # Might overlap, but let's not skip yet, just penalize in cost
                         pass
            
            if L_b < best_initial_score:
                best_initial_score = L_b
                # Store params
                p = []
                for i in range(12):
                    p.extend([pos[i][0], pos[i][1], 0.0])
                best_initial_params = p

    if best_initial_params is None:
        # Fallback
        best_initial_params = [0.0] * 36
        best_initial_score = 10.0

    # 2. Continuous Optimization Phase
    # Use Nelder-Mead to refine
    # We run multiple restarts
    best_final_L = best_initial_score
    best_final_params = best_initial_params
    
    # Number of restarts
    n_restarts = 5
    
    for r in range(n_restarts):
        if time.time() > deadline - 10: break
        
        # Perturb initial params
        if r == 0:
            x0 = np.array(best_initial_params)
        else:
            x0 = np.array(best_initial_params)
            # Add noise
            noise = np.random.normal(0, 0.5, size=x0.shape)
            # Keep angles in [0, 60]
            for i in range(12):
                x0[2*N + i] += noise[2*N + i]
                x0[2*N + i] %= 60.0
            x0[0:2*N] += noise[0:2*N]
        
        # Bounds for Nelder-Mead are not directly supported, but we can use penalty or trust-constr?
        # Nelder-Mead doesn't have bounds. We rely on the penalty function.
        # However, if positions go far away, L_bound increases, so it's self-limiting.
        
        try:
            res = minimize(cost_function, x0, method='Nelder-Mead', 
                           options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 2000, 'maxfev': 2000})
            
            if res.success or res.fun < best_final_L:
                # Check if valid (low penalty)
                pos = np.array(res.x[0:2*N]).reshape(N, 2)
                angs = np.array(res.x[2*N:3*N])
                pen = calc_overlap_penalty(pos, angs)
                if pen < 1e-3:
                    L_val = calc_L_bound(pos, angs)
                    if L_val < best_final_L:
                        best_final_L = L_val
                        best_final_params = res.x
        except Exception as e:
            pass

    # 3. Construct Output
    pos = np.array(best_final_params[0:2*N]).reshape(N, 2)
    angs = np.array(best_final_params[2*N:3*N])
    
    inner = []
    for i in range(N):
        inner.append([pos[i][0], pos[i][1], math.degrees(angs[i])])
    
    return inner, (0.0, 0.0), best_final_L, 0.0

# EVOLVE-BLOCK-END

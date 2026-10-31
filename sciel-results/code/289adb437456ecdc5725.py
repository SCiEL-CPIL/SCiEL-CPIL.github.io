"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts

def compute_required_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_deg)
    outer_center: (x, y)
    outer_angle_deg: rotation of the outer hexagon
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    # Normals for a hexagon with angle 0 (vertices at 0, 60, ...):
    # Outward normals are at 30, 90, 150, 210, 270, 330 degrees.
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_projection = 0.0
    
    # Rotate outer frame to match outer_angle
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            # Translate to outer center
            dx = v[0] - ox
            dy = v[1] - oy
            # Rotate points by -outer_angle to align with standard hex axes
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj
    
    # S * sqrt(3) / 2 >= max_projection
    # S >= 2 * max_projection / sqrt(3)
    return 2.0 * max_projection / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, temp=0.05):
    """
    Compute a smooth approximation of the minimum outer side length using log-sum-exp.
    This provides gradients for optimization.
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    # Collect all projections
    projections = []
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                projections.append(proj)
    
    if not projections:
        return 0.0
    
    # Log-sum-exp for smooth max
    max_proj = max(projections)
    if max_proj < -1000:
        return 0.0
    
    sum_exp = 0.0
    for p in projections:
        sum_exp += math.exp((p - max_proj) / temp)
    
    smooth_max = max_proj + temp * math.log(sum_exp)
    
    # S >= 2 * max_projection / sqrt(3)
    return 2.0 * smooth_max / SQRT3

def sat_overlap_check(v1, v2, eps=1e-9):
    """
    Check if two convex polygons (defined by their vertices) overlap using SAT.
    v1, v2: lists of (x, y) tuples.
    Returns True if they overlap (including touching).
    """
    axes = []
    
    # Get axes from edges of v1
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    # Get axes from edges of v2
    for i in range(len(v2)):
        p1 = v2[i]
        p2 = v2[(i + 1) % len(v2)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    for axis in axes:
        min1 = min(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        max1 = max(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        min2 = min(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        max2 = max(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        
        # Check for gap
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    
    return True

def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    for i in range(n):
        vi = get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
        for j in range(i + 1, n):
            vj = get_hex_vertices(inner[j][0], inner[j][1], 1.0, math.radians(inner[j][2]))
            if sat_overlap_check(vi, vj, eps):
                return False
    return True

def hexagon_distance(v1, v2):
    """
    Compute the minimum distance between two convex hexagons.
    v1, v2: lists of (x, y) tuples.
    Returns 0 if they overlap or touch.
    """
    min_dist = float('inf')
    
    # Check all edge-edge distances
    n1 = len(v1)
    n2 = len(v2)
    
    for i in range(n1):
        p1 = v1[i]
        p2 = v1[(i + 1) % n1]
        
        for j in range(n2):
            q1 = v2[j]
            q2 = v2[(j + 1) % n2]
            
            # Compute distance between two line segments
            d = segment_distance(p1, p2, q1, q2)
            if d < min_dist:
                min_dist = d
    
    return min_dist

def segment_distance(p1, p2, q1, q2):
    """
    Compute minimum distance between two line segments.
    p1, p2, q1, q2: (x, y) tuples.
    """
    # Get segment vectors
    d1 = (p2[0] - p1[0], p2[1] - p1[1])
    d2 = (q2[0] - q1[0], q2[1] - q1[1])
    r = (p1[0] - q1[0], p1[1] - q1[1])
    
    a = d1[0] * d1[0] + d1[1] * d1[1]
    e = d2[0] * d2[0] + d2[1] * d2[1]
    f = d2[0] * r[0] + d2[1] * r[1]
    
    EPS = 1e-12
    
    if a <= EPS and e <= EPS:
        # Both segments are points
        return math.sqrt(r[0] * r[0] + r[1] * r[1])
    
    if a <= EPS:
        # First segment is a point
        t = max(0.0, min(1.0, f / e))
        dist = (r[0] + d2[0] * t, r[1] + d2[1] * t)
        return math.sqrt(dist[0] * dist[0] + dist[1] * dist[1])
    
    c = d1[0] * r[0] + d1[1] * r[1]
    
    if e <= EPS:
        # Second segment is a point
        t = max(0.0, min(1.0, -c / a))
        dist = (r[0] - d1[0] * t, r[1] - d1[1] * t)
        return math.sqrt(dist[0] * dist[0] + dist[1] * dist[1])
    
    b = d1[0] * d2[0] + d1[1] * d2[1]
    denom = a * e - b * b
    
    s = 0.0
    t = 0.0
    
    if denom > EPS:
        s = max(0.0, min(1.0, (b * f - c * e) / denom))
    
    t = max(0.0, min(1.0, (s * b + f) / e))
    
    dist = (r[0] + d1[0] * s - d2[0] * t, r[1] + d1[1] * s - d2[1] * t)
    return math.sqrt(dist[0] * dist[0] + dist[1] * dist[1])

def compute_overlap_penalty(inner, min_dist=SQRT3 - 0.01):
    """
    Compute a smooth penalty for overlaps based on minimum distance between hexagons.
    More accurate than center-distance proxy.
    """
    n = len(inner)
    if n < 2:
        return 0.0
    
    penalty = 0.0
    
    for i in range(n):
        vi = get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
        for j in range(i + 1, n):
            vj = get_hex_vertices(inner[j][0], inner[j][1], 1.0, math.radians(inner[j][2]))
            
            dist = hexagon_distance(vi, vj)
            
            if dist < min_dist:
                diff = min_dist - dist
                penalty += diff * diff
    
    return float(penalty)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    # 1. Define 11 hexagons on a hexagonal lattice
    # Center + 6 neighbors + 4 more to make a compact cluster
    # Axial coordinates (q, r)
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2)
    ]
    
    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]
    
    # 2. Convert axial to Cartesian
    # Standard hex grid: 
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    epsilon = 1e-5
    scale = 1.0 + epsilon
    
    centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        centers.append((x, y))
        
    # Initial layout: All hexagons aligned (angle 0)
    initial_inner = [[x, y, 0.0] for x, y in centers]
    
    best_side = float('inf')
    best_inner = initial_inner
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0

    # Calculate centroid
    cx_avg = sum(c[0] for c in centers) / len(centers)
    cy_avg = sum(c[1] for c in centers) / len(centers)
    
    # 3. Optimize Outer Hexagon parameters (cx, cy, angle)
    # The inner hexagons are fixed in this phase (rigid body assumption)
    
    def objective_outer_smooth(params):
        cx, cy, angle_deg = params
        side = compute_required_side_smooth(initial_inner, (cx, cy), angle_deg, temp=0.05)
        return side
    
    def objective_outer_exact(params):
        cx, cy, angle_deg = params
        side = compute_required_side(initial_inner, (cx, cy), angle_deg)
        return side
    
    # Try multiple diverse initial angles to avoid local minima
    best_result = None
    best_val = float('inf')
    
    # The optimal angle is likely related to the symmetry of the cluster.
    # The cluster has some symmetry, but not full 6-fold.
    # Let's try a range of angles.
    init_angles = [0, 15, 30, 45, 60, 75, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    
    for init_angle in init_angles:
        if time.time() > deadline - 40:
            break
            
        init_params = [cx_avg, cy_avg, float(init_angle)]
        
        try:
            # Use smooth objective for gradient-based optimization
            res = minimize(objective_outer_smooth, init_params, method='L-BFGS-B', 
                          options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.fun < best_val:
                # Verify with exact objective
                exact_val = objective_outer_exact(res.x)
                if exact_val < best_val:
                    best_val = exact_val
                    best_result = res.x
        except Exception:
            pass
            
    if best_result is not None:
        best_outer_center = (best_result[0], best_result[1])
        best_outer_angle = best_result[2]
        best_side = best_val
    else:
        # Fallback
        best_outer_center = (cx_avg, cy_avg)
        best_outer_angle = 0.0
        best_side = compute_required_side(initial_inner, best_outer_center, best_outer_angle)

    # 4. Joint Optimization: Allow small perturbations of inner hexagons
    # This step refines the "rigid" assumption by allowing local adjustments
    # that might squeeze the container slightly tighter.
    
    def objective_joint_smooth(params):
        # Unpack params: [x1, y1, a1, ..., xN, yN, aN, cx, cy, angle]
        inner_list = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            a = params[i*3+2]
            inner_list.append([x, y, a])
        
        ocx = params[N*3]
        ocy = params[N*3+1]
        oangle = params[N*3+2]
        
        side = compute_required_side_smooth(inner_list, (ocx, ocy), oangle, temp=0.05)
        
        # Penalty for overlaps (smooth)
        penalty = compute_overlap_penalty(inner_list)
        
        return side + 500.0 * penalty
    
    def objective_joint_exact(params):
        # Unpack params: [x1, y1, a1, ..., xN, yN, aN, cx, cy, angle]
        inner_list = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            a = params[i*3+2]
            inner_list.append([x, y, a])
        
        ocx = params[N*3]
        ocy = params[N*3+1]
        oangle = params[N*3+2]
        
        side = compute_required_side(inner_list, (ocx, ocy), oangle)
        
        # Penalty for overlaps
        penalty = compute_overlap_penalty(inner_list)
        
        return side + 1000.0 * penalty
    
    # Initial params: current best inner + best outer
    initial_params = []
    for x, y, a in initial_inner:
        initial_params.extend([x, y, a])
    initial_params.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])
    
    # Use L-BFGS-B for the joint optimization with smooth objective
    current_params = np.array(initial_params)
    current_val = objective_joint_smooth(current_params)
    best_params = current_params.copy()
    best_val_joint = current_val
    
    try:
        # Use smooth objective for gradient-based optimization
        res = minimize(objective_joint_smooth, initial_params, method='L-BFGS-B', 
                      options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.fun < best_val_joint:
            # Verify with exact objective
            exact_val = objective_joint_exact(res.x)
            if exact_val < best_val_joint:
                best_val_joint = exact_val
                best_params = res.x
    except Exception:
        pass
    
    # Additional local search with decreasing step sizes for fine-tuning
    current_params = best_params.copy()
    current_val = objective_joint_smooth(current_params)
    best_params_local = current_params.copy()
    best_val_local = current_val
    
    step_size = 0.05
    for _ in range(500):
        if time.time() > deadline - 15:
            break
        
        # Randomly pick a few dimensions to perturb
        num_perturb = 2
        indices = np.random.choice(len(current_params), num_perturb, replace=False)
        
        neighbor = current_params.copy()
        for idx in indices:
            delta = np.random.normal(0, step_size)
            neighbor[idx] += delta
            
        neighbor_val = objective_joint_smooth(neighbor)
        
        if neighbor_val < current_val:
            current_params = neighbor
            current_val = neighbor_val
            if current_val < best_val_local:
                best_params_local = current_params.copy()
                best_val_local = current_val
        
        # Decay step size
        step_size *= 0.995

    # Reconstruct best joint solution
    best_inner_joint = []
    for i in range(N):
        x = best_params_local[i*3]
        y = best_params_local[i*3+1]
        a = best_params_local[i*3+2]
        best_inner_joint.append([x, y, a])
    
    best_ocx = best_params_local[N*3]
    best_ocy = best_params_local[N*3+1]
    best_oangle = best_params_local[N*3+2]
    
    # Verify with SAT
    if check_all_disjoint(best_inner_joint, eps=1e-5):
        best_inner = best_inner_joint
        best_outer_center = (best_ocx, best_ocy)
        best_outer_angle = best_oangle
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)
    else:
        # Fallback to rigid body result if joint search broke disjointness
        best_inner = initial_inner
        best_side = best_val
        # Re-optimize outer for the rigid body if needed? No, we already did that.

    # 5. Final Polish: Tighten the outer container around the fixed inner layout
    def objective_outer_polish_smooth(params):
        cx, cy, angle_deg = params
        return compute_required_side_smooth(best_inner, (cx, cy), angle_deg, temp=0.01)
    
    def objective_outer_polish_exact(params):
        cx, cy, angle_deg = params
        return compute_required_side(best_inner, (cx, cy), angle_deg)
    
    # Try a few local starts around the current best
    polish_starts = [
        [best_outer_center[0], best_outer_center[1], best_outer_angle],
        [best_outer_center[0], best_outer_center[1], best_outer_angle + 0.5],
        [best_outer_center[0], best_outer_center[1], best_outer_angle - 0.5],
        [best_outer_center[0] + 0.1, best_outer_center[1], best_outer_angle],
        [best_outer_center[0] - 0.1, best_outer_center[1], best_outer_angle],
        [best_outer_center[0], best_outer_center[1] + 0.1, best_outer_angle],
        [best_outer_center[0], best_outer_center[1] - 0.1, best_outer_angle],
    ]
    
    best_polish_val = best_side
    best_polish_params = [best_outer_center[0], best_outer_center[1], best_outer_angle]
    
    for ps in polish_starts:
        if time.time() > deadline - 3:
            break
        try:
            # Use smooth objective for gradient-based optimization
            res = minimize(objective_outer_polish_smooth, ps, method='L-BFGS-B',
                          options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12})
            
            if res.fun < best_polish_val:
                # Verify with exact objective
                exact_val = objective_outer_polish_exact(res.x)
                if exact_val < best_polish_val:
                    best_polish_val = exact_val
                    best_polish_params = res.x
        except Exception:
            pass
    
    if best_polish_params is not None:
        best_outer_center = (best_polish_params[0], best_polish_params[1])
        best_outer_angle = best_polish_params[2]
        best_side = best_polish_val
    
    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END

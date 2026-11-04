import numpy as np
import time
from scipy.optimize import minimize

N = 12
TIME_LIMIT = 145.0  # Safety margin for 150s limit
START_TIME = time.time()

def _hex_vertices(center, side, angle_deg):
    """Return vertices of a regular hexagon."""
    x, y = center
    theta = np.deg2rad(angle_deg)
    angles = theta + np.arange(6) * (np.pi / 3)
    return np.array([
        [x + side * np.cos(a), y + side * np.sin(a)] 
        for a in angles
    ])

def _separating_axis_check(v1, v2):
    """Check if two convex polygons are disjoint using separating axis theorem."""
    n1, n2 = len(v1), len(v2)
    
    # Check axes from both polygons
    for i in range(n1):
        edge = v1[(i+1) % n1] - v1[i]
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal = normal / norm
        
        # Project both polygons onto this normal
        proj1 = np.array([np.dot(p, normal) for p in v1])
        proj2 = np.array([np.dot(p, normal) for p in v2])
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            return True  # Separating axis found, polygons are disjoint
    
    for i in range(n2):
        edge = v2[(i+1) % n2] - v2[i]
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal = normal / norm
        
        proj1 = np.array([np.dot(p, normal) for p in v1])
        proj2 = np.array([np.dot(p, normal) for p in v2])
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            return True  # Separating axis found, polygons are disjoint
    
    return False  # No separating axis found, polygons overlap

def _is_inside_hexagon(point, hex_verts):
    """Check if a point is inside a hexagon."""
    n = len(hex_verts)
    for i in range(n):
        a = hex_verts[i]
        b = hex_verts[(i+1) % n]
        cross = (b[0]-a[0])*(point[1]-a[1]) - (b[1]-a[1])*(point[0]-a[0])
        if cross < -1e-9:
            return False
    return True

def _check_validity(inner_hexes, outer_side, outer_center, outer_angle, eps=1e-7):
    """Check if inner hexagons are disjoint and inside outer hexagon."""
    outer_verts = _hex_vertices(outer_center, outer_side, outer_angle)
    inner_verts = [_hex_vertices(h[0:2], 1.0, h[2]) for h in inner_hexes]
    
    # Check disjointness
    for i in range(len(inner_hexes)):
        for j in range(i + 1, len(inner_hexes)):
            if not _separating_axis_check(inner_verts[i], inner_verts[j]):
                return False, "overlap"
        
        # Check containment
        for p in inner_verts[i]:
            if not _is_inside_hexagon(p, outer_verts):
                return False, "outside"
                
    return True, "valid"

def _compute_required_side(centers, angles, outer_angle=0.0):
    """Compute the minimum outer side length required to contain all hexagons."""
    max_proj = 0.0
    # Outer hexagon normals
    for k in range(6):
        angle = np.deg2rad(outer_angle + k * 60)
        normal = np.array([np.cos(angle), np.sin(angle)])
        
        for i in range(N):
            verts = _hex_vertices(centers[i], 1.0, angles[i])
            for v in verts:
                proj = np.dot(v, normal)
                if proj > max_proj:
                    max_proj = proj
                    
    return max_proj * 2.0 / np.sqrt(3.0)

def _sdf_hex_point(point, hex_center, hex_side, hex_angle):
    """Compute signed distance from a point to a hexagon (negative if inside)."""
    verts = _hex_vertices(hex_center, hex_side, hex_angle)
    
    # Check if point is inside
    if _is_inside_hexagon(point, verts):
        # Compute distance to nearest edge
        min_dist = float('inf')
        n = len(verts)
        for i in range(n):
            a = verts[i]
            b = verts[(i+1) % n]
            ab = b - a
            ap = point - a
            t = np.dot(ap, ab) / np.dot(ab, ab)
            t = max(0.0, min(1.0, t))
            closest = a + t * ab
            dist = np.linalg.norm(point - closest)
            if dist < min_dist:
                min_dist = dist
        return -min_dist
    else:
        # Compute distance to nearest edge
        min_dist = float('inf')
        n = len(verts)
        for i in range(n):
            a = verts[i]
            b = verts[(i+1) % n]
            ab = b - a
            ap = point - a
            t = np.dot(ap, ab) / np.dot(ab, ab)
            t = max(0.0, min(1.0, t))
            closest = a + t * ab
            dist = np.linalg.norm(point - closest)
            if dist < min_dist:
                min_dist = dist
        return min_dist

def _smooth_collision_penalty(centers, angles, sigma=0.1):
    """Compute a smooth penalty for overlapping hexagons."""
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            v1 = _hex_vertices(centers[i], 1.0, angles[i])
            v2 = _hex_vertices(centers[j], 1.0, angles[j])
            
            # Use SAT to check overlap and compute minimum distance
            # For smooth penalty, we use a distance-based approach
            # Compute minimum distance between the two hexagons
            min_dist = float('inf')
            
            # Check all edge pairs
            for k1 in range(6):
                p1 = v1[k1]
                p2 = v1[(k1+1) % 6]
                for k2 in range(6):
                    q1 = v2[k2]
                    q2 = v2[(k2+1) % 6]
                    
                    # Segment-segment distance
                    d1 = p2 - p1
                    d2 = q2 - q1
                    r = p1 - q1
                    a = np.dot(d1, d1)
                    e = np.dot(d2, d2)
                    f = np.dot(d2, r)
                    
                    if a < 1e-12 and e < 1e-12:
                        dist = np.linalg.norm(r)
                    elif a < 1e-12:
                        t = max(0.0, min(1.0, f / e))
                        dist = np.linalg.norm(r + t * d2)
                    elif e < 1e-12:
                        s = max(0.0, min(1.0, -np.dot(d1, r) / a))
                        dist = np.linalg.norm(s * d1 - r)
                    else:
                        c = np.dot(d1, r)
                        b = np.dot(d1, d2)
                        denom = a * e - b * b
                        if denom < 1e-12:
                            t = max(0.0, min(1.0, f / e))
                            s = max(0.0, min(1.0, (b * t - c) / a))
                        else:
                            s = (b * f - c * e) / denom
                            if s < 0.0:
                                t = max(0.0, min(1.0, f / e))
                                s = 0.0
                            elif s > 1.0:
                                t = max(0.0, min(1.0, (b - c) / e))
                                s = 1.0
                            else:
                                t = (b * s + f) / e
                                if t < 0.0:
                                    t = 0.0
                                    s = max(0.0, min(1.0, -c / a))
                                elif t > 1.0:
                                    t = 1.0
                                    s = max(0.0, min(1.0, (b - c) / a))
                        dist = np.linalg.norm(s * d1 - r - t * d2)
                    
                    if dist < min_dist:
                        min_dist = dist
            
            # Smooth penalty: use a softplus-like function
            if min_dist < sigma:
                penalty += (sigma - min_dist) ** 2 / (2 * sigma)
    
    return penalty

def _smooth_objective(params, outer_side, outer_angle):
    """Compute the smooth objective: required side + collision penalty."""
    centers = params[:N*2].reshape(N, 2)
    angles = params[N*2:]
    
    # Compute required side
    req_side = _compute_required_side(centers, angles, outer_angle)
    
    # Compute collision penalty
    penalty = _smooth_collision_penalty(centers, angles)
    
    # Objective: minimize required side, penalize overlaps
    return req_side + 10.0 * penalty

def _is_valid_config(centers, angles, outer_side, outer_angle):
    """Check if configuration is valid."""
    inner_hexes = [[centers[i][0], centers[i][1], angles[i]] for i in range(N)]
    is_valid, reason = _check_validity(inner_hexes, outer_side, (0.0, 0.0), outer_angle)
    return is_valid

def _optimize_from_init(init_pos, init_ang, outer_angle, max_time=10.0):
    """Optimize from a given initial configuration."""
    start_time = time.time()
    
    # Flatten parameters: [x0, y0, ..., x11, y11, angle0, ..., angle11]
    params = np.concatenate([init_pos.flatten(), init_ang])
    
    best_params = params.copy()
    best_side = float('inf')
    
    # Use L-BFGS-B optimization
    try:
        result = minimize(
            _smooth_objective,
            params,
            args=(10.0, outer_angle),  # Large initial container
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        
        if time.time() - start_time > max_time:
            return None, None
        
        # Evaluate result
        centers = result.x[:N*2].reshape(N, 2)
        angles = result.x[N*2:]
        
        req_side = _compute_required_side(centers, angles, outer_angle)
        is_valid = _is_valid_config(centers, angles, req_side, outer_angle)
        
        if is_valid and req_side < best_side:
            best_params = result.x.copy()
            best_side = req_side
            
    except Exception as e:
        pass
    
    return best_params, best_side

def construct():
    start_time = time.time()
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    np.random.seed(42)
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # Config 1: Hexagonal packing
    pts = [[0,0]]
    for k in range(6):
        angle = np.pi/3 * k
        pts.append([np.sqrt(3)*np.cos(angle), np.sqrt(3)*np.sin(angle)])
    for k in range(6):
        angle = np.pi/3 * k + np.pi/6
        pts.append([2*np.sqrt(3)*np.cos(angle), 2*np.sqrt(3)*np.sin(angle)])
        if len(pts) == 12:
            break
    initial_configs.append((np.array(pts), np.zeros(N)))
    
    # Config 2: Random
    for _ in range(5):
        pts = np.random.uniform(-4, 4, (N, 2))
        initial_configs.append((pts, np.zeros(N)))
        
    # Config 3: Linear
    pts = np.array([[i*2.0, 0.0] for i in range(N)])
    initial_configs.append((pts, np.zeros(N)))
    
    # Optimization loop
    for outer_angle in [0.0, 30.0]:
        if time.time() - start_time > TIME_LIMIT - 10:
            break
            
        for init_pos, init_ang in initial_configs:
            if time.time() - start_time > TIME_LIMIT - 5:
                break
                
            try:
                # Optimize from this initial configuration
                best_params, req_side = _optimize_from_init(init_pos, init_ang, outer_angle, max_time=10.0)
                
                if best_params is not None and req_side is not None and req_side < best_side:
                    centers = best_params[:N*2].reshape(N, 2)
                    angles = best_params[N*2:]
                    
                    # Verify validity
                    inner_hexes = [[centers[i][0], centers[i][1], angles[i]] for i in range(N)]
                    is_valid, reason = _check_validity(inner_hexes, req_side, (0.0, 0.0), outer_angle)
                    
                    if is_valid:
                        best_side = req_side
                        best_inner = inner_hexes
                        best_outer_center = (0.0, 0.0)
                        best_outer_angle = outer_angle
                        
            except Exception as e:
                pass
                
    if best_inner is not None:
        return (best_inner, best_outer_center, best_side, best_outer_angle)
    
    # Fallback
    return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0

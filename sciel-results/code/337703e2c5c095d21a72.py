# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _hex_vertices(cx, cy, d, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _hex_edge_normals(vertices):
    """Return unit edge normals for a hexagon (6, 2) array."""
    edges = np.roll(vertices, -1, axis=0) - vertices
    normals = np.column_stack([-edges[:, 1], edges[:, 0]])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / norms
    center = vertices.mean(axis=0)
    mids = (vertices + np.roll(vertices, -1, axis=0)) / 2
    dots = np.sum(normals * (center - mids), axis=1)
    normals[dots < 0] *= -1
    return normals

def _check_separation(v1, v2, normals1, normals2, tol=1e-6):
    """Check if two hexagons are separated. Returns True if separated."""
    all_normals = np.vstack([normals1, normals2])
    for n in all_normals:
        p1 = v1 @ n
        p2 = v2 @ n
        min1, max1 = p1.min(), p1.max()
        min2, max2 = p2.min(), p2.max()
        gap = min(max1, max2) - max(min1, min2)
        if gap < -tol:
            return False
    return True

def _required_s(vertices):
    """
    Calculate the minimum side length s of a regular hexagon (centered at origin, 
    aligned with axes such that vertices are at (s,0) etc) 
    that contains all points in vertices (N, 2).
    """
    x = vertices[:, 0]
    y = vertices[:, 1]
    
    # Constraints derived from hexagon inequalities:
    # 1. |x| <= s
    # 2. |sqrt(3)*x + y| <= sqrt(3)*s  => |x + y/sqrt(3)| <= s
    # 3. |sqrt(3)*x - y| <= sqrt(3)*s  => |x - y/sqrt(3)| <= s
    
    s1 = np.max(np.abs(x))
    s2 = np.max(np.abs(x + y / np.sqrt(3)))
    s3 = np.max(np.abs(x - y / np.sqrt(3)))
    
    return max(s1, s2, s3)

def _compute_penalty(centers, angle_inner, tol=1e-6):
    """
    Compute total penalty for a configuration.
    Returns (penalty, s_required, is_feasible)
    """
    penalty = 0.0
    feasible = True
    
    # Precompute inner hex vertices and normals
    inner_verts = []
    inner_normals = []
    all_vertices = []
    
    for i in range(12):
        cx, cy = centers[i]
        v = _hex_vertices(cx, cy, 1.0, angle_inner)
        inner_verts.append(v)
        inner_normals.append(_hex_edge_normals(v))
        all_vertices.append(v)
        
    all_vertices = np.vstack(all_vertices)
    s_req = _required_s(all_vertices)
    
    # Pairwise separation check
    for i in range(12):
        for j in range(i + 1, 12):
            dist_sq = np.sum((centers[i] - centers[j]) ** 2)
            if dist_sq > 9.0: # 3^2, max touching dist is sqrt(3) ~ 1.73, so 9 is safe cutoff
                continue
            all_normals = np.vstack([inner_normals[i], inner_normals[j]])
            max_overlap = 0.0
            for n in all_normals:
                p1 = inner_verts[i] @ n
                p2 = inner_verts[j] @ n
                min1, max1 = p1.min(), p1.max()
                min2, max2 = p2.min(), p2.max()
                gap = min(max1, max2) - max(min1, min2)
                if gap < -tol:
                    max_overlap = max(max_overlap, -gap)
                    feasible = False
            penalty += max_overlap * max_overlap
            
    # We want to minimize s_req, but penalty prevents overlaps.
    # Objective: s_req + large * penalty
    return penalty, s_req, feasible

def hexagon_packing_12():
    start_time = time.time()
    deadline = 625.0
    
    best_score = 0.0
    best_result = None
    best_s = float('inf')
    best_centers = None
    
    angle_inner = 0.0
    angle_outer = 0.0
    
    # Initial configuration: 13-hex cluster minus one
    # Centers:
    # 0: (0,0)
    # 1-6: dist sqrt(3), angles 0, 60, ...
    # 7-12: dist 3, angles 30, 90, ...
    # We use 12 of these.
    
    sqrt3 = np.sqrt(3)
    init_centers = np.zeros((12, 2))
    init_centers[0] = [0, 0]
    
    for i in range(6):
        angle = np.radians(60 * i)
        init_centers[i+1] = [sqrt3 * np.cos(angle), sqrt3 * np.sin(angle)]
        
    # Add 5 from the second ring (angles 30, 90, 150, 210, 270)
    # Skip 330 to make it 12 total? Or just take first 5.
    for i in range(5):
        angle = np.radians(30 + 60 * i)
        init_centers[i+7] = [3 * np.cos(angle), 3 * np.sin(angle)]
        
    current_centers = init_centers.copy()
    
    # SA Parameters
    T0 = 1.0
    T_end = 1e-8
    cooling_rate = 0.9995
    T = T0
    
    best_centers_sa = current_centers.copy()
    best_penalty = float('inf')
    best_s_sa = float('inf')
    
    max_iter = 1000000
    iter_count = 0
    
    # Initial evaluation
    curr_penalty, curr_s, curr_feasible = _compute_penalty(current_centers, angle_inner)
    best_penalty = curr_penalty
    best_s_sa = curr_s
    best_centers_sa = current_centers.copy()
    
    while T > T_end and time.time() < deadline:
        # Perturb
        idx = random.randint(0, 11)
        dx = random.gauss(0, 0.5)
        dy = random.gauss(0, 0.5)
        
        new_centers = current_centers.copy()
        new_centers[idx] = [current_centers[idx][0] + dx, current_centers[idx][1] + dy]
        
        new_penalty, new_s, new_feasible = _compute_penalty(new_centers, angle_inner)
        
        # Objective: s + 10 * penalty
        curr_obj = curr_s + 10 * curr_penalty
        new_obj = new_s + 10 * new_penalty
        
        delta = new_obj - curr_obj
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_centers = new_centers
            curr_penalty = new_penalty
            curr_s = new_s
            
            if curr_obj < best_obj:
                best_obj = curr_obj
                best_penalty = curr_penalty
                best_s_sa = curr_s
                best_centers_sa = current_centers.copy()
                
        T *= cooling_rate
        iter_count += 1
        
        # Check time and update global best
        if iter_count % 1000 == 0:
            if best_feasible_check(best_centers_sa, angle_inner):
                score = 1.0 / best_s_sa
                if score > best_score:
                    best_score = score
                    inner_data = np.column_stack([best_centers_sa, np.full(12, angle_inner)])
                    outer_data = np.array([0, 0, angle_outer])
                    best_result = (inner_data, outer_data, best_s_sa)
                    
    # Final check
    if best_result is None:
        if best_feasible_check(best_centers_sa, angle_inner):
            score = 1.0 / best_s_sa
            if score > best_score:
                best_score = score
                inner_data = np.column_stack([best_centers_sa, np.full(12, angle_inner)])
                outer_data = np.array([0, 0, angle_outer])
                best_result = (inner_data, outer_data, best_s_sa)

    if best_result is None:
        # Fallback
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
                [0, -4, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8
        return inner_hex_data, outer_hex_data, outer_hex_side_length
    
    return best_result

def best_feasible_check(centers, angle_inner, tol=1e-6):
    # Re-use _compute_penalty logic but strict
    penalty, s_req, feasible = _compute_penalty(centers, angle_inner, tol)
    return feasible and penalty < 1e-6

# EVOLVE-BLOCK-END

import numpy as np
import time
import random
import math

N = 12
SQRT3 = np.sqrt(3.0)

# Precompute base vertices for a unit hexagon (flat-top, vertex at angle 0)
# Vertices at angles 0, 60, 120, 180, 240, 300
BASE_ANGLES = np.radians(np.arange(0, 360, 60))
BASE_COS = np.cos(BASE_ANGLES)
BASE_SIN = np.sin(BASE_ANGLES)
BASE_VERTS = np.column_stack([BASE_COS, BASE_SIN])

def get_cluster_positions():
    """
    Generate a compact initial cluster of 12 hexagons.
    We use a 3x4 rectangular grid of hex centers, adjusted for hex geometry.
    Actually, a honeycomb cluster is better.
    Center (0,0)
    Ring 1 (6): (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) in axial coords
    Ring 2 (5): (2,0), (1,1), (0,2), (-1,2), (-2,1)
    Total 12.
    """
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    pos = []
    for q, r in coords:
        x = SQRT3 * (q + r / 2.0)
        y = 1.5 * r
        pos.append([x, y])
    return np.array(pos)

def hexagon_vertices(cx, cy, angle_deg):
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    verts = (R @ BASE_VERTS.T).T
    return verts + np.array([cx, cy])

def min_separation(verts1, verts2):
    """
    Estimate the minimum distance between two convex polygons.
    Returns 0 if they intersect (or touch), positive if separated.
    Uses a simple grid of edge-to-edge and vertex-to-edge checks?
    For speed in SA, we can use a faster approximation or just SAT violation magnitude.
    Let's use SAT to find the maximum penetration depth.
    If max_penetration > 0, they overlap.
    """
    # SAT implementation
    min_proj_diff = float('inf')
    
    # Check edges of p1
    for i in range(6):
        j = (i + 1) % 6
        edge = verts1[j] - verts1[i]
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal /= norm
        proj1 = verts1.dot(normal)
        proj2 = verts2.dot(normal)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # If separated, we don't care for penalty (penalty is 0)
        # If overlapping, penetration = max1 - min2 or max2 - min1
        # We want the MINIMUM separation, so we look for the axis with the LEAST overlap
        # Actually, for penalty, we want the MAXIMUM overlap depth? No.
        # If they are separated, distance is positive.
        # If they overlap, we want a penalty proportional to how much they overlap.
        # The "penetration depth" is the minimum distance one must move to separate them.
        # This is the minimum over all axes of the overlap amount? No.
        # If they overlap, all axes show overlap. The true penetration depth is complex.
        # A common proxy is the sum of overlaps or max overlap.
        # Let's use: if separated, return the separation distance.
        # If overlapping, return the negative of the minimum separation (which is 0) plus penalty.
        
        # Let's compute the gap on this axis.
        # gap = min(max1, max2) - max(min1, min2)
        # If gap < 0, they overlap on this axis by -gap.
        # If gap > 0, they are separated by gap on this axis.
        
        gap = min(max1, max2) - max(min1, min2)
        
        if gap > 0:
             # Separated on this axis. The true distance is at least gap.
             # We can return gap immediately as a lower bound? No, true distance is max of gaps? No.
             # True distance is the minimum distance between any point in P1 and P2.
             # If they are separated, SAT says there is an axis where they are separated.
             # The distance is the maximum of the gaps over all separating axes? No.
             # Actually, if they are disjoint, the distance is the minimum distance between vertices/edges.
             # For penalty purposes, if gap > 0, we can just return a small positive number or the gap itself?
             # If we return the gap, and there are multiple separating axes, the true distance is the max of these gaps? No.
             # The true distance is the minimum over all pairs of points.
             # However, if they are separated, the penalty should be 0.
             # So if any axis shows separation (gap > 0), the polygons are disjoint.
             # Wait, SAT says: if there is ANY axis where projections are disjoint, the polygons are disjoint.
             # So if gap > 0, they are disjoint. Penalty = 0.
             return 0.0
        else:
             # Overlapping on this axis.
             # We track the "least" overlap to estimate penetration depth?
             # Actually, for a penalty, we want to penalize overlap.
             # Let's accumulate the overlap.
             # But we can't just sum.
             # Let's just return a large penalty if they overlap.
             # To make it smooth, we can return the negative gap (which is positive overlap amount).
             # But we need to check all axes. If ANY axis is disjoint, they are disjoint.
             # So we must check all axes. If ALL axes overlap, they intersect.
             # In that case, what is the penalty?
             # A good penalty is the minimum distance required to separate them, which is the min over axes of the overlap? No.
             # It's a hard constraint. Let's use a quadratic penalty on the violation.
             # Violation on axis i is max(0, -gap_i).
             # Total violation = sum of violations? Or max?
             # Max violation is a better measure of "how bad" the overlap is.
             
             # Store the gap for this axis
             min_proj_diff = min(min_proj_diff, gap) # gap is negative here
            
    # If we finished the loop, it means they overlap on ALL axes of P1.
    # We should also check axes of P2.
    for i in range(6):
        j = (i + 1) % 6
        edge = verts2[j] - verts2[i]
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal /= norm
        proj1 = verts1.dot(normal)
        proj2 = verts2.dot(normal)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        gap = min(max1, max2) - max(min1, min2)
        
        if gap > 0:
             return 0.0
        else:
             min_proj_diff = min(min_proj_diff, gap)
             
    # They overlap. min_proj_diff is the most negative gap (largest overlap on a single axis).
    # This is not the true penetration depth, but a good proxy.
    # Penalty = -min_proj_diff (which is positive)
    return -min_proj_diff

def calculate_penalty_and_size(positions, angles):
    """
    Calculate the penalty for overlaps and the required outer hexagon side length.
    """
    # 1. Check overlaps
    total_penalty = 0.0
    vertices_list = []
    for i in range(N):
        x, y = positions[i]
        verts = hexagon_vertices(x, y, angles[i])
        vertices_list.append(verts)
        
    for i in range(N):
        for j in range(i + 1, N):
            # Quick reject: if center distance > 2 * circumradius + epsilon, skip
            # Circumradius of unit hex is 1.
            # If dist > 2.1, they cannot overlap.
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dist_sq = dx*dx + dy*dy
            if dist_sq > 2.25: # 1.5^2
                continue
                
            sep = min_separation(vertices_list[i], vertices_list[j])
            if sep > 0:
                # sep is actually 0 if disjoint, or positive overlap amount if overlapping?
                # My min_separation returns 0 if disjoint.
                # If overlapping, it returns -min_proj_diff which is positive.
                # So if sep > 0, it's an overlap amount.
                total_penalty += sep * sep # Quadratic penalty
            
    # 2. Calculate outer side length
    # Collect all vertices
    all_verts = np.vstack(vertices_list)
    x = all_verts[:, 0]
    y = all_verts[:, 1]
    
    L1 = x * (SQRT3/2) + y * 0.5
    L2 = y
    L3 = -x * (SQRT3/2) + y * 0.5
    L4 = -x * (SQRT3/2) - y * 0.5
    L5 = -y
    L6 = x * (SQRT3/2) - y * 0.5
    
    all_L = np.stack([L1, L2, L3, L4, L5, L6], axis=1)
    max_proj = np.max(all_L)
    side = max_proj * 2.0 / SQRT3
    
    return total_penalty, side

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Initial configuration
    base_pos = get_cluster_positions()
    # Add some random jitter to break symmetry
    pos = base_pos + np.random.normal(0, 0.1, (N, 2))
    angles = np.zeros(N)
    
    # Center the configuration roughly at origin
    pos -= np.mean(pos, axis=0)
    
    current_pen, current_side = calculate_penalty_and_size(pos, angles)
    current_cost = current_side + 10.0 * current_pen
    
    best_pos = pos.copy()
    best_angles = angles.copy()
    best_cost = current_cost
    best_side = current_side
    best_pen = current_pen
    
    # Simulated Annealing Parameters
    T = 0.5
    cooling_rate = 0.995
    max_iters = 5000
    
    rng = np.random.default_rng(42)
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
            
        # Generate new candidate
        new_pos = pos.copy()
        new_angles = angles.copy()
        
        # Move a random hexagon
        idx = rng.integers(0, N)
        
        # Random perturbation
        step_size = 0.2 * T + 0.01
        new_pos[idx] += rng.normal(0, step_size, 2)
        new_angles[idx] += rng.normal(0, step_size * 10) # Rotation in degrees
        
        # Calculate new cost
        new_pen, new_side = calculate_penalty_and_size(new_pos, new_angles)
        new_cost = new_side + 10.0 * new_pen
        
        # Acceptance criterion
        delta_cost = new_cost - current_cost
        if delta_cost < 0 or rng.random() < np.exp(-delta_cost / T):
            pos = new_pos
            angles = new_angles
            current_cost = new_cost
            current_pen = new_pen
            current_side = new_side
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_side = current_side
                best_pen = current_pen
                best_pos = pos.copy()
                best_angles = angles.copy()
        
        T *= cooling_rate
        
        # Optional: If penalty is high, increase temperature or do a larger jump?
        # For now, fixed cooling.
        
    # After SA, try to refine the best solution with a local optimizer?
    # Or just return the best found.
    # Let's do a quick local polish on the best solution if time permits.
    if time.time() < deadline - 10 and best_pen < 1e-6:
        # If it's already valid, we can try to minimize side length further
        # using Nelder-Mead on the valid solution
        from scipy.optimize import minimize
        
        def obj_polish(params):
            p_pos = params[:N*2].reshape(N, 2)
            p_ang = params[N*2:]
            pen, side = calculate_penalty_and_size(p_pos, p_ang)
            if pen > 1e-4:
                return side + 100 * pen
            return side
        
        x0 = np.concatenate([best_pos.flatten(), best_angles])
        res = minimize(obj_polish, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5})
        
        if res.fun < best_side:
            best_side = res.fun
            best_pos = res.x[:N*2].reshape(N, 2)
            best_angles = res.x[N*2:]
            
    # Final validation and fallback
    final_pen, final_side = calculate_penalty_and_size(best_pos, best_angles)
    if final_pen > 1e-4:
        # Fallback to a known valid configuration (scaled up if necessary)
        # The initial cluster is valid if scale >= 1.0
        # Let's just use the initial cluster scaled to fit if SA failed
        init_pos = get_cluster_positions()
        init_angles = np.zeros(N)
        init_pen, init_side = calculate_penalty_and_size(init_pos, init_angles)
        if init_pen < final_pen:
            best_pos = init_pos
            best_angles = init_angles
            best_side = init_side
    
    inner = []
    for i in range(N):
        inner.append([float(best_pos[i, 0]), float(best_pos[i, 1]), float(best_angles[i])])
    
    return inner, (0.0, 0.0), float(best_side), 0.0

# EVOLVE-BLOCK-END

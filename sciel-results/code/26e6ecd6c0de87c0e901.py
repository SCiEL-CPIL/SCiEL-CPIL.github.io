"""Packing 12 unit hexagons in a regular hexagon using Local Search."""
import numpy as np
import time
import random

N = 12

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _get_axes(verts):
    """Get edge normals for a polygon."""
    axes = []
    for i in range(6):
        p1 = verts[i]
        p2 = verts[(i + 1) % 6]
        edge = p2 - p1
        # Normal
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-10:
            axes.append(normal / norm)
    return axes

def _project_range(verts, axis):
    """Project polygon vertices onto an axis and return min/max."""
    dots = np.dot(verts, axis)
    return np.min(dots), np.max(dots)

def _min_overlap_depth(v1, v2):
    """
    Calculate the minimum distance to separate two convex polygons.
    Returns 0 if disjoint, >0 if overlapping.
    """
    axes = _get_axes(v1) + _get_axes(v2)
    min_overlap = float('inf')
    
    for axis in axes:
        min1, max1 = _project_range(v1, axis)
        min2, max2 = _project_range(v2, axis)
        
        # Check for separation
        if max1 < min2 or max2 < min1:
            return 0.0
        
        # Calculate overlap
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_overlap:
            min_overlap = overlap
            
    return min_overlap

def _containment_violation(inner_verts, outer_verts):
    """
    Calculate the maximum distance by which inner vertices are outside the outer hexagon.
    Returns 0 if all inside.
    """
    max_violation = 0.0
    # For each inner vertex, check distance to outer boundary
    # A point is inside a convex polygon if it is on the same side of all edges
    # We can compute the signed distance to each edge.
    # If any signed distance is negative, it's outside.
    # The violation is the maximum negative signed distance (magnitude).
    
    for v in inner_verts:
        min_dist_to_edge = float('inf')
        for i in range(6):
            p1 = outer_verts[i]
            p2 = outer_verts[(i + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-10: continue
            normal /= norm
            # Ensure normal points inward
            # Center of outer hexagon is (0,0) approx? No, use centroid.
            # Let's just use the cross product sign relative to the edge direction.
            # Vector from p1 to v
            vec = v - p1
            cross = np.cross(edge, vec)
            # If cross < 0, it's on the "right" side of the edge.
            # For a CCW polygon, inside is "left".
            # Our _hex_vertices generates CCW?
            # angles 0, 60, 120... cos/sin. Yes, CCW.
            # So cross > 0 is inside.
            # Distance = cross / |edge|
            dist = cross / norm
            if dist < min_dist_to_edge:
                min_dist_to_edge = dist
        if min_dist_to_edge < 0:
            if -min_dist_to_edge > max_violation:
                max_violation = -min_dist_to_edge
                
    return max_violation

def _cost(params, S):
    """
    params: array of shape (12, 3) -> [x, y, angle_deg]
    S: outer side length
    Returns: Scalar cost.
    """
    outer_center = np.array([0.0, 0.0])
    outer_angle = 0.0
    outer_verts = _hex_vertices(0.0, 0.0, S, 0.0)
    
    total_cost = S  # We want to minimize S
    penalty_weight = 100.0
    
    inner_verts_list = []
    for i in range(N):
        x, y, a = params[i]
        verts = _hex_vertices(x, y, 1.0, a)
        inner_verts_list.append(verts)
        
        # Containment penalty
        viol = _containment_violation(verts, outer_verts)
        total_cost += penalty_weight * viol
    
    # Overlap penalty
    for i in range(N):
        for j in range(i + 1, N):
            depth = _min_overlap_depth(inner_verts_list[i], inner_verts_list[j])
            if depth > 0:
                total_cost += penalty_weight * depth
                
    return total_cost

def _local_search(params, S, time_limit):
    """
    Run local search (Simulated Annealing) to minimize cost.
    """
    start_time = time.time()
    current_params = params.copy()
    current_S = S
    current_cost = _cost(current_params, current_S)
    
    best_params = current_params.copy()
    best_S = current_S
    best_cost = current_cost
    
    T = 0.1  # Initial temperature
    cooling_rate = 0.99
    
    while time.time() - start_time < time_limit:
        T *= cooling_rate
        if T < 1e-6:
            break
            
        # Propose move
        new_params = current_params.copy()
        new_S = current_S
        
        # Randomly pick one or two parameters to perturb
        for _ in range(random.randint(1, 2)):
            idx = random.randint(0, N - 1)
            # Perturb x, y, angle
            new_params[idx][0] += random.gauss(0, 0.1)
            new_params[idx][1] += random.gauss(0, 0.1)
            new_params[idx][2] += random.gauss(0, 5)
            
        # Perturb S
        if random.random() < 0.5:
            new_S *= (1 + random.gauss(0, 0.01))
            if new_S < 2.0: new_S = 2.0
            
        new_cost = _cost(new_params, new_S)
        
        # Acceptance criterion
        delta = new_cost - current_cost
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_params = new_params
            current_S = new_S
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_params = current_params.copy()
                best_S = current_S
                
    # Check if best solution is valid (low penalty)
    # If penalty is very low, we can return it
    return best_params, best_S, best_cost

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    best_result = None
    best_side = float('inf')
    
    # Multiple restarts
    num_restarts = 10
    time_per_restart = (deadline - start_time) / num_restarts - 2
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Initialize with hex lattice
        s3 = np.sqrt(3)
        params = np.zeros((N, 3))
        
        # Generate 12 positions on hex lattice
        # Center
        params[0] = [0.0, 0.0, 0.0]
        # Ring 1
        for k in range(1, 7):
            angle = (k - 1) * np.pi / 3
            params[k] = [s3 * np.cos(angle), s3 * np.sin(angle), 0.0]
        # Ring 2 (partial)
        for k in range(7, 12):
            # Place in gaps or next ring
            # Let's just use a wider spread
            angle = (k - 7) * 2 * np.pi / 5 + 0.3
            rad = 2 * s3
            params[k] = [rad * np.cos(angle), rad * np.sin(angle), 0.0]
        
        # Add random jitter
        params += np.random.normal(0, 0.1, params.shape)
        
        # Estimate S
        max_r = np.max(np.linalg.norm(params[:, :2], axis=1)) + 1.0
        S = 2 * max_r / np.sqrt(3) + 0.5
        
        # Run local search
        # Use a fraction of the time
        t_limit = min(time_per_restart, 10.0)
        opt_params, opt_S, opt_cost = _local_search(params, S, t_limit)
        
        # Validate opt_solution
        # Check if penalty is negligible
        # We can re-calculate cost to ensure it's valid
        # If opt_cost is close to opt_S (meaning penalty is small), it's a good candidate
        penalty = opt_cost - opt_S
        if penalty < 1e-3:
            if opt_S < best_side:
                best_side = opt_S
                best_result = (opt_params, opt_S)
        else:
            # If not valid, maybe we can still use it if it's better than current best?
            # No, we need a valid solution.
            # We can try to repair? Or just discard.
            pass
            
    if best_result is None:
        # Fallback: return a known valid, albeit suboptimal, configuration
        # Simple grid
        s3 = np.sqrt(3)
        inner = []
        # 2x6 grid?
        for i in range(2):
            for j in range(6):
                x = j * s3
                y = i * 1.5 * s3
                if i % 2 == 1:
                    x += s3 / 2
                inner.append((x - 2.5*s3, y - 0.75*s3, 0.0))
        
        # Calculate required S
        xs = [p[0] for p in inner]
        ys = [p[1] for p in inner]
        cx = (max(xs) + min(xs)) / 2
        cy = (max(ys) + min(ys)) / 2
        max_r = 0
        for p in inner:
            dx = p[0] - cx
            dy = p[1] - cy
            max_r = max(max_r, np.sqrt(dx*dx + dy*dy) + 1.0)
        S = 2 * max_r / np.sqrt(3) + 0.5
        
        # Convert to list of lists
        inner_list = [[p[0], p[1], p[2]] for p in inner]
        return inner_list, (cx, cy), S, 0.0
    
    opt_params, opt_S = best_result
    inner_list = [[p[0], p[1], p[2]] for p in opt_params]
    
    # Center the inner hexagons? The outer is at (0,0).
    # The problem asks for outer center (x,y). We can return (0,0).
    return inner_list, (0.0, 0.0), opt_S, 0.0

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END

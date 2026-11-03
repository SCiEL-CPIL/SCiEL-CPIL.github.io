import numpy as np
import time
import math

N = 11

def get_hex_vertices(x, y, theta):
    """
    Returns vertices of a unit regular hexagon centered at (x,y) with rotation theta (radians).
    """
    angles = np.array([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3])
    verts = np.vstack([np.cos(angles + theta), np.sin(angles + theta)]).T
    verts += np.array([x, y])
    return verts

def get_hex_normals(theta):
    """
    Returns unit normals to the sides of a unit regular hexagon rotated by theta.
    """
    base_normals = np.array([
        [np.cos(np.pi/6), np.sin(np.pi/6)],
        [np.cos(np.pi/2), np.sin(np.pi/2)],
        [np.cos(5*np.pi/6), np.sin(5*np.pi/6)],
        [np.cos(7*np.pi/6), np.sin(7*np.pi/6)],
        [np.cos(3*np.pi/2), np.sin(3*np.pi/2)],
        [np.cos(11*np.pi/6), np.sin(11*np.pi/6)]
    ])
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return base_normals @ R.T

def min_separation(verts1, normals1, verts2, normals2):
    """
    Returns the minimum separation distance between two convex polygons.
    Positive means separated, negative means overlapping (depth).
    Uses Separating Axis Theorem.
    """
    min_gap = float('inf')
    all_normals = np.vstack([normals1, normals2])
    
    for n in all_normals:
        proj1 = verts1 @ n
        min1 = np.min(proj1)
        max1 = np.max(proj1)
        
        proj2 = verts2 @ n
        min2 = np.min(proj2)
        max2 = np.max(proj2)
        
        gap = max(min1, min2) - max(max1, max2)
        
        if gap > 0:
            return float('inf')
        
        if gap < min_gap:
            min_gap = gap
            
    return min_gap

def calculate_metrics(params):
    """
    params: array of shape (N, 3) -> [x, y, theta]
    Returns: (side_length, overlap_penalty, is_valid)
    """
    xs = params[:, 0]
    ys = params[:, 1]
    thetas = params[:, 2]
    
    outer_normals = np.array([
        [np.cos(np.pi/6), np.sin(np.pi/6)],
        [np.cos(np.pi/2), np.sin(np.pi/2)],
        [np.cos(5*np.pi/6), np.sin(5*np.pi/6)]
    ])
    
    max_proj = 0.0
    for i in range(N):
        verts = get_hex_vertices(xs[i], ys[i], thetas[i])
        projs = np.abs(verts @ outer_normals.T)
        current_max = np.max(projs)
        if current_max > max_proj:
            max_proj = current_max
            
    side = 2.0 * max_proj / np.sqrt(3)
    
    penalty = 0.0
    is_valid = True
    
    verts_list = []
    normals_list = []
    for i in range(N):
        v = get_hex_vertices(xs[i], ys[i], thetas[i])
        n = get_hex_normals(thetas[i])
        verts_list.append(v)
        normals_list.append(n)
        
    for i in range(N):
        for j in range(i+1, N):
            sep = min_separation(verts_list[i], normals_list[i], verts_list[j], normals_list[j])
            if sep < 0:
                is_valid = False
                penalty += (-sep) ** 2
                
    return side, penalty, is_valid

def construct():
    start_time = time.time()
    deadline = 145.0
    
    # Hexagonal lattice packing using axial coordinates
    # For unit hexagons (side=1), center-to-center distance of adjacent hexagons = sqrt(3)
    d = np.sqrt(3)
    
    # Axial coordinate to Cartesian: for flat-top hexagons
    # x = d * (q + r/2)
    # y = d * (3*r/2)
    def axial_to_cartesian(q, r):
        x = d * (q + r/2.0)
        y = d * (3*r/2.0)
        return x, y
    
    # Define 11 positions in a compact hexagonal cluster
    # Center + 6 neighbors (first ring) = 7 hexagons
    # Add 4 more in second ring for compactness
    positions_axial = [
        (0, 0),       # Center
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),  # First ring
        (2, 0), (1, 1), (-2, 1), (-1, -1)  # Select 4 from second ring
    ]
    
    centers = []
    for q, r in positions_axial:
        x, y = axial_to_cartesian(q, r)
        centers.append((x, y))
    
    init_params = np.zeros((N, 3))
    for i in range(N):
        init_params[i, 0] = centers[i][0]
        init_params[i, 1] = centers[i][1]
        init_params[i, 2] = 0.0
    
    # Verify this packing
    side, penalty, is_valid = calculate_metrics(init_params)
    
    if is_valid:
        inner = []
        for i in range(N):
            x, y, ang = init_params[i]
            ang_deg = np.rad2deg(ang)
            inner.append([float(x), float(y), float(ang_deg)])
        return inner, (0.0, 0.0), float(side), 0.0
    
    # Fallback to simulated annealing if lattice packing fails
    best_side = float('inf')
    best_params = None
    best_valid = False
    
    np.random.seed(42)
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        if r == 0:
            params = init_params.copy()
        else:
            if best_params is not None and np.random.rand() < 0.5:
                params = best_params + np.random.normal(0, 0.5, (N, 3))
                params[:, 2] = np.random.uniform(-np.pi, np.pi, N)
            else:
                params = np.zeros((N, 3))
                for i in range(N):
                    if i == 0:
                        params[i] = [0, 0, 0]
                    else:
                        angle = np.random.uniform(0, 2*np.pi)
                        radius = np.random.uniform(1.0, 4.0)
                        params[i] = [radius * np.cos(angle), radius * np.sin(angle), np.random.uniform(-np.pi, np.pi)]
                        
        T = 1.0
        T_min = 0.001
        alpha = 0.995
        steps_per_restart = 5000
        
        current_side, current_penalty, current_valid = calculate_metrics(params)
        current_cost = current_side + 100.0 * current_penalty
        
        for step in range(steps_per_restart):
            if time.time() > deadline - 5:
                break
                
            k = np.random.randint(N)
            old_x, old_y, old_theta = params[k]
            
            move_dist = np.random.uniform(0.1, 0.5)
            move_angle = np.random.uniform(0, 2*np.pi)
            new_x = old_x + move_dist * np.cos(move_angle)
            new_y = old_y + move_dist * np.sin(move_angle)
            
            new_theta = old_theta + np.random.uniform(-0.2, 0.2)
            
            params[k] = [new_x, new_y, new_theta]
            
            new_side, new_penalty, new_valid = calculate_metrics(params)
            new_cost = new_side + 100.0 * new_penalty
            
            delta_cost = new_cost - current_cost
            
            if delta_cost < 0 or np.random.rand() < np.exp(-delta_cost / T):
                current_cost = new_cost
                current_side = new_side
                current_penalty = new_penalty
                current_valid = new_valid
                
                if new_valid and new_side < best_side:
                    best_side = new_side
                    best_params = params.copy()
                    best_valid = True
            else:
                params[k] = [old_x, old_y, old_theta]
                
            T *= alpha
            if T < T_min:
                break
                
    if not best_valid:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
        
    inner = []
    for i in range(N):
        x, y, ang = best_params[i]
        ang_deg = np.rad2deg(ang)
        inner.append([float(x), float(y), float(ang_deg)])
        
    return inner, (0.0, 0.0), float(best_side), 0.0

if __name__ == "__main__":
    inner, center, side, angle = construct()
    print(f"Side: {side}")

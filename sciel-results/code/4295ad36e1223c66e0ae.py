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
    
    # Calculate required outer hexagon side length
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
    
    # Calculate overlap penalty
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
    deadline = 140.0  # 10s safety margin
    
    best_side = float('inf')
    best_params = None
    best_valid = False
    
    # Initial Configuration: Proper hexagonal grid for 11 hexagons
    # Hex grid coordinates for a cluster of 11 hexagons
    d = np.sqrt(3)  # Distance between centers of adjacent unit hexagons
    
    # Using axial coordinates for hex grid
    # Center at (0,0), then 6 neighbors, then 4 more in a ring
    init_centers = [
        (0, 0),
        (d, 0), (d/2, 1.5), (-d/2, 1.5),
        (-d, 0), (-d/2, -1.5), (d/2, -1.5),
        (1.5*d, 3.0), (0, 4.5), (-1.5*d, 3.0),
        (2*d, 0)
    ]
    
    init_params = np.zeros((N, 3))
    for i in range(N):
        init_params[i, 0] = init_centers[i][0]
        init_params[i, 1] = init_centers[i][1]
        init_params[i, 2] = 0.0
        
    # Verify initial configuration is valid
    side, penalty, valid = calculate_metrics(init_params)
    if valid:
        best_side = side
        best_params = init_params.copy()
        best_valid = True
    
    # Simulated Annealing
    np.random.seed(42)
    
    num_restarts = 30
    T_start = 0.5
    T_min = 0.0001
    alpha = 0.99
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Generate initial state
        if r == 0:
            params = init_params.copy()
        elif best_params is not None and np.random.rand() < 0.6:
            params = best_params + np.random.normal(0, 0.3, (N, 3))
            params[:, 2] = np.random.uniform(-np.pi/6, np.pi/6, N)
        else:
            # Random scatter
            params = np.zeros((N, 3))
            params[0] = [0, 0, 0]
            for i in range(1, N):
                angle = np.random.uniform(0, 2*np.pi)
                radius = np.random.uniform(1.0, 3.5)
                params[i] = [radius * np.cos(angle), radius * np.sin(angle), np.random.uniform(-np.pi/6, np.pi/6)]
                
        current_side, current_penalty, current_valid = calculate_metrics(params)
        current_cost = current_side + 50.0 * current_penalty
        
        if current_valid and current_side < best_side:
            best_side = current_side
            best_params = params.copy()
            best_valid = True
            
        T = T_start
        steps_per_restart = 8000
        
        for step in range(steps_per_restart):
            if time.time() > deadline - 2:
                break
                
            # Perturb a random hexagon
            k = np.random.randint(N)
            old_x, old_y, old_theta = params[k]
            
            # Adaptive move size based on temperature
            move_scale = 0.05 + 0.2 * (T / T_start)
            move_dist = np.random.uniform(0.02, move_scale)
            move_angle = np.random.uniform(0, 2*np.pi)
            new_x = old_x + move_dist * np.cos(move_angle)
            new_y = old_y + move_dist * np.sin(move_angle)
            
            # Rotate
            new_theta = old_theta + np.random.uniform(-0.1, 0.1)
            
            params[k] = [new_x, new_y, new_theta]
            
            new_side, new_penalty, new_valid = calculate_metrics(params)
            new_cost = new_side + 50.0 * new_penalty
            
            delta_cost = new_cost - current_cost
            
            if delta_cost < 0 or np.random.rand() < np.exp(-delta_cost / max(T, 1e-10)):
                current_cost = new_cost
                current_side = new_side
                current_penalty = new_penalty
                current_valid = new_valid
                
                if new_valid and new_side < best_side:
                    best_side = new_side
                    best_params = params.copy()
                    best_valid = True
            else:
                # Revert
                params[k] = [old_x, old_y, old_theta]
                
            T *= alpha
            if T < T_min:
                break
                
    # Fallback: return the best valid packing found
    if best_valid:
        inner = []
        for i in range(N):
            x, y, ang = best_params[i]
            ang_deg = np.rad2deg(ang)
            inner.append([float(x), float(y), float(ang_deg)])
        return inner, (0.0, 0.0), float(best_side), 0.0
    else:
        # If no valid packing found, return a large side length
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

if __name__ == "__main__":
    inner, center, side, angle = construct()
    print(f"Side: {side}")

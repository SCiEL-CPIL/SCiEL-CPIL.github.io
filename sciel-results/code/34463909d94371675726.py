import numpy as np
import time

N = 12
HEX_VERTS = np.array([
    [1.0, 0.0],
    [0.5, np.sqrt(3)/2],
    [-0.5, np.sqrt(3)/2],
    [-1.0, 0.0],
    [-0.5, -np.sqrt(3)/2],
    [0.5, -np.sqrt(3)/2]
])

def construct():
    start_time = time.time()
    deadline = 110.0 # 10s margin
    
    # Initial guess: 3x4 grid
    # Rows: 3, Cols: 4
    # dx = sqrt(3), dy = 1.5
    # Offset odd rows by sqrt(3)/2
    
    positions = []
    for j in range(3):
        for i in range(4):
            x = i * np.sqrt(3) + (j % 2) * np.sqrt(3) / 2
            y = j * 1.5
            positions.append([x, y])
    positions = np.array(positions)
    
    # Center the positions
    centroid = np.mean(positions, axis=0)
    positions -= centroid
    
    def get_vertices(positions):
        # positions: (12, 2)
        # Return (72, 2)
        verts = []
        for i in range(12):
            verts.append(positions[i] + HEX_VERTS)
        return np.vstack(verts)

    def calc_L(positions, theta):
        verts = get_vertices(positions)
        center = np.mean(verts, axis=0)
        
        # Normals for theta
        # Flat-top hexagon normals are at 30, 90, 150...
        # If theta is rotation of the hexagon, normals rotate by theta.
        # Standard hexagon (vertex at 0) has flat sides at 30, 90...
        # Wait, standard hexagon vertices at 0, 60...
        # Side 0 is between 0 and 60. Midpoint angle 30. Normal 30.
        # So normals are at theta + 30 + k*60.
        
        angles = np.deg2rad(theta + 30 + np.arange(6) * 60)
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # Projections
        # (72, 2) @ (2, 6) -> (72, 6)
        projs = verts @ normals.T
        
        M = np.max(projs, axis=0)
        m = np.min(projs, axis=0)
        
        # Center of projection interval
        C_k = (M + m) / 2
        # Half-width
        Delta_k = (M - m) / 2
        
        # u_k = center dot normal
        u_k = center @ normals.T
        
        # Required d
        d_k = np.abs(C_k - u_k) + Delta_k
        d = np.max(d_k)
        
        # L = 2 * d / sqrt(3)
        return 2 * d / np.sqrt(3)

    def optimize_theta(positions):
        best_L = float('inf')
        # Scan theta 0 to 30 deg
        for theta in np.linspace(0, 30, 61):
            L = calc_L(positions, theta)
            if L < best_L:
                best_L = L
        return best_L

    # Simulated Annealing
    best_positions = positions.copy()
    best_L = optimize_theta(best_positions)
    best_theta = 0.0 # We only need L for score, but need to return angle?
    # The return interface: inner rows (x, y, angle_degrees), outer center, side, angle.
    # We need to return the actual theta that minimizes L.
    
    def get_best_config(positions):
        best_L = float('inf')
        best_theta = 0.0
        for theta in np.linspace(0, 30, 121): # Finer scan
            L = calc_L(positions, theta)
            if L < best_L:
                best_L = L
                best_theta = theta
        # Refine theta around best
        if best_theta > 0 and best_theta < 30:
             for theta in np.linspace(best_theta - 1, best_theta + 1, 21):
                 if 0 <= theta <= 30:
                     L = calc_L(positions, theta)
                     if L < best_L:
                         best_L = L
                         best_theta = theta
        return best_L, best_theta

    best_L, best_theta = get_best_config(best_positions)
    
    T = 0.5
    T_min = 0.001
    cooling = 0.99
    
    while time.time() - start_time < deadline:
        # Perturb
        new_positions = best_positions.copy()
        idx = np.random.randint(0, 12)
        new_positions[idx] += np.random.normal(0, T * 0.5, 2)
        
        new_L, new_theta = get_best_config(new_positions)
        
        if new_L < best_L:
            best_L = new_L
            best_theta = new_theta
            best_positions = new_positions
        else:
            delta = new_L - best_L
            if np.random.rand() < np.exp(-delta / T):
                best_positions = new_positions
                best_L = new_L
                best_theta = new_theta
                
        T *= cooling
        if T < T_min:
            T = T_min

    # Final optimization
    best_L, best_theta = get_best_config(best_positions)
    
    # Re-center best_positions so outer hex center is (0,0)?
    # The problem asks for outer center.
    # If we calculate L based on centroid, the optimal center is the centroid.
    # So we can return center (0,0) and positions relative to it.
    # Or we can return the actual centroid.
    # Let's return centroid as outer center.
    outer_center = np.mean(get_vertices(best_positions), axis=0)
    
    # Shift positions so they are relative to outer center?
    # The interface: inner rows (x, y, angle).
    # Usually coordinates are absolute.
    # If we return outer center (cx, cy), and inner positions (xi, yi),
    # the evaluator checks if inner hexes are inside outer hex.
    # So we should return absolute coordinates.
    # Our best_positions are centered at 0.
    # So outer center is 0,0.
    
    inner = []
    for i in range(12):
        inner.append([best_positions[i][0], best_positions[i][1], 0.0])
        
    return inner, (0.0, 0.0), best_L, best_theta


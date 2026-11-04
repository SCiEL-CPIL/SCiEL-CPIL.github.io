import numpy as np
import time
from scipy.optimize import minimize

N = 11

def get_hex_vertices(center, angle_deg, side=1.0):
    """
    Get the 6 vertices of a regular hexagon.
    center: (x, y)
    angle_deg: rotation in degrees
    side: side length
    """
    cx, cy = center
    angle_rad = np.deg2rad(angle_deg)
    # Vertices are at radius 'side' from center
    # Angles are 0, 60, 120, 180, 240, 300 relative to the hexagon's local frame
    # If angle_deg is 0, let's say vertex 0 is at (side, 0) relative to center?
    # Standard regular hexagon with "pointy" top or "flat" top?
    # The problem doesn't specify, but "regular hexagon" implies symmetry.
    # Let's assume vertices at angles 0, 60, ...
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack((xs, ys))

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # 100s budget, leave 20s margin
    
    # Initial configuration: Hexagonal lattice
    # Spacing for touching flat sides is sqrt(3)
    d = np.sqrt(3)
    # Generate a grid of potential positions
    # Basis vectors for hex lattice
    v1 = np.array([d, 0.0])
    v2 = np.array([d / 2.0, 1.5])
    
    positions = []
    # Generate enough points
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = i * v1 + j * v2
            positions.append(p)
            
    # Sort by distance to origin and pick 11
    positions = sorted(positions, key=lambda p: np.dot(p, p))
    init_pos = np.array(positions[:N])
    
    # Initial rotations: all 0
    init_thetas = np.zeros(N)
    
    # Flatten parameters
    x0 = np.concatenate([init_pos.flatten(), init_thetas])
    
    # Outer hexagon normals
    # Outer hexagon is centered at (0,0) for calculation purposes?
    # The problem allows the outer hexagon to have a center (x,y).
    # However, for a symmetric packing, the center of mass of the inner hexagons
    # should likely be close to the center of the outer hexagon.
    # We can fix the outer hexagon center at (0,0) and allow the inner ones to move.
    # Or, we can optimize the outer center too.
    # Given the symmetry, fixing outer center at (0,0) is a strong constraint that might be suboptimal
    # if the cluster is asymmetric. But for 11 hexagons, a symmetric cluster is likely optimal.
    # Let's assume outer center is (0,0) to reduce variables.
    
    # Normals for outer hexagon (flat-to-flat distance is S*sqrt(3)/2)
    # Normals point to the middle of the edges.
    # If vertices are at 0, 60..., edges are at 30, 90...
    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    def cost_func(params):
        # Check time
        if time.time() > deadline:
            return 1e9  # Large penalty to stop
        
        # Unpack
        pos = params[:2*N].reshape(N, 2)
        thetas = params[2*N:]
        
        # Calculate vertices of all inner hexagons
        all_vertices = []
        for i in range(N):
            verts = get_hex_vertices(pos[i], thetas[i])
            all_vertices.append(verts)
        
        all_vertices = np.vstack(all_vertices) # Shape (6*N, 2)
        
        # Calculate required S
        # S * sqrt(3)/2 >= max(dot(v, n)) for all v in vertices, n in normals
        # dot products: (6*N, 2) x (2, 6) -> (6*N, 6)
        dots = all_vertices @ normals.T
        max_dot = np.max(dots)
        S_req = (2.0 / np.sqrt(3.0)) * max_dot
        
        # Overlap penalty
        # Conservative check: distance between centers >= 2.0
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(pos[i] - pos[j])
                if dist < 2.0:
                    penalty += (2.0 - dist) ** 2
        
        # We want to minimize S_req.
        # If penalty is large, the configuration is invalid.
        # We add a large multiplier to penalty to discourage overlaps.
        return S_req + 100.0 * penalty

    # Optimization
    best_result = None
    best_cost = float('inf')
    
    # Try a few restarts if time permits
    num_restarts = 1
    for r in range(num_restarts):
        if r > 0:
            # Perturb initial guess
            x0 = x0 + np.random.normal(0, 0.5, size=x0.shape)
        
        try:
            res = minimize(cost_func, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'fatol': 1e-6, 'xatol': 1e-6})
            
            if res.fun < best_cost:
                best_cost = res.fun
                best_result = res.x
                
        except Exception:
            pass
            
        if time.time() > deadline - 20:
            break
            
    if best_result is None:
        # Fallback to initial guess
        best_result = x0
        best_cost = cost_func(x0)
        
    # Extract best configuration
    pos = best_result[:2*N].reshape(N, 2)
    thetas = best_result[2*N:]
    
    # Calculate final S
    all_vertices = []
    for i in range(N):
        verts = get_hex_vertices(pos[i], thetas[i])
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    dots = all_vertices @ normals.T
    max_dot = np.max(dots)
    S_final = (2.0 / np.sqrt(3.0)) * max_dot
    
    # Construct output
    inner = []
    for i in range(N):
        inner.append([pos[i][0], pos[i][1], thetas[i]])
        
    return inner, (0.0, 0.0), S_final, 0.0

# EVOLVE-BLOCK-END

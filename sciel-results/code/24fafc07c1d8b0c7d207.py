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
    # Vertices at angles 0, 60, 120, 180, 240, 300 relative to local frame
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack((xs, ys))

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # Leave 30s margin for safety
    
    # Initial configuration: Hexagonal lattice
    # For unit hexagons, the distance between centers of touching hexagons is sqrt(3)
    d = np.sqrt(3)
    
    # Basis vectors for hex lattice
    v1 = np.array([d, 0.0])
    v2 = np.array([d / 2.0, 1.5])
    
    positions = []
    # Generate enough points for a 11-hexagon cluster
    # A compact cluster of 11 hexagons fits in a 5x5 grid roughly
    for i in range(-3, 4):
        for j in range(-3, 4):
            p = i * v1 + j * v2
            positions.append(p)
            
    # Sort by distance to origin and pick 11
    positions = sorted(positions, key=lambda p: np.dot(p, p))
    init_pos = np.array(positions[:N])
    
    # Initial rotations: all 0
    init_thetas = np.zeros(N)
    
    # Flatten parameters: [x0, y0, ..., x10, y10, theta0, ..., theta10]
    x0 = np.concatenate([init_pos.flatten(), init_thetas])
    
    # Outer hexagon is fixed at center (0,0) with angle 0 for simplicity?
    # The task allows the outer hexagon to rotate and translate.
    # However, due to symmetry, we can fix the outer hexagon center at (0,0) 
    # and allow it to rotate. Or fix rotation at 0 and allow center?
    # Actually, the score is the side length S.
    # If we fix outer center at (0,0) and outer angle at 0, we might miss optimal rotations.
    # Let's optimize the outer hexagon angle as well.
    # Parameters: [inner_pos (2N), inner_thetas (N), outer_angle (1)]
    # Total: 3N + 1 = 34 parameters.
    
    # But wait, if we allow outer translation, it just shifts the inner hexagons.
    # We can fix the outer center at (0,0) without loss of generality by shifting the whole system.
    # So we only need to optimize inner positions, inner rotations, and outer rotation.
    
    def cost_func(params):
        if time.time() > deadline:
            return 1e9
            
        pos = params[:2*N].reshape(N, 2)
        thetas = params[2*N:3*N]
        outer_angle = params[3*N]
        
        # Calculate vertices of all inner hexagons
        all_vertices = []
        for i in range(N):
            verts = get_hex_vertices(pos[i], thetas[i])
            all_vertices.append(verts)
        all_vertices = np.vstack(all_vertices) # Shape (6*N, 2)
        
        # Outer hexagon normals
        # Vertices at 0, 60... -> Edges midpoints at 30, 90...
        # If outer hexagon is rotated by outer_angle, normals are at 30+outer_angle, 90+outer_angle, ...
        normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]) + outer_angle)
        normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
        
        # Calculate required S
        # S * sqrt(3)/2 >= max(dot(v, n)) for all v, n
        dots = all_vertices @ normals.T
        max_dot = np.max(dots)
        
        # Ensure S is positive (if all vertices are at negative dot, S could be negative? No, max_dot will be positive if vertices exist)
        # But if the cluster is entirely in one half-plane relative to a normal, max_dot might be small.
        # Actually, for a convex body containing origin (or close), max_dot over all 6 normals will be the support function.
        # The support function h(n) = max_{v in V} dot(v, n).
        # The width in direction n is h(n) + h(-n).
        # For a regular hexagon centered at origin, h(n) = S * sqrt(3)/2 for all 6 normal directions.
        # So we need S * sqrt(3)/2 >= max(h(n_i)) for i=0..5.
        # Thus S >= (2/sqrt(3)) * max(h(n_i)).
        
        S_req = (2.0 / np.sqrt(3.0)) * max_dot
        
        # Overlap penalty
        # Check distance between centers. 
        # For two unit hexagons, if distance < 2*inradius? No.
        # Inradius = sqrt(3)/2. 
        # If they touch sides, distance = 2 * inradius = sqrt(3).
        # If they touch vertices, distance can be smaller?
        # The minimum distance between two non-overlapping unit hexagons depends on orientation.
        # Simplest robust penalty: use a conservative minimum distance.
        # For arbitrary rotations, the minimum non-overlapping distance is sqrt(3) (touching flat sides).
        # If rotated 30 deg, they can get closer?
        # Actually, if two hexagons are rotated 30 deg relative to each other, they can interlock more tightly?
        # No, they are convex. The minimum distance between two non-overlapping convex polygons is the distance between their boundaries.
        # A simple proxy: if center distance < 1.5, they likely overlap.
        # Let's use 1.8 as a safe threshold for penalty activation.
        # Or better, compute actual polygon intersection? Too slow for Nelder-Mead.
        # Let's use a soft penalty based on center distance < 1.7.
        
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(pos[i] - pos[j])
                if dist < 1.8:
                    penalty += (1.8 - dist) ** 2
        
        return S_req + 50.0 * penalty

    # Optimization
    best_result = None
    best_cost = float('inf')
    
    # Try a few restarts if time permits
    num_restarts = 0
    max_restarts = 5
    
    while num_restarts < max_restarts:
        if num_restarts > 0:
            # Perturb initial guess
            x0 = x0 + np.random.normal(0, 0.5, size=x0.shape)
        
        try:
            res = minimize(cost_func, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'fatol': 1e-6, 'xatol': 1e-6})
            
            if res.fun < best_cost:
                best_cost = res.fun
                best_result = res.x
                
        except Exception:
            pass
            
        num_restarts += 1
        if time.time() > deadline - 20:
            break
            
    if best_result is None:
        # Fallback to initial guess
        best_result = x0
        best_cost = cost_func(x0)
        
    # Extract best configuration
    pos = best_result[:2*N].reshape(N, 2)
    thetas = best_result[2*N:3*N]
    outer_angle = best_result[3*N]
    
    # Calculate final S
    all_vertices = []
    for i in range(N):
        verts = get_hex_vertices(pos[i], thetas[i])
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]) + outer_angle)
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    dots = all_vertices @ normals.T
    max_dot = np.max(dots)
    S_final = (2.0 / np.sqrt(3.0)) * max_dot
    
    # Construct output
    inner = []
    for i in range(N):
        inner.append([pos[i][0], pos[i][1], thetas[i]])
        
    return inner, (0.0, 0.0), S_final, outer_angle

# EVOLVE-BLOCK-END

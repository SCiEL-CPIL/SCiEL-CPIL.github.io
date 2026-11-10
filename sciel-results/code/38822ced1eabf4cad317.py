import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Initial configuration: 1 center, 6 neighbors, 4 more
    # Center at (0,0)
    # Neighbors at distance sqrt(3)
    # We'll use a simple grid-like initialization
    # 1 center
    # 6 around it
    # 4 more attached to the "sides" of the ring
    
    centers = []
    angles = [0.0]
    
    # Center
    centers.append((0.0, 0.0))
    
    # 6 neighbors
    for i in range(6):
        angle = i * np.pi / 3
        x = np.sqrt(3) * np.cos(angle)
        y = np.sqrt(3) * np.sin(angle)
        centers.append((x, y))
        angles.append(0.0)
        
    # 4 more: attach to 4 of the 6 neighbors
    # To keep it symmetric, attach to 0, 90, 180, 270? No, hex symmetry.
    # Attach to neighbors at 0, 60, 180, 240?
    # Let's try attaching to 4 distinct neighbors to spread out.
    # Indices 1, 2, 4, 5?
    # Let's just pick 4 neighbors and add hexes further out.
    # Neighbor 0 is at (sqrt(3), 0). Add at (2*sqrt(3), 0).
    # Neighbor 1 is at (sqrt(3)/2, 3/2). Add further.
    
    # Better: Just use a random initialization and optimize.
    # But we need a good start.
    # Let's try a "3-4-3" strip? No, compact is better.
    
    # Let's use the 7-cluster and add 4 at the "midpoints" of the sides of the 7-cluster?
    # The 7-cluster "sides" are between neighbors.
    # Midpoint of side between N0 and N1 is at angle 30 deg.
    # Distance from center to midpoint of side of 7-cluster?
    # It's not a regular hexagon, so this is tricky.
    
    # Let's just initialize 11 hexes in a tight random cluster near origin.
    np.random.seed(42)
    init_centers = []
    init_angles = []
    
    # 1 center
    init_centers.append([0.0, 0.0])
    init_angles.append(0.0)
    
    # 6 neighbors
    for i in range(6):
        a = i * np.pi / 3
        init_centers.append([np.sqrt(3)*np.cos(a), np.sqrt(3)*np.sin(a)])
        init_angles.append(0.0)
        
    # 4 more: place them further out, e.g. at distance 2*sqrt(3) in 4 directions
    for i in [0, 1, 3, 4]: # Pick 4 directions
        a = i * np.pi / 3
        init_centers.append([2*np.sqrt(3)*np.cos(a), 2*np.sqrt(3)*np.sin(a)])
        init_angles.append(0.0)
        
    # Total 11
    
    def cost(params):
        # params: 11*(x,y,angle) = 33 vars
        # reshape to (11, 3)
        pts = params.reshape(11, 3)
        max_proj = 0.0
        for i in range(11):
            x, y, ang = pts[i]
            # Vertices of hexagon at (x,y) with rotation ang
            # Unit hexagon vertices relative to center:
            # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
            # Rotate by ang
            cos_a = np.cos(ang)
            sin_a = np.sin(ang)
            
            # Pre-computed unit vertices
            # v0 = (1, 0)
            # v1 = (0.5, sqrt(3)/2)
            # etc.
            
            # We need max projection in 6 directions: 0, 60, 120, 180, 240, 300
            # Projection of vertex v onto direction d: (x + vx*cos - vy*sin) * cos_d + (y + vx*sin + vy*cos) * sin_d
            # This is equivalent to max over vertices of dot((x,y,v), d)
            
            # To speed up, we can just calculate the bounding box of the union?
            # No, we need support function in 6 specific directions.
            
            for k in range(6):
                dir_ang = k * np.pi / 3
                cos_d = np.cos(dir_ang)
                sin_d = np.sin(dir_ang)
                
                # Max projection of this hexagon in direction dir_ang
                # The max projection of a rotated hexagon is x*cos_d + y*sin_d + R(ang, dir_ang)
                # where R is the support function of the unit hexagon.
                # For a regular hexagon, the support function in direction phi relative to its normal is:
                # If we align the hexagon with a flat side normal to x-axis, the support is:
                # h(theta) = cos(theta - k*pi/3) for theta in [k*pi/3 - pi/6, k*pi/3 + pi/6]?
                # Actually, simpler: just compute the 6 vertices and take max dot product.
                
                # Unit vertices
                u_v = np.array([
                    [1.0, 0.0],
                    [0.5, np.sqrt(3)/2],
                    [-0.5, np.sqrt(3)/2],
                    [-1.0, 0.0],
                    [-0.5, -np.sqrt(3)/2],
                    [0.5, -np.sqrt(3)/2]
                ])
                
                # Rotate
                rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                verts = u_v @ rot_mat.T
                verts += np.array([x, y])
                
                # Project onto dir
                dir_vec = np.array([cos_d, sin_d])
                projs = verts @ dir_vec
                max_p = np.max(projs)
                
                if max_p > max_proj:
                    max_proj = max_p
                    
        # L = max_proj * 2 / sqrt(3)
        return max_proj * 2.0 / np.sqrt(3)

    # Optimize
    x0 = np.array([v for c in init_centers for v in c] + init_angles)
    
    # Use Nelder-Mead or L-BFGS-B? Nelder-Mead is robust for non-smooth.
    # But 33 variables is a lot.
    # Let's try to reduce variables: fix outer center at (0,0) and rotation 0.
    # We only optimize inner positions and rotations.
    
    # To make it faster, we can use a coarser grid or fewer iterations.
    # 120s limit.
    # Let's try a few random restarts with Nelder-Mead.
    
    best_L = 40.0
    best_params = x0
    
    for _ in range(5):
        # Perturb initial guess
        x0_pert = x0 + np.random.normal(0, 0.5, size=x0.shape)
        
        res = minimize(cost, x0_pert, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-3, 'fatol': 1e-3})
        
        if res.fun < best_L:
            best_L = res.fun
            best_params = res.x
            
    # Extract results
    inner = []
    for i in range(11):
        x, y, ang = best_params[3*i:3*i+3]
        # Convert angle to degrees
        ang_deg = np.degrees(ang) % 360
        inner.append([x, y, ang_deg])
        
    # Outer hexagon center is (0,0), side is best_L, rotation 0
    return inner, (0.0, 0.0), best_L, 0.0
# EVOLVE-BLOCK-END

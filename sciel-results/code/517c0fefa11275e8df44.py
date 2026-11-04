import numpy as np
from scipy.optimize import minimize

def construct():
    N = 12
    
    # Generate 12 centers from a hexagonal tiling
    # Pointy-top hexagons
    centers = []
    for i in range(-1, 2): # -1, 0, 1
        for j in range(-1, 3): # -1, 0, 1, 2
            x = i * np.sqrt(3)
            y = 2 * j + (1 if i % 2 != 0 else 0)
            centers.append([x, y])
            
    centers = np.array(centers)
    # Center the cluster
    centers -= np.mean(centers, axis=0)
    
    # Define the 6 vertices of a unit pointy-top hexagon
    # Angles: 90, 150, 210, 270, 330, 30
    angles = np.deg2rad([90, 150, 210, 270, 330, 30])
    hex_verts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Generate all 72 vertices of the composite body
    all_verts = []
    for c in centers:
        all_verts.append(c + hex_verts)
    all_verts = np.array(all_verts)
    
    def objective(params):
        theta, cx, cy = params
        rot = np.deg2rad(theta)
        c, s = np.cos(rot), np.sin(rot)
        R = np.array([[c, -s], [s, c]])
        
        # Rotate and translate vertices
        rotated_verts = all_verts @ R.T
        translated_verts = rotated_verts + np.array([cx, cy])
        
        x = translated_verts[:, 0]
        y = translated_verts[:, 1]
        
        # Calculate L based on the 6 half-space constraints
        # 1. x + sqrt(3)y <= sqrt(3)L  => L >= (x + sqrt(3)y) / sqrt(3)
        # 2. -x + sqrt(3)y <= sqrt(3)L => L >= (-x + sqrt(3)y) / sqrt(3)
        # 3. -x <= sqrt(3)/2 L         => L >= -x / (sqrt(3)/2) = -2x/sqrt(3)
        # 4. -x - sqrt(3)y <= sqrt(3)L => L >= (-x - sqrt(3)y) / sqrt(3)
        # 5. x - sqrt(3)y <= sqrt(3)L  => L >= (x - sqrt(3)y) / sqrt(3)
        # 6. x <= sqrt(3)/2 L          => L >= x / (sqrt(3)/2) = 2x/sqrt(3)
        
        sqrt3 = np.sqrt(3)
        L1 = (x + sqrt3 * y) / sqrt3
        L2 = (-x + sqrt3 * y) / sqrt3
        L3 = -2 * x / sqrt3
        L4 = (-x - sqrt3 * y) / sqrt3
        L5 = (x - sqrt3 * y) / sqrt3
        L6 = 2 * x / sqrt3
        
        # L must be >= max of all these values for all vertices
        L = np.max(np.maximum(np.maximum(np.maximum(L1, L2), np.maximum(L3, L4)), np.maximum(L5, L6)))
        
        # Ensure L is positive (should be, but just in case)
        return max(L, 0.0)

    # Initial guess
    x0 = [0.0, 0.0, 0.0] # theta, cx, cy
    
    # Use Nelder-Mead for derivative-free optimization
    # We might need to bound theta to [0, 60] due to hexagon symmetry, but NM handles it.
    # To help convergence, we can try a few random starts or just rely on symmetry.
    # The function is periodic in theta with period 60 degrees (for the hexagon shape)
    # but the cluster is not 60-degree symmetric, it's 30-degree? No, the cluster of 12 hexagons
    # in a 3x4 arrangement has a specific symmetry.
    # Let's just run NM from 0.
    
    res = minimize(objective, x0, method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 10000})
    
    # Verify if the result is better than a simple grid search or just use it.
    # We can also try a few other starting points to be safe.
    best_res = res
    best_L = res.fun
    
    # Try a few random starting points for robustness
    np.random.seed(42)
    for _ in range(10):
        theta0 = np.random.uniform(0, 60)
        cx0 = np.random.uniform(-1, 1)
        cy0 = np.random.uniform(-1, 1)
        x0_rand = [theta0, cx0, cy0]
        res_rand = minimize(objective, x0_rand, method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 5000})
        if res_rand.fun < best_L:
            best_L = res_rand.fun
            best_res = res_rand
            
    theta, cx, cy = best_res.x
    
    # Construct output
    # Inner hexagons: (x, y, angle)
    # The angle of each inner hexagon is the global rotation theta
    # The centers are rotated and translated
    
    rot_rad = np.deg2rad(theta)
    c, s = np.cos(rot_rad), np.sin(rot_rad)
    R = np.array([[c, -s], [s, c]])
    
    inner = []
    for c_center in centers:
        rc = R @ c_center
        rc[0] += cx
        rc[1] += cy
        inner.append([rc[0], rc[1], theta])
        
    # Outer hexagon: center (cx, cy), side best_L, rotation 0
    # We fixed outer rotation to 0 in the objective function.
    # The return signature asks for outer rotation. We can return 0.
    
    return inner, (cx, cy), best_L, 0.0

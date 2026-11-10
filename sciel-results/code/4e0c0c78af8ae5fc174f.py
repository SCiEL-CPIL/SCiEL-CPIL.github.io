# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the respective outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    
    # Constants
    N_INNER = 12
    UNIT_SIDE = 1.0
    # Distance between centers of two touching aligned unit hexagons
    MIN_DIST = np.sqrt(3.0)
    
    # Helper: Get vertices of a unit hexagon at (cx, cy) with angle a_rad
    def get_hex_vertices(cx, cy, a_rad):
        angles = a_rad + np.arange(6) * np.pi / 3.0
        return np.column_stack((cx + np.cos(angles), cy + np.sin(angles)))

    # Helper: Calculate minimum outer side length s for given inner centers and outer angle theta
    # Assumes inner hexagons are aligned (angle 0)
    def calc_min_s(centers, theta_rad):
        # Outer hexagon edge normals
        # The normals to the edges of a regular hexagon with angle theta (vertex at theta)
        # Vertices are at theta + i*pi/3.
        # Edge i connects v_i and v_{i+1}.
        # Normal points inward/outward.
        # The apothem direction for edge i is theta + i*pi/3 + pi/6.
        # Let's verify.
        # Vertex 0: (cos(t), sin(t)). Vertex 1: (cos(t+60), sin(t+60)).
        # Midpoint angle: t + 30.
        # So normals are at angles t + 30 + i*60.
        
        normal_angles = theta_rad + np.pi/6 + np.arange(6) * np.pi/3.0
        normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
        
        max_proj = 0.0
        
        # Iterate through all inner hexagons
        for i in range(N_INNER):
            cx, cy = centers[i]
            # Get vertices of this inner hexagon (aligned, angle 0)
            verts = get_hex_vertices(cx, cy, 0.0)
            
            # Project all vertices onto all 6 outer normals
            # We want the max absolute projection relative to the outer center (0,0 for now)
            # The outer hexagon is centered at (0,0) in this calculation frame.
            # We will shift the whole system to center of mass later, or just optimize positions relative to 0,0.
            
            # Project verts onto normals
            # verts: (6, 2), normals: (6, 2)
            # We need max over all v in verts, n in normals of |v . n|
            # This is equivalent to max over all pairs.
            
            # Efficiently:
            # proj = verts @ normals.T -> (6, 6)
            # max_abs = np.max(np.abs(proj))
            
            proj = verts @ normals.T
            max_abs = np.max(np.abs(proj))
            
            if max_abs > max_proj:
                max_proj = max_abs
                
        # max_proj is the required apothem (distance from center to edge)
        # s * sqrt(3)/2 = apothem  => s = apothem * 2 / sqrt(3)
        return max_proj * 2.0 / np.sqrt(3.0)

    # Objective function for optimizer
    # x: flattened array of 12*(2) coordinates
    def objective(x):
        centers = x.reshape((N_INNER, 2))
        
        # Penalty for overlaps
        penalty = 0.0
        for i in range(N_INNER):
            for j in range(i+1, N_INNER):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < MIN_DIST:
                    # Quadratic penalty
                    penalty += (MIN_DIST - dist)**2
        
        # Find optimal theta for this configuration
        # We can sample theta or use a simple minimization over theta
        # Since the function is periodic and smooth-ish, a small grid search is fast enough
        best_s = float('inf')
        # Search theta from 0 to pi/3 (60 deg) with fine steps
        thetas = np.linspace(0, np.pi/3.0, 36)
        for t in thetas:
            s_val = calc_min_s(centers, t)
            if s_val < best_s:
                best_s = s_val
        
        return best_s + penalty * 10.0 # Weight penalty heavily

    # Initial guess: Honeycomb cluster
    # Center at 0,0
    centers = np.zeros((N_INNER, 2))
    # Ring 1: 6 hexagons
    r1 = MIN_DIST
    for i in range(6):
        angle = i * np.pi / 3.0
        centers[i+1] = [r1 * np.cos(angle), r1 * np.sin(angle)]
        
    # Ring 2: 5 hexagons
    # Place them in the gaps or extending the corners
    # A common compact shape for 12 is 1 + 6 + 5.
    # The 5 can be placed at angles 0, 60, 120, 180, 240 but further out?
    # Or fill the "valleys".
    # Let's place them at radius 2 * MIN_DIST * cos(30)? No.
    # Distance from center to ring 2 centers in a perfect hex grid is 2 * r1 * cos(30)?
    # Actually, the next shell in hex grid is at distance 2 * sqrt(3) / sqrt(3) * something?
    # Let's just place them at distance 2 * MIN_DIST from center, at angles offset by 30 deg?
    # Or better, place them touching the ring 1 hexagons.
    
    # Let's use a known good configuration for 12:
    # It's a hexagon of side 2 (which has 19 cells) minus 7 cells?
    # Or just a 3x4 rectangle in hex coords?
    
    # Let's try placing the remaining 5 at:
    # Angles: 0, 60, 120, 180, 240
    # Radius: 2 * MIN_DIST
    # This creates a sparse ring.
    
    for i in range(5):
        angle = i * 2 * np.pi / 5.0 # Evenly spaced? No, hex symmetry is 60 deg.
        # Let's place them at hex grid positions.
        # Hex grid axial coords:
        # Center (0,0)
        # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
        # Ring 2 candidates: (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0)...
        # We need 5 more.
        # Let's pick (2,0), (0,2), (-2,1), (-2,-1), (0,-2)?
        
    # Converting axial to pixel (pointy top or flat top?)
    # Our hexagons are "flat top" if angle 0 means vertex at right?
    # get_hex_vertices with angle 0:
    # v0 = (1, 0). v1 = (0.5, 0.866).
    # This is a "pointy top" hexagon? No, vertex at 0 deg is right.
    # Edge is vertical? No.
    # Let's stick to the optimizer. It will fix the positions.
    
    # Better initial guess for Ring 2:
    # Place them at distance 2 * MIN_DIST from center, at angles 0, 60, 120, 180, 240.
    r2 = 2.0 * MIN_DIST
    for i in range(5):
        angle = i * np.pi / 3.0
        centers[i+7] = [r2 * np.cos(angle), r2 * np.sin(angle)]
        
    # Flatten
    x0 = centers.flatten()
    
    # Optimize
    start_time = time.time()
    timeout = 50.0 # 50 seconds for optimization
    
    best_result = None
    best_val = float('inf')
    
    # Run Nelder-Mead
    try:
        result = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8, 'disp': False})
        
        if result.fun < best_val:
            best_val = result.fun
            best_result = result.x
            
        # Try a few random restarts if time permits
        restarts = 5
        for r in range(restarts):
            if time.time() - start_time > timeout * 0.7:
                break
                
            # Perturb initial guess
            x_rand = x0 + np.random.normal(0, 0.5, size=x0.shape)
            
            res_r = minimize(objective, x_rand, method='Nelder-Mead', 
                             options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'disp': False})
            
            if res_r.fun < best_val:
                best_val = res_r.fun
                best_result = res_r.x
                
    except Exception as e:
        # Fallback to initial guess if optimization fails
        best_result = x0

    if best_result is None:
        best_result = x0

    # Extract best centers
    best_centers = best_result.reshape((N_INNER, 2))
    
    # Find the optimal theta for this final configuration
    best_s_final = float('inf')
    best_theta = 0.0
    thetas_fine = np.linspace(0, np.pi/3.0, 180)
    for t in thetas_fine:
        s_val = calc_min_s(best_centers, t)
        if s_val < best_s_final:
            best_s_final = s_val
            best_theta = t

    # Center the configuration at the origin for the outer hexagon
    # The outer hexagon is at (0,0).
    # We should shift the inner centers so their "center of mass" or bounding box center is at 0?
    # Actually, the calc_min_s assumed outer center at 0.
    # So the best_centers are already relative to the optimal outer center.
    # But wait, the optimizer might have shifted the whole cluster away from 0.
    # The objective function `calc_min_s` computes distance from 0.
    # So if the cluster moves to (100, 100), the s will be huge.
    # The optimizer will naturally keep the cluster near 0 to minimize s.
    # So best_centers are the correct coordinates relative to the outer hex center (0,0).

    # Construct return values
    inner_hex_data = np.zeros((N_INNER, 3))
    for i in range(N_INNER):
        inner_hex_data[i, 0] = best_centers[i, 0]
        inner_hex_data[i, 1] = best_centers[i, 1]
        inner_hex_data[i, 2] = 0.0 # Angle 0
    
    outer_hex_data = np.array([0.0, 0.0, np.degrees(best_theta)])
    
    # Ensure s is not too small (sanity check)
    if best_s_final < np.sqrt(12):
        best_s_final = np.sqrt(12)

    return inner_hex_data, outer_hex_data, best_s_final

# EVOLVE-BLOCK-END

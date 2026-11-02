import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    # Normals for the outer hexagon (fixed orientation)
    # The outer hexagon side length L is what we minimize.
    # Its apothem is A = sqrt(3)/2 * L.
    # The condition is max support of inner cluster in these directions <= A.
    
    # Unit hexagon support function calculation helper
    # Vertices of unit hexagon (side 1) are at angles 0, 60, 120, 180, 240, 300
    # h_unit(theta) = max(cos(theta - 60*k))
    
    def support_hexagon(center, rotation_rad, direction_rad):
        # direction_rad is angle of u
        # center is (x,y)
        # rotation_rad is the rotation of the hexagon itself
        c = np.array(center)
        u = np.array([np.cos(direction_rad), np.sin(direction_rad)])
        
        # Translation part
        trans_part = np.dot(c, u)
        
        # Rotation part
        # The effective angle of the direction relative to the hexagon's local frame
        local_angle = direction_rad - rotation_rad
        # Support of unit hexagon at local_angle
        # max over k in 0..5 of cos(local_angle - k*pi/3)
        angles = local_angle - np.arange(6) * np.pi / 3.0
        rot_part = np.max(np.cos(angles))
        
        return trans_part + rot_part

    def objective(params):
        r1, r2, th1, th2, th_glob = params
        
        # Generate centers
        centers = []
        rotations = []
        
        # 1 Center
        centers.append(np.array([0.0, 0.0]))
        rotations.append(th_glob) # Assume center hexagon rotates with global? Or stays 0? 
        # Actually, if we rotate the whole cluster, the center hexagon orientation relative to global axes changes.
        # Let's say the "cluster frame" is rotated by th_glob.
        # So a hexagon at angle 0 in cluster frame is at th_glob in global frame.
        # Its orientation is also th_glob.
        
        # 5 Inner
        for i in range(5):
            angle_local = th1 + i * 2 * np.pi / 5.0
            x = r1 * np.cos(angle_local + th_glob)
            y = r1 * np.sin(angle_local + th_glob)
            centers.append(np.array([x, y]))
            rotations.append(th_glob) # All inner hexagons aligned with cluster? 
            # Or maybe they should be aligned with the radial direction?
            # "Radial Slicing" might imply alignment with radius.
            # Let's try aligned with radial direction first: rotation = angle_local + th_glob
            # But the seed program used 0. 
            # Let's allow the rotation to be th_glob (parallel to global axes) or angle_local (radial).
            # Given the symmetry, radial alignment is often better for packing.
            # Let's use radial alignment for inner: rot = angle_local + th_glob
            # And radial for outer?
            # Let's stick to one hypothesis: All hexagons are rotated such that their "flat" sides are perpendicular to the radius? 
            # Or just all aligned with th_glob.
            # Let's try all aligned with th_glob first. It's simpler.
            
        # 6 Outer
        for j in range(6):
            angle_local = th2 + j * 2 * np.pi / 6.0
            x = r2 * np.cos(angle_local + th_glob)
            y = r2 * np.sin(angle_local + th_glob)
            centers.append(np.array([x, y]))
            rotations.append(th_glob)
            
        # Calculate max support in the 6 outer hexagon normal directions
        # Outer hexagon normals (fixed): 30, 90, 150, 210, 270, 330 degrees
        outer_normals = np.deg2rad([30, 90, 150, 210, 270, 330])
        
        max_support = 0.0
        for norm_angle in outer_normals:
            # H(u) = max over all inner hexagons of h_i(u)
            h_max = 0.0
            for i in range(N):
                h_val = support_hexagon(centers[i], rotations[i], norm_angle)
                if h_val > h_max:
                    h_max = h_val
            if h_max > max_support:
                max_support = h_max
        
        L_req = (2.0 / np.sqrt(3.0)) * max_support
        
        # Overlap penalties
        viol_sum = 0.0
        min_dist = np.sqrt(3.0)
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < min_dist:
                    viol = min_dist - dist
                    viol_sum += viol * viol
        
        return L_req + 100.0 * viol_sum

    best_L = float('inf')
    best_params = None
    
    # Try a few random starts for the 5 parameters
    for seed in range(20):
        np.random.seed(seed)
        # r1 ~ 1.5-2.5, r2 ~ 3.0-4.5
        r1 = np.random.uniform(1.5, 2.5)
        r2 = np.random.uniform(3.0, 4.5)
        th1 = np.random.uniform(0, 2*np.pi)
        th2 = np.random.uniform(0, 2*np.pi)
        th_glob = np.random.uniform(0, 2*np.pi)
        
        x0 = np.array([r1, r2, th1, th2, th_glob])
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                              options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if result.fun < best_L:
                best_L = result.fun
                best_params = result.x
        except Exception:
            continue
            
    if best_params is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Reconstruct best configuration
    r1, r2, th1, th2, th_glob = best_params
    
    inner_list = []
    # Center
    inner_list.append([0.0, 0.0, np.rad2deg(th_glob)])
    
    # Inner 5
    for i in range(5):
        angle_local = th1 + i * 2 * np.pi / 5.0
        x = r1 * np.cos(angle_local + th_glob)
        y = r1 * np.sin(angle_local + th_glob)
        inner_list.append([x, y, np.rad2deg(th_glob)])
        
    # Outer 6
    for j in range(6):
        angle_local = th2 + j * 2 * np.pi / 6.0
        x = r2 * np.cos(angle_local + th_glob)
        y = r2 * np.sin(angle_local + th_glob)
        inner_list.append([x, y, np.rad2deg(th_glob)])
        
    # Calculate final L for return
    # We need the precise L based on the best_params
    # Re-run objective logic to get just L part?
    # Or just return best_L? The score is the outer side length.
    # The objective included penalties. If valid, best_L is close to L_req.
    # Let's calculate L_req accurately for the best_params.
    
    centers = []
    rotations = []
    centers.append(np.array([0.0, 0.0])); rotations.append(th_glob)
    for i in range(5):
        angle_local = th1 + i * 2 * np.pi / 5.0
        x = r1 * np.cos(angle_local + th_glob)
        y = r1 * np.sin(angle_local + th_glob)
        centers.append(np.array([x, y])); rotations.append(th_glob)
    for j in range(6):
        angle_local = th2 + j * 2 * np.pi / 6.0
        x = r2 * np.cos(angle_local + th_glob)
        y = r2 * np.sin(angle_local + th_glob)
        centers.append(np.array([x, y])); rotations.append(th_glob)
        
    outer_normals = np.deg2rad([30, 90, 150, 210, 270, 330])
    max_support = 0.0
    for norm_angle in outer_normals:
        h_max = 0.0
        for i in range(N):
            h_val = support_hexagon(centers[i], rotations[i], norm_angle)
            if h_val > h_max:
                h_max = h_val
        if h_max > max_support:
            max_support = h_max
            
    L_final = (2.0 / np.sqrt(3.0)) * max_support
    
    return inner_list, (0.0, 0.0), L_final, 0.0

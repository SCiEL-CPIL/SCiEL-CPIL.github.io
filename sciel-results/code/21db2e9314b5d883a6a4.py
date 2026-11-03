import numpy as np
import scipy.optimize as so
import time

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # Vertices of a unit regular hexagon (side length 1)
    # Angle 0: Vertex at (1, 0)
    # This is a "pointy" orientation relative to x-axis?
    # Vertices: (1,0), (0.5, sqrt(3)/2), ...
    v_hex = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    # Normals for the container hexagon
    # Container with side S, apothem a = S * sqrt(3)/2
    # Normals at angles 30, 90, 150, 210, 270, 330 degrees
    # This corresponds to a "flat-top" container (side horizontal at top/bottom? No, normal at 90 is vertical, so side is horizontal).
    # Let's check: Normal at 90 deg is (0,1). Side is y = a.
    # Normal at 30 deg is (sqrt(3)/2, 1/2).
    n_angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    n_vecs = np.array([[np.cos(a), np.sin(a)] for a in n_angles])
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # We will try a few fixed orientations for inner hexagons
    # and solve the LP for each.
    # Since the problem is symmetric, 0 and 30 degrees for inner might be the main candidates.
    # Also container rotation.
    
    inner_angles_to_try = [0.0, 30.0]
    container_angles_to_try = [0.0, 30.0]
    
    for theta_inner in inner_angles_to_try:
        # Rotate vertices by theta_inner
        rad = np.deg2rad(theta_inner)
        R = np.array([
            [np.cos(rad), -np.sin(rad)],
            [np.sin(rad), np.cos(rad)]
        ])
        v_rot = v_hex @ R.T
        
        for phi_container in container_angles_to_try:
            # Rotate normals by phi_container
            rad_c = np.deg2rad(phi_container)
            R_c = np.array([
                [np.cos(rad_c), -np.sin(rad_c)],
                [np.sin(rad_c), np.cos(rad_c)]
            ])
            n_rot = n_vecs @ R_c.T
            
            # Variables: x_0..x_11, y_0..y_11, X, Y, S
            # Total 24 + 2 + 1 = 27 variables
            n_vars = 2 * N + 3
            x_init = np.zeros(n_vars)
            
            # Objective: Minimize S (last variable)
            c = np.zeros(n_vars)
            c[-1] = 1.0
            
            # Constraints:
            # (v_i + v_j) dot n_k - X n_k_x - Y n_k_y <= S * sqrt(3)/2
            # (v_i + v_j) dot n_k - X n_k_x - Y n_k_y - S * sqrt(3)/2 <= 0
            
            A_rows = []
            b_vals = []
            
            for i in range(N):
                for j in range(6):
                    vx = v_rot[j, 0]
                    vy = v_rot[j, 1]
                    for k in range(6):
                        nx = n_rot[k, 0]
                        ny = n_rot[k, 1]
                        
                        row = np.zeros(n_vars)
                        # Coeff for x_i
                        row[2*i] = nx
                        # Coeff for y_i
                        row[2*i + 1] = ny
                        # Coeff for X
                        row[2*N] = -nx
                        # Coeff for Y
                        row[2*N + 1] = -ny
                        # Coeff for S
                        row[-1] = -np.sqrt(3)/2
                        
                        A_rows.append(row)
                        b_vals.append(0.0)
                        
            A = np.array(A_rows)
            b = np.array(b_vals)
            
            # Solve LP
            try:
                res = so.linprog(c, A_ub=A, b_ub=b, bounds=[(None, None)] * n_vars, method='highs')
                if res.success:
                    S_val = res.x[-1]
                    if S_val < best_side:
                        best_side = S_val
                        # Extract inner positions
                        inner_pos = []
                        for i in range(N):
                            x_i = res.x[2*i]
                            y_i = res.x[2*i + 1]
                            inner_pos.append([x_i, y_i, theta_inner])
                        best_inner = inner_pos
                        best_center = (res.x[2*N], res.x[2*N + 1])
                        best_angle = phi_container
            except Exception as e:
                pass
                
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    # If LP failed or didn't find anything, return a safe fallback
    if best_inner is None:
        # Fallback: simple grid
        inner = []
        for i in range(4):
            for j in range(3):
                x = (i - 1.5) * 1.8
                y = (j - 1.0) * 1.732
                inner.append([x, y, 0.0])
        return inner, (0.0, 0.0), 10.0, 0.0

    return best_inner, best_center, best_side, best_angle


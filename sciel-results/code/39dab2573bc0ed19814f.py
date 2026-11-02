import numpy as np
import time
from scipy.optimize import linprog

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_result = None
    best_side = float('inf')
    
    # Try different outer hexagon rotations
    outer_angles = [0.0, 30.0, 15.0, 45.0, 60.0, 20.0, 40.0]
    
    # Hexagonal grid positions for 11 hexagons
    d = np.sqrt(3)  # center-to-center distance for adjacent hexagons
    
    # Generate positions: center + 6 in first ring + 4 in second ring
    positions = []
    positions.append((0.0, 0.0))
    for k in range(6):
        angle = k * np.pi / 3
        positions.append((d * np.cos(angle), d * np.sin(angle)))
    for k in range(6):
        angle = k * np.pi / 3 + np.pi / 6
        positions.append((2 * d * np.cos(angle), 2 * d * np.sin(angle)))
    positions = positions[:N]
    pos_arr = np.array(positions)
    
    for outer_angle in outer_angles:
        if time.time() > deadline:
            break
        
        outer_rot = np.radians(outer_angle)
        
        # 6 normal directions for outer hexagon
        normals = []
        for k in range(6):
            theta = outer_rot + k * np.pi / 3
            normals.append(np.array([np.cos(theta), np.sin(theta)]))
        normals = np.array(normals)
        
        # Support function for a regular hexagon of side `side`, rotated by `rot`, in direction `n`
        def support_hex(side, n, rot=0.0):
            theta_n = np.arctan2(n[1], n[0])
            max_val = -np.inf
            for k in range(6):
                theta_k = rot + k * np.pi / 3
                val = np.cos(theta_n - theta_k)
                if val > max_val:
                    max_val = val
            return side * max_val
        
        # Inner hexagons: side=1, rotation=0
        h_inner = np.array([support_hex(1.0, n, 0.0) for n in normals])
        
        # Outer hexagon: side=s, rotation=outer_rot
        # h_outer(n_k) = s * support_hex(1.0, n_k, outer_rot)
        h_outer_factor = np.array([support_hex(1.0, n, outer_rot) for n in normals])
        
        # LP: minimize s
        # Variables: [cx, cy, s]
        # Constraints: for each i, k:
        #   dot(pos_i, n_k) - cx*n_k[0] - cy*n_k[1] + h_inner[k] <= s * h_outer_factor[k]
        #   => -cx*n_k[0] - cy*n_k[1] - s*h_outer_factor[k] <= -dot(pos_i, n_k) - h_inner[k]
        
        A_rows = []
        b_vals = []
        for i in range(N):
            for k in range(6):
                row = np.array([-normals[k, 0], -normals[k, 1], -h_outer_factor[k]])
                rhs = -np.dot(pos_arr[i], normals[k]) - h_inner[k]
                A_rows.append(row)
                b_vals.append(rhs)
        
        A = np.array(A_rows)
        b = np.array(b_vals)
        c = np.array([0.0, 0.0, 1.0])
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
            if res.success and res.x[2] < best_side:
                best_side = res.x[2]
                best_result = (pos_arr.tolist(), (res.x[0], res.x[1]), res.x[2], outer_angle)
        except Exception:
            pass
    
    if best_result is None:
        # Fallback
        d = np.sqrt(3)
        inner = [[k * d, 0.0, 0.0] for k in range(N)]
        cx = (N - 1) * d / 2.0
        cy = 0.0
        side = (N - 1) * d / 2.0 + 2.0
        best_result = (inner, (cx, cy), side, 0.0)
    
    inner, center, side, angle = best_result
    inner_formatted = [[p[0], p[1], 0.0] for p in inner]
    
    return inner_formatted, (center[0], center[1]), side, angle

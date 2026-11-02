import numpy as np
from itertools import combinations
from scipy.optimize import minimize
import time

N = 11

def construct():
    start_time = time.time()
    deadline = 140.0  # 20s safety margin
    
    # 1. Generate lattice points
    # Axial to Cartesian
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    def axial_to_cart(q, r):
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return np.array([x, y])
    
    # Core 7 points (hexagon of radius 1)
    core_axial = [(0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    core_cart = [axial_to_cart(q, r) for q, r in core_axial]
    
    # Shell 12 points (hexagon of radius 2, excluding core)
    shell_axial = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if q + r < -2 or q + r > 2: continue
            dist = (abs(q) + abs(r) + abs(q+r)) / 2
            if dist == 2:
                shell_axial.append((q, r))
    
    shell_cart = [axial_to_cart(q, r) for q, r in shell_axial]
    
    # 2. Search over combinations of 4 shell points
    best_S = float('inf')
    best_config = None
    best_theta = 0.0
    
    # Pre-compute vertices of a unit hexagon at origin (aligned)
    # Vertices at angles 0, 60, ...
    # Side 1, circumradius 1.
    hex_verts = np.array([[np.cos(np.pi/3 * i), np.sin(np.pi/3 * i)] for i in range(6)])
    
    def calc_min_S(centers, theta):
        # centers: list of (x,y)
        # theta: rotation of outer hexagon
        # Outer hexagon sides normals:
        # n_k = (cos(theta + k*pi/3), sin(theta + k*pi/3))
        # Constraint: (v - center_outer) . n_k <= sqrt(3)/2 * S
        # We assume outer center is (0,0) for this quick pass.
        max_proj = 0.0
        for c in centers:
            # Rotate hex_verts? No, inner hexagons are aligned (angle 0).
            # If we allow inner rotation, this is more complex. 
            # Assume inner rotation 0 for now.
            verts = c + hex_verts
            for k in range(3):
                n = np.array([np.cos(theta + k * np.pi / 3), np.sin(theta + k * np.pi / 3)])
                projs = np.dot(verts, n)
                max_proj = max(max_proj, np.max(np.abs(projs)))
        S = max_proj * 2.0 / np.sqrt(3)
        return S

    # Optimize theta for a given set of centers
    def optimize_theta(centers):
        # S(theta) is max of sinusoids.
        # Sample theta from 0 to pi/3
        thetas = np.linspace(0, np.pi/3, 181)
        Ss = [calc_min_S(centers, th) for th in thetas]
        idx = np.argmin(Ss)
        return Ss[idx], thetas[idx]
        
    # Iterate combinations
    # 12 choose 4 = 495
    for combo in combinations(range(12), 4):
        if time.time() > deadline:
            break
        selected_shell = [shell_cart[i] for i in combo]
        all_centers = core_cart + selected_shell
        
        S_val, th_val = optimize_theta(all_centers)
        
        if S_val < best_S:
            best_S = S_val
            best_config = all_centers
            best_theta = th_val
            
    # 3. Polish the best configuration
    # Variables: x_i, y_i for 11 centers (22 vars)
    # Plus S, theta (2 vars)
    # Total 24 vars.
    
    def flatten_vars(x, y, S, theta):
        return np.concatenate([x, y, [S, theta]])
        
    def unpack_vars(v):
        x = v[0:11]
        y = v[11:22]
        S = v[22]
        theta = v[23]
        return x, y, S, theta

    def objective(v):
        x, y, S, theta = unpack_vars(v)
        if S < 0: return 1e10
        
        # Check disjointness (distance >= sqrt(3))
        centers = np.column_stack((x, y))
        for i in range(11):
            for j in range(i+1, 11):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < np.sqrt(3) - 1e-5:
                    return 1e10 # Penalty for overlap
        
        # Check containment
        # Outer hexagon center (0,0)
        max_proj = 0.0
        for i in range(11):
            c = centers[i]
            verts = c + hex_verts # Inner rotation 0
            for k in range(3):
                n = np.array([np.cos(theta + k * np.pi / 3), np.sin(theta + k * np.pi / 3)])
                projs = np.dot(verts, n)
                max_proj = max(max_proj, np.max(np.abs(projs)))
        
        required_S = max_proj * 2.0 / np.sqrt(3)
        
        if required_S > S:
            return 1e10 + (required_S - S)
            
        return S

    # Initial guess
    init_x = np.array([c[0] for c in best_config])
    init_y = np.array([c[1] for c in best_config])
    init_v = flatten_vars(init_x, init_y, best_S, best_theta)
    
    # Constraints for SLSQP are hard to define with the max projection logic easily in standard form.
    # Instead, use a penalty method or just run minimize with a large penalty for violations.
    # The objective above returns 1e10 if violated.
    
    # Run polish
    if time.time() < deadline - 5:
        res = minimize(objective, init_v, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if res.success and res.fun < best_S:
            best_S = res.fun
            # Extract final positions
            x, y, S_final, theta_final = unpack_vars(res.x)
            # Construct return format
            inner = []
            for i in range(11):
                inner.append([x[i], y[i], 0.0]) # Angle 0
            return inner, (0.0, 0.0), S_final, np.degrees(theta_final)
            
    # Fallback to best lattice config
    inner = []
    for c in best_config:
        inner.append([c[0], c[1], 0.0])
    return inner, (0.0, 0.0), best_S, np.degrees(best_theta)

# EVOLVE-BLOCK-START
def construct():
    return _construct_impl()

def _construct_impl():
    # Import inside to avoid global overhead if not needed, but standard libs are fine
    import numpy as np
    from itertools import combinations
    from scipy.optimize import minimize
    import time

    start_time = time.time()
    deadline = 140.0  # 20s safety margin
    
    # 1. Generate lattice points
    def axial_to_cart(q, r):
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return np.array([x, y])
    
    core_axial = [(0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    core_cart = [axial_to_cart(q, r) for q, r in core_axial]
    
    shell_axial = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if q + r < -2 or q + r > 2: continue
            dist = (abs(q) + abs(r) + abs(q+r)) / 2
            if dist == 2:
                shell_axial.append((q, r))
    
    shell_cart = [axial_to_cart(q, r) for q, r in shell_axial]
    
    best_S = float('inf')
    best_config = None
    best_theta = 0.0
    
    hex_verts = np.array([[np.cos(np.pi/3 * i), np.sin(np.pi/3 * i)] for i in range(6)])
    
    def calc_min_S(centers, theta):
        max_proj = 0.0
        for c in centers:
            verts = c + hex_verts
            for k in range(3):
                n = np.array([np.cos(theta + k * np.pi / 3), np.sin(theta + k * np.pi / 3)])
                projs = np.dot(verts, n)
                max_proj = max(max_proj, np.max(np.abs(projs)))
        S = max_proj * 2.0 / np.sqrt(3)
        return S

    def optimize_theta(centers):
        thetas = np.linspace(0, np.pi/3, 181)
        Ss = [calc_min_S(centers, th) for th in thetas]
        idx = np.argmin(Ss)
        return Ss[idx], thetas[idx]
        
    for combo in combinations(range(12), 4):
        if time.time() > deadline:
            break
        selected_shell = [shell_cart[i] for i in combo]
        all_centers = core_cart + selected_shell
        
        S_val, th_val = optimize_theta(all_centers)
        
        if S_val < best_S:
            best_S = S_val
            best_config = all_centers
            best_theta = th_val
            
    # Polish
    def flatten_vars(x, y, S, theta):
        return np.concatenate([x, y, [S, theta]])
        
    def unpack_vars(v):
        x = v[0:11]
        y = v[11:22]
        S = v[22]
        theta = v[23]
        return x, y, S, theta

    def objective(v):
        x, y, S, theta = unpack_vars(v)
        if S < 0: return 1e10
        
        centers = np.column_stack((x, y))
        for i in range(11):
            for j in range(i+1, 11):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < np.sqrt(3) - 1e-5:
                    return 1e10
        
        max_proj = 0.0
        for i in range(11):
            c = centers[i]
            verts = c + hex_verts
            for k in range(3):
                n = np.array([np.cos(theta + k * np.pi / 3), np.sin(theta + k * np.pi / 3)])
                projs = np.dot(verts, n)
                max_proj = max(max_proj, np.max(np.abs(projs)))
        
        required_S = max_proj * 2.0 / np.sqrt(3)
        
        if required_S > S:
            return 1e10 + (required_S - S)
            
        return S

    if best_config is not None and time.time() < deadline - 5:
        init_x = np.array([c[0] for c in best_config])
        init_y = np.array([c[1] for c in best_config])
        init_v = flatten_vars(init_x, init_y, best_S, best_theta)
        
        res = minimize(objective, init_v, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if res.success and res.fun < best_S:
            best_S = res.fun
            x, y, S_final, theta_final = unpack_vars(res.x)
            inner = []
            for i in range(11):
                inner.append([x[i], y[i], 0.0])
            return inner, (0.0, 0.0), S_final, np.degrees(theta_final)
            
    if best_config is None:
        # Fallback: simple line
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    inner = []
    for c in best_config:
        inner.append([c[0], c[1], 0.0])
    return inner, (0.0, 0.0), best_S, np.degrees(best_theta)

# EVOLVE-BLOCK-END

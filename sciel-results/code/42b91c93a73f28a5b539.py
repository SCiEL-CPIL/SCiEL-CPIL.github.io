import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0  # Safety margin before 150s wall-clock kill

def construct():
    t0 = time.time()
    
    # Precompute unit hexagon vertices (angle 0)
    # Vertices at 0, 60, 120, 180, 240, 300 degrees
    angles = np.radians(np.arange(0, 360, 60))
    unit_hex_verts = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Precompute unit hexagon outward normals
    # Normals are at 30, 90, 150, 210, 270, 330 degrees
    normal_angles = np.radians(np.arange(30, 360, 60))
    unit_hex_normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))

    def get_all_vertices(positions, angles_rad):
        """
        positions: (N, 2)
        angles_rad: (N,)
        returns: (N, 6, 2)
        """
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        
        ux = unit_hex_verts[:, 0]
        uy = unit_hex_verts[:, 1]
        
        # (N, 6)
        rx = ux[None, :] * c[:, None] - uy[None, :] * s[:, None]
        ry = ux[None, :] * s[:, None] + uy[None, :] * c[:, None]
        
        verts_x = rx + positions[:, 0:1]
        verts_y = ry + positions[:, 1:2]
        
        all_verts = np.stack((verts_x, verts_y), axis=-1)
        return all_verts

    def get_all_normals(angles_rad):
        """
        angles_rad: (N,)
        returns: (N, 6, 2)
        """
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        
        ux = unit_hex_normals[:, 0]
        uy = unit_hex_normals[:, 1]
        
        rx = ux[None, :] * c[:, None] - uy[None, :] * s[:, None]
        ry = ux[None, :] * s[:, None] + uy[None, :] * c[:, None]
        
        all_normals = np.stack((rx, ry), axis=-1)
        return all_normals

    def check_disjointness(positions, angles_rad):
        """
        Returns True if all hexagons are disjoint.
        """
        # Get vertices and normals
        verts = get_all_vertices(positions, angles_rad) # (N, 6, 2)
        normals = get_all_normals(angles_rad) # (N, 6, 2)
        
        for i in range(N):
            # Bounding box quick check? 
            # For now, just SAT on all pairs
            for j in range(i + 1, N):
                # Check distance squared for early exit
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist_sq = dx*dx + dy*dy
                if dist_sq > 4.0: # dist > 2.0 (diameter)
                    continue
                
                # SAT Check
                # We need to check normals of i and j
                # Project vertices of i and j onto each normal
                v_i = verts[i]
                v_j = verts[j]
                
                disjoint = False
                # Check normals of i
                for k in range(6):
                    n = normals[i, k]
                    proj_i = np.dot(v_i, n)
                    proj_j = np.dot(v_j, n)
                    if np.max(proj_i) < np.min(proj_j) or np.max(proj_j) < np.min(proj_i):
                        disjoint = True
                        break
                
                if disjoint:
                    continue
                    
                # Check normals of j
                for k in range(6):
                    n = normals[j, k]
                    proj_i = np.dot(v_i, n)
                    proj_j = np.dot(v_j, n)
                    if np.max(proj_i) < np.min(proj_j) or np.max(proj_j) < np.min(proj_i):
                        disjoint = True
                        break
                        
                if not disjoint:
                    return False
                    
        return True

    def compute_objective(params):
        """
        params: (3*N + 3,)
        [x0, y0, a0, ..., x11, y11, a11, cx, cy, phi_outer]
        Returns side length or penalty.
        """
        positions = np.zeros((N, 2))
        angles_deg = np.zeros(N)
        for i in range(N):
            positions[i, 0] = params[3*i]
            positions[i, 1] = params[3*i+1]
            angles_deg[i] = params[3*i+2]
            
        cx = params[3*N]
        cy = params[3*N+1]
        phi_outer_deg = params[3*N+2]
        
        angles_rad = np.radians(angles_deg)
        
        # Check disjointness
        if not check_disjointness(positions, angles_rad):
            return 1e6
            
        # Compute minimal outer side
        all_verts = get_all_vertices(positions, angles_rad).reshape(-1, 2)
        center = np.array([cx, cy])
        offsets = all_verts - center
        
        phi_outer_rad = np.radians(phi_outer_deg)
        # Outer normals
        c_o = np.cos(phi_outer_rad)
        s_o = np.sin(phi_outer_rad)
        
        ux = unit_hex_normals[:, 0]
        uy = unit_hex_normals[:, 1]
        
        rx = ux * c_o - uy * s_o
        ry = ux * s_o + uy * c_o
        outer_normals = np.column_stack((rx, ry)) # (6, 2)
        
        max_extent = 0.0
        for k in range(6):
            n = outer_normals[k]
            proj = np.dot(offsets, n)
            # We need the max projection in the direction of the outward normal
            # Since the hexagon is centered at C, and normals are outward,
            # the constraint is max(offsets . n) <= r
            # Wait, if the hexagon is rotated, the normals rotate.
            # The condition for a point p to be inside hexagon centered at C with inradius r and outward normals n_k is:
            # (p - C) . n_k <= r for all k.
            # So r must be >= max_k (max_p (p-C) . n_k)
            max_proj = np.max(proj)
            if max_proj > max_extent:
                max_extent = max_proj
                
        # r = max_extent
        # side = 2 * r / sqrt(3)
        side = (2.0 / math.sqrt(3)) * max_extent
        return side

    # Initial Configuration: Honeycomb cluster
    # 1 center, 6 ring 1, 5 ring 2 (to make 12)
    # Actually, a 3x4 grid is also good. Let's try a compact cluster.
    # Honeycomb:
    #   1
    #  2 3
    # 4 5 6
    #  7 8
    #   9
    # That's 9. Need 12.
    # Let's use a 3x4 grid of hexagons.
    # Spacing: dx = 3.0 (for flat-top? No, pointy-top or flat-top?)
    # My vertices are at 0, 60... which is flat-top?
    # Vertex at 0 is (1,0). Vertex at 60 is (0.5, 0.866).
    # This is a "pointy-top" hexagon if we consider y-axis up?
    # Usually "flat-top" has flat edges on top/bottom.
    # Vertex at 0 (1,0) means a vertex points right.
    # Let's just use a grid.
    cols = 4
    rows = 3
    # For a hexagon with vertex at (1,0), the width is 2, height is sqrt(3).
    # To tile without overlap:
    # Horizontal spacing: 1.5 * side? No.
    # Distance between centers in a row: 3.0?
    # If I place two hexagons at (0,0) and (3,0) with angle 0.
    # Vertices of first: (1,0), (0.5, 0.866), (-0.5, 0.866), (-1,0), ...
    # Vertices of second: (4,0), (3.5, 0.866), ...
    # Gap between x=1 and x=3 is 2. They are disjoint.
    # Can we pack tighter?
    # If I rotate the second one by 60 degrees?
    # Let's stick to the 3x4 grid with spacing 3.0 x sqrt(3).
    
    initial_positions = np.zeros((N, 2))
    initial_angles = np.zeros(N)
    
    idx = 0
    x0 = -(cols - 1) / 2.0 * 3.0
    y0 = -(rows - 1) / 2.0 * math.sqrt(3)
    
    for r in range(rows):
        for c in range(cols):
            initial_positions[idx, 0] = x0 + c * 3.0
            initial_positions[idx, 1] = y0 + r * math.sqrt(3)
            initial_angles[idx] = 0.0
            idx += 1
            
    # Initial outer params
    init_cx = 0.0
    init_cy = 0.0
    init_phi = 0.0
    
    best_params = np.zeros(3*N + 3)
    for i in range(N):
        best_params[3*i] = initial_positions[i, 0]
        best_params[3*i+1] = initial_positions[i, 1]
        best_params[3*i+2] = initial_angles[i]
    best_params[3*N] = init_cx
    best_params[3*N+1] = init_cy
    best_params[3*N+2] = init_phi
    
    best_score = compute_objective(best_params)
    current_params = best_params.copy()
    current_score = best_score
    
    T = 1.0
    T_min = 0.001
    cooling = 0.9995
    step_size = 0.5
    
    step_count = 0
    max_steps = 50000
    
    # SA Loop
    while time.time() - t0 < DEADLINE * 0.8 and step_count < max_steps:
        step_count += 1
        
        new_params = current_params.copy()
        idx = random.randint(0, 3*N + 2)
        
        # Scale step size based on parameter type
        # Angles are in degrees, positions in units.
        if idx % 3 == 2 and idx < 3*N: # Inner angle
            da = random.gauss(0, 10.0)
            new_params[idx] += da
        elif idx == 3*N + 2: # Outer angle
            da = random.gauss(0, 10.0)
            new_params[idx] += da
        else: # Position or center
            dx = random.gauss(0, step_size)
            new_params[idx] += dx
            
        new_score = compute_objective(new_params)
        
        if new_score < current_score:
            current_params = new_params
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_params = new_params.copy()
        else:
            delta = new_score - current_score
            if delta < 0: delta = 0
            if random.random() < math.exp(-delta / T):
                current_params = new_params
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_params = new_params.copy()
                    
        T *= cooling
        if T < T_min: T = T_min

    # Polish Step
    if time.time() - t0 < DEADLINE * 0.95:
        try:
            res = minimize(compute_objective, best_params, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-7})
            
            if res.fun < best_score:
                best_params = res.x
                best_score = res.fun
        except Exception:
            pass

    # Final Result
    inner = []
    for i in range(N):
        inner.append([float(best_params[3*i]), float(best_params[3*i+1]), float(best_params[3*i+2])])
        
    center = (float(best_params[3*N]), float(best_params[3*N+1]))
    angle = float(best_params[3*N+2])
    side = float(best_score)
    
    # Final validation
    # If the score is a penalty, we might have an invalid config.
    # But SA should keep it valid.
    # Just in case, if side is huge, try to recover?
    # The prompt says "an invalid result... scores 0".
    # So we must ensure the returned config is valid.
    # Let's do a final check.
    positions = np.array([inner[i][:2] for i in range(N)])
    angles_deg = np.array([inner[i][2] for i in range(N)])
    angles_rad = np.radians(angles_deg)
    if not check_disjointness(positions, angles_rad):
        # Fallback to a known valid config (the grid)
        # This shouldn't happen if SA is correct, but safe.
        inner = []
        for i in range(N):
            inner.append([float(initial_positions[i, 0]), float(initial_positions[i, 1]), float(initial_angles[i])])
        center = (0.0, 0.0)
        angle = 0.0
        side = compute_objective(np.array([
            initial_positions[0,0], initial_positions[0,1], initial_angles[0],
            initial_positions[1,0], initial_positions[1,1], initial_angles[1],
            initial_positions[2,0], initial_positions[2,1], initial_angles[2],
            initial_positions[3,0], initial_positions[3,1], initial_angles[3],
            initial_positions[4,0], initial_positions[4,1], initial_angles[4],
            initial_positions[5,0], initial_positions[5,1], initial_angles[5],
            initial_positions[6,0], initial_positions[6,1], initial_angles[6],
            initial_positions[7,0], initial_positions[7,1], initial_angles[7],
            initial_positions[8,0], initial_positions[8,1], initial_angles[8],
            initial_positions[9,0], initial_positions[9,1], initial_angles[9],
            initial_positions[10,0], initial_positions[10,1], initial_angles[10],
            initial_positions[11,0], initial_positions[11,1], initial_angles[11],
            0.0, 0.0, 0.0
        ]))

    return inner, center, side, angle

# EVOLVE-BLOCK-END

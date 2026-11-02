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
        Uses SAT (Separating Axis Theorem).
        """
        verts = get_all_vertices(positions, angles_rad) # (N, 6, 2)
        normals = get_all_normals(angles_rad) # (N, 6, 2)
        
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist_sq = dx*dx + dy*dy
                if dist_sq > 4.0: # dist > 2.0 (diameter)
                    continue
                
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
        params: (3*N + 1,)
        [x0, y0, a0, ..., x11, y11, a11, phi_outer]
        Returns side length or penalty.
        """
        positions = np.zeros((N, 2))
        angles_deg = np.zeros(N)
        for i in range(N):
            positions[i, 0] = params[3*i]
            positions[i, 1] = params[3*i+1]
            angles_deg[i] = params[3*i+2]
            
        phi_outer_deg = params[3*N]
        
        angles_rad = np.radians(angles_deg)
        
        # Check disjointness
        if not check_disjointness(positions, angles_rad):
            return 1e6
            
        # Compute minimal outer side
        all_verts = get_all_vertices(positions, angles_rad).reshape(-1, 2)
        # Center is the centroid of the vertices for optimal containment
        center = np.mean(all_verts, axis=0)
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
            max_proj = np.max(proj)
            if max_proj > max_extent:
                max_extent = max_proj
                
        # r = max_extent
        # side = 2 * r / sqrt(3)
        side = (2.0 / math.sqrt(3)) * max_extent
        return side

    # Initial Configuration: Honeycomb-like cluster
    # 12 hexagons.
    # A 3x4 grid is a good start, but let's try a more compact arrangement.
    # Hexagonal tiling spacing:
    # dx = 1.5 (for pointy top? No, vertex at 0 is flat top? No, vertex at 0 is (1,0).)
    # Let's stick to the grid for robustness, but tighter.
    # For vertex at (1,0), width is 2, height is sqrt(3).
    # Horizontal spacing to touch: 3.0? No.
    # If I put two at (0,0) and (3,0), they don't touch.
    # If I put two at (0,0) and (2.5, 0.866)?
    # Let's use the standard hex grid coordinates.
    # Axial coordinates to cartesian:
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    # This is for pointy-top?
    # My vertices are at 0, 60... so (1,0) is a vertex. This is "pointy" in x?
    # Let's just use a 3x4 grid with spacing 2.0 x sqrt(3) * 0.5?
    # Actually, let's use the grid from the previous version but maybe optimize it.
    # The previous grid was 3x4 with spacing 3.0 x sqrt(3).
    # Let's try a 2x6 grid? Or 4x3?
    # Let's try to generate a honeycomb patch.
    
    initial_positions = np.zeros((N, 2))
    initial_angles = np.zeros(N)
    
    # Generate 12 hexagons in a honeycomb pattern
    # Center at (0,0)
    # Ring 0: 1 hex
    # Ring 1: 6 hexes
    # Ring 2: 5 hexes (to make 12)
    
    hex_centers = []
    # Center
    hex_centers.append((0.0, 0.0))
    
    # Ring 1
    # Neighbors of (0,0) in a hex grid with spacing sqrt(3) (distance between centers)
    # Wait, distance between centers of touching hexagons is 2 * inradius = 2 * sqrt(3)/2 = sqrt(3).
    # My hexagons have side 1, inradius sqrt(3)/2.
    # So center distance is sqrt(3).
    # Directions: 0, 60, 120, 180, 240, 300?
    # If vertices are at 0, 60..., the flat sides are at 30, 90...
    # So the "flat" neighbors are in directions 30, 90, 150...
    # Distance sqrt(3).
    for k in range(6):
        ang = math.radians(30 + k * 60)
        x = math.cos(ang) * math.sqrt(3)
        y = math.sin(ang) * math.sqrt(3)
        hex_centers.append((x, y))
        
    # Ring 2
    # We need 5 more.
    # Let's just pick 5 from the ring 2 positions.
    # Ring 2 positions are at distance 2*sqrt(3) or sqrt(3)*sqrt(3)?
    # Actually, let's just use a simple grid for initialization to be safe.
    # 3 rows, 4 cols.
    # Spacing x: 3.0, y: sqrt(3).
    # This is valid.
    
    cols = 4
    rows = 3
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
    init_phi = 0.0
    
    best_params = np.zeros(3*N + 1)
    for i in range(N):
        best_params[3*i] = initial_positions[i, 0]
        best_params[3*i+1] = initial_positions[i, 1]
        best_params[3*i+2] = initial_angles[i]
    best_params[3*N] = init_phi
    
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
        idx = random.randint(0, 3*N)
        
        # Scale step size based on parameter type
        if idx % 3 == 2 and idx < 3*N: # Inner angle
            da = random.gauss(0, 10.0)
            new_params[idx] += da
        elif idx == 3*N: # Outer angle
            da = random.gauss(0, 10.0)
            new_params[idx] += da
        else: # Position
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
        
    # Compute center and side for the final configuration
    positions = np.array([inner[i][:2] for i in range(N)])
    angles_deg = np.array([inner[i][2] for i in range(N)])
    angles_rad = np.radians(angles_deg)
    
    # Final validation
    if not check_disjointness(positions, angles_rad):
        # Fallback to a known valid config (the grid)
        inner = []
        for i in range(N):
            inner.append([float(initial_positions[i, 0]), float(initial_positions[i, 1]), float(initial_angles[i])])
        positions = initial_positions
        angles_rad = np.zeros(N)
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
            0.0
        ]))
    else:
        side = best_score

    # Compute center
    all_verts = get_all_vertices(positions, angles_rad).reshape(-1, 2)
    center = tuple(np.mean(all_verts, axis=0))
    
    angle = float(best_params[3*N])
    side = float(side)

    return inner, center, side, angle

# EVOLVE-BLOCK-END

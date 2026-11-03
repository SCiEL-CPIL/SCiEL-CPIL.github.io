"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Helper: get vertices of a regular hexagon with side length 1
    def hex_vertices(cx, cy, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            x = cx + math.cos(a)
            y = cy + math.sin(a)
            verts.append((x, y))
        return np.array(verts)
    
    # Helper: check if two hexagons overlap using SAT (Separating Axis Theorem)
    # Returns True if they overlap
    def hexagons_overlap(v1, v2):
        axes = []
        for i in range(6):
            edge = v1[i] - v1[(i+1) % 6]
            normal = np.array([-edge[1], edge[0]])
            axes.append(normal)
            edge2 = v2[i] - v2[(i+1) % 6]
            normal2 = np.array([-edge2[1], edge2[0]])
            axes.append(normal2)
        
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            if max1 < min2 or max2 < min1:
                return False
        return True

    # Helper: compute minimum outer hexagon side length needed to contain all inner hexagons
    # Uses Linear Programming to find the optimal center and size for a fixed outer angle
    def compute_min_side(positions, angles, outer_angle_deg):
        # Get all vertices of all inner hexagons
        all_verts = []
        for i in range(N):
            cx, cy = positions[i]
            a = angles[i]
            verts = hex_vertices(cx, cy, a)
            all_verts.append(verts)
        
        if not all_verts:
            return 0.0
            
        all_verts = np.vstack(all_verts)
        
        theta = math.radians(outer_angle_deg)
        # Normals for the 3 unique directions (0, 60, 120 deg)
        # The outer hexagon has 6 sides, but constraints come in pairs.
        # We need to satisfy constraints for 3 axes.
        normals = []
        for k in range(3):
            a = theta + k * math.pi / 3
            normals.append(np.array([math.cos(a), math.sin(a)]))
        
        # Calculate min and max projection for each normal
        M = np.zeros(3)
        P = np.zeros(3)
        for k in range(3):
            proj = np.dot(all_verts, normals[k])
            M[k] = proj.min()
            P[k] = proj.max()
        
        # We want to minimize a (apothem)
        # Variables: Cx, Cy, a
        # Constraints:
        # n_k . C - a <= M_k
        # n_k . C + a >= P_k  =>  -n_k . C - a <= -P_k
        
        # A matrix for linprog: A_ub @ x <= b_ub
        # x = [Cx, Cy, a]
        A_ub = np.zeros((6, 3))
        b_ub = np.zeros(6)
        
        for k in range(3):
            nx, ny = normals[k]
            # n_k . C - a <= M_k
            A_ub[2*k, 0] = nx
            A_ub[2*k, 1] = ny
            A_ub[2*k, 2] = -1
            b_ub[2*k] = M[k]
            
            # -n_k . C - a <= -P_k
            A_ub[2*k+1, 0] = -nx
            A_ub[2*k+1, 1] = -ny
            A_ub[2*k+1, 2] = -1
            b_ub[2*k+1] = -P[k]
        
        # Objective: minimize a
        c = np.array([0, 0, 1])
        
        # Bounds: a >= 0, Cx, Cy free
        bounds = [(-np.inf, np.inf), (-np.inf, np.inf), (0, np.inf)]
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                a_min = res.x[2]
                return 2 * a_min / math.sqrt(3)
            else:
                return 1e10
        except Exception:
            return 1e10

    # Objective function
    def objective(params):
        if time.time() > deadline:
            return 1e10
        
        positions = []
        angles = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i + 1]
            a = params[3*i + 2]
            positions.append((x, y))
            angles.append(a)
        
        outer_angle = params[3*N]
        
        # Check for overlaps
        verts = [hex_vertices(p[0], p[1], a) for p, a in zip(positions, angles)]
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if hexagons_overlap(verts[i], verts[j]):
                    overlap_penalty += 1.0
        
        if overlap_penalty > 0:
            return 100.0 + overlap_penalty
        
        # Compute required outer side length
        side = compute_min_side(positions, angles, outer_angle)
        return side

    # Initial guess: Honeycomb lattice
    init_params = []
    # 4 rows of 3
    # dy = 1.5
    # dx = sqrt(3)
    # Offset odd rows by sqrt(3)/2
    dx = math.sqrt(3)
    dy = 1.5
    
    for row in range(4):
        for col in range(3):
            x = col * dx + (row % 2) * (dx / 2)
            y = row * dy
            a = 0.0
            init_params.extend([x, y, a])
    
    # Outer angle initial
    init_params.append(0.0)
    
    init_params = np.array(init_params)
    
    best_result = None
    best_side = 1e10
    
    # Run multiple restarts
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            params0 = init_params.copy()
        else:
            # Perturb
            params0 = init_params + np.random.randn(len(init_params)) * 0.5
        
        try:
            result = minimize(objective, params0, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            
            if result.fun < best_side and result.fun < 100.0:
                best_side = result.fun
                best_result = result.x.copy()
                
        except Exception:
            continue
        
        if time.time() > deadline - 20:
            break
    
    if best_result is None:
        best_result = init_params
    
    # Extract final positions
    positions = []
    angles = []
    for i in range(N):
        x = best_result[3*i]
        y = best_result[3*i + 1]
        a = best_result[3*i + 2]
        positions.append((x, y))
        angles.append(a)
    
    outer_angle = best_result[3*N]
    
    # Build inner list
    inner = [[positions[i][0], positions[i][1], angles[i]] for i in range(N)]
    
    # Compute the actual side length
    side = compute_min_side(positions, angles, outer_angle)
    
    # Center of the outer hexagon is not returned in the interface?
    # "outer hexagon center (x, y), side length and rotation"
    # We need to return the center.
    # The LP finds the optimal center. We need to re-solve or store it.
    # Let's re-solve for the center to be accurate.
    
    # Re-calculate center for the final configuration
    all_verts = []
    for i in range(N):
        cx, cy = positions[i]
        a = angles[i]
        verts = hex_vertices(cx, cy, a)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    theta = math.radians(outer_angle)
    normals = []
    for k in range(3):
        a = theta + k * math.pi / 3
        normals.append(np.array([math.cos(a), math.sin(a)]))
        
    M = np.zeros(3)
    P = np.zeros(3)
    for k in range(3):
        proj = np.dot(all_verts, normals[k])
        M[k] = proj.min()
        P[k] = proj.max()
        
    A_ub = np.zeros((6, 3))
    b_ub = np.zeros(6)
    for k in range(3):
        nx, ny = normals[k]
        A_ub[2*k, 0] = nx
        A_ub[2*k, 1] = ny
        A_ub[2*k, 2] = -1
        b_ub[2*k] = M[k]
        
        A_ub[2*k+1, 0] = -nx
        A_ub[2*k+1, 1] = -ny
        A_ub[2*k+1, 2] = -1
        b_ub[2*k+1] = -P[k]
        
    c = np.array([0, 0, 1])
    bounds = [(-np.inf, np.inf), (-np.inf, np.inf), (0, np.inf)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            outer_x = res.x[0]
            outer_y = res.x[1]
        else:
            # Fallback to centroid
            outer_x = sum(p[0] for p in positions) / N
            outer_y = sum(p[1] for p in positions) / N
    except Exception:
        outer_x = sum(p[0] for p in positions) / N
        outer_y = sum(p[1] for p in positions) / N

    return inner, (outer_x, outer_y), side, outer_angle
# EVOLVE-BLOCK-END

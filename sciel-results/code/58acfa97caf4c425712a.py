"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from scipy.optimize import minimize, differential_evolution

N = 12
DEADLINE = 145.0

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    def hexagon_vertices(cx, cy, side, angle_rad):
        """Vertices of a regular hexagon (flat-top when angle=0)."""
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def min_vertex_edge_distance(hex1_verts, hex2_verts):
        """Minimum distance from vertices of hex1 to edges of hex2 and vice versa.
        Negative if overlapping."""
        min_dist = np.inf
        # Check vertices of hex1 against edges of hex2
        for v in hex1_verts:
            for i in range(6):
                e1 = hex2_verts[i]
                e2 = hex2_verts[(i + 1) % 6]
                d_vec = e2 - e1
                L2 = np.dot(d_vec, d_vec)
                if L2 < 1e-12:
                    d = np.linalg.norm(v - e1)
                else:
                    t = np.dot(v - e1, d_vec) / L2
                    t = max(0.0, min(1.0, t))
                    closest = e1 + t * d_vec
                    d = np.linalg.norm(v - closest)
                if d < min_dist:
                    min_dist = d
        # Check vertices of hex2 against edges of hex1
        for v in hex2_verts:
            for i in range(6):
                e1 = hex1_verts[i]
                e2 = hex1_verts[(i + 1) % 6]
                d_vec = e2 - e1
                L2 = np.dot(d_vec, d_vec)
                if L2 < 1e-12:
                    d = np.linalg.norm(v - e1)
                else:
                    t = np.dot(v - e1, d_vec) / L2
                    t = max(0.0, min(1.0, t))
                    closest = e1 + t * d_vec
                    d = np.linalg.norm(v - closest)
                if d < min_dist:
                    min_dist = d
        return min_dist

    def polygons_disjoint_sat(poly1, poly2, tol=1e-9):
        """Check if two convex polygons are disjoint using SAT."""
        for poly in [poly1, poly2]:
            for i in range(len(poly)):
                v1 = poly[i]
                v2 = poly[(i + 1) % len(poly)]
                edge = v2 - v1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-10:
                    continue
                normal = normal / norm
                
                proj1 = np.dot(poly1, normal)
                proj2 = np.dot(poly2, normal)
                
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                if max1 < min2 - tol or max2 < min1 - tol:
                    return True
        return False

    def compute_outer_side(inner_hexes, center=(0.0, 0.0), angle_rad=0.0):
        """
        Compute minimum outer hexagon side length to contain all inner hexagons.
        The outer hexagon is centered at `center` with rotation `angle_rad`.
        """
        max_required = 0.0
        # Normals for a flat-top hexagon centered at origin
        # Edges are perpendicular to angles 0, 60, 120, 180, 240, 300
        normals = np.array([
            [np.cos(angle_rad + np.radians(a)), np.sin(angle_rad + np.radians(a))] 
            for a in [0, 60, 120, 180, 240, 300]
        ])
        
        for hex_verts in inner_hexes:
            for v in hex_verts:
                offset = v - np.array(center)
                # Project vertex onto normals
                dists = np.dot(normals, offset)
                # The required apothem is the max projection
                # Side S = 2 * apothem / sqrt(3)
                max_proj = np.max(dists)
                required_S = max_proj * 2.0 / np.sqrt(3.0)
                if required_S > max_required:
                    max_required = required_S
        return max_required

    def objective_rigid(params):
        """
        Objective for rigid lattice optimization.
        params: [X, Y, theta]
        """
        if time.time() > deadline:
            return 1e6
            
        X, Y, theta = params
        
        # Generate lattice positions
        # 3x4 parallelogram of hexagons
        # Basis vectors for centers: u = (2, 0), v = (1, sqrt(3))
        # Positions in local frame:
        local_pos = []
        for i in range(4):
            for j in range(3):
                lx = 2*i + j
                ly = np.sqrt(3)*j
                local_pos.append([lx, ly])
        local_pos = np.array(local_pos)
        
        # Rotate and translate
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        
        global_pos = (local_pos @ R.T) + np.array([X, Y])
        
        inner_hexes = []
        for i in range(N):
            verts = hexagon_vertices(global_pos[i, 0], global_pos[i, 1], 1.0, theta)
            inner_hexes.append(verts)
            
        side = compute_outer_side(inner_hexes)
        
        # Check validity (optional in rigid mode if we trust lattice, but good for safety)
        # In a perfect lattice, distance is exactly 2, so they touch.
        # SAT with tol=1e-9 should pass.
        # We don't add penalty here to keep it fast, but we verify at the end.
        
        return side

    def objective_full(params):
        """
        Objective for full local search.
        params: x1, y1, a1, ..., x12, y12, a12
        """
        if time.time() > deadline:
            return 1e6
        
        inner_x = params[0:12]
        inner_y = params[12:24]
        inner_a = params[24:36]
        
        inner_hexes = []
        for i in range(N):
            verts = hexagon_vertices(inner_x[i], inner_y[i], 1.0, inner_a[i])
            inner_hexes.append(verts)
            
        side = compute_outer_side(inner_hexes)
        
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dist = min_vertex_edge_distance(inner_hexes[i], inner_hexes[j])
                if dist < 0:
                    penalty += 10.0 * dist * dist
        
        return side + penalty

    # Phase 1: Rigid Lattice Optimization
    # Initial guess: Centered, theta=0
    # The lattice is not centered at 0,0.
    # Local coords: x in [0, 2*3+2=8], y in [0, sqrt(3)*2]
    # Center of mass approx: x=4, y=sqrt(3)
    init_X = -4.0
    init_Y = -np.sqrt(3)
    init_theta = 0.0
    
    # Optimize theta, X, Y
    # X and Y can be adjusted to center the bounding box, but the objective
    # compute_outer_side uses center (0,0) by default.
    # Actually, compute_outer_side takes center as arg. 
    # In objective_rigid, I passed center (0,0) implicitly? 
    # No, compute_outer_side has default center=(0,0).
    # So we are minimizing the hexagon centered at (0,0) that contains the cluster.
    # This implicitly optimizes X and Y as well.
    
    bounds_rigid = [(-10, 10), (-10, 10), (-np.pi/2, np.pi/2)]
    
    best_rigid_params = None
    best_rigid_obj = np.inf
    
    try:
        de_result = differential_evolution(
            objective_rigid, bounds_rigid,
            seed=42, maxiter=50, popsize=15,
            tol=1e-6, mut=(0.5, 1.0), recomb=0.8,
            polish=False
        )
        if de_result.fun < best_rigid_obj:
            best_rigid_obj = de_result.fun
            best_rigid_params = de_result.x
    except Exception:
        pass
        
    # Refine rigid with local search
    if best_rigid_params is not None and time.time() < deadline - 20:
        try:
            res = minimize(objective_rigid, best_rigid_params, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_rigid_obj:
                best_rigid_obj = res.fun
                best_rigid_params = res.x
        except Exception:
            pass

    # Extract rigid solution to use as initial guess for full search
    X, Y, theta = best_rigid_params
    local_pos = []
    for i in range(4):
        for j in range(3):
            lx = 2*i + j
            ly = np.sqrt(3)*j
            local_pos.append([lx, ly])
    local_pos = np.array(local_pos)
    
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    global_pos = (local_pos @ R.T) + np.array([X, Y])
    
    init_full = np.concatenate([
        global_pos[:, 0], 
        global_pos[:, 1], 
        np.full(N, theta)
    ])
    
    best_full_params = init_full
    best_full_obj = objective_full(best_full_params)
    
    # Phase 2: Full Local Search
    if time.time() < deadline - 10:
        try:
            res = minimize(objective_full, best_full_params, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
            if res.fun < best_full_obj:
                best_full_obj = res.fun
                best_full_params = res.x
        except Exception:
            pass
            
    # Try a few random restarts if time permits
    if time.time() < deadline - 5:
        for _ in range(3):
            if time.time() > deadline - 2:
                break
            perturbed = best_full_params + np.random.normal(0, 0.1, len(best_full_params))
            try:
                res = minimize(objective_full, perturbed, method='Nelder-Mead',
                               options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})
                if res.fun < best_full_obj:
                    best_full_obj = res.fun
                    best_full_params = res.x
            except Exception:
                pass

    # Extract final solution
    params = best_full_params
    inner_x = params[0:12]
    inner_y = params[12:24]
    inner_a = params[24:36]
    
    inner = [[inner_x[i], inner_y[i], np.degrees(inner_a[i])] for i in range(N)]
    
    # Verify validity
    inner_hexes = []
    for i in range(N):
        verts = hexagon_vertices(inner_x[i], inner_y[i], 1.0, inner_a[i])
        inner_hexes.append(verts)
    
    valid = True
    for i in range(N):
        for j in range(i + 1, N):
            if not polygons_disjoint_sat(inner_hexes[i], inner_hexes[j], 1e-7):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to rigid solution if full search broke validity
        # Re-construct rigid solution
        X, Y, theta = best_rigid_params
        local_pos = []
        for i in range(4):
            for j in range(3):
                lx = 2*i + j
                ly = np.sqrt(3)*j
                local_pos.append([lx, ly])
        local_pos = np.array(local_pos)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        global_pos = (local_pos @ R.T) + np.array([X, Y])
        
        inner_hexes = []
        for i in range(N):
            verts = hexagon_vertices(global_pos[i, 0], global_pos[i, 1], 1.0, theta)
            inner_hexes.append(verts)
        
        # Check rigid validity
        rigid_valid = True
        for i in range(N):
            for j in range(i + 1, N):
                if not polygons_disjoint_sat(inner_hexes[i], inner_hexes[j], 1e-7):
                    rigid_valid = False
                    break
            if not rigid_valid:
                break
                
        if rigid_valid:
            inner = [[global_pos[i, 0], global_pos[i, 1], np.degrees(theta)] for i in range(N)]
            exact_side = compute_outer_side(inner_hexes)
            return inner, (0.0, 0.0), exact_side, 0.0
        else:
            # Ultimate fallback: widely spaced
            inner = [[3.5 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            return inner, (0.0, 0.0), 35.0, 0.0

    # Recompute exact side length without penalty
    exact_side = compute_outer_side(inner_hexes)
    
    # Return: inner, (center_x, center_y), side, angle_degrees
    return inner, (0.0, 0.0), exact_side, 0.0

# EVOLVE-BLOCK-END

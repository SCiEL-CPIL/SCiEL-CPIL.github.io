"""Packing 11 unit hexagons into a minimal regular hexagon with decoupled container optimization."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 11
SCALE = 1.002  # Tiny scale to ensure gap between touching hexagons

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    def hex_vertices(cx, cy, angle, s=1.0):
        """Return the 6 vertices of a regular hexagon."""
        angles = angle + np.arange(6) * np.pi / 3
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def min_enclosing_hex_side(points, cx, cy, angle):
        """
        Given a set of 2D points and a container hexagon center (cx, cy) with rotation angle,
        compute the minimum side length S of the container hexagon to contain all points.
        """
        apothem = math.sqrt(3) / 2.0
        max_extent = 0.0
        # The 6 outward normals of a regular hexagon with rotation `angle`
        for k in range(6):
            theta = angle + k * math.pi / 3 + math.pi / 6
            nx = math.cos(theta)
            ny = math.sin(theta)
            # Project all points onto this normal
            projs = points[:, 0] * nx + points[:, 1] * ny
            center_proj = cx * nx + cy * ny
            max_rel = np.max(projs) - center_proj
            min_rel = np.min(projs) - center_proj
            extent = max(abs(max_rel), abs(min_rel))
            if extent > max_extent:
                max_extent = extent
        return max_extent / apothem

    def sat_overlap_penalty(v1, v2):
        """
        Calculate a smooth penalty for overlap between two convex polygons.
        Returns 0 if disjoint, > 0 if overlapping.
        Uses the 12 edge normals from both polygons.
        """
        penalty = 0.0
        # Get all edge normals from both polygons
        # Poly 1 normals
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-9:
                continue
            normal /= norm_len
            
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for separation
            if max1 < min2 or max2 < min1:
                return 0.0 # Disjoint
            
            # Calculate overlap depth on this axis
            # The amount by which the intervals [min1, max1] and [min2, max2] overlap
            overlap = min(max1, max2) - max(min1, min2)
            if overlap > 0:
                penalty += overlap ** 2
        
        # Poly 2 normals
        for i in range(6):
            p1 = v2[i]
            p2 = v2[(i + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-9:
                continue
            normal /= norm_len
            
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for separation
            if max1 < min2 or max2 < min1:
                return 0.0 # Disjoint
            
            # Calculate overlap depth on this axis
            overlap = min(max1, max2) - max(min1, min2)
            if overlap > 0:
                penalty += overlap ** 2
                
        return penalty

    def compute_S(vertices_list, ocx, ocy, oangle):
        """
        Compute the minimum side length S of the outer hexagon 
        centered at (ocx, ocy) with rotation oangle to contain all vertices.
        """
        all_points = np.vstack(vertices_list)
        return min_enclosing_hex_side(all_points, ocx, ocy, oangle)

    def objective(params):
        """
        params: [cx1, cy1, a1, ..., cxN, cyN, aN, ocx, ocy, oangle]
        """
        n_inner = N
        inner_params = params[:3 * n_inner]
        cont_params = params[3 * n_inner:]
        ocx, ocy, oangle = cont_params
        
        vertices_list = []
        for i in range(n_inner):
            cx = inner_params[3 * i]
            cy = inner_params[3 * i + 1]
            angle = inner_params[3 * i + 2]
            verts = hex_vertices(cx, cy, angle, SCALE)
            vertices_list.append(verts)
        
        S = compute_S(vertices_list, ocx, ocy, oangle)
        
        # Overlap penalty
        penalty = 0.0
        for i in range(n_inner):
            for j in range(i + 1, n_inner):
                # Quick rejection
                c1 = np.array([inner_params[3*i], inner_params[3*i+1]])
                c2 = np.array([inner_params[3*j], inner_params[3*j+1]])
                if np.linalg.norm(c1 - c2) > 2.5:
                    continue
                p = sat_overlap_penalty(vertices_list[i], vertices_list[j])
                penalty += p
        
        return S + 100.0 * penalty

    # Initialize with hexagonal lattice
    np.random.seed(42)
    params = np.zeros(3 * N + 3)
    
    # Inner hexagons
    # Center
    params[0:3] = [0, 0, 0]
    
    # Ring 1
    r1 = np.sqrt(3)
    for k in range(6):
        theta = k * np.pi / 3
        cx = r1 * np.cos(theta)
        cy = r1 * np.sin(theta)
        idx = 1 + k
        params[3 * idx] = cx
        params[3 * idx + 1] = cy
        params[3 * idx + 2] = 0
        
    # Ring 2 (4 more)
    # Place them in the gaps at 30, 90, 150, 210 degrees
    r2 = 2 * np.sqrt(3)
    angles2 = [np.pi/6, np.pi/2, 5*np.pi/6, 7*np.pi/6]
    for k in range(4):
        theta = angles2[k]
        cx = r2 * np.cos(theta)
        cy = r2 * np.sin(theta)
        idx = 7 + k
        params[3 * idx] = cx
        params[3 * idx + 1] = cy
        params[3 * idx + 2] = 0
        
    # Container
    params[3 * N] = 0.0
    params[3 * N + 1] = 0.0
    params[3 * N + 2] = 0.0

    current_params = params.copy()
    current_energy = objective(current_params)
    best_params = current_params.copy()
    best_energy = current_energy
    
    # Simulated Annealing
    T = 0.1
    T_min = 0.001
    cooling = 0.995
    
    # Number of steps depends on time
    # Let's run for a fixed number of iterations or until time limit
    while time.time() < deadline - 20.0: # Leave 20s for refinement
        # Perturb
        new_params = current_params.copy()
        
        # Randomly select an inner hexagon to move
        idx = np.random.randint(0, N)
        # Move x, y, angle
        new_params[3 * idx] += np.random.normal(0, 0.1)
        new_params[3 * idx + 1] += np.random.normal(0, 0.1)
        new_params[3 * idx + 2] += np.random.normal(0, 0.05)
        
        # Occasionally perturb container
        if np.random.random() < 0.05:
            new_params[3 * N] += np.random.normal(0, 0.1)
            new_params[3 * N + 1] += np.random.normal(0, 0.1)
            new_params[3 * N + 2] += np.random.normal(0, 0.05)
            
        new_energy = objective(new_params)
        
        delta_e = new_energy - current_energy
        if delta_e < 0 or np.random.random() < np.exp(-delta_e / T):
            current_params = new_params
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_params = current_params.copy()
                
        T *= cooling
        if T < T_min:
            break

    # Refinement using Nelder-Mead on the best parameters found
    if time.time() < deadline - 15.0:
        try:
            max_iter = 200
            res = minimize(objective, best_params, method='Nelder-Mead', 
                           options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.fun < best_energy:
                best_params = res.x
                best_energy = res.fun
        except Exception as e:
            pass # Fallback to SA result
    
    # Decoupled container optimization: fix inner positions, optimize only container
    if time.time() < deadline - 5.0:
        try:
            def container_obj(cont_params):
                ocx, ocy, oangle = cont_params
                vertices_list = []
                for i in range(N):
                    cx = best_params[3 * i]
                    cy = best_params[3 * i + 1]
                    angle = best_params[3 * i + 2]
                    verts = hex_vertices(cx, cy, angle, SCALE)
                    vertices_list.append(verts)
                return compute_S(vertices_list, ocx, ocy, oangle)
            
            init_cont = best_params[3*N:3*N+3]
            res_cont = minimize(container_obj, init_cont, method='Nelder-Mead',
                               options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if res_cont.fun < compute_S([hex_vertices(best_params[3*i], best_params[3*i+1], best_params[3*i+2], SCALE) for i in range(N)],
                                        best_params[3*N], best_params[3*N+1], best_params[3*N+2]):
                best_params[3*N:3*N+3] = res_cont.x
        except Exception as e:
            pass
    
    # Final high-precision polish: optimize all parameters with tight tolerance
    if time.time() < deadline - 2.0:
        try:
            res_final = minimize(objective, best_params, method='Nelder-Mead',
                                 options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if res_final.fun < best_energy:
                best_params = res_final.x
                best_energy = res_final.fun
        except Exception as e:
            pass

    # Extract results
    inner = []
    vertices_list = []
    for i in range(N):
        cx = best_params[3 * i]
        cy = best_params[3 * i + 1]
        angle = best_params[3 * i + 2]
        inner.append([cx, cy, np.degrees(angle)])
        vertices_list.append(hex_vertices(cx, cy, angle, SCALE))

    ocx = best_params[3 * N]
    ocy = best_params[3 * N + 1]
    oangle = best_params[3 * N + 2]
    
    # Final S calculation
    S = compute_S(vertices_list, ocx, ocy, oangle)
    
    return inner, (ocx, ocy), S, np.degrees(oangle)
# EVOLVE-BLOCK-END

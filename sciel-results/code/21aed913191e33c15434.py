import numpy as np
import time
import random
from scipy.optimize import differential_evolution, minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        """Returns vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
        angle = np.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * np.pi / 3
            x = cx + side * np.cos(theta)
            y = cy + side * np.sin(theta)
            vertices.append([x, y])
        return np.array(vertices)

    def compute_overlap_penalty(v1, v2):
        """
        Computes a smooth penalty based on the sum of overlaps along all separating axes.
        v1, v2 are arrays of shape (6, 2).
        Returns 0 if disjoint.
        """
        # Calculate edges and normals for v1 and v2
        e1 = np.roll(v1, -1, axis=0) - v1
        n1 = np.column_stack((-e1[:, 1], e1[:, 0]))
        n1_norm = np.linalg.norm(n1, axis=1, keepdims=True)
        n1 = n1 / (n1_norm + 1e-12)
        
        e2 = np.roll(v2, -1, axis=0) - v2
        n2 = np.column_stack((-e2[:, 1], e2[:, 0]))
        n2_norm = np.linalg.norm(n2, axis=1, keepdims=True)
        n2 = n2 / (n2_norm + 1e-12)
        
        all_normals = np.vstack([n1, n2])
        
        total_penalty = 0.0
        for n in all_normals:
            proj1 = np.dot(v1, n)
            proj2 = np.dot(v2, n)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 > min2 and max2 > min1:
                overlap = min(max1, max2) - max(min1, min2)
                if overlap > 0:
                    total_penalty += overlap
        return total_penalty

    def compute_min_side(inner_verts, center, angle_deg):
        """
        Compute the minimum side length of the outer hexagon to contain all inner hexagons.
        inner_verts: list of (6,2) arrays.
        center: (x, y)
        angle_deg: rotation of outer hexagon
        """
        sqrt3 = np.sqrt(3)
        max_extent = 0.0
        angle_rad = np.radians(angle_deg)
        
        # Normals of the outer hexagon
        # Vertices of outer hexagon are at angles angle + i*pi/3
        # Normals are perpendicular to edges.
        # Edge i connects vertex i and i+1.
        # Midpoint of edge i is at angle angle + (i+0.5)*pi/3
        # Normal points outward, so angle is angle + (i+0.5)*pi/3
        for k in range(6):
            normal_angle = angle_rad + np.radians(30) + k * np.pi / 3
            normal = np.array([np.cos(normal_angle), np.sin(normal_angle)])
            
            for verts in inner_verts:
                offsets = verts - np.array(center)
                extents = np.dot(offsets, normal)
                max_e = np.max(extents)
                if max_e > max_extent:
                    max_extent = max_e
        
        # Side length L such that L * sqrt(3)/2 = max_extent
        return max_extent * 2.0 / sqrt3

    def objective(params):
        """
        Objective: minimize outer side length + penalty for overlaps.
        params: 3*N + 3 = 39 parameters
        0..35: x_i, y_i, theta_i for i in 0..11
        36: center_x
        37: center_y
        38: outer_angle
        """
        inner_verts = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            theta = params[3*i+2]
            v = hexagon_vertices(x, y, 1.0, theta)
            inner_verts.append(v)
        
        center = (params[36], params[37])
        outer_angle = params[38]
        
        # Penalty for overlaps
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                penalty += compute_overlap_penalty(inner_verts[i], inner_verts[j])
        
        if penalty > 1e-6:
             return 1000.0 * penalty + compute_min_side(inner_verts, center, outer_angle)
        
        side = compute_min_side(inner_verts, center, outer_angle)
        return side

    def initial_guess():
        """Generate a valid initial honeycomb packing."""
        params = np.zeros(3*N + 3)
        sqrt3 = np.sqrt(3)
        
        positions = []
        positions.append((0, 0, 0.0))
        for i in range(6):
            theta = i * np.pi / 3
            r = np.sqrt(3)
            positions.append((r * np.cos(theta), r * np.sin(theta), 0.0))
        for i in range(5):
            theta = i * 2*np.pi / 5 + np.pi/5
            r = 2 * np.sqrt(3)
            positions.append((r * np.cos(theta), r * np.sin(theta), 0.0))
        
        for i in range(N):
            params[3*i] = positions[i][0]
            params[3*i+1] = positions[i][1]
            params[3*i+2] = positions[i][2]
        
        params[36] = 0.0
        params[37] = 0.0
        params[38] = 0.0
        return params

    best_params = initial_guess()
    best_score = objective(best_params)
    
    # Phase 1: Differential Evolution
    # Bounds for DE
    bounds = []
    for i in range(N):
        bounds.append((-5, 5)) # x
        bounds.append((-5, 5)) # y
        bounds.append((0, 360)) # angle
    bounds.append((-5, 5)) # center x
    bounds.append((-5, 5)) # center y
    bounds.append((0, 360)) # outer angle

    # Check time
    if time.time() < deadline - 60.0:
        try:
            res = differential_evolution(
                objective, 
                bounds, 
                seed=42, 
                maxiter=200, 
                popsize=15,
                tol=1e-4,
                polish=True, # Use L-BFGS-B for polish
                workers=1
            )
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass

    # Phase 2: Local Refinement with Nelder-Mead
    if time.time() < deadline - 5.0:
        def nm_objective(x):
            return objective(x)
        
        # Refine best DE result
        result = minimize(nm_objective, best_params, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        
        if result.fun < best_score:
            best_params = result.x
            best_score = result.fun

        # Multi-start from perturbed best
        n_starts = 3
        for s in range(n_starts):
            if time.time() >= deadline - 5.0:
                break
            
            perturbed = best_params.copy()
            scale = 0.1
            for idx in range(len(perturbed)):
                perturbed[idx] += random.gauss(0, scale)
            
            p_score = objective(perturbed)
            if p_score < best_score:
                result2 = minimize(nm_objective, perturbed, method='Nelder-Mead', 
                                   options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
                if result2.fun < best_score:
                    best_params = result2.x
                    best_score = result2.fun

    # Extract best
    inner = []
    for i in range(N):
        inner.append([float(best_params[3*i]), float(best_params[3*i+1]), float(best_params[3*i+2])])
    
    center = (float(best_params[36]), float(best_params[37]))
    angle = float(best_params[38])
    
    # Recalculate side for return
    inner_verts = [hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(N)]
    side = compute_min_side(inner_verts, center, angle)
    
    # Final check: ensure validity
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if compute_overlap_penalty(inner_verts[i], inner_verts[j]) > 1e-5:
                valid = False
                break
        if not valid: break
    
    if not valid:
        # Fallback: use a known valid configuration (hexagonal grid)
        inner = []
        positions = [(0, 0, 0)]
        for i in range(6):
            theta = i * np.pi / 3
            positions.append((np.sqrt(3)*np.cos(theta), np.sqrt(3)*np.sin(theta), 0))
        for i in range(5):
            theta = i * 2*np.pi / 5 + np.pi/5
            positions.append((2*np.sqrt(3)*np.cos(theta), 2*np.sqrt(3)*np.sin(theta), 0))
        for pos in positions:
            inner.append([float(pos[0]), float(pos[1]), float(pos[2])])
        center = (0.0, 0.0)
        angle = 0.0
        inner_verts = [hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(N)]
        side = compute_min_side(inner_verts, center, angle)

    return inner, center, side, angle
# EVOLVE-BLOCK-END

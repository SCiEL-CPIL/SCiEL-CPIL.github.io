"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    def hex_vertices(cx, cy, angle, s=1.0):
        """Return the 6 vertices of a regular hexagon."""
        angles = angle + np.arange(6) * np.pi / 3
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def compute_S_for_container(vertices_list, ocx, ocy, oangle):
        """Compute the minimum side length S of the outer hexagon (centered at ocx, ocy, rot oangle) to contain all vertices."""
        # Transform vertices to container's local coordinate system
        cos_a = np.cos(oangle)
        sin_a = np.sin(oangle)
        S = 0.0
        apothem = np.sqrt(3) / 2
        for verts in vertices_list:
            for v in verts:
                # Translate
                lx = v[0] - ocx
                ly = v[1] - ocy
                # Rotate by -oangle to align container axes
                rx = lx * cos_a + ly * sin_a
                ry = -lx * sin_a + ly * cos_a
                # Project onto 6 normals
                for k in range(6):
                    theta_n = np.pi / 6 + k * np.pi / 3
                    proj = rx * np.cos(theta_n) + ry * np.sin(theta_n)
                    S = max(S, abs(proj))
        return S / apothem

    def sat_overlap(v1, v2):
        """Check if two convex polygons overlap using Separating Axis Theorem."""
        polys = [v1, v2]
        for poly in polys:
            n = len(poly)
            for i in range(n):
                p1 = poly[i]
                p2 = poly[(i + 1) % n]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                proj1 = v1 @ normal
                proj2 = v2 @ normal
                if proj1.max() < proj2.min() or proj2.max() < proj1.min():
                    return False
        return True

    def objective(params):
        """Compute the outer side length S and penalty for overlaps. Assumes container at (0,0) rot 0 for inner optimization."""
        n = len(params) // 3
        vertices_list = []
        for i in range(n):
            cx = params[3 * i]
            cy = params[3 * i + 1]
            angle = params[3 * i + 2]
            verts = hex_vertices(cx, cy, angle)
            vertices_list.append(verts)
        
        # Use fixed container for inner optimization to keep it simple, 
        # we will refine container in post-processing
        S = compute_S_for_container(vertices_list, 0.0, 0.0, 0.0)
        
        # Overlap penalty
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                c1 = vertices_list[i].mean(axis=0)
                c2 = vertices_list[j].mean(axis=0)
                dist = np.linalg.norm(c1 - c2)
                if dist < 2.0: # Only check SAT if potentially overlapping
                    if sat_overlap(vertices_list[i], vertices_list[j]):
                        # Strong penalty to push apart
                        penalty += 1000.0 * (2.0 - dist)
        
        return S + penalty

    def find_best_container(vertices_list):
        """Find the minimal enclosing hexagon for a set of vertices."""
        # Initial guess: centroid, 0 angle
        cx, cy = 0.0, 0.0
        angle = 0.0
        
        def cont_obj(params_c):
            ocx, ocy, oangle = params_c
            return compute_S_for_container(vertices_list, ocx, ocy, oangle)
        
        # Optimize container parameters
        res = minimize(cont_obj, [cx, cy, angle], method='Powell', options={'maxiter': 100, 'maxfev': 500})
        return res.x, res.fun

    # Initial guess: hexagonal lattice
    params = np.zeros(3 * N)
    params[0:3] = [0, 0, 0]
    idx = 1
    for k in range(6):
        theta = k * np.pi / 3
        cx = np.sqrt(3) * np.cos(theta)
        cy = np.sqrt(3) * np.sin(theta)
        params[3 * idx] = cx
        params[3 * idx + 1] = cy
        params[3 * idx + 2] = 0
        idx += 1
    for k in range(4):
        theta = k * np.pi / 3
        cx = 2 * np.sqrt(3) * np.cos(theta)
        cy = 2 * np.sqrt(3) * np.sin(theta)
        params[3 * idx] = cx
        params[3 * idx + 1] = cy
        params[3 * idx + 2] = 0
        idx += 1

    best_params = params
    best_inner_score = objective(params)
    
    # Multi-start optimization of inner positions
    np.random.seed(42) # For reproducibility in debugging, but we want diversity
    for i in range(10):
        p = params.copy()
        if i > 0:
            p += np.random.normal(0, 0.2, size=p.size)
        
        res = minimize(objective, p, method='Powell', 
                      options={'maxiter': 2000, 'maxfev': 10000})
        
        if res.fun < best_inner_score:
            best_inner_score = res.fun
            best_params = res.x

    # Extract inner positions
    inner = []
    vertices_list = []
    for i in range(N):
        cx = best_params[3 * i]
        cy = best_params[3 * i + 1]
        angle = best_params[3 * i + 2]
        inner.append([cx, cy, np.degrees(angle)])
        vertices_list.append(hex_vertices(cx, cy, angle))

    # Refine the outer container
    cont_params, S = find_best_container(vertices_list)
    ocx, ocy, oangle = cont_params
    oangle_deg = np.degrees(oangle)
    
    return inner, (ocx, ocy), S, oangle_deg
# EVOLVE-BLOCK-END

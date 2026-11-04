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

    def compute_S(vertices_list):
        """Compute the minimum side length S of the outer hexagon (centered at 0,0, rot 0) to contain all vertices."""
        S = 0.0
        apothem = np.sqrt(3) / 2
        for verts in vertices_list:
            for v in verts:
                for k in range(6):
                    theta_n = np.pi / 6 + k * np.pi / 3
                    proj = v[0] * np.cos(theta_n) + v[1] * np.sin(theta_n)
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
        """Compute the outer side length S and penalty for overlaps."""
        n = len(params) // 3
        vertices_list = []
        for i in range(n):
            cx = params[3 * i]
            cy = params[3 * i + 1]
            angle = params[3 * i + 2]
            verts = hex_vertices(cx, cy, angle)
            vertices_list.append(verts)
        
        S = compute_S(vertices_list)
        
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

    # Use Powell method which is often more robust for these problems
    result = minimize(objective, params, method='Powell', 
                      options={'maxiter': 2000, 'maxfev': 10000})
    
    # Try a few random perturbations if the first result is bad
    best_params = result.x
    best_score = result.fun
    
    # Simple multi-start
    for _ in range(5):
        p = params.copy()
        p += np.random.normal(0, 0.1, size=p.size)
        res = minimize(objective, p, method='Powell', options={'maxiter': 1000, 'maxfev': 5000})
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x

    opt_params = best_params
    inner = []
    for i in range(N):
        cx = opt_params[3 * i]
        cy = opt_params[3 * i + 1]
        angle = np.degrees(opt_params[3 * i + 2])
        inner.append([cx, cy, angle])
    
    # Compute the final S
    vertices_list = []
    for i in range(N):
        cx = opt_params[3 * i]
        cy = opt_params[3 * i + 1]
        angle = opt_params[3 * i + 2]
        verts = hex_vertices(cx, cy, angle)
        vertices_list.append(verts)
    S = compute_S(vertices_list)
    
    return inner, (0.0, 0.0), S, 0.0
# EVOLVE-BLOCK-END

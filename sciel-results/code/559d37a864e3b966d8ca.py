import numpy as np
from scipy.optimize import minimize

def construct():
    N = 12
    sqrt3 = np.sqrt(3)
    
    # Generate initial centers from a hexagonal tiling pattern
    centers = []
    for i in range(-1, 2):
        for j in range(-1, 3):
            x = i * sqrt3
            y = 2 * j + (1 if i % 2 != 0 else 0)
            centers.append([x, y])
            
    centers = np.array(centers)
    centers -= np.mean(centers, axis=0)
    
    # Initial angles: 0
    angles = np.zeros(N)
    
    # Variables: x1, y1, a1, x2, y2, a2, ...
    x0 = np.zeros(36)
    for i in range(N):
        x0[i*3] = centers[i, 0]
        x0[i*3+1] = centers[i, 1]
        x0[i*3+2] = 0.0
        
    # Base hexagon vertices (unit side, pointy-top orientation)
    base_angles = np.deg2rad([90, 150, 210, 270, 330, 30])
    base_verts = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    
    def get_inner_verts(params):
        verts_list = []
        for i in range(N):
            cx = params[i*3]
            cy = params[i*3+1]
            ang = params[i*3+2]
            c, s = np.cos(ang), np.sin(ang)
            R = np.array([[c, -s], [s, c]])
            v = (R @ base_verts.T).T + np.array([cx, cy])
            verts_list.append(v)
        return np.array(verts_list)
        
    def get_edges(verts):
        """Get all edges from vertices. Each edge is (p1, p2)."""
        edges = []
        for i in range(N):
            v = verts[i]
            for j in range(6):
                edges.append((v[j], v[(j+1) % 6]))
        return np.array(edges)  # Shape (72, 2, 2)
        
    def point_to_segment_distance(p, a, b):
        """Distance from point p to segment ab."""
        ab = b - a
        t = np.dot(p - a, ab) / np.dot(ab, ab)
        t = np.clip(t, 0, 1)
        closest = a + t * ab
        return np.linalg.norm(p - closest)
        
    def min_edge_distance(e1, e2):
        """Minimum distance between two segments e1 and e2."""
        # Check if endpoints of e1 are close to e2
        d1 = point_to_segment_distance(e1[0], e2[0], e2[1])
        d2 = point_to_segment_distance(e1[1], e2[0], e2[1])
        # Check if endpoints of e2 are close to e1
        d3 = point_to_segment_distance(e2[0], e1[0], e1[1])
        d4 = point_to_segment_distance(e2[1], e1[0], e1[1])
        return min(d1, d2, d3, d4)
        
    def calc_L_from_verts(verts):
        """Calculate minimum outer hexagon side length to contain all vertices."""
        x = verts[:, :, 0]
        y = verts[:, :, 1]
        
        term1 = x + y / sqrt3
        term2 = 2 * y / sqrt3
        term3 = -x + y / sqrt3
        term4 = -x - y / sqrt3
        term5 = -2 * y / sqrt3
        term6 = x - y / sqrt3
        
        L_points = np.max(np.stack([term1, term2, term3, term4, term5, term6]), axis=0)
        return np.max(L_points)
        
    def overlap_penalty(params):
        """Differentiable penalty based on pairwise edge distances."""
        verts = get_inner_verts(params)
        edges = get_edges(verts)
        
        # Only check edges from different hexagons
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                # Edges from hex i: indices i*6 to i*6+5
                # Edges from hex j: indices j*6 to j*6+5
                for ei in range(6):
                    for ej in range(6):
                        idx_i = i * 6 + ei
                        idx_j = j * 6 + ej
                        d = min_edge_distance(edges[idx_i], edges[idx_j])
                        if d < 0.1:
                            penalty += (0.1 - d) ** 2
        return penalty
        
    def objective(params):
        verts = get_inner_verts(params)
        L = calc_L_from_verts(verts)
        pen = overlap_penalty(params)
        return L + 5.0 * pen
        
    # Optimize using Nelder-Mead
    res = minimize(objective, x0, method='Nelder-Mead', 
                   options={'xatol': 1e-5, 'fatol': 1e-5, 'maxiter': 5000, 'adaptive': True})
    
    best_params = res.x
    best_L = calc_L_from_verts(get_inner_verts(best_params))
    
    inner = []
    for i in range(N):
        cx = best_params[i*3]
        cy = best_params[i*3+1]
        ang_deg = np.rad2deg(best_params[i*3+2])
        inner.append([cx, cy, ang_deg])
        
    return inner, (0.0, 0.0), best_L, 0.0

"""Kinematic closure: 12 flat-top unit hexagons in a 4x3 honeycomb, optimize outer rotation and size."""
import numpy as np
from scipy.optimize import linprog

N = 12

def construct():
    sqrt3 = np.sqrt(3.0)
    
    # Flat-top unit hexagons in a 4-column x 3-row honeycomb
    centers = []
    for col in range(4):
        for row in range(3):
            x = 1.5 * col
            y = sqrt3 * row + (sqrt3 / 2.0) * (col % 2)
            centers.append((x, y))
    
    # Center the cluster at origin
    cx = np.mean([c[0] for c in centers])
    cy = np.mean([c[1] for c in centers])
    centers = [(c[0] - cx, c[1] - cy) for c in centers]
    
    # Flat-top unit hexagon vertices: angles 0, 60, 120, 180, 240, 300
    angles_vertex = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    vertex_offsets = np.column_stack([np.cos(angles_vertex), np.sin(angles_vertex)])
    
    # All vertices of all 12 hexagons
    all_vertices = []
    for (hx, hy) in centers:
        for (vx, vy) in vertex_offsets:
            all_vertices.append((hx + vx, hy + vy))
    all_vertices = np.array(all_vertices)
    
    def min_inradius(theta):
        edge_angles = theta + np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
        normals = np.column_stack([np.cos(edge_angles), np.sin(edge_angles)])
        
        A_rows = []
        b_vals = []
        for p in all_vertices:
            for n in normals:
                A_rows.append([-n[0], -n[1], -1.0])
                b_vals.append(-np.dot(p, n))
        
        A = np.array(A_rows)
        b = np.array(b_vals)
        c_obj = np.array([0.0, 0.0, 1.0])
        bounds = [(None, None), (None, None), (0, None)]
        
        res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if res.success:
            return res.fun
        else:
            return 1e10
    
    # Search over theta in [0, 60) degrees
    best_r = 1e10
    best_theta = 0.0
    
    for theta_deg in np.linspace(0, 60, 61):
        r = min_inradius(theta_deg * np.pi / 180.0)
        if r < best_r:
            best_r = r
            best_theta = theta_deg * np.pi / 180.0
    
    lo = max(0, best_theta - 2 * np.pi / 180.0)
    hi = min(np.pi / 3.0, best_theta + 2 * np.pi / 180.0)
    
    for _ in range(20):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        r1 = min_inradius(m1)
        r2 = min_inradius(m2)
        if r1 < r2:
            hi = m2
        else:
            lo = m1
    
    best_theta = (lo + hi) / 2.0
    best_r = min_inradius(best_theta)
    side = 2.0 * best_r / sqrt3
    
    # Solve for center
    edge_angles = best_theta + np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    normals = np.column_stack([np.cos(edge_angles), np.sin(edge_angles)])
    
    A_rows = []
    b_vals = []
    for p in all_vertices:
        for n in normals:
            A_rows.append([-n[0], -n[1], -1.0])
            b_vals.append(-np.dot(p, n))
    
    A = np.array(A_rows)
    b = np.array(b_vals)
    c_obj = np.array([0.0, 0.0, 1.0])
    bounds = [(None, None), (None, None), (0, None)]
    res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    
    if res.success:
        outer_cx = res.x[0]
        outer_cy = res.x[1]
    else:
        outer_cx = 0.0
        outer_cy = 0.0
    
    inner = []
    for (hx, hy) in centers:
        inner.append([hx, hy, 0.0])
    
    return inner, (outer_cx, outer_cy), side, np.degrees(best_theta)


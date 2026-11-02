import numpy as np
from scipy.optimize import minimize
import time

N = 12
DEADLINE = 140.0

def hex_vertices(cx, cy, side, angle_deg):
    angle = np.radians(angle_deg)
    r = side
    angles = angle + np.arange(6) * np.pi / 3
    return np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)])

def enclosing_hexagon_side(centers, angles_deg, ocx, ocy, orot_deg):
    """Compute side length of enclosing hexagon."""
    all_pts = []
    rot = np.radians(orot_deg)
    R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    for i in range(len(centers)):
        verts = hex_vertices(centers[i][0], centers[i][1], 1.0, angles_deg[i])
        pts = (verts - np.array([ocx, ocy])) @ R.T
        all_pts.append(pts)
    all_pts = np.vstack(all_pts)
    normals = np.array([[np.cos(np.pi/3 * k), np.sin(np.pi/3 * k)] for k in range(6)])
    projections = all_pts @ normals.T
    max_proj = np.max(np.abs(projections), axis=0)
    return np.max(max_proj) / (np.sqrt(3) / 2)

def construct():
    start_time = time.time()
    
    # Best known: honeycomb-like arrangement
    # For 12 hexagons, try a compact cluster
    # Center + 6 ring1 + 5 ring2
    
    d = np.sqrt(3)  # center-to-center distance for touching hexagons (same orientation)
    
    # Configuration: hexagonal cluster
    positions = [(0.0, 0.0)]
    for k in range(6):
        a = np.pi / 3 * k
        positions.append((d * np.cos(a), d * np.sin(a)))
    # Add 5 from ring 2 (axial positions)
    for k in range(5):
        a = np.pi / 3 * k
        positions.append((2 * d * np.cos(a), 2 * d * np.sin(a)))
    
    centers = np.array(positions)
    angles = [0.0] * N
    
    # Compute initial enclosing hexagon
    all_verts = np.vstack([hex_vertices(c[0], c[1], 1.0, 0.0) for c in centers])
    cx0, cy0 = np.mean(all_verts, axis=0)
    
    best_side = float('inf')
    best_params = None
    
    # Optimize outer hexagon (center and rotation) for fixed inner positions
    def outer_obj(params):
        ocx, ocy, orot = params
        return enclosing_hexagon_side(centers, angles, ocx, ocy, orot)
    
    for rot_init in range(0, 60, 5):
        x0 = [cx0, cy0, float(rot_init)]
        res = minimize(outer_obj, x0, method='Nelder-Mead',
                      options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-9})
        if res.fun < best_side:
            best_side = res.fun
            best_params = res.x
    
    if time.time() - start_time > DEADLINE - 10:
        cx, cy, orot = best_params
        inner = [[float(centers[i][0]), float(centers[i][1]), 0.0] for i in range(N)]
        return inner, (float(cx), float(cy)), float(best_side), float(orot)
    
    # Now try to optimize inner positions to reduce enclosing hexagon
    # Keep outer hexagon fixed, optimize inner positions
    
    ocx, ocy, orot = best_params
    
    def inner_obj(params):
        new_centers = params.reshape(N, 2)
        return enclosing_hexagon_side(new_centers, angles, ocx, ocy, orot)
    
    init_inner = centers.flatten()
    res2 = minimize(inner_obj, init_inner, method='Nelder-Mead',
                   options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-9})
    
    if res2.fun < best_side:
        best_side = res2.fun
        centers = res2.x.reshape(N, 2)
    
    # Try a few more outer rotations with updated centers
    for rot_try in range(0, 60, 10):
        def outer_obj2(params):
            ocx2, ocy2, orot2 = params
            return enclosing_hexagon_side(centers, angles, ocx2, ocy2, orot2)
        res3 = minimize(outer_obj2, [cx0, cy0, float(rot_try)], method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-9})
        if res3.fun < best_side:
            best_side = res3.fun
            ocx, ocy, orot = res3.x
    
    inner = [[float(centers[i][0]), float(centers[i][1]), 0.0] for i in range(N)]
    return inner, (float(ocx), float(ocy)), float(best_side), float(orot)

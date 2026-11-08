"""Packing 11 unit regular hexagons in a minimum-size regular hexagon."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11

def hex_vertices(cx, cy, side=1.0, angle_deg=0.0):
    a = np.radians(angle_deg)
    angles = a + np.pi / 3 * np.arange(6)
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _point_in_convex(px, py, verts):
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if cross < -1e-10:
            return False
    return True

def hexagons_disjoint(verts_list, tol=1e-9):
    n = len(verts_list)
    for i in range(n):
        for j in range(i + 1, n):
            vi, vj = verts_list[i], verts_list[j]
            for k in range(6):
                if _point_in_convex(vi[k, 0], vi[k, 1], vj):
                    return False
            for k in range(6):
                if _point_in_convex(vj[k, 0], vj[k, 1], vi):
                    return False
    return True

def min_outer_side(verts_list):
    """Find minimum side length of a regular hexagon (any center, any rotation) containing all points."""
    all_pts = np.vstack(verts_list)
    M = len(all_pts)
    
    def side_for_center_angle(ocx, ocy, oang):
        dx = all_pts[:, 0] - ocx
        dy = all_pts[:, 1] - ocy
        normal_angles = oang + (np.arange(6) + 0.5) * np.pi / 3
        cos_n = np.cos(normal_angles)
        sin_n = np.sin(normal_angles)
        dots = np.outer(dx, cos_n) + np.outer(dy, sin_n)
        max_dot = np.max(dots)
        return max_dot * 2.0 / np.sqrt(3)
    
    return side_for_center_angle

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    sqrt3 = np.sqrt(3)
    
    # Initial hexagonal lattice arrangement
    positions = []
    for i in range(3):
        positions.append((i * sqrt3, 0.0))
    for i in range(4):
        positions.append((sqrt3/2 + i * sqrt3, 1.5))
    for i in range(3):
        positions.append((i * sqrt3, 3.0))
    positions.append((sqrt3/2, 4.5))
    
    angles_init = [0.0] * N
    cx = np.mean([p[0] for p in positions])
    cy = np.mean([p[1] for p in positions])
    
    def full_objective(x):
        if time.time() > deadline - 5:
            return 1e10
        pos = x[:2*N].reshape(N, 2)
        ang = x[2*N:3*N]
        ocx, ocy, oang = x[3*N], x[3*N+1], x[3*N+2]
        
        verts_list = []
        for i in range(N):
            v = hex_vertices(pos[i, 0], pos[i, 1], 1.0, ang[i])
            verts_list.append(v)
        
        # Disjointness penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                vi, vj = verts_list[i], verts_list[j]
                for k in range(6):
                    if _point_in_convex(vi[k, 0], vi[k, 1], vj):
                        penalty += 5.0
                for k in range(6):
                    if _point_in_convex(vj[k, 0], vj[k, 1], vi):
                        penalty += 5.0
        
        # Outer hex side length
        all_pts = np.vstack(verts_list)
        dx = all_pts[:, 0] - ocx
        dy = all_pts[:, 1] - ocy
        normal_angles = oang + (np.arange(6) + 0.5) * np.pi / 3
        cos_n = np.cos(normal_angles)
        sin_n = np.sin(normal_angles)
        dots = np.outer(dx, cos_n) + np.outer(dy, sin_n)
        max_dot = np.max(dots)
        s = max_dot * 2.0 / np.sqrt(3)
        
        return s + penalty
    
    x0 = np.zeros(3*N + 3)
    for i in range(N):
        x0[2*i] = positions[i][0] - cx
        x0[2*i+1] = positions[i][1] - cy
        x0[2*N + i] = angles_init[i]
    
    best_x = x0.copy()
    best_val = full_objective(x0)
    
    # Nelder-Mead
    try:
        res = minimize(full_objective, x0, method='Nelder-Mead',
                      options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-7})
        if res.fun < best_val and time.time() < deadline - 10:
            best_val = res.fun
            best_x = res.x
    except:
        pass
    
    # L-BFGS-B
    if time.time() < deadline - 20:
        try:
            res2 = minimize(full_objective, best_x, method='L-BFGS-B',
                           options={'maxiter': 1000, 'ftol': 1e-10})
            if res2.fun < best_val:
                best_val = res2.fun
                best_x = res2.x
        except:
            pass
    
    pos = best_x[:2*N].reshape(N, 2)
    ang = best_x[2*N:3*N]
    ocx, ocy, oang = best_x[3*N], best_x[3*N+1], best_x[3*N+2]
    
    verts_list = []
    for i in range(N):
        v = hex_vertices(pos[i, 0], pos[i, 1], 1.0, ang[i] * 180.0 / np.pi)
        verts_list.append(v)
    
    inner = []
    for i in range(N):
        inner.append([float(pos[i, 0]), float(pos[i, 1]), float(ang[i] * 180.0 / np.pi)])
    
    all_pts = np.vstack(verts_list)
    dx = all_pts[:, 0] - ocx
    dy = all_pts[:, 1] - ocy
    normal_angles = oang + (np.arange(6) + 0.5) * np.pi / 3
    cos_n = np.cos(normal_angles)
    sin_n = np.sin(normal_angles)
    dots = np.outer(dx, cos_n) + np.outer(dy, sin_n)
    s = np.max(dots) * 2.0 / np.sqrt(3)
    
    if not hexagons_disjoint(verts_list):
        # Fallback
        inner = [[float(positions[i][0] - cx), float(positions[i][1] - cy), 0.0] for i in range(N)]
        verts_list = [hex_vertices(positions[i][0] - cx, positions[i][1] - cy, 1.0, 0.0) for i in range(N)]
        all_pts = np.vstack(verts_list)
        # Find best outer hex
        best_s = float('inf')
        for oang_try in np.linspace(0, np.pi/3, 13):
            for ocx_try in np.linspace(-2, 2, 9):
                for ocy_try in np.linspace(-2, 2, 9):
                    dx = all_pts[:, 0] - ocx_try
                    dy = all_pts[:, 1] - ocy_try
                    na = oang_try + (np.arange(6) + 0.5) * np.pi / 3
                    d = np.outer(dx, np.cos(na)) + np.outer(dy, np.sin(na))
                    s_try = np.max(d) * 2.0 / np.sqrt(3)
                    if s_try < best_s:
                        best_s = s_try
        s = best_s
        ocx, ocy, oang = 0.0, 0.0, 0.0
    
    return inner, (float(ocx), float(ocy)), float(s), float(oang * 180.0 / np.pi)

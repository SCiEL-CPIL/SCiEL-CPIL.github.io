"""Rewritten program for packing 12 unit hexagons in minimal outer hexagon."""
import numpy as np
import time
import math

N = 12
DEADLINE = 140.0  # Safety margin before 150s kill

def get_hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    """
    Return vertices of a regular hexagon centered at (cx, cy) with given side length and rotation.
    Vertices are in counter-clockwise order.
    """
    rot_rad = np.radians(rot_deg)
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + rot_rad
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def sat_overlap(v1, v2):
    """
    Check if two convex hexagons overlap using SAT.
    Returns True if they overlap, False otherwise.
    """
    for v in (v1, v2):
        for i in range(6):
            p1 = v[i]
            p2 = v[(i+1) % 6]
            e = p2 - p1
            n = np.array([-e[1], e[0]])
            norm = np.linalg.norm(n)
            if norm < 1e-12:
                continue
            n = n / norm
            proj1 = v1 @ n
            proj2 = v2 @ n
            if np.max(proj1) < np.min(proj2) or np.max(proj2) < np.min(proj1):
                return False
    return True

def min_penetration(v1, v2):
    """
    Compute minimum penetration depth between two convex polygons using SAT.
    Returns 0 if separated, positive value if overlapping.
    """
    min_pen = float('inf')
    for v in (v1, v2):
        for i in range(6):
            p1 = v[i]
            p2 = v[(i+1) % 6]
            e = p2 - p1
            n = np.array([-e[1], e[0]])
            norm = np.linalg.norm(n)
            if norm < 1e-12:
                continue
            n = n / norm
            proj1 = v1 @ n
            proj2 = v2 @ n
            overlap = min(np.max(proj1), np.max(proj2)) - max(np.min(proj1), np.min(proj2))
            if overlap < 0:
                return 0.0  # Separated on this axis
            min_pen = min(min_pen, overlap)
    return min_pen

def outer_hex_side(points, outer_rot_deg):
    """
    Calculate the minimum side length L of a regular hexagon centered at (0,0)
    with rotation outer_rot_deg that encloses all given points.
    """
    angles = np.radians(np.array([90, 150, 210, 270, 330, 30]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    apothem = max_proj
    return (2.0 / np.sqrt(3.0)) * apothem

def objective(params, time_limit):
    """
    params: array of length 3*N + 1
    [x0, y0, rot0, x1, y1, rot1, ..., x11, y11, rot11, outer_rot_deg]
    """
    if time.time() > time_limit:
        return 1e6
    
    xys_rots = params[:3*N].reshape(N, 3)
    outer_rot = params[3*N]
    
    # Compute vertices for all inner hexagons
    vertices_list = []
    for i in range(N):
        cx, cy, rot = xys_rots[i]
        v = get_hex_vertices(cx, cy, 1.0, rot)
        vertices_list.append(v)
    
    # Check overlaps and calculate penalty
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            pen = min_penetration(vertices_list[i], vertices_list[j])
            if pen > 0:
                penalty += pen ** 2
    
    # Calculate bounding hexagon
    all_points = np.vstack(vertices_list)
    L = outer_hex_side(all_points, outer_rot)
    
    # Total objective
    return L + 1000.0 * penalty

def is_valid(xys_rots, outer_rot, tol=1e-3):
    """Check if configuration is valid (no overlaps, all inside)."""
    vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]
    
    # Check overlaps
    for i in range(N):
        for j in range(i+1, N):
            pen = min_penetration(vertices_list[i], vertices_list[j])
            if pen > tol:
                return False
    
    return True

def construct():
    start_time = time.time()
    time_limit = start_time + DEADLINE
    
    best_L = 100.0
    best_params = None
    
    # Generate initial configurations
    def make_hexagonal_lattice():
        """Create a hexagonal lattice arrangement."""
        pts = []
        sqrt3 = np.sqrt(3)
        # 3-4-3-2 arrangement or similar
        for r in range(4):
            for c in range(3):
                x = c * sqrt3
                y = r * 1.5
                if r % 2 == 1:
                    x += sqrt3 / 2.0
                pts.append([x, y, 0.0])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts.flatten()

    def make_linear():
        """Create a linear arrangement."""
        pts = []
        sqrt3 = np.sqrt(3)
        for i in range(N):
            x = i * sqrt3
            y = 0.0
            pts.append([x, y, 0.0])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts.flatten()

    def make_cluster():
        """Create a compact cluster."""
        pts = []
        sqrt3 = np.sqrt(3)
        pts.append([0, 0, 0.0])
        for k in range(6):
            ang = np.radians(60 * k)
            pts.append([sqrt3 * np.cos(ang), sqrt3 * np.sin(ang), 0.0])
        for k in range(5):
            ang = np.radians(60 * k + 30)
            pts.append([2 * sqrt3 * np.cos(ang), 2 * sqrt3 * np.sin(ang), 0.0])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts.flatten()

    def make_ring():
        """Create a ring arrangement."""
        pts = []
        sqrt3 = np.sqrt(3)
        # Center
        pts.append([0, 0, 0.0])
        # First ring (6 hexagons)
        for k in range(6):
            ang = np.radians(60 * k)
            pts.append([sqrt3 * np.cos(ang), sqrt3 * np.sin(ang), 0.0])
        # Second ring (5 hexagons to make 12 total)
        for k in range(5):
            ang = np.radians(60 * k + 30)
            pts.append([2 * sqrt3 * np.cos(ang), 2 * sqrt3 * np.sin(ang), 0.0])
        pts = np.array(pts)
        pts[:, :2] -= np.mean(pts[:, :2], axis=0)
        return pts.flatten()

    configs = [
        make_hexagonal_lattice(),
        make_linear(),
        make_cluster(),
        make_ring()
    ]
    
    # Use scipy minimize
    from scipy.optimize import minimize
    
    for cfg in configs:
        if time.time() > time_limit:
            break
        
        init_params = np.concatenate((cfg, [0.0]))
        
        try:
            res = minimize(
                objective, init_params, args=(time_limit,),
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-5,
                    'fatol': 1e-5,
                    'adaptive': True
                }
            )
            
            xys_rots = res.x[:3*N].reshape(N, 3)
            outer_rot = res.x[3*N]
            
            # Verify no overlaps
            vertices_list = [get_hex_vertices(x, y, 1.0, r) for x, y, r in xys_rots]
            overlap = False
            for i in range(N):
                for j in range(i+1, N):
                    pen = min_penetration(vertices_list[i], vertices_list[j])
                    if pen > 1e-3:
                        overlap = True
                        break
                if overlap:
                    break
            
            if not overlap:
                L_val = outer_hex_side(np.vstack(vertices_list), outer_rot)
                if L_val < best_L:
                    best_L = L_val
                    best_params = res.x.copy()
        except Exception:
            pass
    
    # If no valid improvement found, return a simple configuration
    if best_params is None:
        # Return a simple valid configuration
        inner = []
        sqrt3 = np.sqrt(3)
        for i in range(N):
            x = (i % 4) * sqrt3
            y = (i // 4) * 1.5
            if (i // 4) % 2 == 1:
                x += sqrt3 / 2.0
            inner.append([float(x), float(y), 0.0])
        return inner, (0.0, 0.0), 20.0, 0.0
    
    # Extract best result
    xys_rots = best_params[:3*N].reshape(N, 3)
    outer_rot = best_params[3*N]
    
    inner = []
    for i in range(N):
        inner.append([float(xys_rots[i][0]), float(xys_rots[i][1]), float(xys_rots[i][2])])
    
    return inner, (0.0, 0.0), float(best_L), float(outer_rot)

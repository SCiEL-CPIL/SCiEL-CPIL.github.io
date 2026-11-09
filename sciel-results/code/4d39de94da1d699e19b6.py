"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    sqrt3 = math.sqrt(3.0)

    def get_hex_vertices(cx, cy, angle_deg, side=1.0):
        """Get vertices of a regular hexagon."""
        vertices = []
        angle_rad = math.radians(angle_deg)
        for i in range(6):
            theta = angle_rad + i * math.pi / 3.0
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return vertices

    def point_in_hex(px, py, poly, eps=1e-7):
        """Check if point (px, py) is inside convex hexagon poly."""
        # poly is list of (x, y) in CCW order
        inside = True
        for k in range(6):
            x1, y1 = poly[k]
            x2, y2 = poly[(k + 1) % 6]
            # Vector edge
            ex, ey = x2 - x1, y2 - y1
            # Vector to point
            px1, py1 = px - x1, py - y1
            cross = ex * py1 - ey * px1
            if cross < -eps:  # If strictly to the right (outside)
                return False
        return True

    def check_overlap(centers, angles, eps=1e-6):
        """Check if any two inner hexagons overlap."""
        for i in range(N):
            for j in range(i + 1, N):
                # Quick distance check: if centers are far apart, no overlap
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist_sq = dx*dx + dy*dy
                # If dist > 2.0, they definitely don't overlap? No, max width is 2.0.
                # If dist >= 2.0, they might touch but not overlap.
                # Let's use a safe threshold.
                if dist_sq > 4.0:
                    continue
                
                vi = get_hex_vertices(centers[i][0], centers[i][1], angles[i])
                vj = get_hex_vertices(centers[j][0], centers[j][1], angles[j])
                
                # Check if any vertex of i is inside j, or vice versa
                for vx, vy in vi:
                    if point_in_hex(vx, vy, vj):
                        return False
                for vx, vy in vj:
                    if point_in_hex(vx, vy, vi):
                        return False
        return True

    def enclosing_side_length(centers, angles, outer_angle=0.0):
        """
        Compute the minimal side length of a regular hexagon (centered at origin, 
        with given rotation) that contains all inner unit hexagons.
        """
        # 6 outward normal directions for outer hexagon with rotation outer_angle
        normals = []
        for k in range(6):
            angle = math.radians(outer_angle + 30 + k * 60)
            normals.append((math.cos(angle), math.sin(angle)))
        
        max_extent = 0.0
        for i in range(N):
            cx, cy = centers[i]
            inner_angle_rad = math.radians(angles[i])
            
            for nk in range(6):
                nx, ny = normals[nk]
                n_angle = math.atan2(ny, nx)
                max_dot = 0.0
                for j in range(6):
                    v_angle = inner_angle_rad + j * math.pi / 3.0
                    dot_val = math.cos(v_angle - n_angle)
                    if dot_val > max_dot:
                        max_dot = dot_val
                
                extent = cx * nx + cy * ny + max_dot
                if extent > max_extent:
                    max_extent = extent
        
        return 2.0 * max_extent / sqrt3

    # Initialize with a valid hexagonal lattice configuration
    # Hex grid spacing for side 1 hexagons touching side-to-side is sqrt(3).
    
    def grid_to_cart(q, r):
        x = sqrt3 * q + sqrt3/2 * r
        y = 1.5 * r
        return (x, y)
    
    init_grid_coords = [
        (0,0),
        (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
        (2,0), (1,1), (0,2), (-1,2), (-2,1)
    ]
    
    centers = [grid_to_cart(q, r) for q, r in init_grid_coords]
    angles = [0.0] * N
    
    # Verify this initial config is valid
    if not check_overlap(centers, angles):
        # Fallback to a very sparse config if the grid is somehow invalid (shouldn't happen)
        centers = [(4.0 * k, 0.0) for k in range(-(N // 2), N - N // 2)]
        angles = [0.0] * N

    best_L = enclosing_side_length(centers, angles, 0.0)
    best_centers = [c[:] for c in centers]
    best_angles = angles[:]
    best_outer_angle = 0.0
    
    # Nelder-Mead Optimization
    def objective(params):
        if time.time() > deadline - 5.0:
            return 1e6 # Stop optimization
        
        # Unpack parameters: x0, y0, a0, x1, y1, a1, ..., x11, y11, a11, outer_angle
        # Total 37 parameters
        new_centers = []
        new_angles = []
        for i in range(N):
            new_centers.append((params[3*i], params[3*i+1]))
            new_angles.append(params[3*i+2])
        outer_angle = params[3*N]
        
        # Check overlap
        if not check_overlap(new_centers, new_angles):
            return 1e6
        
        # Compute side length
        L = enclosing_side_length(new_centers, new_angles, outer_angle)
        return L

    # Initial parameters
    init_params = []
    for i in range(N):
        init_params.append(centers[i][0])
        init_params.append(centers[i][1])
        init_params.append(angles[i])
    init_params.append(best_outer_angle)
    
    # Run Nelder-Mead
    try:
        result = minimize(objective, init_params, method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4, 'disp': False})
        
        if result.fun < best_L:
            best_L = result.fun
            for i in range(N):
                best_centers[i] = (result.x[3*i], result.x[3*i+1])
                best_angles[i] = result.x[3*i+2]
            best_outer_angle = result.x[3*N]
    except Exception:
        pass # Keep previous best if optimization fails

    # Final check and return
    if not check_overlap(best_centers, best_angles):
        # If best is invalid, use the initial valid one
        best_centers = [grid_to_cart(q, r) for q, r in init_grid_coords]
        best_angles = [0.0] * N
        best_L = enclosing_side_length(best_centers, best_angles, 0.0)
        best_outer_angle = 0.0

    inner = []
    for i in range(N):
        inner.append([best_centers[i][0], best_centers[i][1], best_angles[i]])
    
    center = (0.0, 0.0)
    side = best_L
    angle = best_outer_angle
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END

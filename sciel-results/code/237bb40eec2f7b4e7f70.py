import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    sqrt3 = np.sqrt(3.0)
    a1 = np.array([1.5, sqrt3/2])
    a2 = np.array([-1.5, sqrt3/2])
    
    # Multiple hexagonal cluster shapes for N=11
    shapes = [
        # 4-3-4
        [(-2,0), (-1,0), (0,0), (1,0), (-1,1), (0,1), (1,1), (-2,2), (-1,2), (0,2), (1,2)],
        # 3-5-3
        [(-1,0), (0,0), (1,0), (-2,1), (-1,1), (0,1), (1,1), (2,1), (-1,2), (0,2), (1,2)],
        # 1-3-4-3
        [(0,0), (-1,1), (0,1), (1,1), (-2,2), (-1,2), (0,2), (1,2), (-1,3), (0,3), (1,3)],
        # 3-4-4
        [(-1,0), (0,0), (1,0), (-2,1), (-1,1), (0,1), (1,1), (-1,2), (0,2), (1,2), (2,2)],
        # 2-4-5? Let's do 2-3-4-2
        [(-1,0), (0,0), (-2,1), (-1,1), (0,1), (1,1), (-1,2), (0,2), (1,2), (-1,3), (0,3)],
        # 4-4-3
        [(-2,0), (-1,0), (0,0), (1,0), (-1,1), (0,1), (1,1), (2,1), (-1,2), (0,2), (1,2)],
    ]
    
    # Hexagon vertices for unit hexagon with angle 0
    hex_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
    hex_vertices = np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])
    
    best_side = float('inf')
    best_inner = None
    best_center = None
    best_angle = 0.0
    
    for shape in shapes:
        centers = np.array([i*a1 + j*a2 for i,j in shape], dtype=float)
        com = np.mean(centers, axis=0)
        
        all_vertices = np.vstack([c + hex_vertices for c in centers])
        
        # Coarse search
        theta_vals = np.linspace(0, np.pi/3, 25)
        dx_range = np.linspace(-1.5, 1.5, 11)
        dy_range = np.linspace(-1.5, 1.5, 11)
        
        for theta in theta_vals:
            normal_angles = theta + np.arange(6) * np.pi/3
            normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
            
            for dx in dx_range:
                for dy in dy_range:
                    C = com + np.array([dx, dy])
                    diffs = all_vertices - C
                    projections = diffs @ normals.T
                    max_proj = np.max(projections)
                    s = max_proj / (sqrt3/2)
                    if s < best_side:
                        best_side = s
                        best_inner = centers.copy()
                        best_center = C.copy()
                        best_angle = theta
        
        if time.time() > deadline - 30:
            break
    
    # Refine around best
    if best_inner is not None and time.time() < deadline - 10:
        centers = best_inner
        all_vertices = np.vstack([c + hex_vertices for c in centers])
        com = best_center
        
        # Fine search
        theta_vals = np.linspace(best_angle - 0.05, best_angle + 0.05, 21)
        dx_range = np.linspace(-0.3, 0.3, 13)
        dy_range = np.linspace(-0.3, 0.3, 13)
        
        for theta in theta_vals:
            normal_angles = theta + np.arange(6) * np.pi/3
            normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
            
            for dx in dx_range:
                for dy in dy_range:
                    C = com + np.array([dx, dy])
                    diffs = all_vertices - C
                    projections = diffs @ normals.T
                    max_proj = np.max(projections)
                    s = max_proj / (sqrt3/2)
                    if s < best_side:
                        best_side = s
                        best_center = C.copy()
                        best_angle = theta
    
    inner = [[float(c[0]), float(c[1]), 0.0] for c in best_inner]
    center = (float(best_center[0]), float(best_center[1]))
    side = float(best_side)
    angle_deg = float(np.degrees(best_angle))
    
    return inner, center, side, angle_deg
# EVOLVE-BLOCK-END

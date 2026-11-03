import numpy as np
import time

N = 12

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    sqrt3 = np.sqrt(3)
    apothem_factor = np.sqrt(3) / 2  # apothem = side * apothem_factor
    
    def compute_min_side(all_verts, n_coarse=301, n_refine=201, n_refine_iters=3):
        """Find minimum outer hexagon side length and optimal angle."""
        best_side = float('inf')
        best_angle = 0.0
        for test_angle in np.linspace(0, 60, n_coarse):
            angle_rad = np.radians(test_angle)
            normal_angles = angle_rad + np.arange(6) * np.pi / 3 - np.pi / 6
            normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
            dots = np.abs(all_verts @ normals.T)
            max_dot = dots.max()
            s = max_dot / apothem_factor
            if s < best_side:
                best_side = s
                best_angle = test_angle
        for _ in range(n_refine_iters):
            local_best_angle = best_angle
            local_best_side = best_side
            for test_angle in np.linspace(best_angle - 2, best_angle + 2, n_refine):
                angle_rad = np.radians(test_angle)
                normal_angles = angle_rad + np.arange(6) * np.pi / 3 - np.pi / 6
                normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
                dots = np.abs(all_verts @ normals.T)
                max_dot = dots.max()
                s = max_dot / apothem_factor
                if s < local_best_side:
                    local_best_side = s
                    local_best_angle = test_angle
            best_angle = local_best_angle
            best_side = local_best_side
        return best_side, best_angle
    
    def make_arrangement(rows, cols, inner_angle_deg=0.0, stagger=True):
        positions = []
        for row in range(rows):
            for col in range(cols):
                x = col * 1.5
                y = row * sqrt3
                if stagger and row % 2 == 1:
                    x += 0.75
                positions.append([x, y, inner_angle_deg])
        xs = np.array([p[0] for p in positions])
        ys = np.array([p[1] for p in positions])
        cx = (xs.min() + xs.max()) / 2
        cy = (ys.min() + ys.max()) / 2
        for p in positions:
            p[0] -= cx
            p[1] -= cy
        all_verts = []
        for pos in positions:
            base_angle = np.radians(pos[2])
            angles = np.arange(6) * np.pi / 3 + base_angle
            verts = np.column_stack([pos[0] + np.cos(angles), pos[1] + np.sin(angles)])
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        return positions, all_verts
    
    best_overall_side = float('inf')
    best_overall_inner = None
    best_overall_angle = 0.0
    
    arrangements = [
        (4, 3, 0.0, True),
        (3, 4, 0.0, True),
        (6, 2, 0.0, True),
        (2, 6, 0.0, True),
        (4, 3, 30.0, True),
        (3, 4, 30.0, True),
        (4, 3, 0.0, False),
        (3, 4, 0.0, False),
    ]
    
    for rows, cols, angle, stagger in arrangements:
        if time.time() > deadline - 10:
            break
        positions, all_verts = make_arrangement(rows, cols, angle, stagger)
        side, ang = compute_min_side(all_verts)
        if side < best_overall_side:
            best_overall_side = side
            best_overall_inner = positions
            best_overall_angle = ang
    
    if best_overall_inner is None:
        positions, all_verts = make_arrangement(4, 3, 0.0, True)
        best_overall_inner = positions
        best_overall_side, best_overall_angle = compute_min_side(all_verts)
    
    return best_overall_inner, (0.0, 0.0), best_overall_side, best_overall_angle

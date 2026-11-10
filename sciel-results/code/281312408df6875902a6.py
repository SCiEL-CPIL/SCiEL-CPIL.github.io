# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from collections import deque
import time

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 600.0

    def _min_side_for_inner(inner_xy, outer_center, angle_deg):
        """Compute minimum side length of outer hexagon (at given center and angle)
        that contains all vertices of unit hexagons at inner_xy (angle 0)."""
        a = np.deg2rad(angle_deg)
        # 6 outward normals for outer hexagon edges
        n_angs = a + np.deg2rad(60.0 * np.arange(6) + 30.0)
        normals = np.column_stack([np.cos(n_angs), np.sin(n_angs)])  # (6,2)
        
        # Inner hexagon vertices (angle 0, unit side)
        v_angs = np.deg2rad(60.0 * np.arange(6))
        v_offsets = np.column_stack([np.cos(v_angs), np.sin(v_angs)])  # (6,2)
        
        # All inner vertices: (11, 6, 2)
        all_verts = inner_xy[:, np.newaxis, :] + v_offsets[np.newaxis, :, :]  # (11,6,2)
        all_verts = all_verts.reshape(-1, 2)  # (66, 2)
        
        # Shift by outer center
        all_verts = all_verts - outer_center  # (66, 2)
        
        # Max projection onto each normal: (66,2) @ (2,6) -> (66,6), then max over rows
        dots = all_verts @ normals  # (66, 6)
        max_dot = dots.max(axis=0)  # (6,)
        # Apothem = side * cos(30) = side * sqrt(3)/2
        # Need: max_dot <= side * sqrt(3)/2 for all edges
        # => side >= max(max_dot) / (sqrt(3)/2)
        side = max_dot.max() / (np.sqrt(3.0) / 2.0)
        return max(side, 0.0)

    def _separation_violation(inner_xy):
        """Return max violation for pair separation. Positive means overlap."""
        n = len(inner_xy)
        n_angs = np.deg2rad(60.0 * np.arange(6))
        normals = np.column_stack([np.cos(n_angs), np.sin(n_angs)])  # (6,2)
        max_viol = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                d = inner_xy[i] - inner_xy[j]
                proj_diffs = np.abs(d @ normals.T)  # (6,)
                min_proj = proj_diffs.min()
                viol = 2.0 - min_proj
                if viol > max_viol:
                    max_viol = viol
        return max_viol

    def _objective(params, outer_center, outer_angle):
        inner_xy = params.reshape(11, 2)
        viol = _separation_violation(inner_xy)
        if viol > 1e-6:
            return 1e6 + viol * 10
        side = _min_side_for_inner(inner_xy, outer_center, outer_angle)
        return side

    # Build initial cluster of 11 hexagons using hex lattice growth
    centers = [[0.0, 0.0]]
    used = set([(0.0, 0.0)])
    queue = deque([(0.0, 0.0)])
    dirs = np.array([[2.0, 0.0], [1.0, np.sqrt(3.0)], [-1.0, np.sqrt(3.0)],
                     [-2.0, 0.0], [-1.0, -np.sqrt(3.0)], [1.0, -np.sqrt(3.0)]])
    
    while len(centers) < 11 and queue:
        cx, cy = queue.popleft()
        for d in dirs:
            nx, ny = cx + d[0], cy + d[1]
            key = (round(nx, 8), round(ny, 8))
            if key not in used:
                used.add(key)
                centers.append([nx, ny])
                queue.append((nx, ny))
                if len(centers) == 11:
                    break
        if len(centers) == 11:
            break
    
    inner_xy0 = np.array(centers, dtype=float)
    
    # Try multiple outer angles and optimize
    best_side = np.inf
    best_inner = None
    best_outer_angle = 0.0
    outer_center = np.array([0.0, 0.0])
    
    for outer_angle in [0.0, 30.0, 15.0]:
        if time.time() > deadline - 60.0:
            break
        
        x0 = inner_xy0.flatten()
        try:
            res = minimize(
                _objective, x0,
                args=(outer_center, outer_angle),
                method='Nelder-Mead',
                options={'maxiter': 8000, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True}
            )
            side = res.fun
            if side < best_side and side < 1e5:
                best_side = side
                best_inner = res.x.reshape(11, 2)
                best_outer_angle = outer_angle
        except Exception:
            pass
    
    if best_inner is None:
        best_inner = inner_xy0
        best_outer_angle = 0.0
        best_side = _min_side_for_inner(best_inner, outer_center, best_outer_angle)
    
    # Final validation and fallback
    viol = _separation_violation(best_inner)
    if viol > 1e-6:
        # Use initial layout
        best_inner = inner_xy0
        best_outer_angle = 0.0
        best_side = _min_side_for_inner(best_inner, outer_center, best_outer_angle)
    
    # Recompute side
    best_side = _min_side_for_inner(best_inner, outer_center, best_outer_angle)
    
    # Ensure side >= sqrt(11)
    min_side = np.sqrt(11.0)
    if best_side < min_side:
        best_side = min_side
    
    inner_hex_data = np.zeros((11, 3))
    inner_hex_data[:, 0] = best_inner[:, 0]
    inner_hex_data[:, 1] = best_inner[:, 1]
    inner_hex_data[:, 2] = 0.0
    
    outer_hex_data = np.array([0.0, 0.0, best_outer_angle])
    
    return inner_hex_data, outer_hex_data, float(best_side)

# EVOLVE-BLOCK-END

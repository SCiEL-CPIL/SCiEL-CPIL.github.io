"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import numpy as np
from scipy.optimize import minimize
import time
import random

N = 12

def _hex_vertices(cx, cy, s, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy) with side s, rotated angle_deg."""
    a = np.radians(angle_deg)
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180 + a
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _support_hexagon(cx, cy, s, angle_deg, direction):
    """Support function of a regular hexagon in a given direction (unit vector)."""
    a = np.radians(angle_deg)
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180 + a
    vx = s * np.cos(angles)
    vy = s * np.sin(angles)
    proj = vx * direction[0] + vy * direction[1]
    return cx * direction[0] + cy * direction[1] + np.max(proj)

def _hex_separated(p1, p2, tol=1e-9):
    """Check if two unit hexagons are separated (non-overlapping) using SAT."""
    x1, y1, a1 = p1
    x2, y2, a2 = p2
    
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
    v1_rel = np.column_stack([np.cos(base_angles + np.radians(a1)), 
                               np.sin(base_angles + np.radians(a1))])
    v2_rel = np.column_stack([np.cos(base_angles + np.radians(a2)), 
                               np.sin(base_angles + np.radians(a2))])
    
    v1 = v1_rel + np.array([x1, y1])
    v2 = v2_rel + np.array([x2, y2])
    
    for v in [v1, v2]:
        for i in range(6):
            ex = v[(i+1)%6, 0] - v[i, 0]
            ey = v[(i+1)%6, 1] - v[i, 1]
            length = np.sqrt(ex*ex + ey*ey)
            if length < 1e-12:
                continue
            nx = -ey / length
            ny = ex / length
            
            proj1 = v1 @ np.array([nx, ny])
            proj2 = v2 @ np.array([nx, ny])
            
            if np.max(proj1) < np.min(proj2) + tol or np.max(proj2) < np.min(proj1) + tol:
                return True
    
    return False

def _polygons_overlap(p1, p2):
    """Check if two convex polygons overlap using SAT."""
    return not _hex_separated(p1, p2)

def _compute_min_side(inner, outer_center, outer_angle):
    """Compute minimum outer side length for given inner config and outer center/angle."""
    cx, cy = outer_center
    theta = np.radians(outer_angle)
    face_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180 + theta
    normals = np.column_stack([np.cos(face_angles), np.sin(face_angles)])
    
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
    
    min_s = 0.0
    for ix, iy, ia in inner:
        ia_rad = np.radians(ia)
        v_rel = np.column_stack([np.cos(base_angles + ia_rad), 
                                  np.sin(base_angles + ia_rad)])
        v_abs = v_rel + np.array([ix, iy])
        
        for k in range(6):
            dots = (v_abs[:, 0] - cx) * normals[k, 0] + (v_abs[:, 1] - cy) * normals[k, 1]
            max_dot = np.max(dots)
            if max_dot > 0:
                s_needed = max_dot / (np.sqrt(3) / 2)
                min_s = max(min_s, s_needed)
    
    return min_s

def _overlap_penalty(inner_pts, tol=1e-6):
    """Compute total overlap penalty between all pairs of inner hexagons."""
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            if not _hex_separated(inner_pts[i], inner_pts[j], tol):
                # Estimate overlap depth using center distance
                dx = inner_pts[i][0] - inner_pts[j][0]
                dy = inner_pts[i][1] - inner_pts[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                # For touching hexagons, center distance should be >= 2*sqrt(3)
                min_dist = 2 * np.sqrt(3)
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
    return penalty

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_score = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_side = 40.0
    best_angle = 0.0
    
    np.random.seed(42)
    random.seed(42)
    
    def make_hex_lattice(rows, cols, rotation_deg=0.0):
        spacing = 2 * np.sqrt(3)
        pts = []
        for r in range(rows):
            for c in range(cols):
                x = c * spacing
                y = r * spacing * 1.5
                if r % 2 == 1:
                    x += spacing / 2
                pts.append([x, y, rotation_deg])
        return pts
    
    def center_config(config):
        xs = [p[0] for p in config]
        ys = [p[1] for p in config]
        cx0 = (min(xs) + max(xs)) / 2
        cy0 = (min(ys) + max(ys)) / 2
        return [[p[0] - cx0, p[1] - cy0, p[2]] for p in config]
    
    def is_valid_config(inner_pts):
        for i in range(N):
            for j in range(i+1, N):
                if not _hex_separated(inner_pts[i], inner_pts[j]):
                    return False
        return True
    
    def contact_graph_sa(config, max_iter=5000, t_start=1.0, t_end=0.01):
        """Simulated annealing that evolves contact graph topology."""
        current = [list(p) for p in config]
        current_penalty = _overlap_penalty(current)
        current_side = _compute_min_side(current, (0.0, 0.0), 0.0)
        current_obj = current_side + 100 * current_penalty
        
        best = [list(p) for p in current]
        best_obj = current_obj
        
        t = t_start
        cooling = (t_end / t_start) ** (1.0 / max_iter)
        
        for it in range(max_iter):
            if time.time() > deadline - 2.0:
                break
            
            t *= cooling
            
            # Pick a random hexagon to move
            idx = random.randint(0, N-1)
            old_pos = list(current[idx])
            
            # Random perturbation
            dx = random.gauss(0, 0.3)
            dy = random.gauss(0, 0.3)
            da = random.gauss(0, 10)
            
            current[idx] = [old_pos[0] + dx, old_pos[1] + dy, old_pos[2] + da]
            
            new_penalty = _overlap_penalty(current)
            new_side = _compute_min_side(current, (0.0, 0.0), 0.0)
            new_obj = new_side + 100 * new_penalty
            
            delta = new_obj - current_obj
            if delta < 0 or random.random() < np.exp(-delta / max(t, 1e-10)):
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best = [list(p) for p in current]
            else:
                current[idx] = old_pos
        
        return best
    
    def nelder_mead_refine(config, max_iter=2000):
        """Nelder-Mead refinement of a configuration."""
        def objective(x):
            if time.time() > deadline - 1.0:
                return 1e10
            inner_pts = []
            for i in range(N):
                ix = x[3*i]
                iy = x[3*i+1]
                ia = np.degrees(x[3*i+2])
                inner_pts.append([ix, iy, ia])
            
            penalty = _overlap_penalty(inner_pts)
            s = _compute_min_side(inner_pts, (0.0, 0.0), 0.0)
            return s + 100 * penalty
        
        x0 = []
        for p in config:
            x0.extend([p[0], p[1], np.radians(p[2])])
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                            options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-6,
                                   'adaptive': True})
            if result.fun < 1e7:
                inner_pts = []
                for i in range(N):
                    ix = result.x[3*i]
                    iy = result.x[3*i+1]
                    ia = np.degrees(result.x[3*i+2])
                    inner_pts.append([ix, iy, ia])
                return inner_pts
        except Exception:
            pass
        return None
    
    # Generate diverse initial configurations
    configs = []
    configs.append(center_config(make_hex_lattice(3, 4)))
    configs.append(center_config(make_hex_lattice(4, 3)))
    configs.append(center_config(make_hex_lattice(2, 6)))
    configs.append(center_config(make_hex_lattice(6, 2)))
    configs.append(center_config(make_hex_lattice(1, 12)))
    configs.append(center_config(make_hex_lattice(12, 1)))
    
    # Add ring + center configurations
    for radius in [1, 2]:
        ring_config = [[0.0, 0.0, 0.0]]
        spacing = 2 * np.sqrt(3) * radius
        for k in range(6):
            angle = k * 60
            x = spacing * np.cos(np.radians(angle))
            y = spacing * np.sin(np.radians(angle))
            ring_config.append([x, y, 0.0])
        if len(ring_config) >= N:
            configs.append(center_config(ring_config[:N]))
    
    # Process each configuration
    for config_idx, config in enumerate(configs):
        if time.time() > deadline - 30.0:
            break
        
        # Step 1: Contact graph simulated annealing
        sa_result = contact_graph_sa(config, max_iter=3000)
        if sa_result is not None:
            sa_obj = _compute_min_side(sa_result, (0.0, 0.0), 0.0) + 100 * _overlap_penalty(sa_result)
            if sa_obj < 1e7:
                # Step 2: Nelder-Mead refinement
                nm_result = nelder_mead_refine(sa_result, max_iter=1500)
                if nm_result is not None:
                    nm_obj = _compute_min_side(nm_result, (0.0, 0.0), 0.0) + 100 * _overlap_penalty(nm_result)
                    if nm_obj < sa_obj:
                        sa_result = nm_result
                
                # Validate and update best
                if is_valid_config(sa_result):
                    s_final = _compute_min_side(sa_result, (0.0, 0.0), 0.0)
                    if s_final < best_score:
                        best_score = s_final
                        best_inner = [list(p) for p in sa_result]
                        best_center = (0.0, 0.0)
                        best_side = s_final
                        best_angle = 0.0
        
        # Also try direct NM on original config
        if time.time() < deadline - 20.0:
            nm_direct = nelder_mead_refine(config, max_iter=1000)
            if nm_direct is not None and is_valid_config(nm_direct):
                s_direct = _compute_min_side(nm_direct, (0.0, 0.0), 0.0)
                if s_direct < best_score:
                    best_score = s_direct
                    best_inner = [list(p) for p in nm_direct]
                    best_center = (0.0, 0.0)
                    best_side = s_direct
                    best_angle = 0.0
    
    # Final polish: optimize outer angle and center
    if best_inner is not None and time.time() < deadline - 10.0:
        def outer_objective(x):
            ox, oy, oangle = x
            s = _compute_min_side(best_inner, (ox, oy), np.degrees(oangle))
            return s
        
        x0_outer = [0.0, 0.0, 0.0]
        try:
            result_outer = minimize(outer_objective, x0_outer, method='Nelder-Mead',
                                   options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if result_outer.fun < best_score:
                ox, oy, oangle = result_outer.x
                s_final = _compute_min_side(best_inner, (ox, oy), np.degrees(oangle))
                
                if s_final < best_score:
                    best_score = s_final
                    best_center = (ox, oy)
                    best_side = s_final
                    best_angle = np.degrees(oangle)
        except Exception:
            pass
    
    # If we found something, return it
    if best_inner is not None:
        return best_inner, best_center, best_side, best_angle
    
    # Fallback
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0

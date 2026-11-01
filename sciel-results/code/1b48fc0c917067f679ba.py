"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time
import random

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        dx = px - cx
        dy = py - cy
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a
        
        apothem = side * np.sqrt(3) / 2
        normal_angles = np.radians(np.arange(30, 360, 60))
        
        for na in normal_angles:
            nx = np.cos(na)
            ny = np.sin(na)
            dist = local_x * nx + local_y * ny
            if dist > apothem + 1e-9:
                return False
        return True

    def polygons_disjoint_sat(v1, v2):
        """
        Check if two convex polygons are disjoint using Separating Axis Theorem.
        v1, v2: Nx2 arrays of vertices.
        """
        axes = []
        for i in range(len(v1)):
            p1 = v1[i]
            p2 = v1[(i + 1) % len(v1)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) > 1e-9:
                axes.append(normal / np.linalg.norm(normal))
        
        for i in range(len(v2)):
            p1 = v2[i]
            p2 = v2[(i + 1) % len(v2)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) > 1e-9:
                axes.append(normal / np.linalg.norm(normal))
        
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for gap
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return True
        return False

    def hexagons_disjoint(v1, s1, a1, v2, s2, a2):
        verts1 = hexagon_vertices(v1[0], v1[1], s1, a1)
        verts2 = hexagon_vertices(v2[0], v2[1], s2, a2)
        return polygons_disjoint_sat(verts1, verts2)

    def all_inside_outer(inner_positions, outer_center, outer_side, outer_angle):
        for i in range(len(inner_positions)):
            cx, cy, angle = inner_positions[i]
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            for vx, vy in verts:
                if not point_in_hexagon(vx, vy, outer_center[0], outer_center[1], outer_side, outer_angle):
                    return False
        return True

    def pairwise_disjoint(inner_positions):
        for i in range(len(inner_positions)):
            for j in range(i+1, len(inner_positions)):
                ci = (inner_positions[i][0], inner_positions[i][1])
                cj = (inner_positions[j][0], inner_positions[j][1])
                ai = inner_positions[i][2]
                aj = inner_positions[j][2]
                if not hexagons_disjoint(ci, 1.0, ai, cj, 1.0, aj):
                    return False
        return True

    def objective(params):
        n_inner = N
        inner_positions = []
        for i in range(n_inner):
            x = params[3*i]
            y = params[3*i+1]
            a = params[3*i+2]
            inner_positions.append((x, y, a))
        
        ox = params[3*n_inner]
        oy = params[3*n_inner+1]
        os = params[3*n_inner+2]
        oa = params[3*n_inner+3]
        
        outer_center = (ox, oy)
        outer_side = max(os, 0.1)
        outer_angle = oa
        
        inside_ok = all_inside_outer(inner_positions, outer_center, outer_side, outer_angle)
        disjoint_ok = pairwise_disjoint(inner_positions)
        
        if inside_ok and disjoint_ok:
            return outer_side
        
        penalty = 0.0
        if not inside_ok:
            for i in range(n_inner):
                cx, cy, angle = inner_positions[i]
                verts = hexagon_vertices(cx, cy, 1.0, angle)
                for vx, vy in verts:
                    if not point_in_hexagon(vx, vy, outer_center[0], outer_center[1], outer_side, outer_angle):
                        penalty += 1.0
        if not disjoint_ok:
            for i in range(n_inner):
                for j in range(i+1, n_inner):
                    ci = (inner_positions[i][0], inner_positions[i][1])
                    cj = (inner_positions[j][0], inner_positions[j][1])
                    ai = inner_positions[i][2]
                    aj = inner_positions[j][2]
                    if not hexagons_disjoint(ci, 1.0, ai, cj, 1.0, aj):
                        penalty += 1.0
                        
        return outer_side + penalty * 10.0

    # Generate multiple diverse initial configurations
    def make_grid_config():
        """4x3 grid configuration"""
        positions = []
        x_coords = [-3.0, -1.0, 1.0, 3.0]
        y_coords = [-1.8, 0.0, 1.8]
        for y in y_coords:
            for x in x_coords:
                positions.append([x, y, 0.0])
        return positions, [0.0, 0.0], 5.0, 30.0

    def make_hex_cluster_config():
        """Hexagonal cluster: 1 center + 6 ring + 5 more"""
        positions = []
        # Center
        positions.append([0.0, 0.0, 0.0])
        # First ring (6 hexagons)
        for i in range(6):
            angle = np.radians(60 * i)
            x = 2.0 * np.cos(angle)
            y = 2.0 * np.sin(angle)
            positions.append([x, y, 0.0])
        # Second ring partial (5 more)
        for i in range(5):
            angle = np.radians(60 * i + 30)
            x = 3.5 * np.cos(angle)
            y = 3.5 * np.sin(angle)
            positions.append([x, y, 0.0])
        return positions, [0.0, 0.0], 5.5, 30.0

    def make_random_config():
        """Random scattered configuration"""
        positions = []
        for _ in range(N):
            x = random.uniform(-4, 4)
            y = random.uniform(-4, 4)
            a = random.uniform(0, 60)
            positions.append([x, y, a])
        return positions, [0.0, 0.0], 6.0, 30.0

    def make_staggered_config():
        """Staggered grid with hexagonal symmetry"""
        positions = []
        # Row 0: 4 hexagons
        for j in range(4):
            x = (j - 1.5) * 2.0
            y = -1.8
            positions.append([x, y, 0.0])
        # Row 1: 3 hexagons offset
        for j in range(3):
            x = (j - 1.0) * 2.0
            y = 0.0
            positions.append([x, y, 0.0])
        # Row 2: 4 hexagons
        for j in range(4):
            x = (j - 1.5) * 2.0
            y = 1.8
            positions.append([x, y, 0.0])
        return positions, [0.0, 0.0], 5.0, 30.0

    # Try multiple initial configurations and pick the best
    configs = [
        make_grid_config(),
        make_hex_cluster_config(),
        make_staggered_config(),
        make_random_config()
    ]
    
    best_params = None
    best_score = float('inf')
    
    for positions, oc, os_, oa in configs:
        params = []
        for pos in positions:
            params.extend([pos[0], pos[1], pos[2]])
        params.extend([oc[0], oc[1], os_, oa])
        
        # Validate
        inner_init = [(p[0], p[1], p[2]) for p in positions]
        if all_inside_outer(inner_init, (oc[0], oc[1]), os_, oa) and pairwise_disjoint(inner_init):
            score = objective(params)
            if score < best_score:
                best_score = score
                best_params = params.copy()
    
    # If no valid config found, use grid with larger outer
    if best_params is None:
        positions, oc, os_, oa = make_grid_config()
        os_ = 6.0
        params = []
        for pos in positions:
            params.extend([pos[0], pos[1], pos[2]])
        params.extend([oc[0], oc[1], os_, oa])
        best_params = params
        best_score = objective(params)
    
    # Phase 1: Simulated Annealing for broad exploration
    if time.time() < deadline - 30:
        current_params = best_params.copy()
        current_score = best_score
        T = 1.0
        cooling_rate = 0.995
        max_iters = 3000
        
        for it in range(max_iters):
            if time.time() > deadline - 30:
                break
            
            # Random perturbation
            idx = random.randint(0, 3*N + 3)
            if idx < 3*N:
                # Perturb inner hexagon
                dim = idx % 3
                if dim < 2:
                    current_params[idx] += random.gauss(0, 0.5)
                else:
                    current_params[idx] += random.gauss(0, 10)
            elif idx < 3*N + 2:
                # Perturb outer center
                current_params[idx] += random.gauss(0, 0.3)
            elif idx == 3*N + 2:
                # Perturb outer side
                current_params[idx] += random.gauss(0, 0.2)
            else:
                # Perturb outer angle
                current_params[idx] += random.gauss(0, 5)
            
            new_score = objective(current_params)
            delta = new_score - current_score
            
            if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-6)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_params = current_params.copy()
            
            T *= cooling_rate
    
    # Phase 2: Nelder-Mead local refinement
    if time.time() < deadline - 10:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead', 
                          options={'maxiter': 1000, 'xatol': 0.005, 'fatol': 0.005, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Phase 3: Fine-tuning with smaller steps
    if time.time() < deadline - 5:
        try:
            res2 = minimize(objective, best_params, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 0.001, 'fatol': 0.001, 'adaptive': True})
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x
        except Exception:
            pass
    
    final_params = best_params
    
    inner = []
    for i in range(N):
        x = final_params[3*i]
        y = final_params[3*i+1]
        a = final_params[3*i+2]
        inner.append([x, y, a])
    
    ox = final_params[3*N]
    oy = final_params[3*N+1]
    os = max(final_params[3*N+2], 0.1)
    oa = final_params[3*N+3]
    
    # Final validation
    if not (all_inside_outer(inner, (ox, oy), os, oa) and pairwise_disjoint(inner)):
        # Fallback: expand outer side until valid
        for _ in range(100):
            os += 0.1
            if all_inside_outer(inner, (ox, oy), os, oa) and pairwise_disjoint(inner):
                break
    
    return inner, (ox, oy), os, oa
# EVOLVE-BLOCK-END

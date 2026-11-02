"""Improved packing: 12 unit hexagons in minimal outer hexagon using simulated annealing."""
import numpy as np
import time
import random

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    HEX_ANGLES = np.deg2rad(np.arange(0, 360, 60))
    HEX_NORMALS = np.array([
        [np.cos(np.deg2rad(30 + k * 60)), np.sin(np.deg2rad(30 + k * 60))]
        for k in range(6)
    ])
    APOTHEM = np.sqrt(3) / 2
    
    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + HEX_ANGLES
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])
    
    def point_in_hexagon(pts, cx, cy, side, angle_deg):
        """Check if points (n,2) are inside hexagon. Returns boolean array."""
        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a
        apothem = side * APOTHEM
        dists = local_x * HEX_NORMALS[:, 0] + local_y * HEX_NORMALS[:, 1]
        return np.all(dists <= apothem + 1e-9, axis=1)
    
    def sat_separated(v1, v2, eps=1e-9):
        """SAT check: returns True if polygons are separated."""
        edges1 = np.roll(v1, -1, axis=0) - v1
        edges2 = np.roll(v2, -1, axis=0) - v2
        normals1 = np.column_stack([-edges1[:, 1], edges1[:, 0]])
        normals2 = np.column_stack([-edges2[:, 1], edges2[:, 0]])
        
        norms1 = np.linalg.norm(normals1, axis=1)
        norms2 = np.linalg.norm(normals2, axis=1)
        
        all_normals = np.vstack([
            normals1[norms1 > 1e-9] / norms1[norms1 > 1e-9, None],
            normals2[norms2 > 1e-9] / norms2[norms2 > 1e-9, None]
        ])
        
        for axis in all_normals:
            proj1 = v1 @ axis
            proj2 = v2 @ axis
            if np.max(proj1) < np.min(proj2) - eps or np.max(proj2) < np.min(proj1) - eps:
                return True
        return False
    
    def compute_outer_side(inner_positions, outer_center, outer_angle_deg):
        """Compute minimal side length of outer hexagon to contain all inner hexagons."""
        all_verts = []
        for cx, cy, angle in inner_positions:
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        
        # Translate so outer center is at origin
        all_verts = all_verts - np.array(outer_center)
        
        angle_rad = np.radians(outer_angle_deg)
        normal_angles = angle_rad + np.deg2rad(np.arange(30, 360, 60))
        
        max_side = 0.0
        for na in normal_angles:
            nx = np.cos(na)
            ny = np.sin(na)
            dists = all_verts[:, 0] * nx + all_verts[:, 1] * ny
            max_dist = np.max(np.abs(dists))
            if max_dist > 0:
                needed_side = max_dist / APOTHEM
                max_side = max(max_side, needed_side)
        
        return max_side
    
    def make_hex_cluster():
        """Hexagonal cluster: 1 center + 6 first ring + 5 from second ring."""
        positions = [(0.0, 0.0, 0.0)]
        for k in range(6):
            angle = np.radians(60 * k)
            x = 2.0 * np.cos(angle)
            y = 2.0 * np.sin(angle)
            positions.append((x, y, 0.0))
        
        sqrt3 = np.sqrt(3)
        second_ring = [
            (4.0, 0.0), (3.0, sqrt3), (1.0, 2*sqrt3), (-1.0, 2*sqrt3),
            (-3.0, sqrt3), (-4.0, 0.0), (-3.0, -sqrt3), (-1.0, -2*sqrt3),
            (1.0, -2*sqrt3), (3.0, -sqrt3)
        ]
        
        best_combo = None
        best_side = float('inf')
        from itertools import combinations
        for combo in combinations(range(10), 5):
            test = positions.copy()
            for idx in combo:
                x, y = second_ring[idx]
                test.append((x, y, 0.0))
            side = compute_outer_side(test, (0.0, 0.0), 0.0)
            if side < best_side:
                best_side = side
                best_combo = combo
        
        for idx in best_combo:
            x, y = second_ring[idx]
            positions.append((x, y, 0.0))
        return positions
    
    def make_staggered():
        """Staggered rows configuration."""
        positions = []
        sqrt3 = np.sqrt(3)
        for row in range(4):
            y_off = row * 2 * sqrt3 - 3 * sqrt3
            x_off = (row % 2) * 2.0 - 4.0
            for col in range(3):
                x = col * 4.0 + x_off
                y = y_off
                positions.append((x, y, 0.0))
        return positions
    
    def make_grid():
        """Grid configuration."""
        positions = []
        sqrt3 = np.sqrt(3)
        for i in range(3):
            for j in range(4):
                x = j * 4.0 - 6.0
                y = i * 2 * sqrt3 - 2 * sqrt3
                positions.append((x, y, 0.0))
        return positions
    
    # Generate initial configurations
    configs = [make_hex_cluster(), make_staggered(), make_grid()]
    
    best_config = None
    best_side = float('inf')
    for cfg in configs:
        for oa in [0.0, 15.0, 30.0, 45.0, 60.0]:
            side = compute_outer_side(cfg, (0.0, 0.0), oa)
            if side < best_side:
                best_side = side
                best_config = (cfg, (0.0, 0.0), oa)
    
    inner_init, oc_init, oa_init = best_config
    
    # Simulated annealing
    best_inner = [p[:] for p in inner_init]
    best_oc = oc_init[:]
    best_oa = oa_init
    best_score = best_side
    
    def evaluate(inner, oc, oa):
        """Compute outer side length, with penalty for invalid configurations."""
        side = compute_outer_side(inner, oc, oa)
        
        # Check if all inner hexagons are inside outer
        outer_verts_check = True
        for cx, cy, angle in inner:
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            if not np.all(point_in_hexagon(verts, oc[0], oc[1], side, oa)):
                outer_verts_check = False
                break
        
        # Check pairwise disjointness
        disjoint_check = True
        verts_list = [hexagon_vertices(p[0], p[1], 1.0, p[2]) for p in inner]
        for i in range(N):
            for j in range(i + 1, N):
                if not sat_separated(verts_list[i], verts_list[j]):
                    disjoint_check = False
                    break
            if not disjoint_check:
                break
        
        penalty = 0.0
        if not outer_verts_check:
            penalty += 10.0
        if not disjoint_check:
            penalty += 10.0
        
        return side + penalty
    
    current_inner = [p[:] for p in inner_init]
    current_oc = oc_init[:]
    current_oa = oa_init
    current_score = evaluate(current_inner, current_oc, current_oa)
    
    # Simulated annealing parameters
    T_start = 2.0
    T_end = 0.01
    n_iterations = 8000
    cooling = (T_end / T_start) ** (1.0 / n_iterations)
    
    for it in range(n_iterations):
        if time.time() > deadline - 10:
            break
        
        T = T_start * (cooling ** it)
        
        # Generate perturbation
        new_inner = [p[:] for p in current_inner]
        new_oc = current_oc[:]
        new_oa = current_oa
        
        # Perturb one or two hexagons
        n_perturb = random.randint(1, 2)
        for _ in range(n_perturb):
            idx = random.randint(0, N - 1)
            new_inner[idx][0] += random.gauss(0, 0.3)
            new_inner[idx][1] += random.gauss(0, 0.3)
            new_inner[idx][2] += random.gauss(0, 15)
        
        # Sometimes perturb outer center/angle
        if random.random() < 0.3:
            new_oc[0] += random.gauss(0, 0.2)
            new_oc[1] += random.gauss(0, 0.2)
        if random.random() < 0.2:
            new_oa += random.gauss(0, 5)
        
        new_score = evaluate(new_inner, new_oc, new_oa)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
            current_inner = new_inner
            current_oc = new_oc
            current_oa = new_oa
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_inner = [p[:] for p in current_inner]
                best_oc = current_oc[:]
                best_oa = current_oa
    
    # Final polish with smaller steps
    if time.time() < deadline - 15:
        T_start2 = 0.05
        T_end2 = 0.001
        n_iter2 = 3000
        cooling2 = (T_end2 / T_start2) ** (1.0 / n_iter2)
        
        for it in range(n_iter2):
            if time.time() > deadline - 5:
                break
            
            T = T_start2 * (cooling2 ** it)
            
            new_inner = [p[:] for p in best_inner]
            new_oc = best_oc[:]
            new_oa = best_oa
            
            idx = random.randint(0, N - 1)
            new_inner[idx][0] += random.gauss(0, 0.05)
            new_inner[idx][1] += random.gauss(0, 0.05)
            new_inner[idx][2] += random.gauss(0, 3)
            
            if random.random() < 0.2:
                new_oc[0] += random.gauss(0, 0.03)
                new_oc[1] += random.gauss(0, 0.03)
            if random.random() < 0.1:
                new_oa += random.gauss(0, 1)
            
            new_score = evaluate(new_inner, new_oc, new_oa)
            
            delta = new_score - best_score
            if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
                best_score = new_score
                best_inner = new_inner
                best_oc = new_oc
                best_oa = new_oa
    
    # Final validation and fallback
    final_inner = best_inner
    final_oc = best_oc
    final_oa = best_oa
    final_side = compute_outer_side(final_inner, final_oc, final_oa)
    
    # Verify validity
    valid = True
    outer_verts_check = True
    for cx, cy, angle in final_inner:
        verts = hexagon_vertices(cx, cy, 1.0, angle)
        if not np.all(point_in_hexagon(verts, final_oc[0], final_oc[1], final_side, final_oa)):
            outer_verts_check = False
            break
    
    if outer_verts_check:
        verts_list = [hexagon_vertices(p[0], p[1], 1.0, p[2]) for p in final_inner]
        for i in range(N):
            for j in range(i + 1, N):
                if not sat_separated(verts_list[i], verts_list[j]):
                    valid = False
                    break
            if not valid:
                break
    else:
        valid = False
    
    if not valid:
        # Fall back to initial configuration
        final_inner = inner_init
        final_oc = oc_init
        final_oa = oa_init
        final_side = compute_outer_side(final_inner, final_oc, final_oa)
        # Expand outer if needed
        while True:
            outer_verts_check = True
            for cx, cy, angle in final_inner:
                verts = hexagon_vertices(cx, cy, 1.0, angle)
                if not np.all(point_in_hexagon(verts, final_oc[0], final_oc[1], final_side, final_oa)):
                    outer_verts_check = False
                    break
            if outer_verts_check:
                break
            final_side += 0.05
            if final_side > 15:
                break
    
    return final_inner, final_oc, final_side, final_oa
# EVOLVE-BLOCK-END

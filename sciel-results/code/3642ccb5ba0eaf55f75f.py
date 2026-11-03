"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Helper: get vertices of a regular hexagon with side length 1
    def hex_vertices(cx, cy, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            x = cx + math.cos(a)
            y = cy + math.sin(a)
            verts.append((x, y))
        return np.array(verts)
    
    # Helper: check if two hexagons overlap using SAT (Separating Axis Theorem)
    def hexagons_overlap(v1, v2):
        """Check if two convex hexagons overlap using SAT."""
        axes = []
        for i in range(6):
            edge = v1[i] - v1[(i+1) % 6]
            normal = np.array([-edge[1], edge[0]])
            axes.append(normal)
            edge2 = v2[i] - v2[(i+1) % 6]
            normal2 = np.array([-edge2[1], edge2[0]])
            axes.append(normal2)
        
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            if max1 < min2 or max2 < min1:
                return False
        return True
        
    def check_all_overlaps(positions, angles):
        """Check if any pair of hexagons overlaps."""
        verts = [hex_vertices(p[0], p[1], a) for p, a in zip(positions, angles)]
        for i in range(N):
            for j in range(i+1, N):
                if hexagons_overlap(verts[i], verts[j]):
                    return True
        return False

    # Helper: compute minimum outer hexagon side length needed to contain all inner hexagons
    def compute_min_side(positions, angles, outer_angle_deg):
        """
        positions: list of (x, y) for each inner hexagon
        angles: list of rotation angles for each inner hexagon
        outer_angle_deg: rotation of outer hexagon
        Returns the minimum side length of outer hexagon to contain all inner hexagons.
        """
        all_verts = []
        for i in range(N):
            cx, cy = positions[i]
            a = angles[i]
            verts = hex_vertices(cx, cy, a)
            all_verts.append(verts)
        
        if not all_verts:
            return 0.0
            
        all_verts = np.vstack(all_verts)
        
        theta = math.radians(outer_angle_deg)
        normals = []
        for k in range(6):
            a = theta + k * math.pi / 3
            normals.append(np.array([math.cos(a), math.sin(a)]))
        
        max_proj = 0
        for normal in normals:
            proj = np.dot(all_verts, normal)
            max_proj = max(max_proj, proj.max())
        
        side = 2 * max_proj / math.sqrt(3)
        return side

    def get_config_score(positions, angles, outer_angle):
        """Return side length if valid, else infinity."""
        if check_all_overlaps(positions, angles):
            return float('inf')
        return compute_min_side(positions, angles, outer_angle)

    def make_honeycomb(rows, cols, angle_deg=0.0, outer_angle=0.0):
        """Generate a honeycomb grid configuration."""
        positions = []
        angles = []
        for r in range(rows):
            for c in range(cols):
                x = c * math.sqrt(3) + (r % 2) * math.sqrt(3) / 2
                y = r * 1.5
                positions.append((x, y))
                angles.append(angle_deg)
        
        if not positions:
            return [], [], outer_angle
        cx = sum(p[0] for p in positions) / len(positions)
        cy = sum(p[1] for p in positions) / len(positions)
        positions = [(x - cx, y - cy) for x, y in positions]
        
        return positions, angles, outer_angle

    def make_circular(radius, angle_deg=0.0, outer_angle=0.0):
        """Generate a circular arrangement."""
        positions = []
        angles = []
        for i in range(N):
            a = 2 * math.pi * i / N
            x = radius * math.cos(a)
            y = radius * math.sin(a)
            positions.append((x, y))
            angles.append(angle_deg)
        return positions, angles, outer_angle

    def physics_optimize(positions, angles, outer_angle, max_iters=2000, tol=1e-6):
        """
        Physics-based optimization: use repulsive forces between overlapping hexagons
        and attractive forces toward center to find a valid packing with minimal spread.
        """
        pos = np.array(positions, dtype=float)
        ang = np.array(angles, dtype=float)
        
        # Track best configuration
        best_pos = pos.copy()
        best_ang = ang.copy()
        best_score = get_config_score(list(map(tuple, pos)), list(ang), outer_angle)
        
        # Precompute initial overlaps
        verts_list = [hex_vertices(pos[i][0], pos[i][1], ang[i]) for i in range(N)]
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Compute forces
            forces = np.zeros_like(pos)
            
            # Repulsive forces between overlapping pairs
            for i in range(N):
                for j in range(i+1, N):
                    if hexagons_overlap(verts_list[i], verts_list[j]):
                        # Compute penetration depth approximation using center distance
                        d = pos[j] - pos[i]
                        dist = np.linalg.norm(d)
                        if dist < 1e-6:
                            d = np.random.randn(2) * 0.1
                            dist = np.linalg.norm(d)
                        d_norm = d / dist
                        # Strong repulsion
                        rep = 5.0 * d_norm
                        forces[i] -= rep
                        forces[j] += rep
            
            # Attractive force toward center (to minimize spread)
            center = pos.mean(axis=0)
            for i in range(N):
                forces[i] -= 0.5 * (pos[i] - center)
            
            # Adaptive step size
            step = 0.1
            new_pos = pos + step * forces
            
            # Check if new configuration is valid
            new_verts_list = [hex_vertices(new_pos[i][0], new_pos[i][1], ang[i]) for i in range(N)]
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    if hexagons_overlap(new_verts_list[i], new_verts_list[j]):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                pos = new_pos
                verts_list = new_verts_list
                score = get_config_score(list(map(tuple, pos)), list(ang), outer_angle)
                if score < best_score:
                    best_score = score
                    best_pos = pos.copy()
                    best_ang = ang.copy()
            else:
                # Reduce step size
                step *= 0.5
                if step < 1e-8:
                    break
        
        return list(map(tuple, best_pos)), list(best_ang), outer_angle, best_score

    def nelder_mead_optimize(positions, angles, outer_angle, max_iter=500):
        """
        Use Nelder-Mead to optimize positions for minimal outer hexagon side length,
        with penalty for overlaps.
        """
        # Flatten positions to 2D array
        pos_arr = np.array(positions, dtype=float).flatten()
        
        def objective(x):
            if time.time() > deadline - 5:
                return 1e10
            pos = x.reshape(N, 2)
            # Check overlaps
            verts_list = [hex_vertices(pos[i][0], pos[i][1], angles[i]) for i in range(N)]
            overlap_penalty = 0
            for i in range(N):
                for j in range(i+1, N):
                    if hexagons_overlap(verts_list[i], verts_list[j]):
                        overlap_penalty += 10.0
            if overlap_penalty > 0:
                return overlap_penalty + compute_min_side(list(map(tuple, pos)), angles, outer_angle)
            return compute_min_side(list(map(tuple, pos)), angles, outer_angle)
        
        try:
            result = minimize(objective, pos_arr, method='Nelder-Mead', 
                             options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-4})
            final_pos = result.x.reshape(N, 2)
            
            # Verify validity
            verts_list = [hex_vertices(final_pos[i][0], final_pos[i][1], angles[i]) for i in range(N)]
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    if hexagons_overlap(verts_list[i], verts_list[j]):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                score = compute_min_side(list(map(tuple, final_pos)), angles, outer_angle)
                return list(map(tuple, final_pos)), angles, outer_angle, score
            else:
                return None
        except Exception:
            return None

    best_score = float('inf')
    best_positions = None
    best_angles = None
    best_outer_angle = 0.0

    # List of candidate configurations to try
    candidates = []
    
    # 1. Honeycomb 3x4 (standard)
    candidates.append(make_honeycomb(4, 3, 0.0, 0.0))
    # 2. Honeycomb 3x4 (rotated hexagons)
    candidates.append(make_honeycomb(4, 3, 30.0, 0.0))
    # 3. Honeycomb 2x6
    candidates.append(make_honeycomb(2, 6, 0.0, 0.0))
    # 4. Honeycomb 6x2
    candidates.append(make_honeycomb(6, 2, 0.0, 0.0))
    # 5. Circular
    candidates.append(make_circular(2.0, 0.0, 0.0))
    candidates.append(make_circular(2.5, 0.0, 0.0))
    candidates.append(make_circular(3.0, 0.0, 0.0))
    # 6. Try different outer angles for honeycomb
    base_pos, base_ang, _ = make_honeycomb(4, 3, 0.0, 0.0)
    for oa in [0, 15, 30, 45, 60, 75, 90]:
        candidates.append((base_pos, base_ang, oa))

    # Evaluate candidates and run optimizations
    for pos, ang, oa in candidates:
        if time.time() > deadline - 15:
            break
            
        score = get_config_score(pos, ang, oa)
        
        if score < best_score:
            best_score = score
            best_positions = pos
            best_angles = ang
            best_outer_angle = oa
            
        # If the initial candidate is valid, try to improve it
        if score < float('inf') and time.time() < deadline - 20:
            # Try physics optimization
            new_pos, new_ang, new_oa, new_score = physics_optimize(pos, ang, oa, max_iters=1000)
            if new_score < best_score:
                best_score = new_score
                best_positions = new_pos
                best_angles = new_ang
                best_outer_angle = new_oa
            
            # Try Nelder-Mead if time permits
            if time.time() < deadline - 30:
                nm_result = nelder_mead_optimize(pos, ang, oa, max_iter=300)
                if nm_result is not None:
                    nm_pos, nm_ang, nm_oa, nm_score = nm_result
                    if nm_score < best_score:
                        best_score = nm_score
                        best_positions = nm_pos
                        best_angles = nm_ang
                        best_outer_angle = nm_oa

    # Fallback: If no valid configuration was found
    if best_positions is None or best_score == float('inf'):
        positions = []
        angles = []
        for i in range(N):
            r = i // 4
            c = i % 4
            positions.append((c * 4.0, r * 4.0))
            angles.append(0.0)
        best_positions = positions
        best_angles = angles
        best_outer_angle = 0.0
        best_score = compute_min_side(positions, angles, 0.0)

    # Final cleanup
    if best_positions:
        cx = sum(p[0] for p in best_positions) / N
        cy = sum(p[1] for p in best_positions) / N
        final_positions = [(x - cx, y - cy) for x, y in best_positions]
    else:
        final_positions = [(0,0)] * N
        best_angles = [0] * N
        best_score = 10.0

    inner = [[final_positions[i][0], final_positions[i][1], best_angles[i]] for i in range(N)]
    
    return inner, (0.0, 0.0), best_score, best_outer_angle
# EVOLVE-BLOCK-END

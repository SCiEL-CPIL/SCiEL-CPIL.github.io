from scipy.optimize import differential_evolution, minimize_scalar
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    N = 12

    def hexagon_vertices(cx, cy, side, angle_rad):
        """Vertices of a regular hexagon (flat-top when angle=0)."""
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def polygon_disjoint_sat(v1, v2):
        """Check if two convex polygons are disjoint using SAT."""
        axes = []
        for v in [v1, v2]:
            for i in range(len(v)):
                j = (i + 1) % len(v)
                edge = v[j] - v[i]
                normal = np.array([-edge[1], edge[0]])
                if np.linalg.norm(normal) > 1e-12:
                    normal /= np.linalg.norm(normal)
                    axes.append(normal)
        
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return True
        
        return False

    def hexagons_disjoint(positions, angles):
        """Check if all hexagons are pairwise disjoint."""
        n = len(positions)
        for i in range(n):
            vi = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            for j in range(i + 1, n):
                vj = hexagon_vertices(positions[j, 0], positions[j, 1], 1.0, angles[j])
                if not polygon_disjoint_sat(vi, vj):
                    return False
        return True

    def calc_min_outer_side(all_verts, theta):
        """Calculate minimum outer hexagon side length for a given orientation theta."""
        angles = theta + np.radians(np.arange(0, 360, 60))
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        dots = all_verts @ normals.T
        widths = dots.max(axis=0) - dots.min(axis=0)
        max_width = np.max(widths)
        apothem = max_width / 2.0
        side = 2 * apothem / np.sqrt(3)
        return side

    def find_best_outer_orientation(all_verts):
        """Find orientation and side length for minimum enclosing hexagon."""
        res = minimize_scalar(lambda th: calc_min_outer_side(all_verts, th), 
                              bounds=(0, np.pi/3), method='bounded', 
                              options={'xatol': 1e-4})
        return res.fun, res.x

    def get_overlap_penalty(positions, angles):
        """Calculate a smooth penalty for overlaps using SAT penetration depth approximation."""
        penalty = 0.0
        n = len(positions)
        for i in range(n):
            vi = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            for j in range(i + 1, n):
                vj = hexagon_vertices(positions[j, 0], positions[j, 1], 1.0, angles[j])
                
                # Get axes
                axes = []
                for v in [vi, vj]:
                    for k in range(len(v)):
                        m = (k + 1) % len(v)
                        edge = v[m] - v[k]
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm > 1e-12:
                            normal /= norm
                            axes.append(normal)
                
                min_overlap = float('inf')
                for axis in axes:
                    proj1 = np.dot(vi, axis)
                    proj2 = np.dot(vj, axis)
                    min1, max1 = proj1.min(), proj1.max()
                    min2, max2 = proj2.min(), proj2.max()
                    
                    # If separated, no overlap on this axis
                    if max1 < min2 or max2 < min1:
                        min_overlap = 0.0
                        break
                    
                    # Calculate overlap on this axis
                    overlap = min(max1, max2) - max(min1, min2)
                    if overlap < min_overlap:
                        min_overlap = overlap
                
                # If min_overlap > 0, they are intersecting
                if min_overlap > 0:
                    penalty += min_overlap * min_overlap
        
        return penalty

    def approx_outer_side(positions, angles):
        """Fast approximation of outer side length using fixed orientations."""
        all_verts = []
        for i in range(len(positions)):
            v = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        
        # Check a few orientations
        min_side = float('inf')
        for theta in [0, np.pi/12, np.pi/6, np.pi/4, np.pi/3]:
            side = calc_min_outer_side(all_verts, theta)
            if side < min_side:
                min_side = side
        return min_side

    def full_objective(params):
        """Objective: approx outer side + penalty for overlap."""
        if time.time() > deadline:
            return 1e10
            
        n = len(params) // 3
        positions = np.array([[params[i*3], params[i*3+1]] for i in range(n)])
        angles = np.array([params[i*3+2] for i in range(n)])
        
        # Soft penalty
        penalty = get_overlap_penalty(positions, angles)
        if penalty > 0.01:
             return 10.0 + 100.0 * penalty
             
        side = approx_outer_side(positions, angles)
        return side

    def generate_honeycomb_config(seed_offset=0):
        """Generate a honeycomb-like configuration with slight random perturbation."""
        rng = np.random.RandomState(42 + seed_offset)
        d = 2.05
        positions = []
        angles = []
        
        positions.append([0.0, 0.0])
        angles.append(0.0)
        
        for k in range(6):
            ang = np.radians(60 * k)
            positions.append([d * np.cos(ang), d * np.sin(ang)])
            angles.append(0.0)
        
        gap_angles = [30, 90, 150, 210, 270]
        for k in range(5):
            ang = np.radians(gap_angles[k])
            dist = d * np.sqrt(3)
            positions.append([dist * np.cos(ang), dist * np.sin(ang)])
            angles.append(0.0)
        
        # Add small random perturbation
        for i in range(12):
            positions[i][0] += rng.uniform(-0.1, 0.1)
            positions[i][1] += rng.uniform(-0.1, 0.1)
            angles[i] += rng.uniform(-0.05, 0.05)
        
        positions = np.array(positions)
        angles = np.array(angles)
        
        # Ensure valid
        if not hexagons_disjoint(positions, angles):
            positions = positions * 1.1
            if not hexagons_disjoint(positions, angles):
                positions = positions * 1.2
        
        return positions, angles

    def generate_staggered_rows_config(seed_offset=0):
        """Generate a staggered rows configuration."""
        rng = np.random.RandomState(100 + seed_offset)
        positions = []
        angles = []
        
        # 3 rows of 4 hexagons each, staggered
        row_y = [-2.0, 0.0, 2.0]
        row_x_offset = [0.0, 1.0, 0.0]
        
        idx = 0
        for r in range(3):
            for c in range(4):
                x = (c - 1.5) * 2.1 + row_x_offset[r]
                y = row_y[r]
                x += rng.uniform(-0.05, 0.05)
                y += rng.uniform(-0.05, 0.05)
                positions.append([x, y])
                angles.append(rng.uniform(-0.1, 0.1))
                idx += 1
        
        positions = np.array(positions)
        angles = np.array(angles)
        
        if not hexagons_disjoint(positions, angles):
            # Scale up slightly
            positions = positions * 1.05
            if not hexagons_disjoint(positions, angles):
                positions = positions * 1.1
        
        return positions, angles

    def generate_ring_config(seed_offset=0):
        """Generate a ring configuration: center + inner ring + outer ring."""
        rng = np.random.RandomState(200 + seed_offset)
        positions = []
        angles = []
        
        # Center
        positions.append([0.0, 0.0])
        angles.append(0.0)
        
        # Inner ring (6 hexagons)
        d1 = 2.05
        for k in range(6):
            ang = np.radians(60 * k + rng.uniform(-5, 5))
            positions.append([d1 * np.cos(ang), d1 * np.sin(ang)])
            angles.append(rng.uniform(-0.1, 0.1))
        
        # Outer ring (5 hexagons in gaps)
        d2 = d1 * np.sqrt(3)
        gap_angles = [30, 90, 150, 210, 270]
        for k in range(5):
            ang = np.radians(gap_angles[k] + rng.uniform(-5, 5))
            positions.append([d2 * np.cos(ang), d2 * np.sin(ang)])
            angles.append(rng.uniform(-0.1, 0.1))
        
        positions = np.array(positions)
        angles = np.array(angles)
        
        if not hexagons_disjoint(positions, angles):
            positions = positions * 1.05
            if not hexagons_disjoint(positions, angles):
                positions = positions * 1.1
        
        return positions, angles

    # Generate multiple diverse initial configurations
    configs = []
    for i in range(4):
        pos, ang = generate_honeycomb_config(i)
        configs.append((pos, ang))
    for i in range(2):
        pos, ang = generate_staggered_rows_config(i)
        configs.append((pos, ang))
    for i in range(2):
        pos, ang = generate_ring_config(i)
        configs.append((pos, ang))

    best_params = None
    best_valid_side = 1e9
    best_positions = None
    best_angles = None

    bounds = []
    for i in range(12):
        bounds.append((-4.0, 4.0))
        bounds.append((-4.0, 4.0))
        bounds.append((0.0, np.pi))

    def callback(x, convergence):
        nonlocal best_valid_side, best_params, best_positions, best_angles
        if time.time() > deadline:
            return True
        pos = np.array([[x[i*3], x[i*3+1]] for i in range(12)])
        ang = np.array([x[i*3+2] for i in range(12)])
        if hexagons_disjoint(pos, ang):
            all_verts = []
            for i in range(12):
                v = hexagon_vertices(pos[i, 0], pos[i, 1], 1.0, ang[i])
                all_verts.append(v)
            all_verts = np.vstack(all_verts)
            side, _ = find_best_outer_orientation(all_verts)
            if side < best_valid_side:
                best_valid_side = side
                best_params = x.copy()
                best_positions = pos.copy()
                best_angles = ang.copy()
        return False

    # Run DE for each configuration
    for cfg_idx, (init_pos, init_ang) in enumerate(configs):
        if time.time() > deadline:
            break
        
        init_params = np.concatenate([init_pos.flatten(), init_ang])
        
        # Quick check if initial is already good
        if hexagons_disjoint(init_pos, init_ang):
            all_verts = []
            for i in range(12):
                v = hexagon_vertices(init_pos[i, 0], init_pos[i, 1], 1.0, init_ang[i])
                all_verts.append(v)
            all_verts = np.vstack(all_verts)
            side, _ = find_best_outer_orientation(all_verts)
            if side < best_valid_side:
                best_valid_side = side
                best_params = init_params.copy()
                best_positions = init_pos.copy()
                best_angles = init_ang.copy()
        
        # Early termination if we already have a good solution
        if best_valid_side < 4.2:
            break
        
        # Run DE with limited iterations to stay within time budget
        time_remaining = deadline - time.time()
        maxiter = max(10, min(60, int(time_remaining / 2.0)))
        popsize = max(8, min(15, int(time_remaining / 5.0)))
        
        try:
            result = differential_evolution(
                full_objective,
                bounds=bounds,
                seed=42 + cfg_idx,
                maxiter=maxiter,
                popsize=popsize,
                tol=1e-5,
                polish=False,
                callback=callback,
                init=init_params
            )
            
            # Check final result
            if result.fun < 10.0:
                pos = np.array([[result.x[i*3], result.x[i*3+1]] for i in range(12)])
                ang = np.array([result.x[i*3+2] for i in range(12)])
                if hexagons_disjoint(pos, ang):
                    all_verts = []
                    for i in range(12):
                        v = hexagon_vertices(pos[i, 0], pos[i, 1], 1.0, ang[i])
                        all_verts.append(v)
                    all_verts = np.vstack(all_verts)
                    side, _ = find_best_outer_orientation(all_verts)
                    if side < best_valid_side:
                        best_valid_side = side
                        best_params = result.x.copy()
                        best_positions = pos.copy()
                        best_angles = ang.copy()

        except Exception as e:
            pass

    # Final refinement if we have time and a good solution
    if best_valid_side < 5.0 and time.time() < deadline - 10.0:
        try:
            final_params = best_params.copy()
            result = differential_evolution(
                full_objective,
                bounds=bounds,
                seed=999,
                maxiter=30,
                popsize=10,
                tol=1e-6,
                polish=False,
                callback=callback,
                init=final_params
            )
            if result.fun < 10.0:
                pos = np.array([[result.x[i*3], result.x[i*3+1]] for i in range(12)])
                ang = np.array([result.x[i*3+2] for i in range(12)])
                if hexagons_disjoint(pos, ang):
                    all_verts = []
                    for i in range(12):
                        v = hexagon_vertices(pos[i, 0], pos[i, 1], 1.0, ang[i])
                        all_verts.append(v)
                    all_verts = np.vstack(all_verts)
                    side, _ = find_best_outer_orientation(all_verts)
                    if side < best_valid_side:
                        best_valid_side = side
                        best_positions = pos.copy()
                        best_angles = ang.copy()
        except Exception as e:
            pass

    # Extract best solution
    if best_positions is not None and best_angles is not None:
        all_verts = []
        for i in range(12):
            v = hexagon_vertices(best_positions[i, 0], best_positions[i, 1], 1.0, best_angles[i])
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        
        outer_side, outer_angle_rad = find_best_outer_orientation(all_verts)
        
        # Calculate center
        angles = outer_angle_rad + np.radians(np.arange(0, 360, 60))
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        dots = all_verts @ normals.T
        mins = dots.min(axis=0)
        maxs = dots.max(axis=0)
        mids = (mins + maxs) / 2.0
        c, _, _, _ = np.linalg.lstsq(normals.T, mids, rcond=None)
        outer_center = tuple(c)
        outer_angle_deg = np.degrees(outer_angle_rad)
    else:
        # Fallback to first config
        pos, ang = configs[0]
        all_verts = []
        for i in range(12):
            v = hexagon_vertices(pos[i, 0], pos[i, 1], 1.0, ang[i])
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        outer_side, outer_angle_rad = find_best_outer_orientation(all_verts)
        angles = outer_angle_rad + np.radians(np.arange(0, 360, 60))
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        dots = all_verts @ normals.T
        mins = dots.min(axis=0)
        maxs = dots.max(axis=0)
        mids = (mins + maxs) / 2.0
        c, _, _, _ = np.linalg.lstsq(normals.T, mids, rcond=None)
        outer_center = tuple(c)
        outer_angle_deg = np.degrees(outer_angle_rad)
        best_positions = pos
        best_angles = ang

    inner = [[best_positions[i, 0], best_positions[i, 1], np.degrees(best_angles[i])] for i in range(12)]
    
    return inner, outer_center, outer_side, outer_angle_deg
# EVOLVE-BLOCK-END

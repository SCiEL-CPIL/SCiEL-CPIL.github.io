import numpy as np
import time

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def hex_verts(cx, cy, s, rot_deg):
        rot_rad = np.deg2rad(rot_deg)
        angles = rot_rad + np.deg2rad(np.arange(6) * 60)
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def point_in_hexagon(px, py, cx, cy, s, rot_deg):
        """Check if point (px,py) is inside hexagon centered at (cx,cy) with side s and rotation rot_deg."""
        # Transform to hexagon's local coordinate system
        dx = px - cx
        dy = py - cy
        rot_rad = np.deg2rad(rot_deg)
        cos_r = np.cos(-rot_rad)
        sin_r = np.sin(-rot_rad)
        lx = cos_r * dx - sin_r * dy
        ly = sin_r * dx + cos_r * dy
        
        # Regular hexagon with vertex at angle 0 in local coords
        # Flat-top: vertices at 0, 60, 120, 180, 240, 300
        # The hexagon has 6 half-planes
        # For a hexagon with side s, the distance from center to each edge (inradius) is s*sqrt(3)/2
        # Edge normals are at 30, 90, 150, 210, 270, 330 degrees
        r_in = s * np.sqrt(3) / 2.0
        for k in range(6):
            angle = np.deg2rad(30 + k * 60)
            nx = np.cos(angle)
            ny = np.sin(angle)
            # Distance from point to edge line
            dist = lx * nx + ly * ny
            if dist > r_in + 1e-10:
                return False
        return True

    def hex_overlap_depth(v1, v2):
        """
        Estimate overlap depth between two convex hexagons.
        Returns 0 if no overlap, positive value indicating how deep they overlap.
        Uses a simple approach: check if any vertex of one is inside the other,
        and estimate the penetration depth.
        """
        overlap = 0.0
        # Check vertices of v1 inside v2
        for i in range(6):
            px, py = v1[i]
            # Find center and rotation of v2
            cx2 = np.mean(v2[:, 0])
            cy2 = np.mean(v2[:, 1])
            # Estimate rotation from first vertex
            vx = v2[0, 0] - cx2
            vy = v2[0, 1]
            rot2 = np.rad2deg(np.arctan2(vy, vx))
            s2 = np.sqrt(vx**2 + vy**2)
            
            if point_in_hexagon(px, py, cx2, cy2, s2, rot2):
                # Estimate penetration: distance from vertex to closest edge of v2
                min_dist_to_edge = float('inf')
                for j in range(6):
                    p1 = v2[j]
                    p2 = v2[(j+1) % 6]
                    # Point-to-segment distance
                    v = p2 - p1
                    w = v1[i] - p1
                    c1 = np.dot(w, v)
                    c2 = np.dot(v, v)
                    if c1 <= 0:
                        d = np.linalg.norm(w)
                    elif c1 >= c2:
                        d = np.linalg.norm(w - v)
                    else:
                        d = np.linalg.norm(w - (c1 / c2) * v)
                    min_dist_to_edge = min(min_dist_to_edge, d)
                overlap = max(overlap, min_dist_to_edge)
        
        # Check vertices of v2 inside v1
        for i in range(6):
            px, py = v2[i]
            cx1 = np.mean(v1[:, 0])
            cy1 = np.mean(v1[:, 1])
            vx = v1[0, 0] - cx1
            vy = v1[0, 1] - cy1
            rot1 = np.rad2deg(np.arctan2(vy, vx))
            s1 = np.sqrt(vx**2 + vy**2)
            
            if point_in_hexagon(px, py, cx1, cy1, s1, rot1):
                min_dist_to_edge = float('inf')
                for j in range(6):
                    p1 = v1[j]
                    p2 = v1[(j+1) % 6]
                    v = p2 - p1
                    w = v2[i] - p1
                    c1 = np.dot(w, v)
                    c2 = np.dot(v, v)
                    if c1 <= 0:
                        d = np.linalg.norm(w)
                    elif c1 >= c2:
                        d = np.linalg.norm(w - v)
                    else:
                        d = np.linalg.norm(w - (c1 / c2) * v)
                    min_dist_to_edge = min(min_dist_to_edge, d)
                overlap = max(overlap, min_dist_to_edge)
        
        return overlap

    def check_feasibility(params, outer_s):
        """
        Check if the configuration is valid.
        params: [x1, y1, a1, x2, y2, a2, ..., x12, y12, a12] where a is angle in degrees
        Returns (is_valid, violation) where violation is a measure of infeasibility.
        """
        violation = 0.0
        n_inner = 12
        
        # Extract positions and angles
        positions = []
        angles = []
        for i in range(n_inner):
            x = params[3 * i]
            y = params[3 * i + 1]
            a = params[3 * i + 2]
            positions.append((x, y))
            angles.append(a)
        
        # Generate inner hexagon vertices
        inner_verts = []
        for i in range(n_inner):
            v = hex_verts(positions[i][0], positions[i][1], 1.0, angles[i])
            inner_verts.append(v)
        
        # Containment check
        r = outer_s * np.sqrt(3) / 2.0
        normals = np.deg2rad(30 + np.arange(6) * 60)
        n_vecs = np.column_stack([np.cos(normals), np.sin(normals)])
        
        for i in range(n_inner):
            for v in inner_verts[i]:
                dists = np.dot(v, n_vecs.T)
                out = np.maximum(0, dists - r)
                violation += np.sum(out)
        
        # Overlap check
        for i in range(n_inner):
            for j in range(i + 1, n_inner):
                ov = hex_overlap_depth(inner_verts[i], inner_verts[j])
                if ov > 1e-6:
                    violation += ov ** 2 * 10.0
                # Also add center-distance penalty for smoothness
                cx1, cy1 = positions[i]
                cx2, cy2 = positions[j]
                c_dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
                threshold = np.sqrt(3)
                if c_dist < threshold:
                    violation += (threshold - c_dist) ** 2 * 5.0
        
        is_valid = violation < 1e-3
        return is_valid, violation

    def get_initial_positions():
        """Get initial positions for the 12 hexagons."""
        params = np.zeros(36)
        # Place in a hexagonal-ish pattern
        offsets = [
            (0, 0),
            (2.0, 0), (-1.0, np.sqrt(3)),
            (-1.0, -np.sqrt(3)), (1.0, np.sqrt(3)), (1.0, -np.sqrt(3)),
            (2.0, np.sqrt(3)*1.5), (2.0, -np.sqrt(3)*1.5),
            (-2.0, np.sqrt(3)*1.5), (-2.0, -np.sqrt(3)*1.5),
            (3.0, 0), (-3.0, 0)
        ]
        for i in range(12):
            params[3 * i] = offsets[i][0]
            params[3 * i + 1] = offsets[i][1]
            params[3 * i + 2] = 0.0
        return params

    # Binary search on outer side length
    LB = 3.0
    UB = 12.0
    
    best_solution = None
    best_s = float('inf')
    
    max_iter_nm = 300
    tol = 1e-4
    
    # Simulated annealing / gradient-free optimization
    for step in range(25):
        if time.time() > deadline:
            break
            
        mid_s = (LB + UB) / 2.0
        
        # Try to find a valid configuration for this S
        found_valid = False
        best_violation = float('inf')
        best_params = None
        
        # Multiple restarts
        for restart in range(5):
            if time.time() > deadline:
                break
                
            x0 = get_initial_positions()
            x0 += np.random.normal(0, 0.5, x0.shape)
            
            # Simple iterative improvement
            current = x0.copy()
            current_violation = check_feasibility(current, mid_s)[1]
            
            for it in range(max_iter_nm):
                if time.time() > deadline:
                    break
                    
                # Random perturbation
                direction = np.random.normal(0, 1.0, current.shape)
                step_size = 0.3 * (1 - it / max_iter_nm) + 0.01
                new_params = current + direction * step_size
                
                # Ensure angles stay in reasonable range
                for i in range(12):
                    if new_params[3 * i + 2] > 90:
                        new_params[3 * i + 2] = 90
                    elif new_params[3 * i + 2] < -90:
                        new_params[3 * i + 2] = -90
                
                new_violation = check_feasibility(new_params, mid_s)[1]
                
                if new_violation < current_violation:
                    current = new_params
                    current_violation = new_violation
                    if current_violation < 1e-3:
                        found_valid = True
                        best_params = current.copy()
                        break
                else:
                    # Accept with some probability (simulated annealing-like)
                    if np.random.random() < 0.3 and it < max_iter_nm // 2:
                        current = new_params
                        current_violation = new_violation
            
            if current_violation < best_violation:
                best_violation = current_violation
                best_params = current.copy()
            
            if found_valid:
                break
        
        if found_valid:
            UB = mid_s
            best_solution = best_params.copy()
            best_s = mid_s
        else:
            LB = mid_s
    
    if best_solution is None:
        # Fallback: return a large valid box
        inner = []
        for i in range(12):
            x = (i % 4) * 2.5 - 3.75
            y = (i // 4) * 2.2 - 2.2
            inner.append([x, y, 0.0])
        return inner, (0.0, 0.0), 12.0, 0.0
    
    # Extract best solution
    inner = []
    for i in range(12):
        x = best_solution[3 * i]
        y = best_solution[3 * i + 1]
        a = best_solution[3 * i + 2]
        inner.append([x, y, a])
        
    return inner, (0.0, 0.0), best_s, 0.0


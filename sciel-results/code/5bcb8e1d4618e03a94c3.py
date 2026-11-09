import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_11():
    start_time = time.time()
    deadline = 620  # 10s safety margin

    def get_hex_vertices(cx, cy, angle_deg):
        """Get vertices of a unit hexagon."""
        a = np.radians(angle_deg)
        angles = a + np.radians([0, 60, 120, 180, 240, 300])
        return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])
    
    def hex_edge_normals(angle_deg):
        """Get the 6 edge normals of a unit hexagon."""
        a = np.radians(angle_deg)
        # Edge normals are at 30, 90, 150, 210, 270, 330 degrees relative to hex angle
        angles = a + np.radians([30, 90, 150, 210, 270, 330])
        return np.column_stack([np.cos(angles), np.sin(angles)])
    
    def is_pair_separated(vertices_i, vertices_j, epsilon=1e-6):
        """Check if two convex polygons are separated using 12 edge normals."""
        # Get normals from both polygons
        normals_i = hex_edge_normals(0)  # For unit hex at origin
        normals_j = hex_edge_normals(0)
        
        # But we need to account for the actual angles... 
        # Actually, for the separation test, we can just use all 12 normals
        # from both hexagons' edge directions
        
        # Simpler: use the 6 normals at 30,90,...,330 for each hexagon's orientation
        # But since we're testing actual vertex positions, let's compute properly
        
        # Get all 12 normals (6 from each hexagon's orientation)
        # For simplicity in dynamics, we'll use a simpler overlap test
        
        # Check if any vertex of i is inside j or vice versa using winding
        # But that's expensive. Use the projection method.
        
        # The 6 edge normals for a hexagon with angle a are at a+30, a+90, etc.
        # For unit hexagons, the max projection on normal n is apothem = sqrt(3)/2
        
        # For two hexagons, we need to check all 12 edge normals
        # Normal k of hex i: the edge between vertex k-1 and k
        # This is complex. Let's use a simpler sufficient test:
        # If centers are far enough apart, they're separated
        # Otherwise, check if any edge of one intersects any edge of other
        
        # Actually, the problem states: "on at least one of their 12 edge normals 
        # the projected intervals are apart by more than 1e-6"
        # So we need the 6 edge normals of hex i and 6 of hex j = 12 total
        
        # For hexagon i with angle ai, its edge normals are at ai + 30 + 60*k
        # For hexagon j with angle aj, its edge normals are at aj + 30 + 60*k
        
        # But we don't store angles separately in vertices. Let me restructure.
        
        # For now, use a simpler approach: check if minimum distance between 
        # any vertex of i and any edge of j is > 0, and vice versa.
        # This is O(36) per pair which is fine for 55 pairs.
        
        def point_to_segment_dist(p, a, b):
            """Distance from point p to segment ab."""
            ab = b - a
            ap = p - a
            t = np.dot(ap, ab) / np.dot(ab, ab) if np.dot(ab, ab) > 1e-15 else 0
            t = np.clip(t, 0, 1)
            closest = a + t * ab
            return np.linalg.norm(p - closest)
        
        # Check if any vertex of i is inside j
        for vi in vertices_i:
            # Check if vi is inside j using winding number
            inside = True
            for k in range(6):
                p1 = vertices_j[k]
                p2 = vertices_j[(k+1) % 6]
                cross = np.cross(p2 - p1, vi - p1)
                if cross < -1e-9:  # Strictly outside
                    inside = False
                    break
            if inside:
                return False
        
        # Check if any vertex of j is inside i
        for vj in vertices_j:
            inside = True
            for k in range(6):
                p1 = vertices_i[k]
                p2 = vertices_i[(k+1) % 6]
                cross = np.cross(p2 - p1, vj - p1)
                if cross < -1e-9:
                    inside = False
                    break
            if inside:
                return False
        
        # Check edge intersections
        for k in range(6):
            p1 = vertices_i[k]
            p2 = vertices_i[(k+1) % 6]
            for l in range(6):
                p3 = vertices_j[l]
                p4 = vertices_j[(l+1) % 6]
                # Check if segments p1-p2 and p3-p4 intersect
                d1 = p2 - p1
                d2 = p4 - p3
                cross_dd = d1[0]*d2[1] - d1[1]*d2[0]
                if abs(cross_dd) < 1e-15:
                    continue  # Parallel
                t = ((p3[0]-p1[0])*d2[1] - (p3[1]-p1[1])*d2[0]) / cross_dd
                u = ((p3[0]-p1[0])*d1[1] - (p3[1]-p1[1])*d1[0]) / cross_dd
                if 0 < t < 1 and 0 < u < 1:
                    return False
        
        # If we get here, they're separated
        return True
    
    def compute_min_enclosing_side(positions, angles):
        """Compute the minimum side length of a regular hexagon that contains all hexagons."""
        # The outer hexagon can have any angle. We need to find min over outer angle.
        # For a given outer angle a, the constraint is:
        # For each inner vertex p, for each outer edge, cross(edge_vec, p - edge_start) >= -eps
        # 
        # Alternatively: the apothem of the outer hex is S * sqrt(3)/2
        # The constraint is that for each of the 6 outward normals n_k of the outer hex:
        #   max over all inner vertices of dot(p, n_k) <= S * sqrt(3)/2
        # So S >= (max_dot) / (sqrt(3)/2) for each k, and S = max over k of these.
        # Then minimize over outer angle.
        
        all_vertices = []
        for i in range(11):
            v = get_hex_vertices(positions[i, 0], positions[i, 1], angles[i])
            all_vertices.append(v)
        all_vertices = np.vstack(all_vertices)  # (66, 2)
        
        # For outer hex with angle a, normals are at a + 30 + 60*k
        best_S = np.inf
        best_a = 0
        
        # Sample outer angles
        for a_deg in np.arange(0, 60, 5):  # 12 samples, exploit 60-degree symmetry
            a = np.radians(a_deg)
            normals = []
            for k in range(6):
                ang = a + np.radians(30 + 60*k)
                normals.append([np.cos(ang), np.sin(ang)])
            normals = np.array(normals)  # (6, 2)
            
            # Max projection on each normal
            max_projs = np.max(all_vertices @ normals.T, axis=0)  # (6,)
            S_required = np.max(max_projs) / (np.sqrt(3) / 2)
            
            if S_required < best_S:
                best_S = S_required
                best_a = a_deg
        
        return best_S, best_a
    
    def validate(inner, outer, side):
        """Full validation as per task contract."""
        if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(side):
            return False
        if side < np.sqrt(11) - 1e-9:
            return False
        
        ox, oy, oa = outer
        inner_shifted = inner.copy()
        inner_shifted[:, 0] -= ox
        inner_shifted[:, 1] -= oy
        
        # Check edge lengths
        for i in range(11):
            v = get_hex_vertices(inner_shifted[i, 0], inner_shifted[i, 1], inner_shifted[i, 2])
            diffs = np.diff(v, axis=0, append=v[:1])
            lengths = np.linalg.norm(diffs, axis=1)
            if not np.all(np.abs(lengths - 1.0) < 1e-8):
                return False
        
        # Check non-overlap using the 12-edge-normal method
        all_vertices = []
        all_angles = []
        for i in range(11):
            v = get_hex_vertices(inner_shifted[i, 0], inner_shifted[i, 1], inner_shifted[i, 2])
            all_vertices.append(v)
            all_angles.append(inner_shifted[i, 2])
        
        for i in range(11):
            vi = all_vertices[i]
            ai = all_angles[i]
            # Get 6 edge normals for hex i
            ai_rad = np.radians(ai)
            normals_i = np.column_stack([
                np.cos(ai_rad + np.radians(30 + 60*k)),
                np.sin(ai_rad + np.radians(30 + 60*k))
            ] for k in range(6))
            
            for j in range(i+1, 11):
                vj = all_vertices[j]
                aj = all_angles[j]
                aj_rad = np.radians(aj)
                normals_j = np.column_stack([
                    np.cos(aj_rad + np.radians(30 + 60*k)),
                    np.sin(aj_rad + np.radians(30 + 60*k))
                ] for k in range(6))
                
                all_normals = np.vstack([normals_i, normals_j])  # (12, 2)
                
                separated = False
                for n in all_normals:
                    proj_i = vi @ n
                    proj_j = vj @ n
                    gap = min(np.min(proj_j) - np.max(proj_i), np.min(proj_i) - np.max(proj_j))
                    if gap > 1e-6:
                        separated = True
                        break
                
                if not separated:
                    return False
        
        # Check containment
        oa_rad = np.radians(oa)
        outer_v = np.column_stack([
            side * np.cos(oa_rad + np.radians(60*k)),
            side * np.sin(oa_rad + np.radians(60*k))
        ] for k in range(6))
        
        for i in range(6):
            p1 = outer_v[i]
            p2 = outer_v[(i+1) % 6]
            for i_h in range(11):
                vi = all_vertices[i_h]
                for vp in vi:
                    c = np.cross(p2 - p1, vp - p1)
                    if c < -1e-6:
                        return False
        
        return True
    
    # Initial configurations
    
    def hex_lattice_config():
        """Standard hexagonal lattice: center + 6 in ring + 4 more."""
        positions = np.zeros((11, 2))
        angles = np.zeros(11)
        
        # Center
        positions[0] = [0, 0]
        
        # First ring at distance sqrt(3) (touching)
        for i in range(1, 7):
            ang = np.radians(60 * (i-1))
            r = np.sqrt(3)
            positions[i] = [r * np.cos(ang), r * np.sin(ang)]
        
        # Second ring - place in gaps
        # Gaps are at 30, 90, 150, 210, 270, 330 degrees
        # Distance for touching: 2*sqrt(3) from center? No.
        # For hexagonal lattice with side 1, the distance between adjacent centers is sqrt(3)
        # Second ring centers are at distance 2*sqrt(3) from center
        # But we only need 4 more. Place them at 4 of the 6 gap positions.
        
        gap_angles = [30, 150, 210, 330]  # Pick 4
        for i, a_deg in enumerate(gap_angles):
            ang = np.radians(a_deg)
            r = 2 * np.sqrt(3)
            positions[7+i] = [r * np.cos(ang), r * np.sin(ang)]
        
        return positions, angles
    
    def symmetric_11_config():
        """Try a different symmetric arrangement."""
        positions = np.zeros((11, 2))
        angles = np.zeros(11)
        
        # Center
        positions[0] = [0, 0]
        
        # 6 in first ring
        for i in range(1, 7):
            ang = np.radians(60 * i)
            r = np.sqrt(3)
            positions[i] = [r * np.cos(ang), r * np.sin(ang)]
        
        # 4 in second ring at 45-degree offsets? No, hexagonal symmetry.
        # Place at 60, 180, 240, 300? No, those are occupied.
        # Place at 0, 120, 240, 360? But 0 is close to ring hexes.
        
        # Let's try: 4 hexes at the "corners" of a larger hexagon
        # At angles 0, 120, 240 and one more? That's not symmetric.
        
        # Try: 6 in ring, then 5 more at specific positions
        # Actually, let's just use the lattice config for now
        return positions, angles
    
    # Dynamics-based optimization
    
    def dynamics_optimize(positions, angles, max_time=120):
        """
        Rigid-body dynamics with:
        - Repulsive potential for overlap
        - Attractive force toward center
        - Shrinking container
        """
        if time.time() - start_time > max_time:
            return positions, angles
        
        n = 11
        # State: [x0, y0, angle0, x1, y1, angle1, ..., vx0, vy0, vangle0, ...]
        # Or simpler: just positions and angles, use gradient descent
        
        # Use a penalty-based approach with scipy minimize
        def objective(params):
            pos = params[:22].reshape(11, 2)
            ang = params[22:33]
            
            # Penalty for overlap
            penalty = 0.0
            all_vertices = []
            for i in range(n):
                v = get_hex_vertices(pos[i, 0], pos[i, 1], ang[i])
                all_vertices.append(v)
            
            for i in range(n):
                for j in range(i+1, n):
                    # Check if overlapping
                    # Simple check: min distance between vertices
                    # If any vertex of i is inside j, penalty
                    vi = all_vertices[i]
                    vj = all_vertices[j]
                    
                    for vp in vi:
                        # Check if vp is inside vj
                        inside = True
                        for k in range(6):
                            p1 = vj[k]
                            p2 = vj[(k+1) % 6]
                            cross = np.cross(p2 - p1, vp - p1)
                            if cross < -1e-9:
                                inside = False
                                break
                        if inside:
                            penalty += 1000.0
                            break
                    else:
                        continue
                    break
                
                # Also check edge intersections
                # For simplicity, if we found overlap above, skip
                # Otherwise, check if edges intersect
                for k in range(6):
                    p1 = vi[k]
                    p2 = vi[(k+1) % 6]
                    for l in range(6):
                        p3 = vj[l]
                        p4 = vj[(l+1) % 6]
                        d1 = p2 - p1
                        d2 = p4 - p3
                        cross_dd = d1[0]*d2[1] - d1[1]*d2[0]
                        if abs(cross_dd) < 1e-15:
                            continue
                        t = ((p3[0]-p1[0])*d2[1] - (p3[1]-p1[1])*d2[0]) / cross_dd
                        u = ((p3[0]-p1[0])*d1[1] - (p3[1]-p1[1])*d1[0]) / cross_dd
                        if 0 < t < 1 and 0 < u < 1:
                            penalty += 1000.0
                            break
                    else:
                        continue
                    break
            
            # Penalty for being too spread out (use current bounding size)
            S_current, _ = compute_min_enclosing_side(pos, ang)
            
            # Objective: minimize S with penalty for overlaps
            return S_current + penalty
        
        # Try multiple initializations
        best_S = np.inf
        best_params = None
        
        if time.time() - start_time > max_time - 30:
            return positions, angles
        
        # Run Nelder-Mead with penalty
        try:
            params0 = np.concatenate([positions.flatten(), angles])
            res = minimize(objective, params0, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            
            if res.fun < 100:  # No significant penalty
                opt_params = res.x
                opt_pos = opt_params[:22].reshape(11, 2)
                opt_ang = opt_params[22:33]
                
                # Center the configuration
                opt_pos -= np.mean(opt_pos, axis=0)
                
                S_opt, _ = compute_min_enclosing_side(opt_pos, opt_ang)
                if S_opt < best_S:
                    best_S = S_opt
                    best_params = (opt_pos.copy(), opt_ang.copy())
        except:
            pass
        
        if best_params is not None:
            return best_params
        
        return positions, angles
    
    # Main algorithm
    
    best_result = None
    best_score = -np.inf
    
    # Start with hex lattice
    positions, angles = hex_lattice_config()
    
    # Compute initial S
    S_init, a_init = compute_min_enclosing_side(positions, angles)
    
    # Try dynamics optimization
    pos_opt, ang_opt = dynamics_optimize(positions, angles, max_time=180)
    
    if pos_opt is not None and ang_opt is not None:
        pos_opt = pos_opt - np.mean(pos_opt, axis=0)
        S_opt, a_opt = compute_min_enclosing_side(pos_opt, ang_opt)
        
        inner = np.column_stack([pos_opt, ang_opt])
        outer = np.array([0, 0, a_opt])
        
        if validate(inner, outer, S_opt):
            score = 1.0 / S_opt
            if score > best_score:
                best_score = score
                best_result = (inner, outer, S_opt)
    
    # Try with different initial configurations and more optimization rounds
    if time.time() < deadline - 60:
        # Try perturbing the lattice
        for trial in range(3):
            if time.time() > deadline - 30:
                break
            
            # Perturb positions
            np.random.seed(42 + trial)
            pos_pert = positions + np.random.normal(0, 0.3, (11, 2))
            ang_pert = angles + np.random.uniform(-10, 10, 11)
            
            pos_opt, ang_opt = dynamics_optimize(pos_pert, ang_pert, max_time=90)
            
            if pos_opt is not None and ang_opt is not None:
                pos_opt = pos_opt - np.mean(pos_opt, axis=0)
                S_opt, a_opt = compute_min_enclosing_side(pos_opt, ang_opt)
                
                inner = np.column_stack([pos_opt, ang_opt])
                outer = np.array([0, 0, a_opt])
                
                if validate(inner, outer, S_opt):
                    score = 1.0 / S_opt
                    if score > best_score:
                        best_score = score
                        best_result = (inner, outer, S_opt)
    
    # Fallback to seed if nothing worked
    if best_result is None:
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8
        return inner_hex_data, outer_hex_data, outer_hex_side_length
    
    return best_result

# EVOLVE-BLOCK-END

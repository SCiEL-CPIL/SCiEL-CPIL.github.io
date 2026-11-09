import numpy as np
import time

def _hex_vertices(cx, cy, d, a):
    a_rad = np.deg2rad(a)
    angles = a_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _min_outer_side(positions, angle_deg):
    """
    Compute the minimum side length of a regular hexagon with the given angle
    that encloses all unit hexagons at the given positions.
    Assumes inner hexagons are aligned with angle 0.
    """
    unit_hex_verts = _hex_vertices(0, 0, 1, 0)
    
    # All inner hexagon vertices
    all_verts = positions[:, np.newaxis, :] + unit_hex_verts[np.newaxis, :, :]
    all_verts_flat = all_verts.reshape(-1, 2)
    
    a_rad = np.deg2rad(angle_deg)
    
    # Outward normals of the outer hexagon edges
    # Edge i connects vertex i and i+1.
    # Outward normal angle is a + 60i + 30 degrees.
    normals = []
    for i in range(6):
        n_angle = a_rad + i * np.pi/3 + np.pi/6
        n_unit = np.array([np.cos(n_angle), np.sin(n_angle)])
        normals.append(n_unit)
    normals = np.array(normals)
    
    # Project all vertices onto normals
    projections = all_verts_flat @ normals.T # (36, 6)
    max_projections = np.max(projections, axis=0) # (6,)
    
    # s * sqrt(3)/2 >= max_proj  =>  s >= 2 * max_proj / sqrt(3)
    min_s = np.max(max_projections) * 2 / np.sqrt(3)
    
    return min_s

def _check_validity(inner, outer, s):
    """
    Validates the solution according to the task contract.
    Returns True if valid, False otherwise.
    """
    # Check finite
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(s):
        return False
    
    if s < np.sqrt(12) - 1e-9:
        return False
        
    # Check outer vertices distinct
    outer_verts = _hex_vertices(outer[0], outer[1], s, outer[2])
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-9:
                return False
                
    # Check unit edge length of inner hexagons
    # "after shifting all centers by minus the outer center"
    # This implies we check the geometry relative to the outer center?
    # Actually, the text says: "after shifting all centers by minus the outer center, 
    # a unit edge whose computed length differs from 1 by more than 1e-8"
    # This phrasing is slightly ambiguous. Does it mean the inner hexagons must have side 1?
    # Yes, "Pack 12 unit regular hexagons".
    # So inner hexagons must have side 1.
    # Let's verify the side length of the defined inner hexagons.
    for i in range(12):
        cx, cy, ang = inner[i]
        verts = _hex_vertices(cx, cy, 1, ang)
        for k in range(6):
            v1 = verts[k]
            v2 = verts[(k+1)%6]
            if abs(np.linalg.norm(v2 - v1) - 1.0) > 1e-8:
                return False

    # Check separation (SAT)
    # "two inner hexagons not separated: some edge normal of either one (separating axis test) 
    # must give projection intervals more than 1e-6 apart"
    # Wait, the invalid condition is: "two inner hexagons not separated".
    # This means they MUST be separated.
    # Separation test: For every pair, there must exist an axis (from either polygon's edges)
    # such that the projections are disjoint by > 1e-6?
    # Actually, SAT says: Polygons are disjoint IF AND ONLY IF there exists an axis where projections are disjoint.
    # So "not separated" means NO axis has disjoint projections.
    # So we need to ensure that for EVERY pair, there IS an axis with disjoint projections.
    
    inner_verts_list = []
    for i in range(12):
        cx, cy, ang = inner[i]
        inner_verts_list.append(_hex_vertices(cx, cy, 1, ang))
        
    for i in range(12):
        for j in range(i+1, 12):
            va = inner_verts_list[i]
            vb = inner_verts_list[j]
            
            separated = False
            # Check axes from A
            for k in range(6):
                v1 = va[k]
                v2 = va[(k+1)%6]
                e = v2 - v1
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm < 1e-12: continue
                n = n / norm
                proj_a = np.dot(va, n)
                proj_b = np.dot(vb, n)
                if np.max(proj_a) < np.min(proj_b) - 1e-6 or np.max(proj_b) < np.min(proj_a) - 1e-6:
                    separated = True
                    break
            if separated: continue
            
            # Check axes from B
            for k in range(6):
                v1 = vb[k]
                v2 = vb[(k+1)%6]
                e = v2 - v1
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm < 1e-12: continue
                n = n / norm
                proj_a = np.dot(va, n)
                proj_b = np.dot(vb, n)
                if np.max(proj_a) < np.min(proj_b) - 1e-6 or np.max(proj_b) < np.min(proj_a) - 1e-6:
                    separated = True
                    break
            if not separated:
                return False
                
    # Check inner vertices inside outer
    # "an inner vertex v outside the outer hexagon: for every outer edge (p1, p2) in vertex order, 
    # cross(p2 - p1, v - p1) must be >= -1e-6"
    # Note: cross product sign depends on orientation.
    # Standard hexagon vertices are CCW.
    # For a CCW polygon, interior is to the left of edges.
    # cross(edge, point - start) > 0 means left.
    # So we need cross >= -1e-6.
    
    for i in range(12):
        for v in inner_verts_list[i]:
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1)%6]
                e = p2 - p1
                # cross(e, v - p1) = e_x*(v_y - p1_y) - e_y*(v_x - p1_x)
                cross_val = e[0] * (v[1] - p1[1]) - e[1] * (v[0] - p1[0])
                if cross_val < -1e-6:
                    inside = False
                    break
            if not inside:
                return False
                
    return True

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 590 # Safety margin
    
    best_score = 0.0
    best_inner = None
    best_outer = None
    best_s = None
    
    # Simulated Annealing
    # We will run multiple restarts with different random initializations.
    
    num_restarts = 50
    steps_per_restart = 2000
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize positions
        # Start with a loose hexagonal lattice, add random noise
        h = np.sqrt(3)/2
        init_pos = np.array([
            [0, 0],
            [2*h, 0],
            [h, 1.5],
            [-h, 1.5],
            [-2*h, 0],
            [-h, -1.5],
            [h, -1.5],
            [3*h, 1.5],
            [0, 3],
            [-3*h, 1.5],
            [-3*h, -1.5],
            [3*h, -1.5]
        ])
        
        # Add noise
        noise_scale = 0.5 + 0.5 * (r / num_restarts)
        pos = init_pos + np.random.randn(12, 2) * noise_scale
        
        # Random outer angle
        angle = np.random.uniform(0, 180)
        
        current_pos = pos.copy()
        current_angle = angle
        current_s = _min_outer_side(current_pos, current_angle)
        current_score = 1.0 / current_s
        
        # Track best in this restart
        restart_best_pos = current_pos.copy()
        restart_best_angle = current_angle
        restart_best_s = current_s
        restart_best_score = current_score
        
        T = 1.0
        cooling = 0.999
        
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
                
            # Perturb
            new_pos = current_pos.copy()
            new_angle = current_angle + np.random.randn() * 0.1
            new_pos += np.random.randn(12, 2) * 0.1 * T
            
            # Compute new s
            new_s = _min_outer_side(new_pos, new_angle)
            new_score = 1.0 / new_s
            
            # Check validity approximately?
            # Full validity check is expensive.
            # We can do a quick check: min distance between centers > 1.5?
            # Or just rely on the final check.
            # To avoid getting stuck in invalid states, we can penalize overlaps.
            # But SAT is expensive.
            # Let's just use the score. If they overlap, the packing is invalid, 
            # but the score might still be high if s is small.
            # We MUST ensure the final result is valid.
            # Strategy: During SA, we minimize s. 
            # At the end, we check validity. If invalid, we try to fix or discard.
            # Better: Add a penalty to the objective if overlap is detected.
            # Quick overlap check: Min distance between any two vertices of different hexagons?
            # Or just center distance < 1.8?
            
            # Let's use a simple penalty: 
            # If any pair of centers is closer than 1.7, add penalty.
            penalty = 0
            for i in range(12):
                for j in range(i+1, 12):
                    d = np.linalg.norm(new_pos[i] - new_pos[j])
                    if d < 1.7:
                        penalty += (1.7 - d)**2
            
            effective_new_score = new_score - penalty * 10
            
            delta_score = effective_new_score - current_score
            if delta_score > 0 or np.random.rand() < np.exp(delta_score / T):
                current_pos = new_pos
                current_angle = new_angle
                current_s = new_s
                current_score = new_score
                
                if current_score > restart_best_score:
                    restart_best_pos = current_pos.copy()
                    restart_best_angle = current_angle
                    restart_best_s = current_s
                    restart_best_score = current_score
                    
            T *= cooling
            
        # End of restart. Check validity of best found in this restart.
        # We need to construct the full inner array.
        # We assumed inner angles are 0.
        inner_candidate = np.zeros((12, 3))
        inner_candidate[:, :2] = restart_best_pos
        inner_candidate[:, 2] = 0.0
        
        outer_candidate = np.array([0, 0, restart_best_angle])
        
        # Recompute s exactly
        s_candidate = _min_outer_side(restart_best_pos, restart_best_angle)
        
        # Check validity
        if _check_validity(inner_candidate, outer_candidate, s_candidate):
            score_candidate = 1.0 / s_candidate
            if score_candidate > best_score:
                best_score = score_candidate
                best_inner = inner_candidate
                best_outer = outer_candidate
                best_s = s_candidate
        else:
            # Try to repair? Or just discard.
            # We can try to expand s slightly to fix "inside" violations, 
            # but overlap violations cannot be fixed by expanding s.
            # If invalid due to overlap, this restart failed.
            pass
            
    # If we found a valid solution, return it
    if best_inner is not None:
        return best_inner, best_outer, best_s
    else:
        # Fallback to a known valid configuration (seed-like)
        # We need to ensure the fallback is valid.
        # Let's construct a very loose packing that is definitely valid.
        # 12 hexagons in a 4x3 grid?
        # Spacing 2.5 is safe.
        h = np.sqrt(3)/2
        inner_fallback = np.array([
            [0, 0, 0],
            [2.5, 0, 0],
            [-2.5, 0, 0],
            [1.25, 2.165, 0],
            [-1.25, 2.165, 0],
            [1.25, -2.165, 0],
            [-1.25, -2.165, 0],
            [3.75, 2.165, 0],
            [-3.75, 2.165, 0],
            [3.75, -2.165, 0],
            [-3.75, -2.165, 0],
            [0, 4.33, 0]
        ])
        outer_fallback = np.array([0, 0, 0])
        s_fallback = _min_outer_side(inner_fallback[:, :2], 0)
        
        # Ensure s is large enough
        if s_fallback < np.sqrt(12):
            s_fallback = np.sqrt(12)
            
        return inner_fallback, outer_fallback, s_fallback

# EVOLVE-BLOCK-END

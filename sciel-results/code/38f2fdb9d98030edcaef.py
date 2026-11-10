# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side d, angle in degrees."""
    a = math.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = a + math.radians(60 * i)
        verts[i, 0] = cx + d * math.cos(theta)
        verts[i, 1] = cy + d * math.sin(theta)
    return verts

def _check_separation(hex1, hex2):
    """Check if two hexagons (each (6,2) arrays) are separated by SAT."""
    for verts in (hex1, hex2):
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i + 1) % 6]
            edge = p2 - p1
            n = np.array([-edge[1], edge[0]])
            n = n / (np.linalg.norm(n) + 1e-15)
            proj1 = hex1 @ n
            proj2 = hex2 @ n
            if proj1.max() <= proj2.min() + 1e-6 or proj2.max() <= proj1.min() + 1e-6:
                return True
    return False

def _check_inside(hex_verts, outer_verts):
    """Check if all vertices of hex are inside outer hexagon."""
    for v in hex_verts:
        for i in range(6):
            p1 = outer_verts[i]
            p2 = outer_verts[(i + 1) % 6]
            cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
            if cross < -1e-6:
                return False
    return True

def _is_valid(inner_data, outer_data, s):
    """Full validity check."""
    outer_verts = _hex_vertices(outer_data[0], outer_data[1], s, outer_data[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False
    inner_verts_list = []
    for i in range(12):
        v = _hex_vertices(inner_data[i, 0], inner_data[i, 1], 1.0, inner_data[i, 2])
        inner_verts_list.append(v)
        if not _check_inside(v, outer_verts):
            return False
    for i in range(12):
        for j in range(i + 1, 12):
            if not _check_separation(inner_verts_list[i], inner_verts_list[j]):
                return False
    return True

def _get_all_inner_vertices(inner_data):
    """Get all 72 vertices from all 12 inner hexagons."""
    all_verts = []
    for i in range(12):
        v = _hex_vertices(inner_data[i, 0], inner_data[i, 1], 1.0, inner_data[i, 2])
        all_verts.append(v)
    return all_verts

def _compute_s_and_gradient(inner_data, outer_angle_deg):
    """
    Compute the required outer side length s and the subgradient with respect to inner positions.
    Returns:
    s: float
    grad: (12, 2) array, subgradient of s w.r.t each hexagon center.
    active_hex: index of the hexagon contributing to the max projection (for potential refinement)
    """
    a = math.radians(outer_angle_deg)
    # Normals for the outer hexagon edges (pointing inward from center)
    # Vertices are at a + 60i. Edges are between v_i and v_{i+1}.
    # The normal to edge i points inwards.
    # For a hexagon centered at origin, edge i is perpendicular to vector at angle a + 30 + 60i?
    # Let's derive.
    # Vertex i: (cos(a+60i), sin(a+60i)).
    # Edge i connects v_i and v_{i+1}.
    # Midpoint angle is a + 60i + 30.
    # The normal pointing OUT is in direction a + 60i + 30.
    # The normal pointing IN is opposite.
    # The constraint is that all inner vertices v must satisfy:
    # (v - center) . n_out <= apothem_outer
    # n_out is unit vector.
    # apothem_outer = s * sqrt(3) / 2.
    # So s >= (v - center) . n_out / (sqrt(3)/2) for all v, n_out.
    # Since we center the outer hex at 0,0 for calculation (we shift inner positions relative to outer center),
    # we assume outer center is (0,0).
    
    normals_out = np.empty((6, 2))
    for i in range(6):
        theta = a + math.radians(30 + 60 * i)
        normals_out[i] = [math.cos(theta), math.sin(theta)]
    
    max_proj = 0.0
    grad = np.zeros((12, 2))
    
    # Vectorized calculation
    # inner_data[:, :2] are centers.
    # For each hex i, vertices are center_i + R * v_local
    # v_local are fixed for angle 0 (since inner angles are 0).
    # v_local = _hex_vertices(0, 0, 1, 0)
    
    v_local = _hex_vertices(0, 0, 1.0, 0.0)
    
    # For each hexagon i:
    # verts_i = centers[i] + v_local
    # proj = verts_i @ normals_out
    # We need max over i, k (vertex index), j (normal index) of proj[i,k,j]
    
    # To find gradient, we need to know which (i, k, j) is the max.
    # If multiple are max, we can sum gradients or pick one.
    
    # Let's compute all projections.
    # Shape of centers: (12, 2)
    # Shape of v_local: (6, 2)
    # Shape of normals_out: (6, 2)
    
    # proj[i, k, j] = (centers[i] + v_local[k]) @ normals_out[j]
    # = centers[i] @ normals_out[j] + v_local[k] @ normals_out[j]
    
    # c_proj[i, j] = centers[i] @ normals_out[j] -> (12, 6)
    c_proj = inner_data[:, :2] @ normals_out.T
    
    # v_proj[k, j] = v_local[k] @ normals_out[j] -> (6, 6)
    v_proj = v_local @ normals_out.T
    
    # total_proj[i, k, j] = c_proj[i, j] + v_proj[k, j]
    # We want max over i, k, j.
    
    # Expand c_proj to (12, 6, 1) and v_proj to (1, 6, 6)
    # Then add.
    # Actually, we can loop over j (normals) to save memory, or just use broadcasting.
    # (12, 1, 6) + (1, 6, 6) -> (12, 6, 6)
    total_proj = c_proj[:, np.newaxis, :] + v_proj[np.newaxis, :, :]
    
    # Find the max value and its indices
    max_val = np.max(total_proj)
    # Get indices
    # Flatten and argmax
    idx_flat = np.argmax(total_proj)
    i_idx, k_idx, j_idx = np.unravel_index(idx_flat, total_proj.shape)
    
    s_val = max_val / (math.sqrt(3) / 2)
    
    # Gradient w.r.t center i_idx
    # d s / d center[i_idx] = (1 / (sqrt(3)/2)) * normals_out[j_idx]
    # Other hexagons have 0 gradient for this specific max term.
    # However, if we move a hexagon, it might become the new max.
    # For a subgradient step, we use the current active constraint.
    
    grad[i_idx] = normals_out[j_idx] / (math.sqrt(3) / 2)
    
    return s_val, grad, i_idx

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    start_time = time.time()
    deadline = start_time + 580

    # Basis vectors for hex lattice
    e1 = np.array([math.sqrt(3), 0])
    e2 = np.array([math.sqrt(3)/2, 1.5])
    
    # Initial configuration: Compact cluster
    # Let's try the 3x4 grid first, but centered.
    positions = []
    for i in range(3):
        for j in range(4):
            pos = i * e1 + j * e2
            positions.append([pos[0], pos[1], 0.0])
            
    inner = np.array(positions)
    com = np.mean(inner[:, :2], axis=0)
    inner[:, :2] -= com
    
    # Optimize outer angle for this initial config
    best_s = float('inf')
    best_inner = inner.copy()
    best_outer = np.array([0, 0, 0.0])
    
    for angle_deg in np.linspace(0, 60, 121):
        s_cand, _, _ = _compute_s_and_gradient(inner, angle_deg)
        if s_cand < best_s:
            best_s = s_cand
            best_inner = inner.copy()
            best_outer = np.array([0, 0, angle_deg])
            
    current_inner = best_inner.copy()
    current_outer = best_outer.copy()
    current_s = best_s
    
    # Local optimization using subgradient descent
    step_size = 0.05
    no_improve_count = 0
    max_no_improve = 50
    T = 0.1
    
    np.random.seed(42)
    
    for iteration in range(20000):
        if time.time() > deadline:
            break
            
        # Compute gradient
        s_val, grad, active_idx = _compute_s_and_gradient(current_inner, current_outer[2])
        
        # Try to move the active hexagon
        # Move in direction -grad[active_idx]
        # But we must ensure we don't collide.
        
        # Also try random moves to escape local minima
        if np.random.random() < 0.1:
            idx = np.random.randint(0, 12)
            dx = np.random.uniform(-step_size, step_size)
            dy = np.random.uniform(-step_size, step_size)
        else:
            idx = active_idx
            # Scale gradient step
            dx = -grad[idx, 0] * step_size
            dy = -grad[idx, 1] * step_size
            
        trial_inner = current_inner.copy()
        trial_inner[idx, 0] += dx
        trial_inner[idx, 1] += dy
        
        # Check separation
        valid = True
        trial_verts = _hex_vertices(trial_inner[idx, 0], trial_inner[idx, 1], 1.0, trial_inner[idx, 2])
        for j in range(12):
            if j == idx:
                continue
            other_verts = _hex_vertices(trial_inner[j, 0], trial_inner[j, 1], 1.0, trial_inner[j, 2])
            if not _check_separation(trial_verts, other_verts):
                valid = False
                break
        
        if valid:
            # Calculate new s
            new_s, _, _ = _compute_s_and_gradient(trial_inner, current_outer[2])
            
            # We can also try to adjust the outer angle slightly
            best_trial_s = new_s
            best_trial_angle = current_outer[2]
            # Small angle search
            for angle_offset in np.linspace(-1, 1, 5):
                trial_angle = current_outer[2] + angle_offset
                s_cand, _, _ = _compute_s_and_gradient(trial_inner, trial_angle)
                if s_cand < best_trial_s:
                    best_trial_s = s_cand
                    best_trial_angle = trial_angle
            
            delta_s = current_s - best_trial_s
            if delta_s > 0:
                trial_outer = np.array([0, 0, best_trial_angle])
                if _is_valid(trial_inner, trial_outer, best_trial_s):
                    current_inner = trial_inner
                    current_outer = trial_outer
                    current_s = best_trial_s
                    no_improve_count = 0
                    T = max(T * 0.99, 0.01)
            else:
                if np.random.random() < math.exp(-delta_s / T):
                    trial_outer = np.array([0, 0, best_trial_angle])
                    if _is_valid(trial_inner, trial_outer, best_trial_s):
                        current_inner = trial_inner
                        current_outer = trial_outer
                        current_s = best_trial_s
                        T = max(T * 0.99, 0.01)
                else:
                    no_improve_count += 1
        else:
            no_improve_count += 1
            
        if no_improve_count >= max_no_improve:
            step_size *= 0.9
            no_improve_count = 0
            T *= 0.9
            if step_size < 0.001:
                break
                
    if not _is_valid(current_inner, current_outer, current_s):
        current_inner = best_inner
        current_outer = best_outer
        current_s = best_s

    return current_inner, current_outer, current_s

# EVOLVE-BLOCK-END

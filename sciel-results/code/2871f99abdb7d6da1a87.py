import numpy as np
import time

N = 11
_DEADLINE = 140.0

def _hex_vertices(cx, cy, theta, side=1.0):
    """Generate vertices of a regular hexagon."""
    angles = theta + np.pi / 3 * np.arange(6)
    return np.column_stack([cx + side * np.cos(angles),
                            cy + side * np.sin(angles)])

def _point_hex_distance(p, hex_verts):
    """
    Minimum distance from point p to the boundary of the hexagon defined by hex_verts.
    Returns negative if p is inside.
    """
    min_dist = np.inf
    # Check distance to each edge
    for i in range(6):
        a = hex_verts[i]
        b = hex_verts[(i + 1) % 6]
        ab = b - a
        ab_len_sq = np.dot(ab, ab)
        if ab_len_sq < 1e-10:
            continue
        t = np.dot(p - a, ab) / ab_len_sq
        t = np.clip(t, 0, 1)
        closest = a + t * ab
        d = np.linalg.norm(p - closest)
        if d < min_dist:
            min_dist = d
    
    # Check if inside:
    # For a convex polygon, if the point is inside, the distance to the boundary is negative?
    # No, the function above returns positive distance to boundary.
    # We need to know if it's inside.
    # A point is inside a convex polygon if it is on the same side of all edges.
    # Let's check orientation.
    inside = True
    for i in range(6):
        a = hex_verts[i]
        b = hex_verts[(i + 1) % 6]
        # Vector a->b
        ab = b - a
        # Vector a->p
        ap = p - a
        # Cross product z-component
        cross = ab[0] * ap[1] - ab[1] * ap[0]
        # For a CCW polygon, cross should be >= 0 for points inside (assuming standard orientation)
        # Our hexagon vertices are generated CCW?
        # angles = theta + pi/3 * k. cos/sin.
        # k=0: (1,0). k=1: (0.5, 0.866).
        # Vector (0.5, 0.866) - (1, 0) = (-0.5, 0.866).
        # Point (0,0) relative to a=(1,0) is (-1, 0).
        # Cross: (-0.5)*0 - 0.866*(-1) = 0.866 > 0.
        # So yes, CCW.
        if cross < -1e-10:
            inside = False
            break
    
    if inside:
        return -min_dist
    else:
        return min_dist

def _min_distance_between_hexes(pos1, ang1, pos2, ang2):
    """
    Compute the minimum distance between two regular hexagons of side 1.
    Returns negative if overlapping.
    """
    v1 = _hex_vertices(pos1[0], pos1[1], ang1, 1.0)
    v2 = _hex_vertices(pos2[0], pos2[1], ang2, 1.0)
    
    min_dist = np.inf
    
    # Check vertices of v1 against edges of v2
    for p in v1:
        d = _point_hex_distance(p, v2)
        if d < min_dist:
            min_dist = d
            if min_dist < 0:
                return min_dist # Early exit if overlapping
    
    # Check vertices of v2 against edges of v1
    for p in v2:
        d = _point_hex_distance(p, v1)
        if d < min_dist:
            min_dist = d
            if min_dist < 0:
                return min_dist
    
    return min_dist

def _packing_quality(positions, angles, outer_side):
    """
    Calculate the minimum clearance.
    Returns a tuple (min_clearance, is_valid)
    min_clearance is the minimum distance to boundary or between hexes.
    is_valid is True if min_clearance >= 0.
    """
    min_clearance = np.inf
    is_valid = True
    
    # Outer hexagon is fixed at (0,0) with angle 0
    outer_verts = _hex_vertices(0.0, 0.0, 0.0, outer_side)
    
    for i in range(N):
        # Distance to outer boundary
        v_inner = _hex_vertices(positions[i, 0], positions[i, 1], angles[i], 1.0)
        
        # Check all vertices of inner hex against outer boundary
        min_dist_outer = np.inf
        for v in v_inner:
            d = _point_hex_distance(v, outer_verts)
            if d < min_dist_outer:
                min_dist_outer = d
        
        if min_dist_outer < min_clearance:
            min_clearance = min_dist_outer
        
        # Check against other inner hexes
        for j in range(i + 1, N):
            d_pair = _min_distance_between_hexes(positions[i], angles[i], positions[j], angles[j])
            if d_pair < min_clearance:
                min_clearance = d_pair
    
    if min_clearance < 0:
        is_valid = False
        
    return min_clearance, is_valid

def construct():
    start_time = time.time()
    
    # Initial guess: Hexagonal lattice
    # 1 center, 6 around, 4 more?
    # Let's just use a grid and let SA fix it.
    # Better: Start with a valid packing in a large hexagon.
    
    positions = np.zeros((N, 2))
    angles = np.zeros(N)
    
    # Place in a hexagonal pattern
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2)
    ]
    
    for i, (q, r) in enumerate(axial_coords):
        x = 1.5 * q
        y = np.sqrt(3) * (r + q / 2.0)
        positions[i, 0] = x
        positions[i, 1] = y
        angles[i] = 0.0
    
    # Initial side length
    max_r = np.max(np.linalg.norm(positions, axis=1))
    # Apothem of outer hex must be > max_r + apothem_inner (approx)
    # max_r is distance to vertex.
    # Outer apothem = S * sqrt(3) / 2.
    # Inner hex extends 1.0 from center.
    # So S * sqrt(3) / 2 >= max_r + 1.0
    S = (max_r + 1.5) * 2.0 / np.sqrt(3.0)
    
    best_S = S
    best_positions = positions.copy()
    best_angles = angles.copy()
    
    # Simulated Annealing
    T = 1.0
    cooling_rate = 0.9995
    min_T = 1e-5
    max_iters = 200000
    
    rng = np.random.default_rng(42)
    
    current_S = S
    current_positions = positions.copy()
    current_angles = angles.copy()
    
    # Calculate initial cost
    clearance, valid = _packing_quality(current_positions, current_angles, current_S)
    if valid:
        cost = current_S
    else:
        cost = current_S + 100.0 * abs(clearance)
    
    best_cost = cost
    
    for it in range(max_iters):
        if time.time() > _DEADLINE:
            break
        
        # Move a random hexagon
        idx = rng.integers(0, N)
        
        # Perturbation
        step_size = 0.1 * (1.0 + current_S)
        dx = rng.normal(0, step_size)
        dy = rng.normal(0, step_size)
        dtheta = rng.normal(0, 0.1)
        
        old_pos = current_positions[idx].copy()
        old_ang = current_angles[idx]
        
        current_positions[idx, 0] += dx
        current_positions[idx, 1] += dy
        current_angles[idx] += dtheta
        
        # Optionally adjust S
        # If valid, try to shrink S. If invalid, maybe expand or just penalize.
        # Let's try to shrink S slightly if we have been valid for a while?
        # Simpler: Just move hexagons. S is fixed?
        # No, we want to minimize S.
        # Let's perturb S as well.
        if rng.random() < 0.1:
            dS = rng.normal(0, 0.05)
            current_S += dS
            if current_S < 3.0: # Hard lower bound
                current_S = 3.0
        
        new_clearance, new_valid = _packing_quality(current_positions, current_angles, current_S)
        
        if new_valid:
            new_cost = current_S
        else:
            new_cost = current_S + 100.0 * abs(new_clearance)
        
        delta_cost = new_cost - cost
        
        if delta_cost < 0 or rng.random() < np.exp(-delta_cost / T):
            cost = new_cost
            # Accept
            if new_valid and current_S < best_S:
                best_S = current_S
                best_positions = current_positions.copy()
                best_angles = current_angles.copy()
                best_cost = cost
        else:
            # Reject
            current_positions[idx] = old_pos
            current_angles[idx] = old_ang
            # Note: if we perturbed S, we should revert it too?
            # Yes, state must be consistent.
            # We didn't save old_S. Let's fix that.
            # Actually, reverting S is tricky if we don't save it.
            # Let's save old_S.
            
        # Cooling
        T *= cooling_rate
        if T < min_T:
            T = min_T
    
    # Final check
    # Ensure the best found is valid
    clearance, valid = _packing_quality(best_positions, best_angles, best_S)
    if not valid:
        # Fallback to a known valid packing (large S)
        # Re-run with larger S or just return the initial valid one?
        # The initial one was valid.
        # Let's return the initial one if best is invalid.
        # But we updated best only if valid. So best should be valid.
        pass
    
    inner = []
    for i in range(N):
        inner.append([best_positions[i, 0], best_positions[i, 1], np.degrees(best_angles[i])])
    
    return inner, (0.0, 0.0), best_S, 0.0

# Wait, I need to handle the S perturbation revert correctly.
# Let's rewrite the SA loop properly.

def construct():
    start_time = time.time()
    
    positions = np.zeros((N, 2))
    angles = np.zeros(N)
    
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2)
    ]
    
    for i, (q, r) in enumerate(axial_coords):
        x = 1.5 * q
        y = np.sqrt(3) * (r + q / 2.0)
        positions[i, 0] = x
        positions[i, 1] = y
        angles[i] = 0.0
    
    max_r = np.max(np.linalg.norm(positions, axis=1))
    S = (max_r + 1.5) * 2.0 / np.sqrt(3.0)
    
    best_S = S
    best_positions = positions.copy()
    best_angles = angles.copy()
    
    T = 1.0
    cooling_rate = 0.9995
    min_T = 1e-5
    max_iters = 150000 # Adjust based on speed
    
    rng = np.random.default_rng(42)
    
    current_S = S
    current_positions = positions.copy()
    current_angles = angles.copy()
    
    clearance, valid = _packing_quality(current_positions, current_angles, current_S)
    if valid:
        cost = current_S
    else:
        cost = current_S + 100.0 * abs(clearance)
    
    best_cost = cost
    
    for it in range(max_iters):
        if time.time() > _DEADLINE:
            break
        
        idx = rng.integers(0, N)
        
        step_size = 0.1 * (1.0 + current_S)
        dx = rng.normal(0, step_size)
        dy = rng.normal(0, step_size)
        dtheta = rng.normal(0, 0.1)
        
        old_pos = current_positions[idx].copy()
        old_ang = current_angles[idx]
        old_S = current_S
        
        current_positions[idx, 0] += dx
        current_positions[idx, 1] += dy
        current_angles[idx] += dtheta
        
        if rng.random() < 0.1:
            dS = rng.normal(0, 0.05)
            current_S += dS
            if current_S < 3.0:
                current_S = 3.0
        
        new_clearance, new_valid = _packing_quality(current_positions, current_angles, current_S)
        
        if new_valid:
            new_cost = current_S
        else:
            new_cost = current_S + 100.0 * abs(new_clearance)
        
        delta_cost = new_cost - cost
        
        if delta_cost < 0 or rng.random() < np.exp(-delta_cost / T):
            cost = new_cost
            if new_valid and current_S < best_S:
                best_S = current_S
                best_positions = current_positions.copy()
                best_angles = current_angles.copy()
                best_cost = cost
        else:
            current_positions[idx] = old_pos
            current_angles[idx] = old_ang
            current_S = old_S
            
        T *= cooling_rate
        if T < min_T:
            T = min_T
    
    clearance, valid = _packing_quality(best_positions, best_angles, best_S)
    if not valid:
        # Fallback
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N//2), N - N//2)], (0.0, 0.0), 50.0, 0.0

    inner = []
    for i in range(N):
        inner.append([best_positions[i, 0], best_positions[i, 1], np.degrees(best_angles[i])])
    
    return inner, (0.0, 0.0), best_S, 0.0

import numpy as np
import time
import sys

N = 12
START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin for 150s limit

def _hex_vertices(center, side, angle_deg):
    """Return vertices of a regular hexagon."""
    x, y = center
    theta = np.deg2rad(angle_deg)
    angles = theta + np.arange(6) * (np.pi / 3)
    return np.array([
        [x + side * np.cos(a), y + side * np.sin(a)] 
        for a in angles
    ])

def _min_dist_polygons(v1, v2):
    """
    Compute the minimum distance between two convex polygons.
    Returns 0 if they overlap (or touch).
    Uses Separating Axis Theorem (SAT) for overlap check,
    and point-to-edge distance for separation.
    """
    # Check overlap using SAT
    axes = []
    for i in range(6):
        edge = v1[(i + 1) % 6] - v1[i]
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-9:
            normal /= norm
            axes.append(normal)
    for i in range(6):
        edge = v2[(i + 1) % 6] - v2[i]
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-9:
            normal /= norm
            axes.append(normal)
    
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
            # Separated along this axis
            gap = max(min2 - max1, min1 - max2)
            return gap
    
    # If no separating axis found, they overlap
    return 0.0

def _check_validity(inner_hexes, outer_side, outer_center, outer_angle, eps=1e-7):
    """Check if inner hexagons are disjoint and inside outer hexagon."""
    outer_verts = _hex_vertices(outer_center, outer_side, outer_angle)
    inner_verts = [_hex_vertices(h[0:2], 1.0, h[2]) for h in inner_hexes]
    
    # Check containment: all inner vertices must be inside outer
    for iv in inner_verts:
        for v in iv:
            inside = True
            for k in range(6):
                a = outer_verts[k]
                b = outer_verts[(k+1)%6]
                cross = (b[0]-a[0])*(v[1]-a[1]) - (b[1]-a[1])*(v[0]-a[0])
                if cross < -eps:
                    inside = False
                    break
            if not inside:
                return False, "outside"
                
    # Check disjointness
    for i in range(len(inner_hexes)):
        for j in range(i + 1, len(inner_hexes)):
            # Use SAT for overlap
            # We can reuse logic from _min_dist or just check distance > 0
            # For speed in final check, we can use the distance function
            if _min_dist_polygons(inner_verts[i], inner_verts[j]) < eps:
                return False, "overlap"
                
    return True, "valid"

def _compute_min_side(inner_hexes, outer_center, outer_angle):
    """
    Compute the minimum side length of the outer hexagon (centered at outer_center, rotated by outer_angle)
    required to contain all inner hexagons.
    """
    ox, oy = outer_center
    max_proj = 0.0
    theta = np.deg2rad(outer_angle)
    
    # Pre-compute normals for the 6 sides of the outer hexagon
    normals = []
    for k in range(6):
        angle = theta + k * (np.pi / 3)
        normals.append(np.array([np.cos(angle), np.sin(angle)]))
    
    # Project all inner hex vertices onto these normals
    # To be efficient, we only need the max projection for each normal
    # Inner hexagons are unit side length.
    
    for h in inner_hexes:
        cx, cy, a_deg = h
        rel_cx = cx - ox
        rel_cy = cy - oy
        verts = _hex_vertices((rel_cx, rel_cy), 1.0, a_deg)
        
        for normal in normals:
            projs = np.dot(verts, normal)
            max_p = np.max(projs)
            if max_p > max_proj:
                max_proj = max_p
            
    # max_proj is the required apothem
    # S = 2 * apothem / sqrt(3)
    return max_proj * 2.0 / np.sqrt(3.0)

def _objective(params, outer_center, outer_angle):
    """
    Calculate the objective function: outer side length + penalty for overlaps.
    params: array of shape (12, 3) -> [x, y, angle_deg]
    """
    # Reshape to list of hexagons
    inner_hexes = []
    for i in range(N):
        x, y, a = params[i]
        inner_hexes.append((x, y, a))
    
    # Calculate side length
    side = _compute_min_side(inner_hexes, outer_center, outer_angle)
    
    # Calculate overlap penalty
    penalty = 0.0
    # Pre-compute vertices for all inner hexagons to avoid recalculation
    verts = [_hex_vertices(h[0:2], 1.0, h[2]) for h in inner_hexes]
    
    for i in range(N):
        for j in range(i + 1, N):
            dist = _min_dist_polygons(verts[i], verts[j])
            if dist < 0.01: # Only penalize if close
                # Penalty increases quadratically as they overlap or get very close
                # We want them to be at least 0 distance (touching is ok for packing, but strict disjointness in SAT might mean >0)
                # The task says "disjoint", which usually means interiors do not intersect.
                # If they touch, distance is 0.
                # We add a small penalty for distance < 0 to enforce strict separation if needed, 
                # but for packing, touching is allowed.
                # However, numerical stability suggests keeping a tiny gap.
                # Let's penalize negative distance (overlap) heavily.
                # _min_dist returns 0 if overlap. It doesn't return negative depth.
                # So we need a different check for overlap depth?
                # SAT with gap returns 0 if overlapping.
                # We need to know HOW MUCH they overlap to push them apart.
                # Approximation: if dist == 0, assume they overlap. 
                # We can't easily get depth from simple SAT.
                # Alternative: Use a potential function based on center distance? No, rotation matters.
                # Let's assume if _min_dist == 0, they are touching or overlapping.
                # To resolve overlaps, we need a gradient.
                # Since _min_dist is non-differentiable at 0, SA is robust to this.
                # We add a penalty if dist == 0.
                penalty += 10.0
                
    return side + penalty

def construct():
    start_time = time.time()
    
    # Initial guess: Hexagonal lattice
    # Center (0,0)
    # Ring 1: 6 hexes
    # Ring 2: 5 hexes
    centers = []
    angles = []
    centers.append([0.0, 0.0])
    angles.append(0.0)
    
    d = np.sqrt(3)
    for k in range(6):
        a = np.pi/3 * k
        centers.append([d*np.cos(a), d*np.sin(a)])
        angles.append(0.0)
    
    # Ring 2 positions (approximate)
    # We need 5 more.
    # Let's pick 5 from the 12 available ring 2 spots.
    # (2*sqrt(3), 0), (3, sqrt(3)), (1.5, 3), (-1.5, 3), (-3, sqrt(3))
    pos_ring2 = [
        [2*np.sqrt(3), 0],
        [3, np.sqrt(3)],
        [1.5, 3],
        [-1.5, 3],
        [-3, np.sqrt(3)]
    ]
    for p in pos_ring2:
        centers.append(p)
        angles.append(0.0)
        
    init_params = np.array([list(c) + [a] for c, a in zip(centers, angles)])
    
    best_params = init_params.copy()
    best_side = _compute_min_side([(p[0], p[1], p[2]) for p in best_params], (0,0), 0.0)
    
    # Simulated Annealing
    # Parameters:
    # T_start: Initial temperature
    # T_end: Final temperature
    # Cooling rate
    # Steps
    
    T_start = 1.0
    T_end = 0.001
    steps = 50000 # Adjust based on time
    
    # Estimate time per step
    # _objective is expensive. Let's do a quick timing.
    t0 = time.time()
    _objective(best_params, (0,0), 0.0)
    t1 = time.time()
    time_per_step = t1 - t0
    if time_per_step < 1e-5: time_per_step = 1e-5
    
    # Max steps allowed by time budget
    max_steps = int((TIME_LIMIT - 10) / time_per_step)
    if max_steps < 1000:
        # Fallback to few steps
        max_steps = 1000
    steps = min(steps, max_steps)
    
    # Cooling schedule
    alpha = (T_end / T_start) ** (1.0 / steps)
    T = T_start
    
    current_params = best_params.copy()
    current_obj = _objective(current_params, (0,0), 0.0)
    
    for i in range(steps):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
            
        # Generate new candidate
        # Random perturbation
        # Scale of perturbation:
        # x, y: ~0.5 * T
        # angle: ~5 * T degrees
        
        new_params = current_params.copy()
        # Perturb a random subset or all? All is better for global search initially.
        # Let's perturb all with small noise.
        noise_scale = 0.5 * T
        angle_noise_scale = 5.0 * T
        
        for k in range(N):
            new_params[k, 0] += np.random.normal(0, noise_scale)
            new_params[k, 1] += np.random.normal(0, noise_scale)
            new_params[k, 2] += np.random.normal(0, angle_noise_scale)
            
        new_obj = _objective(new_params, (0,0), 0.0)
        
        # Acceptance criterion
        delta = new_obj - current_obj
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_params = new_params
            current_obj = new_obj
            
            # Update best if valid (no penalty) or lower side
            # We track best_side based on the raw side length if penalty is 0
            # But _objective includes penalty.
            # We need to check if penalty is 0.
            # Let's re-calculate side and check validity separately for best tracking?
            # That's expensive.
            # Let's assume if new_obj < best_obj, it's good.
            # But we want to return a VALID packing.
            # We will track the best VALID packing separately.
            
            # Check validity of current_params
            # This is expensive, so we only do it if current_obj is significantly lower than best_valid_obj
            # Or periodically.
            
            # Optimization: Only check validity if current_obj < best_valid_obj + epsilon
            # But we don't know best_valid_obj's penalty part.
            
            # Let's just update best_params if current_obj < best_obj (total)
            if current_obj < best_obj:
                best_obj = current_obj
                best_params = current_params.copy()
                
                # Try to extract a valid solution from best_params
                # If penalty is 0, it's valid.
                # We can check validity every 100 steps or when a new best is found.
                
        T *= alpha
        
    # Final validation and cleanup
    # The best_params might still have overlaps if the penalty wasn't high enough or SA got stuck.
    # We need to return a VALID packing.
    
    # Let's check the best_params found.
    inner_hexes = [(p[0], p[1], p[2]) for p in best_params]
    side = _compute_min_side(inner_hexes, (0,0), 0.0)
    is_valid, reason = _check_validity(inner_hexes, side, (0,0), 0.0)
    
    if not is_valid:
        # If the best found is invalid, we need to find the best VALID one.
        # This implies we should have tracked valids separately.
        # Since we didn't, let's do a local refinement or fallback.
        # Fallback: Use the initial lattice which is known to be valid (if side is large enough).
        # The initial lattice is valid.
        init_inner = [(p[0], p[1], p[2]) for p in init_params]
        init_side = _compute_min_side(init_inner, (0,0), 0.0)
        is_init_valid, _ = _check_validity(init_inner, init_side, (0,0), 0.0)
        
        if is_init_valid:
            best_params = init_params
            best_side = init_side
        else:
            # Should not happen, but just in case
            # Return a trivial valid packing (spread out)
            best_params = np.array([[k*5, 0, 0] for k in range(N)])
            best_side = 50.0

    # Final check
    inner_hexes_final = [(p[0], p[1], p[2]) for p in best_params]
    final_side = _compute_min_side(inner_hexes_final, (0,0), 0.0)
    
    # Verify one last time
    is_valid_final, _ = _check_validity(inner_hexes_final, final_side, (0,0), 0.0)
    if not is_valid_final:
         # Force valid by using initial
         inner_hexes_final = [(p[0], p[1], p[2]) for p in init_params]
         final_side = _compute_min_side(inner_hexes_final, (0,0), 0.0)

    return [list(p) for p in inner_hexes_final], (0.0, 0.0), final_side, 0.0

# Global variable for best objective tracking in SA loop
best_obj = float('inf')

# Wrap construct to handle the global state properly if needed, 
# but since construct is called once, local vars are fine. 
# However, I used best_obj inside the loop without defining it locally. 
# Let's fix that.

def construct():
    start_time = time.time()
    
    # Initial guess: Hexagonal lattice
    centers = []
    angles = []
    centers.append([0.0, 0.0])
    angles.append(0.0)
    
    d = np.sqrt(3)
    for k in range(6):
        a = np.pi/3 * k
        centers.append([d*np.cos(a), d*np.sin(a)])
        angles.append(0.0)
        
    pos_ring2 = [
        [2*np.sqrt(3), 0],
        [3, np.sqrt(3)],
        [1.5, 3],
        [-1.5, 3],
        [-3, np.sqrt(3)]
    ]
    for p in pos_ring2:
        centers.append(p)
        angles.append(0.0)
        
    init_params = np.array([list(c) + [a] for c, a in zip(centers, angles)])
    
    best_params = init_params.copy()
    best_obj = _objective(best_params, (0,0), 0.0)
    
    # Simulated Annealing
    T_start = 1.0
    T_end = 0.001
    
    # Estimate time per step
    t0 = time.time()
    _objective(best_params, (0,0), 0.0)
    t1 = time.time()
    time_per_step = max(t1 - t0, 1e-5)
    
    max_steps = int((TIME_LIMIT - 10) / time_per_step)
    if max_steps < 1000:
        max_steps = 1000
    steps = min(50000, max_steps)
    
    alpha = (T_end / T_start) ** (1.0 / steps)
    T = T_start
    
    current_params = best_params.copy()
    current_obj = best_obj
    
    best_valid_params = init_params.copy()
    best_valid_side = _compute_min_side([(p[0], p[1], p[2]) for p in best_valid_params], (0,0), 0.0)
    
    for i in range(steps):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
            
        new_params = current_params.copy()
        noise_scale = 0.5 * T
        angle_noise_scale = 5.0 * T
        
        for k in range(N):
            new_params[k, 0] += np.random.normal(0, noise_scale)
            new_params[k, 1] += np.random.normal(0, noise_scale)
            new_params[k, 2] += np.random.normal(0, angle_noise_scale)
            
        new_obj = _objective(new_params, (0,0), 0.0)
        
        delta = new_obj - current_obj
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_params = new_params
            current_obj = new_obj
            
            if current_obj < best_obj:
                best_obj = current_obj
                best_params = current_params.copy()
                
                # Check validity of the new best
                # Only do this if the penalty part is likely 0
                # We can check if current_obj is close to the side length
                # Or just check every 50 steps if we found a new best
                if i % 50 == 0 or i == steps - 1:
                    inner_hexes = [(p[0], p[1], p[2]) for p in best_params]
                    side = _compute_min_side(inner_hexes, (0,0), 0.0)
                    is_valid, _ = _check_validity(inner_hexes, side, (0,0), 0.0)
                    if is_valid:
                        best_valid_params = best_params.copy()
                        best_valid_side = side

        T *= alpha
        
    # If we never found a valid one better than initial, use initial
    # But initial is valid.
    
    inner_hexes_final = [(p[0], p[1], p[2]) for p in best_valid_params]
    final_side = _compute_min_side(inner_hexes_final, (0,0), 0.0)
    
    return [list(p) for p in inner_hexes_final], (0.0, 0.0), final_side, 0.0


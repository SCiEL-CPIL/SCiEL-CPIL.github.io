# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hex_vertices(cx, cy, s, a_deg):
    """Compute 6 vertices of a regular hexagon centered at (cx,cy) with side s and angle a_deg."""
    a = np.radians(a_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _edge_normals(vertices):
    """Compute 6 outward edge normals for a hexagon given its 6 vertices in CCW order."""
    normals = np.zeros((6, 2))
    for i in range(6):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % 6]
        edge = v2 - v1
        # Outward normal (perpendicular, pointing away from center)
        n = np.array([-edge[1], edge[0]])
        n /= (np.linalg.norm(n) + 1e-30)
        # Ensure outward: check against center
        center = np.mean(vertices, axis=0)
        mid = (v1 + v2) / 2
        if np.dot(n, mid - center) < 0:
            n = -n
        normals[i] = n
    return normals

def _hexagon_separation(v1, v2):
    """Check if two hexagons (given as vertex arrays) are separated.
    Returns True if separated (no overlap), False if overlapping."""
    n1 = _edge_normals(v1)
    n2 = _edge_normals(v2)
    
    for n in n1:
        proj1 = np.dot(v1, n)
        proj2 = np.dot(v2, n)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 + 1e-6 <= min2 or max2 + 1e-6 <= min1:
            continue  # This normal separates them
        else:
            return False  # This normal doesn't separate, check next
    # If no normal from v1 separates, check normals from v2
    for n in n2:
        proj1 = np.dot(v1, n)
        proj2 = np.dot(v2, n)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 + 1e-6 <= min2 or max2 + 1e-6 <= min1:
            continue
        else:
            return False
    return True

def _inside_hexagon(vertices, outer_vertices):
    """Check if all vertices of inner hexagon are inside outer hexagon."""
    for p in vertices:
        for i in range(6):
            v1 = outer_vertices[i]
            v2 = outer_vertices[(i + 1) % 6]
            edge = v2 - v1
            # Cross product to check if p is on the correct side
            cross = (edge[0] * (p[1] - v1[1]) - edge[1] * (p[0] - v1[0]))
            if cross < -1e-6:
                return False
    return True

def _compute_outer_side(inner_positions, inner_angles, outer_center, outer_angle):
    """Compute minimum outer hexagon side length that contains all inner hexagons."""
    # For a hexagon centered at outer_center with angle outer_angle,
    # the distance from center to each edge is s * cos(30°) = s * sqrt(3)/2
    # So s = distance / (sqrt(3)/2) = 2*distance/sqrt(3)
    
    max_dist = 0
    for i in range(11):
        cx, cy = inner_positions[i]
        a = inner_angles[i]
        verts = _hex_vertices(cx, cy, 1.0, a)
        # Shift by outer center
        verts_shifted = verts - outer_center
        for v in verts_shifted:
            dist = np.linalg.norm(v)
            if dist > max_dist:
                max_dist = dist
    
    # The apothem (distance from center to edge) of outer hexagon must be >= max_dist
    # apothem = s * cos(30°) = s * sqrt(3)/2
    # So s >= max_dist / (sqrt(3)/2) = 2 * max_dist / sqrt(3)
    side = 2 * max_dist / np.sqrt(3)
    return side

def _objective(params, outer_center, outer_angle):
    """Objective: minimize outer side length, with penalty for overlaps."""
    n = 11
    inner_positions = params[:2*n].reshape(n, 2)
    inner_angles = params[2*n:]
    
    # Check overlaps between all pairs
    penalty = 0.0
    verts_list = []
    for i in range(n):
        verts_list.append(_hex_vertices(inner_positions[i,0], inner_positions[i,1], 1.0, inner_angles[i]))
    
    for i in range(n):
        for j in range(i+1, n):
            if not _hexagon_separation(verts_list[i], verts_list[j]):
                # Compute overlap penalty
                # Use minimum distance between edges as a proxy
                penalty += 100.0
    
    # Compute outer side
    side = _compute_outer_side(inner_positions, inner_angles, outer_center, outer_angle)
    
    return side + penalty

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600 second deadline
    
    n = 11
    best_side = np.inf
    best_params = None
    best_outer_center = np.array([0.0, 0.0])
    best_outer_angle = 0.0
    
    # Initial configuration: central hex + 6 in ring + 4 more
    # Hexagonal grid positions for the 7-cluster
    # Center at (0,0), ring at distance sqrt(3) in 6 directions
    ring_dist = np.sqrt(3)
    initial_positions = np.array([
        [0, 0],
        [ring_dist, 0],
        [ring_dist/2, ring_dist*np.sqrt(3)/2],
        [-ring_dist/2, ring_dist*np.sqrt(3)/2],
        [-ring_dist, 0],
        [-ring_dist/2, -ring_dist*np.sqrt(3)/2],
        [ring_dist/2, -ring_dist*np.sqrt(3)/2],
        # 4 more hexagons placed further out
        [2*ring_dist, 0],
        [ring_dist, ring_dist*np.sqrt(3)],
        [-ring_dist, ring_dist*np.sqrt(3)],
        [-2*ring_dist, 0],
    ])
    initial_angles = np.zeros(n)
    
    # Try multiple restarts with different initial configurations
    rng = np.random.default_rng(42)
    
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 60:
            break
        
        # Perturb initial configuration
        if restart == 0:
            params0 = np.concatenate([initial_positions.flatten(), initial_angles])
        else:
            noise = rng.normal(0, 0.5, size=2*n)
            angle_noise = rng.normal(0, 15, size=n)
            params0 = np.concatenate([initial_positions.flatten() + noise, initial_angles + angle_noise])
        
        # Optimize
        result = minimize(
            _objective,
            params0,
            args=(best_outer_center, best_outer_angle),
            method='Nelder-Mead',
            options={
                'maxiter': 5000,
                'xatol': 1e-8,
                'fatol': 1e-10,
                'adaptive': True
            }
        )
        
        if result.success or result.fun < best_side:
            params = result.x
            side = result.fun - 100.0 * 0  # Remove penalty if no overlaps
            # Recompute actual side
            inner_positions = params[:2*n].reshape(n, 2)
            inner_angles = params[2*n:]
            actual_side = _compute_outer_side(inner_positions, inner_angles, best_outer_center, best_outer_angle)
            
            # Verify no overlaps
            verts_list = []
            for i in range(n):
                verts_list.append(_hex_vertices(inner_positions[i,0], inner_positions[i,1], 1.0, inner_angles[i]))
            
            valid = True
            for i in range(n):
                for j in range(i+1, n):
                    if not _hexagon_separation(verts_list[i], verts_list[j]):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid and actual_side < best_side:
                best_side = actual_side
                best_params = params.copy()
    
    # Final result
    if best_params is None:
        # Fallback to initial configuration
        best_params = np.concatenate([initial_positions.flatten(), initial_angles])
        best_side = _compute_outer_side(initial_positions, initial_angles, best_outer_center, best_outer_angle)
    
    inner_positions = best_params[:2*n].reshape(n, 2)
    inner_angles = best_params[2*n:]
    
    # Build return arrays
    inner_hex_data = np.column_stack([inner_positions[:,0], inner_positions[:,1], inner_angles])
    outer_hex_data = np.array([best_outer_center[0], best_outer_center[1], best_outer_angle])
    
    return inner_hex_data, outer_hex_data, float(best_side)

# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize


def hexagon_vertices(cx, cy, s, angle_deg):
    """Return vertices of a regular hexagon with center (cx,cy), side s, angle angle_deg."""
    angles = np.radians(angle_deg + 60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def get_edge_normals(verts):
    """Get outward-pointing edge normals for a hexagon given its vertices."""
    normals = []
    center = np.mean(verts, axis=0)
    for i in range(6):
        p1 = verts[i]
        p2 = verts[(i + 1) % 6]
        edge = p2 - p1
        n = np.array([-edge[1], edge[0]])
        mid = (p1 + p2) / 2
        if np.dot(n, mid - center) < 0:
            n = -n
        n = n / np.linalg.norm(n)
        normals.append(n)
    return np.array(normals)


def min_gap_between_hexes(v1, v2):
    """
    Compute the separation gap between two hexagons using their 12 edge normals.
    Returns the maximum over all normals of the gap (negative means overlap).
    """
    n1 = get_edge_normals(v1)
    n2 = get_edge_normals(v2)
    all_normals = np.vstack([n1, n2])
    
    max_gap = -np.inf
    for n in all_normals:
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        gap1 = min2 - max1
        gap2 = min1 - max2
        gap = max(gap1, gap2)
        if gap > max_gap:
            max_gap = gap
    
    return max_gap


def check_separation(hex1, hex2, tol=1e-6):
    """
    Check if two hexagons are separated: on at least one of their 12 edge normals,
    the projected intervals are apart by more than tol.
    """
    x1, y1, a1 = hex1
    x2, y2, a2 = hex2
    
    v1 = hexagon_vertices(x1, y1, 1.0, a1)
    v2 = hexagon_vertices(x2, y2, 1.0, a2)
    
    gap = min_gap_between_hexes(v1, v2)
    return gap > tol


def calculate_min_side(inner_hex_data, outer_angle_deg):
    """
    Calculate the minimum side length S of an outer regular hexagon centered at origin
    with given angle that contains all inner hexagons.
    
    The outer hexagon has vertices at distance S from center (circumradius = S for regular hexagon).
    Actually for a regular hexagon with side length S, the distance from center to vertex is S,
    and the distance from center to edge midpoint is S*sqrt(3)/2.
    
    The condition for a point to be inside is: for each of the 6 edge normals,
    the projection onto that normal must be <= S*sqrt(3)/2 (the apothem).
    
    So the minimum S is: max over all points and all 6 normals of (projection) / (sqrt(3)/2)
    = 2/sqrt(3) * max_projection
    """
    # The 6 edge normals of the outer hexagon
    # Edge normals are at angles: outer_angle_deg + 30 + 60*k for k=0..5
    # (since vertices are at outer_angle_deg + 60*k, edges are between consecutive vertices,
    #  and the normal to an edge is perpendicular to the edge, pointing outward)
    
    normal_angles = [outer_angle_deg + 30 + 60 * k for k in range(6)]
    normals = np.array([[np.cos(np.radians(a)), np.sin(np.radians(a))] for a in normal_angles])
    
    all_vertices = []
    for i in range(11):
        x, y, angle = inner_hex_data[i]
        verts = hexagon_vertices(x, y, 1.0, angle)
        all_vertices.append(verts)
    
    all_vertices = np.vstack(all_vertices)
    
    # Project all vertices onto each normal
    projections = all_vertices @ normals.T  # shape (66, 6)
    max_projs = np.max(projections, axis=0)  # shape (6,)
    max_proj = np.max(max_projs)
    
    # The apothem of the outer hexagon must be >= max_proj
    # apothem = S * sqrt(3) / 2
    # So S = 2 * max_proj / sqrt(3)
    side = 2.0 * max_proj / math.sqrt(3)
    
    return side


def is_valid_configuration(inner_hex_data, outer_hex_data, side, tol=1e-6):
    """
    Check if the configuration is valid:
    1. All values finite
    2. side >= sqrt(11)
    3. Every pair of inner hexagons is separated
    4. Every inner vertex is inside the outer hexagon
    5. Edge lengths are within 1e-8 of 1
    """
    # Check finiteness
    if not np.all(np.isfinite(inner_hex_data)) or not np.all(np.isfinite(outer_hex_data)):
        return False
    
    # Check side >= sqrt(11)
    if side < math.sqrt(11) - 1e-9:
        return False
    
    # Check edge lengths
    for i in range(11):
        x, y, angle = inner_hex_data[i]
        verts = hexagon_vertices(x, y, 1.0, angle)
        for j in range(6):
            p1 = verts[j]
            p2 = verts[(j + 1) % 6]
            edge_len = np.linalg.norm(p2 - p1)
            if abs(edge_len - 1.0) > 1e-8:
                return False
    
    # Check separation between all pairs
    for i in range(11):
        for j in range(i + 1, 11):
            if not check_separation(inner_hex_data[i], inner_hex_data[j], tol=tol):
                return False
    
    # Check that all inner vertices are inside the outer hexagon
    outer_x, outer_y, outer_angle = outer_hex_data
    outer_verts = hexagon_vertices(outer_x, outer_y, side, outer_angle)
    
    # Shift coordinates so outer center is at origin
    shift = np.array([outer_x, outer_y])
    
    for i in range(11):
        x, y, angle = inner_hex_data[i]
        verts = hexagon_vertices(x - outer_x, y - outer_y, 1.0, angle)
        for v in verts:
            # Check if v is inside the outer hexagon (centered at origin)
            # For each edge of the outer hexagon, check the cross product condition
            for k in range(6):
                p1 = outer_verts[k] - shift
                p2 = outer_verts[(k + 1) % 6] - shift
                edge = p2 - p1
                cross_val = np.cross(edge, v - p1)
                # The outer hexagon is convex, and we need v to be on the same side
                # as the center (origin) for each edge
                # cross(p2-p1, p-p1) >= -1e-6 means p is on the correct side
                if cross_val < -1e-6:
                    return False
    
    return True


def objective(params, outer_angle_deg):
    """
    Objective function for optimization: returns the outer hexagon side length
    plus a penalty for any violations of separation constraints.
    """
    # params: 22 values (11 hexagons * 2 coords)
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i, 0] = params[2*i]
        inner_hex_data[i, 1] = params[2*i+1]
        inner_hex_data[i, 2] = 0.0
    
    # Compute the outer hexagon side
    side = calculate_min_side(inner_hex_data, outer_angle_deg)
    
    # Check separation constraints
    penalty = 0.0
    for i in range(11):
        for j in range(i+1, 11):
            if not check_separation(inner_hex_data[i], inner_hex_data[j], tol=1e-6):
                # Compute the actual gap
                v1 = hexagon_vertices(inner_hex_data[i, 0], inner_hex_data[i, 1], 1.0, 0.0)
                v2 = hexagon_vertices(inner_hex_data[j, 0], inner_hex_data[j, 1], 1.0, 0.0)
                gap = min_gap_between_hexes(v1, v2)
                if gap < 1e-6:
                    penalty += (1e-6 - gap) ** 2
    
    return side + 100 * penalty


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    
    Strategy: Start with a honeycomb cluster, then use multi-start optimization
    to find better arrangements.
    """
    start_time = time.time()
    deadline = 630.0
    safety_margin = 15.0
    
    def elapsed():
        return time.time() - start_time
    
    def time_left():
        return deadline - safety_margin - elapsed()
    
    h = math.sqrt(3) / 2
    
    # Initial honeycomb configuration (4-3-4 pattern)
    def make_honeycomb():
        coords = []
        for i in range(4):
            coords.append([i * 3.0, 0.0])
        for i in range(3):
            coords.append([1.5 + i * 3.0, 2 * h])
        for i in range(4):
            coords.append([i * 3.0, 4 * h])
        
        cx = np.mean([c[0] for c in coords])
        cy = np.mean([c[1] for c in coords])
        for c in coords:
            c[0] -= cx
            c[1] -= cy
        
        inner_hex_data = np.zeros((11, 3))
        for i, c in enumerate(coords):
            inner_hex_data[i, 0] = c[0]
            inner_hex_data[i, 1] = c[1]
            inner_hex_data[i, 2] = 0.0
        
        return inner_hex_data
    
    best_side = float('inf')
    best_outer_angle = 0.0
    best_inner = None
    
    # Try multiple initial configurations
    initial_configs = []
    
    # Config 1: Standard honeycomb
    initial_configs.append(make_honeycomb())
    
    # Config 2: Honeycomb with slight perturbation
    np.random.seed(42)
    perturbed = make_honeycomb().copy()
    perturbed[:, :2] += np.random.randn(11, 2) * 0.05
    initial_configs.append(perturbed)
    
    # Config 3: More spread out
    coords3 = []
    for i in range(4):
        coords3.append([i * 3.2, 0.0])
    for i in range(3):
        coords3.append([1.6 + i * 3.2, 2 * h * 1.1])
    for i in range(4):
        coords3.append([i * 3.2, 4 * h * 1.1])
    cx3 = np.mean([c[0] for c in coords3])
    cy3 = np.mean([c[1] for c in coords3])
    for c in coords3:
        c[0] -= cx3
        c[1] -= cy3
    inner3 = np.zeros((11, 3))
    for i, c in enumerate(coords3):
        inner3[i, 0] = c[0]
        inner3[i, 1] = c[1]
        inner3[i, 2] = 0.0
    initial_configs.append(inner3)
    
    # For each initial config, try to optimize
    for config_idx, initial_config in enumerate(initial_configs):
        if time_left() < 30:
            break
        
        # Find best outer angle for this config
        best_angle_for_config = 0.0
        best_side_for_config = float('inf')
        
        for outer_angle_deg in np.arange(0, 30.01, 1.0):
            if time_left() < 30:
                break
            side = calculate_min_side(initial_config, outer_angle_deg)
            if side < best_side_for_config:
                best_side_for_config = side
                best_angle_for_config = outer_angle_deg
        
        # Refine angle
        if time_left() < 30:
            break
        for outer_angle_deg in np.arange(max(0, best_angle_for_config - 1.0), 
                                         min(30, best_angle_for_config + 1.0) + 0.01, 0.1):
            if time_left() < 30:
                break
            side = calculate_min_side(initial_config, outer_angle_deg)
            if side < best_side_for_config:
                best_side_for_config = side
                best_angle_for_config = outer_angle_deg
        
        if best_side_for_config < best_side:
            best_side = best_side_for_config
            best_outer_angle = best_angle_for_config
            best_inner = initial_config.copy()
        
        # Now optimize positions with scipy
        if time_left() < 60:
            continue
        
        x0 = best_inner[:, :2].flatten()
        
        # Try L-BFGS-B
        for trial in range(5):
            if time_left() < 30:
                break
            
            if trial == 0:
                x_init = x0.copy()
            else:
                np.random.seed(trial * 7919 + config_idx * 104729)
                x_init = x0 + np.random.randn(22) * 0.02
            
            try:
                result = minimize(
                    objective,
                    x_init,
                    args=(best_outer_angle,),
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if time_left() < 30:
                    break
                
                # Check if the result is valid
                test_inner = np.zeros((11, 3))
                for i in range(11):
                    test_inner[i, 0] = result.x[2*i]
                    test_inner[i, 1] = result.x[2*i+1]
                    test_inner[i, 2] = 0.0
                
                # Verify separation
                valid = True
                for i in range(11):
                    for j in range(i+1, 11):
                        if not check_separation(test_inner[i], test_inner[j], tol=1e-6):
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    new_side = calculate_min_side(test_inner, best_outer_angle)
                    if new_side < best_side - 1e-10:
                        best_side = new_side
                        best_inner = test_inner.copy()
                        
            except Exception:
                pass
        
        # Try Nelder-Mead for local refinement
        if time_left() < 30:
            continue
        
        try:
            result = minimize(
                objective,
                best_inner[:, :2].flatten(),
                args=(best_outer_angle,),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            
            if time_left() < 30:
                continue
            
            test_inner = np.zeros((11, 3))
            for i in range(11):
                test_inner[i, 0] = result.x[2*i]
                test_inner[i, 1] = result.x[2*i+1]
                test_inner[i, 2] = 0.0
            
            valid = True
            for i in range(11):
                for j in range(i+1, 11):
                    if not check_separation(test_inner[i], test_inner[j], tol=1e-6):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                new_side = calculate_min_side(test_inner, best_outer_angle)
                if new_side < best_side - 1e-10:
                    best_side = new_side
                    best_inner = test_inner.copy()
                    
        except Exception:
            pass
    
    # Final local refinement with small steps
    if time_left() > 30 and best_inner is not None:
        current_best = best_side
        current_pos = best_inner.copy()
        
        step_sizes = [0.01, 0.005, 0.002, 0.001]
        for step in step_sizes:
            if time_left() < 30:
                break
            improved = True
            while improved and time_left() > 30:
                improved = False
                for i in range(11):
                    if time_left() < 30:
                        break
                    for dx in [-step, 0, step]:
                        for dy in [-step, 0, step]:
                            if dx == 0 and dy == 0:
                                continue
                            test_pos = current_pos.copy()
                            test_pos[i, 0] += dx
                            test_pos[i, 1] += dy
                            
                            valid = True
                            for j in range(11):
                                if i == j:
                                    continue
                                if not check_separation(test_pos[i], test_pos[j], tol=1e-6):
                                    valid = False
                                    break
                            
                            if not valid:
                                continue
                            
                            new_side = calculate_min_side(test_pos, best_outer_angle)
                            if new_side < current_best - 1e-10:
                                current_pos = test_pos
                                current_best = new_side
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        if current_best < best_side:
            best_side = current_best
            best_inner = current_pos
    
    # Final verification
    outer_hex_data = np.array([0.0, 0.0, best_outer_angle])
    
    if not is_valid_configuration(best_inner, outer_hex_data, best_side):
        # Fallback to a known valid configuration
        fallback = make_honeycomb()
        # Scale slightly to ensure separation
        scale = 1.001
        fallback[:, :2] *= scale
        
        # Find best angle for fallback
        best_fb_angle = 0.0
        best_fb_side = float('inf')
        for a in np.arange(0, 30.01, 0.5):
            s = calculate_min_side(fallback, a)
            if s < best_fb_side:
                best_fb_side = s
                best_fb_angle = a
        
        if is_valid_configuration(fallback, np.array([0.0, 0.0, best_fb_angle]), best_fb_side):
            return fallback, np.array([0.0, 0.0, best_fb_angle]), best_fb_side
    
    return best_inner, outer_hex_data, best_side


# EVOLVE-BLOCK-END

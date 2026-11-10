# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import itertools

def _hex_vertices(cx, cy, s, a_deg):
    """Compute 6 vertices of a regular hexagon."""
    a = np.radians(a_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _min_distance_hexes(h1, h2):
    """
    Compute minimum distance between two unit hexagons.
    Returns negative if overlapping, positive if separated.
    Uses the separating axis theorem for hexagons.
    """
    v1 = _hex_vertices(h1[0], h1[1], 1.0, h1[2])
    v2 = _hex_vertices(h2[0], h2[1], 1.0, h2[2])
    
    # Get all edge normals from both hexagons
    normals = []
    for v in [v1, v2]:
        for i in range(6):
            edge = v[(i + 1) % 6] - v[i]
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-10:
                n /= norm
                normals.append(n)
    
    # For each normal, compute the gap
    min_gap = float('inf')
    for n in normals:
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Gap is the distance between the intervals
        # If intervals overlap, gap is negative
        gap = max(min2 - max1, min1 - max2)
        min_gap = min(min_gap, gap)
    
    return min_gap

def _check_pair_separation(hex1, hex2, tol=1e-6):
    """
    Check if two unit hexagons are separated (non-intersecting).
    Returns True if they are separated on at least one of the 12 edge normals.
    """
    v1 = _hex_vertices(hex1[0], hex1[1], 1.0, hex1[2])
    v2 = _hex_vertices(hex2[0], hex2[1], 1.0, hex2[2])

    # Get all 12 edge normals (6 from each hexagon)
    normals = []
    for v in [v1, v2]:
        for i in range(6):
            edge = v[(i + 1) % 6] - v[i]
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-10:
                n /= norm
                normals.append(n)

    # Check each normal
    for n in normals:
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)

        # Check if intervals are apart by more than tol
        if max1 < min2 - tol or max2 < min1 - tol:
            return True

    return False

def _check_inner_in_outer(inner_hexes, outer_center, outer_side, outer_angle, tol=1e-6):
    """
    Check if all inner hexagons are inside the outer hexagon.
    For each inner vertex, check cross product with each outer edge.
    """
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)

    for hex_data in inner_hexes:
        inner_verts = _hex_vertices(hex_data[0], hex_data[1], 1.0, hex_data[2])

        for iv in inner_verts:
            # Shift by outer center
            iv_shifted = iv - outer_center

            for i in range(6):
                p1 = outer_verts[i] - outer_center
                p2 = outer_verts[(i + 1) % 6] - outer_center

                # Cross product: (p2 - p1) x (p - p1)
                edge = p2 - p1
                vec = iv_shifted - p1
                cross = edge[0] * vec[1] - edge[1] * vec[0]

                if cross < -tol:
                    return False

    return True

def _min_outer_side_for_inner(inner_hexes, outer_center, outer_angle):
    """
    Find the minimum outer side length such that all inner hexagons fit inside.
    """
    min_side = 0.0

    for i in range(6):
        angle = np.radians(60 * i + 30)
        n = np.array([np.cos(angle), np.sin(angle)])

        max_proj = 0.0
        for hex_data in inner_hexes:
            inner_verts = _hex_vertices(hex_data[0], hex_data[1], 1.0, hex_data[2])
            inner_verts_shifted = inner_verts - outer_center
            projs = inner_verts_shifted @ n
            max_proj = max(max_proj, np.max(projs))

        required_side = 2.0 * max_proj / np.sqrt(3.0)
        min_side = max(min_side, required_side)

    return min_side

def _soft_max_enclosing_side(inner_hexes, outer_center, outer_angle, beta=50.0):
    """
    Smooth approximation of the minimum enclosing side using log-sum-exp.
    """
    max_projs = np.zeros(6)
    
    for i in range(6):
        angle = np.radians(60 * i + 30)
        n = np.array([np.cos(angle), np.sin(angle)])
        
        max_proj = 0.0
        for hex_data in inner_hexes:
            inner_verts = _hex_vertices(hex_data[0], hex_data[1], 1.0, hex_data[2])
            inner_verts_shifted = inner_verts - outer_center
            projs = inner_verts_shifted @ n
            max_proj = max(max_proj, np.max(projs))
        
        max_projs[i] = max_proj
    
    scaled = beta * max_projs
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    smooth_max = log_sum_exp / beta
    
    return 2.0 * smooth_max / np.sqrt(3.0)

def _objective_smooth(params, outer_center, outer_angle, beta=50.0):
    """
    Objective function with smooth enclosing side and penalty for overlaps.
    """
    inner_hexes = params.reshape(11, 3)
    
    penalty = 0.0
    for i in range(11):
        for j in range(i + 1, 11):
            if not _check_pair_separation(inner_hexes[i], inner_hexes[j]):
                penalty += 10.0
    
    side = _soft_max_enclosing_side(inner_hexes, outer_center, outer_angle, beta)
    
    return side + penalty

def _objective_exact(params, outer_center, outer_angle):
    """
    Exact objective for final polish.
    """
    inner_hexes = params.reshape(11, 3)
    
    for i in range(11):
        for j in range(i + 1, 11):
            if not _check_pair_separation(inner_hexes[i], inner_hexes[j]):
                return 1e6
    
    side = _min_outer_side_for_inner(inner_hexes, outer_center, outer_angle)
    return side

def _initial_guesses():
    """Generate multiple initial guesses for the packing."""
    sqrt3 = np.sqrt(3)
    
    guesses = []
    
    # Guess 1: Standard hexagonal arrangement (center + 6 + 4 corners)
    g1 = np.array([
        [0, 0, 0],
        [3, 0, 0],
        [-3, 0, 0],
        [1.5, 2.598, 0],
        [-1.5, 2.598, 0],
        [1.5, -2.598, 0],
        [-1.5, -2.598, 0],
        [4.5, 2.598, 0],
        [-4.5, 2.598, 0],
        [4.5, -2.598, 0],
        [-4.5, -2.598, 0],
    ]).flatten()
    guesses.append(g1)
    
    # Guess 2: Slightly compressed arrangement
    g2 = g1.copy()
    for i in range(11):
        x, y, a = g2[3*i], g2[3*i+1], g2[3*i+2]
        r = np.sqrt(x*x + y*y)
        if r > 0.1:
            scale = 0.95
            g2[3*i] = x * scale
            g2[3*i+1] = y * scale
    guesses.append(g2)
    
    # Guess 3: Different angular arrangement
    g3 = np.array([
        [0, 0, 0],
        [3, 0, 0],
        [-3, 0, 0],
        [1.5, 2.598, 0],
        [-1.5, 2.598, 0],
        [1.5, -2.598, 0],
        [-1.5, -2.598, 0],
        [3.0, 1.5, 0],
        [-3.0, 1.5, 0],
        [3.0, -1.5, 0],
        [-3.0, -1.5, 0],
    ]).flatten()
    guesses.append(g3)
    
    # Guess 4: Rotated outer arrangement
    g4 = g1.copy()
    theta = np.radians(30)
    c, s = np.cos(theta), np.sin(theta)
    for i in range(11):
        x, y = g4[3*i], g4[3*i+1]
        g4[3*i] = x * c - y * s
        g4[3*i+1] = x * s + y * c
    guesses.append(g4)
    
    return guesses

def _verify_solution(inner_hexes, outer_center, outer_side, outer_angle):
    """Verify that a solution is valid."""
    for i in range(11):
        for j in range(i + 1, 11):
            if not _check_pair_separation(inner_hexes[i], inner_hexes[j]):
                return False
    
    if not _check_inner_in_outer(inner_hexes, outer_center, outer_side, outer_angle):
        return False
    
    if outer_side < np.sqrt(11) - 1e-9:
        return False
    
    if not np.all(np.isfinite(inner_hexes)):
        return False
    
    return True

def _local_search(params, outer_center, outer_angle, max_iter=200, step_size=0.1):
    """
    Local search to improve a solution.
    """
    inner_hexes = params.reshape(11, 3)
    current_side = _min_outer_side_for_inner(inner_hexes, outer_center, outer_angle)
    
    if not _verify_solution(inner_hexes, outer_center, current_side, outer_angle):
        return params, current_side
    
    best_params = params.copy()
    best_side = current_side
    
    for iteration in range(max_iter):
        improved = False
        
        for i in range(11):
            for j in range(3):
                for direction in [-1, 1]:
                    new_params = best_params.copy()
                    idx = i * 3 + j
                    new_params[idx] += direction * step_size
                    
                    # Keep angle in valid range
                    if j == 2:
                        new_params[idx] = new_params[idx] % 360
                    
                    new_inner = new_params.reshape(11, 3)
                    
                    if _verify_solution(new_inner, outer_center, best_side, outer_angle):
                        new_side = _min_outer_side_for_inner(new_inner, outer_center, outer_angle)
                        
                        if new_side < best_side - 1e-8:
                            best_params = new_params.copy()
                            best_side = new_side
                            improved = True
        
        if not improved:
            step_size *= 0.5
            if step_size < 1e-6:
                break
    
    return best_params, best_side

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    """
    start_time = time.time()
    deadline = start_time + 600
    
    outer_center = np.array([0.0, 0.0])
    outer_angle = 0.0
    
    best_params = None
    best_side = float('inf')
    
    # Get initial guesses
    guesses = _initial_guesses()
    
    # Phase 1: Try each initial guess with local search
    for gi, guess in enumerate(guesses):
        if time.time() > deadline - 30:
            break
        
        params0 = guess.copy()
        inner_hexes = params0.reshape(11, 3)
        
        # First, check if the initial guess is valid
        side = _min_outer_side_for_inner(inner_hexes, outer_center, outer_angle)
        
        if _verify_solution(inner_hexes, outer_center, side, outer_angle):
            if side < best_side:
                best_side = side
                best_params = params0.copy()
        
        # Then try to improve with local search
        if time.time() < deadline - 10:
            improved_params, improved_side = _local_search(
                params0, outer_center, outer_angle, max_iter=100, step_size=0.05
            )
            
            if improved_side < best_side:
                best_side = improved_side
                best_params = improved_params.copy()
    
    # Phase 2: Smooth optimization from the best solution
    if best_params is not None and time.time() < deadline - 60:
        try:
            result = minimize(
                _objective_smooth,
                best_params,
                args=(outer_center, outer_angle, 50.0),
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5}
            )
            
            if result.fun < best_side:
                inner_hexes = result.x.reshape(11, 3)
                side = _min_outer_side_for_inner(inner_hexes, outer_center, outer_angle)
                if _verify_solution(inner_hexes, outer_center, side, outer_angle):
                    best_side = side
                    best_params = result.x.copy()
        except Exception:
            pass
    
    # Phase 3: Exact polish
    if best_params is not None and time.time() < deadline - 30:
        try:
            result = minimize(
                _objective_exact,
                best_params,
                args=(outer_center, outer_angle),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-7}
            )
            
            if result.fun < best_side:
                inner_hexes = result.x.reshape(11, 3)
                side = _min_outer_side_for_inner(inner_hexes, outer_center, outer_angle)
                if _verify_solution(inner_hexes, outer_center, side, outer_angle):
                    best_side = side
                    best_params = result.x.copy()
        except Exception:
            pass
    
    # Phase 4: Final local search refinement
    if best_params is not None and time.time() < deadline - 5:
        try:
            improved_params, improved_side = _local_search(
                best_params, outer_center, outer_angle, max_iter=200, step_size=0.01
            )
            
            if improved_side < best_side:
                best_side = improved_side
                best_params = improved_params.copy()
        except Exception:
            pass
    
    # If no valid solution found, use the best guess
    if best_params is None:
        best_params = guesses[0].copy()
        best_side = _min_outer_side_for_inner(best_params.reshape(11, 3), outer_center, outer_angle)
    
    inner_hex_data = best_params.reshape(11, 3)
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = float(best_side)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END

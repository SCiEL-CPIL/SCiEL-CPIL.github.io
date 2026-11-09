# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time


def _hex_vertices(cx, cy, s, angle_deg):
    """Compute 6 vertices of a regular hexagon."""
    angles = np.radians(angle_deg + 60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _edge_normals(vertices):
    """Compute 6 edge outward normals (unit) for a hexagon given its 6 CCW vertices."""
    n = 6
    normals = np.zeros((n, 2))
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        edge = p2 - p1
        nx, ny = edge[1], -edge[0]
        norm = np.linalg.norm([nx, ny])
        if norm > 1e-12:
            normals[i] = [nx / norm, ny / norm]
    return normals


def _separation_check(v1, n1, v2, n2):
    """Check if two hexagons are separated on at least one of 12 edge normals."""
    all_normals = np.vstack([n1, n2])
    for i in range(12):
        n = all_normals[i]
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
            return True
    return False


def _all_pairs_separated(inner_hexes):
    """Check if all pairs of inner hexagons are separated."""
    n = len(inner_hexes)
    vertices_list = []
    normals_list = []
    for i in range(n):
        x, y, a = inner_hexes[i]
        v = _hex_vertices(x, y, 1.0, a)
        nm = _edge_normals(v)
        vertices_list.append(v)
        normals_list.append(nm)
    
    for i in range(n):
        for j in range(i + 1, n):
            if not _separation_check(vertices_list[i], normals_list[i], vertices_list[j], normals_list[j]):
                return False
    return True


def _outer_contains_inner(outer_center, outer_side, outer_angle, inner_hexes, tol=1e-6):
    """Check if all inner hexagon vertices are inside the outer hexagon."""
    outer_v = _hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    for i in range(len(inner_hexes)):
        x, y, a = inner_hexes[i]
        v = _hex_vertices(x, y, 1.0, a)
        for p in v:
            for e in range(6):
                p1 = outer_v[e]
                p2 = outer_v[(e + 1) % 6]
                cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                if cross < -tol:
                    return False
    return True


def _compute_side_length(inner_hexes, outer_center, outer_angle):
    """Compute the minimum side length of outer hexagon that contains all inner hexagons."""
    outer_v_template = _hex_vertices(0, 0, 1.0, outer_angle)
    
    max_s = 0.0
    for e in range(6):
        p1 = outer_v_template[e]
        p2 = outer_v_template[(e + 1) % 6]
        edge = p2 - p1
        nx, ny = edge[1], -edge[0]
        norm = np.linalg.norm([nx, ny])
        if norm < 1e-12:
            continue
        n = np.array([nx / norm, ny / norm])
        
        for i in range(len(inner_hexes)):
            x, y, a = inner_hexes[i]
            v = _hex_vertices(x, y, 1.0, a)
            for p in v:
                dp = p - outer_center
                proj = dp @ n
                if proj > 0:
                    required_s = proj * 2.0 / np.sqrt(3.0)
                    max_s = max(max_s, required_s)
    
    return max_s


def _objective(params, outer_center, outer_angle):
    """Objective: minimize the side length of the outer hexagon."""
    inner_hexes = params.reshape(11, 3)
    
    if not _all_pairs_separated(inner_hexes):
        return 1e6
    
    side = _compute_side_length(inner_hexes, outer_center, outer_angle)
    
    if not _outer_contains_inner(outer_center, side, outer_angle, inner_hexes):
        return 1e6
    
    return side


def _make_symmetric_init(ring1_dist, ring2_dist):
    """Create a symmetric initial configuration with 1 center + 6 ring1 + 4 ring2."""
    sqrt3 = np.sqrt(3)
    init = np.array([
        [0, 0, 0],
        [ring1_dist, 0, 0],
        [ring1_dist/2, ring1_dist*sqrt3/2, 0],
        [-ring1_dist/2, ring1_dist*sqrt3/2, 0],
        [-ring1_dist, 0, 0],
        [-ring1_dist/2, -ring1_dist*sqrt3/2, 0],
        [ring1_dist/2, -ring1_dist*sqrt3/2, 0],
        # 4 outer positions (top-right, top-left, bottom-right, bottom-left)
        [ring2_dist*3/2, ring2_dist*sqrt3/2, 0],
        [-ring2_dist*3/2, ring2_dist*sqrt3/2, 0],
        [ring2_dist*3/2, -ring2_dist*sqrt3/2, 0],
        [-ring2_dist*3/2, -ring2_dist*sqrt3/2, 0],
    ])
    return init


def _free_body_simulate(inner_hexes, outer_center, outer_angle, max_steps=500, dt=0.01):
    """
    Free-body simulation: push hexagons apart if overlapping, pull toward center.
    Returns modified inner_hexes.
    """
    n = len(inner_hexes)
    hexes = inner_hexes.copy().astype(float)
    sqrt3 = np.sqrt(3)
    
    for step in range(max_steps):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                # Check if hexagons i and j are separated
                vi = _hex_vertices(hexes[i, 0], hexes[i, 1], 1.0, hexes[i, 2])
                vj = _hex_vertices(hexes[j, 0], hexes[j, 1], 1.0, hexes[j, 2])
                ni = _edge_normals(vi)
                nj = _edge_normals(vj)
                
                if _separation_check(vi, ni, vj, nj):
                    continue  # Already separated
                
                # Compute push direction: from j to i
                dx = hexes[i, 0] - hexes[j, 0]
                dy = hexes[i, 1] - hexes[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-8:
                    dx, dy = 1.0, 0.0
                    dist = 1.0
                else:
                    dx, dy = dx/dist, dy/dist
                
                # Push apart
                push = dt * 0.5
                hexes[i, 0] += dx * push
                hexes[i, 1] += dy * push
                hexes[j, 0] -= dx * push
                hexes[j, 1] -= dy * push
                moved = True
        
        # Pull all hexagons toward center (weak spring)
        for i in range(n):
            dx = outer_center[0] - hexes[i, 0]
            dy = outer_center[1] - hexes[i, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist > 1e-8:
                pull = dt * 0.1
                hexes[i, 0] += dx/dist * pull
                hexes[i, 1] += dy/dist * pull
                moved = True
        
        if not moved:
            break
    
    return hexes


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    Uses free-body simulation for initial refinement, then differential evolution and local optimization.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    outer_center = np.array([0.0, 0.0])
    
    best_params = None
    best_side = float('inf')
    best_angle = 0.0
    
    # Try different outer angles and initial configurations
    angles_to_try = [0, 30, 60, 90, 120, 150]
    
    # Initial symmetric configurations with different spacing
    init_configs = []
    for r1 in [2.0, 2.1, 2.2, 2.3]:
        for r2 in [3.5, 3.8, 4.0, 4.2]:
            init_configs.append(_make_symmetric_init(r1, r2))
    
    # Also add the original grid-based init
    sqrt3 = np.sqrt(3)
    init_configs.append(np.array([
        [0, 0, 0],
        [2.0, 0, 0],
        [1.0, sqrt3, 0],
        [-1.0, sqrt3, 0],
        [-2.0, 0, 0],
        [-1.0, -sqrt3, 0],
        [1.0, -sqrt3, 0],
        [3.0, sqrt3, 0],
        [0, 2*sqrt3, 0],
        [-3.0, sqrt3, 0],
        [3.0, -sqrt3, 0],
    ]))
    
    # Evaluate all initial configs and keep the best, using free-body simulation
    for init in init_configs:
        if time.time() > deadline - 60:
            break
        for angle in angles_to_try[:4]:  # Try more angles for initial evaluation
            if time.time() > deadline - 60:
                break
            # Apply free-body simulation to refine
            refined = _free_body_simulate(init, outer_center, float(angle), max_steps=200, dt=0.02)
            
            side = _compute_side_length(refined, outer_center, float(angle))
            if _all_pairs_separated(refined) and _outer_contains_inner(outer_center, side, float(angle), refined):
                if side < best_side:
                    best_params = refined.copy()
                    best_side = side
                    best_angle = float(angle)
    
    if best_params is None:
        # Fallback to first config
        best_params = init_configs[0]
        best_angle = 0.0
        best_side = _compute_side_length(best_params, outer_center, best_angle)
    
    # Phase 1: Differential evolution for global optimization
    if time.time() < deadline - 120:
        # Bounds for each hexagon: x in [-5,5], y in [-5,5], angle in [0,180]
        bounds = [(-5, 5)] * 11 + [(-5, 5)] * 11 + [(0, 180)] * 11
        
        try:
            de_result = differential_evolution(
                _objective,
                bounds,
                args=(outer_center, best_angle),
                maxiter=300,
                popsize=20,
                tol=1e-8,
                seed=42,
                polish=False,
                init='sobol'
            )
            
            if de_result.fun < best_side:
                candidate = de_result.x.reshape(11, 3)
                if _all_pairs_separated(candidate):
                    side = _compute_side_length(candidate, outer_center, best_angle)
                    if _outer_contains_inner(outer_center, side, best_angle, candidate):
                        best_params = candidate
                        best_side = side
        except Exception:
            pass
    
    # Phase 2: Local refinement with Nelder-Mead from best found
    if time.time() < deadline - 80:
        try:
            nm_result = minimize(
                _objective,
                best_params.flatten(),
                args=(outer_center, best_angle),
                method='Nelder-Mead',
                options={'maxiter': 8000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if nm_result.fun < best_side:
                candidate = nm_result.x.reshape(11, 3)
                if _all_pairs_separated(candidate):
                    side = _compute_side_length(candidate, outer_center, best_angle)
                    if _outer_contains_inner(outer_center, side, best_angle, candidate):
                        best_params = candidate
                        best_side = side
        except Exception:
            pass
    
    # Phase 3: Free-body simulation from best found for further refinement
    if time.time() < deadline - 40:
        refined = _free_body_simulate(best_params, outer_center, best_angle, max_steps=1000, dt=0.005)
        side = _compute_side_length(refined, outer_center, best_angle)
        if _all_pairs_separated(refined) and _outer_contains_inner(outer_center, side, best_angle, refined):
            if side < best_side:
                best_params = refined
                best_side = side
    
    # Phase 4: Try other outer angles with the best configuration
    if time.time() < deadline - 30:
        for angle in angles_to_try:
            if time.time() > deadline - 15:
                break
            if abs(angle - best_angle) < 1:
                continue
            try:
                side = _compute_side_length(best_params, outer_center, float(angle))
                if _all_pairs_separated(best_params) and _outer_contains_inner(outer_center, side, float(angle), best_params):
                    if side < best_side:
                        best_side = side
                        best_angle = float(angle)
            except Exception:
                continue
    
    # Final refinement with best angle
    if time.time() < deadline - 15:
        try:
            nm_result2 = minimize(
                _objective,
                best_params.flatten(),
                args=(outer_center, best_angle),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            
            if nm_result2.fun < best_side:
                candidate = nm_result2.x.reshape(11, 3)
                if _all_pairs_separated(candidate):
                    side = _compute_side_length(candidate, outer_center, best_angle)
                    if _outer_contains_inner(outer_center, side, best_angle, candidate):
                        best_params = candidate
                        best_side = side
        except Exception:
            pass
    
    # Final verification and fallback
    final_side = _compute_side_length(best_params, outer_center, best_angle)
    
    if not _all_pairs_separated(best_params) or not _outer_contains_inner(outer_center, final_side, best_angle, best_params):
        # Fallback to a known valid configuration
        sqrt3 = np.sqrt(3)
        best_params = np.array([
            [0, 0, 0],
            [2.0, 0, 0],
            [1.0, sqrt3, 0],
            [-1.0, sqrt3, 0],
            [-2.0, 0, 0],
            [-1.0, -sqrt3, 0],
            [1.0, -sqrt3, 0],
            [3.0, sqrt3, 0],
            [0, 2*sqrt3, 0],
            [-3.0, sqrt3, 0],
            [3.0, -sqrt3, 0],
        ])
        best_angle = 0.0
        final_side = _compute_side_length(best_params, outer_center, best_angle)
    
    inner_hex_data = best_params
    outer_hex_data = np.array([0.0, 0.0, float(best_angle)])
    outer_hex_side_length = float(final_side)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _hex_vertices(cx, cy, d, angle_deg):
    """Compute vertices of a regular hexagon."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60) * np.arange(6)
    return np.column_stack([
        cx + d * np.cos(angles),
        cy + d * np.sin(angles)
    ])


def _support_function(points, directions):
    """Compute support function: max of dot products for each direction."""
    # points: (N, 2), directions: (M, 2)
    # returns: (M,) array of max dot products
    return np.max(points @ directions.T, axis=1)


def _min_outer_side(translations, outer_center, outer_angle_deg):
    """
    Find minimum side length s of outer hexagon centered at outer_center
    with given angle that contains all 12 unit hexagons.
    """
    outer_a = np.radians(outer_angle_deg)
    # Outer hexagon edge normals (6 directions)
    outer_normals = np.column_stack([
        np.cos(outer_a + np.radians(60) * np.arange(6)),
        np.sin(outer_a + np.radians(60) * np.arange(6))
    ])
    
    # For each inner hexagon, compute its vertices
    all_vertices = []
    for i in range(12):
        tx, ty = translations[i]
        verts = _hex_vertices(tx, ty, 1.0, 0.0)
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)  # (72, 2)
    
    # For containment, we need: for each outer edge normal n,
    # max_v (v - outer_center) . n <= s * (distance from center to edge along n)
    # The distance from center to edge in direction n is s * cos(0) = s * 1
    # Wait, for a regular hexagon with side s, the apothem (distance from center to edge midpoint) is s * sqrt(3)/2
    # The support in direction of edge normal is s * sqrt(3)/2
    
    shifted = all_vertices - outer_center
    supports = _support_function(shifted, outer_normals)  # (6,)
    # Need s * sqrt(3)/2 >= max(supports)
    # So s >= max(supports) / (sqrt(3)/2) = 2 * max(supports) / sqrt(3)
    s_min = 2.0 * np.max(supports) / np.sqrt(3.0)
    return s_min


def _check_validity(inner_data, outer_data, s, tol=1e-6):
    """Check if the packing is valid."""
    # Check s >= sqrt(12)
    if s < np.sqrt(12.0):
        return False
    
    # Check finite values
    if not (np.all(np.isfinite(inner_data)) and np.all(np.isfinite(outer_data))):
        return False
    
    # Get outer center
    ox, oy, oa = outer_data
    outer_center = np.array([ox, oy])
    
    # Check inner hexagons are separated
    for i in range(12):
        for j in range(i + 1, 12):
            cx1, cy1, ca1 = inner_data[i]
            cx2, cy2, ca2 = inner_data[j]
            v1 = _hex_vertices(cx1, cy1, 1.0, ca1)
            v2 = _hex_vertices(cx2, cy2, 1.0, ca2)
            
            # Separating axis test: check all 12 edge normals (6 from each)
            for verts in [v1, v2]:
                for k in range(6):
                    p1 = verts[k]
                    p2 = verts[(k + 1) % 6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    normal = normal / (np.linalg.norm(normal) + 1e-15)
                    
                    proj1 = v1 @ normal
                    proj2 = v2 @ normal
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    
                    # If intervals are separated by more than tol, OK
                    # If they overlap, not separated
                    gap = max(min1 - max2, min2 - max1)
                    if gap < -tol:
                        # Overlapping - not separated
                        return False
    
    # Check all inner vertices are inside outer hexagon
    outer_verts = _hex_vertices(ox, oy, s, oa)
    for i in range(12):
        cx, cy, ca = inner_data[i]
        verts = _hex_vertices(cx, cy, 1.0, ca)
        for v in verts:
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k + 1) % 6]
                cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
                if cross < -1e-6:
                    return False
    
    return True


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length using convex optimization.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600 second limit
    
    # Initial positions: hexagonal lattice arrangement
    # 12 hexagons in a roughly circular/hexagonal cluster
    # Using a pattern: center + 6 around + 5 more
    init_translations = np.array([
        [0, 0],           # center
        [2.5, 0],         # right
        [-2.5, 0],        # left
        [1.25, 2.165],    # top-right
        [-1.25, 2.165],   # top-left
        [1.25, -2.165],   # bottom-right
        [-1.25, -2.165],  # bottom-left
        [0, 4.33],        # top
        [0, -4.33],       # bottom
        [3.75, 2.165],    # far top-right
        [-3.75, 2.165],   # far top-left
        [3.75, -2.165],   # far bottom-right
    ])
    # Wait, that's 12 but let me recount and use a better arrangement
    # Standard hexagonal packing: 1 center, 6 first ring, 5 from second ring
    init_translations = np.array([
        [0, 0],            # 0: center
        [2.5, 0],          # 1
        [-2.5, 0],         # 2
        [1.25, 2.165],     # 3
        [-1.25, 2.165],    # 4
        [1.25, -2.165],    # 5
        [-1.25, -2.165],   # 6
        [0, 4.33],         # 7: top
        [0, -4.33],        # 8: bottom
        [3.75, 2.165],     # 9
        [-3.75, 2.165],    # 10
        [3.75, -2.165],    # 11
    ])
    
    best_s = np.inf
    best_translations = init_translations.copy()
    best_outer_center = np.array([0.0, 0.0])
    best_outer_angle = 0.0
    
    # We'll optimize translations and outer hexagon parameters
    # Variables: 12*2 translations + 2 outer center + 1 outer angle = 27 variables
    # But angle is periodic, so let's fix outer angle to 0 and just optimize translations + center
    
    # Actually, let's try a few outer angles and pick the best
    outer_angles_to_try = [0.0, 30.0]
    
    for outer_angle in outer_angles_to_try:
        if time.time() > deadline - 60.0:
            break
        
        # Objective: minimize s (the outer hexagon side length)
        # Given translations and outer center, s is determined
        
        def objective(params):
            translations = params[:24].reshape(12, 2)
            outer_center = params[24:26]
            return _min_outer_side(translations, outer_center, outer_angle)
        
        x0 = np.zeros(26)
        x0[:24] = init_translations.flatten()
        x0[24:26] = [0.0, 0.0]
        
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 1000}
            )
            
            if result.success or result.fun < best_s:
                best_s = result.fun
                best_translations = result.x[:24].reshape(12, 2)
                best_outer_center = result.x[24:26]
                best_outer_angle = outer_angle
        except Exception:
            pass
    
    # Now we have best positions, but we need to verify separation
    # The optimization only enforced containment, not separation
    # Let's add separation constraints via penalty or local adjustment
    
    # Check if current solution is valid
    inner_data = np.column_stack([
        best_translations[:, 0],
        best_translations[:, 1],
        np.zeros(12)
    ])
    outer_data = np.array([best_outer_center[0], best_outer_center[1], best_outer_angle])
    
    # Local refinement: try to push apart overlapping hexagons
    for iteration in range(50):
        if time.time() > deadline - 30.0:
            break
        
        # Check for overlaps and push apart
        pushed = False
        for i in range(12):
            for j in range(i + 1, 12):
                cx1, cy1 = best_translations[i]
                cx2, cy2 = best_translations[j]
                v1 = _hex_vertices(cx1, cy1, 1.0, 0.0)
                v2 = _hex_vertices(cx2, cy2, 1.0, 0.0)
                
                # Check if separated
                separated = True
                for verts in [v1, v2]:
                    for k in range(6):
                        p1 = verts[k]
                        p2 = verts[(k + 1) % 6]
                        edge = p2 - p1
                        normal = np.array([-edge[1], edge[0]])
                        normal = normal / (np.linalg.norm(normal) + 1e-15)
                        
                        proj1 = v1 @ normal
                        proj2 = v2 @ normal
                        min1, max1 = np.min(proj1), np.max(proj1)
                        min2, max2 = np.min(proj2), np.max(proj2)
                        
                        gap = max(min1 - max2, min2 - max1)
                        if gap < -1e-6:
                            separated = False
                            # Push apart along normal
                            # Determine which way to push
                            center1 = np.array([cx1, cy1])
                            center2 = np.array([cx2, cy2])
                            diff = center2 - center1
                            if np.dot(diff, normal) < 0:
                                normal = -normal
                            push = 0.01
                            best_translations[i] -= push * normal
                            best_translations[j] += push * normal
                            pushed = True
                            break
                    if not separated:
                        break
                if not separated:
                    break
            if not separated:
                break
        
        if not pushed:
            break
        
        # Recompute s after pushing
        best_s = _min_outer_side(best_translations, best_outer_center, best_outer_angle)
    
    # Final validation
    inner_data = np.column_stack([
        best_translations[:, 0],
        best_translations[:, 1],
        np.zeros(12)
    ])
    outer_data = np.array([best_outer_center[0], best_outer_center[1], best_outer_angle])
    
    if not _check_validity(inner_data, outer_data, best_s):
        # Fallback to the seed solution if optimization failed
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
                [0, -4, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length
    
    return inner_data, outer_data, best_s


# EVOLVE-BLOCK-END

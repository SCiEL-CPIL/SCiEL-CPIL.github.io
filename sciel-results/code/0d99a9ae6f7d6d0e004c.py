"""Packing unit hexagons inside a regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)
DEADLINE = 135.0  # safety margin before 150s kill


def logsumexp(a, b=1.0):
    """Stable log-sum-exp for array a with temperature b."""
    a = np.asarray(a, dtype=np.float64)
    amax = np.max(a)
    if not np.isfinite(amax):
        return amax
    return amax + math.log(np.sum(np.exp((a - amax) / b))) * b


def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts


def compute_required_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_deg)
    outer_center: (x, y)
    outer_angle_deg: rotation of the outer hexagon
    """
    if not inner:
        return 0.0

    ox, oy = outer_center
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))

    max_projection = 0.0

    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))

    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a

            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj

    return 2.0 * max_projection / SQRT3


def sat_overlap_check(v1, v2, eps=1e-9):
    """
    Check if two convex polygons (defined by their vertices) overlap using SAT.
    v1, v2: lists of (x, y) tuples.
    Returns True if they overlap (including touching).
    """
    axes = []

    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0] ** 2 + normal[1] ** 2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))

    for i in range(len(v2)):
        p1 = v2[i]
        p2 = v2[(i + 1) % len(v2)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0] ** 2 + normal[1] ** 2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))

    for axis in axes:
        min1 = min(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        max1 = max(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        min2 = min(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        max2 = max(p[0] * axis[0] + p[1] * axis[1] for p in v2)

        if max1 < min2 + eps or max2 < min1 + eps:
            return False

    return True


def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    for i in range(n):
        for j in range(i + 1, n):
            vi = get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
            vj = get_hex_vertices(inner[j][0], inner[j][1], 1.0, math.radians(inner[j][2]))
            if sat_overlap_check(vi, vj, eps):
                return False
    return True


def check_inside_outer(inner, outer_center, outer_side, outer_angle_deg, eps=1e-6):
    """Check all inner hexagons are inside the outer hexagon."""
    ox, oy = outer_center
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))

    apothem = outer_side * SQRT3 / 2.0

    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a

            for i in range(6):
                theta = math.radians(30 + i * 60)
                nx = math.cos(theta)
                ny = math.sin(theta)
                if lx * nx + ly * ny > apothem + eps:
                    return False
    return True


def generate_polyhexes(n):
    """Generate all n-cell polyhexes in axial coordinates."""
    if n == 1:
        return {frozenset([(0, 0)])}
    
    prev = generate_polyhexes(n - 1)
    result = set()
    directions = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    for poly in prev:
        for cell in poly:
            for dq, dr in directions:
                new_cell = (cell[0] + dq, cell[1] + dr)
                if new_cell not in poly:
                    new_poly = frozenset(list(poly) + [new_cell])
                    # Normalize: shift so min q is 0, then min r is 0
                    min_q = min(c[0] for c in new_poly)
                    min_r = min(c[1] for c in new_poly)
                    normalized = frozenset([(c[0] - min_q, c[1] - min_r) for c in new_poly])
                    result.add(normalized)
    
    return result


def compute_min_enclosing_hex_side(centers, angle_deg=0.0):
    """
    Given a set of hexagon centers (all with angle 0), compute the minimum
    outer hexagon side length needed to contain them, given outer rotation.
    The outer center is optimized to minimize the side length.
    """
    # For a fixed rotation, the optimal center is the one that minimizes
    # the maximum support function. This is a linear programming problem,
    # but we can use the fact that for a hexagon, the optimal center
    # balances the projections.
    
    # Actually, for a given set of points and a fixed hexagon orientation,
    # the minimum enclosing hexagon center can be found by noting that
    # the constraint is: for each of the 6 normals n_i, max_i (p · n_i) - min_i (p · n_i) <= 2 * apothem
    # where apothem = side * sqrt(3) / 2
    
    # The width in direction n is max(p·n) - min(p·n), which is independent of center.
    # So the minimum side is determined by the maximum width over all 6 directions.
    
    # Wait, that's not quite right. The hexagon is centered, so:
    # For each normal n_i: max_j (p_j · n_i) - c · n_i <= apothem
    # and min_j (p_j · n_i) - c · n_i >= -apothem
    # So: c · n_i >= max_j (p_j · n_i) - apothem
    # and: c · n_i <= min_j (p_j · n_i) + apothem
    # This requires: max_j (p_j · n_i) - min_j (p_j · n_i) <= 2 * apothem
    # i.e., width_i <= 2 * apothem = side * sqrt(3)
    # So side >= max_i (width_i) / sqrt(3)
    
    # But this is for a hexagon with specific orientation. The widths depend on orientation.
    
    # Compute all vertex positions
    all_verts = []
    for cx, cy in centers:
        for i in range(6):
            theta = i * math.pi / 3.0
            vx = cx + math.cos(theta)
            vy = cy + math.sin(theta)
            all_verts.append((vx, vy))
    
    all_verts = np.array(all_verts)
    
    # For the given outer angle, compute the 6 normals in world coordinates
    cos_a = math.cos(math.radians(angle_deg))
    sin_a = math.sin(math.radians(angle_deg))
    
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        nx = math.cos(theta)
        ny = math.sin(theta)
        # Rotate normal by outer angle
        rnx = nx * cos_a - ny * sin_a
        rny = nx * sin_a + ny * cos_a
        normals.append((rnx, rny))
    
    max_width = 0.0
    for nx, ny in normals:
        projs = all_verts[:, 0] * nx + all_verts[:, 1] * ny
        width = np.max(projs) - np.min(projs)
        if width > max_width:
            max_width = width
    
    return max_width / SQRT3


def construct():
    start_time = time.time()
    
    # Generate candidate polyhex shapes
    # For 11 cells, exhaustive generation may be large, so we use a subset
    # of well-known compact shapes
    
    # Shape 1: 3-4-3 hexagonal cluster (original)
    shape_343 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2),
    ]
    
    # Shape 2: More compact arrangement
    # Try a 2-3-3-2 or similar
    shape_compact1 = [
        (0, 0), (1, 0), (0, 1),
        (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2),
        (2, 1),
    ]
    
    # Shape 3: Another variant
    shape_compact2 = [
        (0, 0), (1, 0), (0, 1),
        (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2),
        (2, -1),
    ]
    
    # Shape 4: Tighter packing
    shape_compact3 = [
        (0, 0), (1, 0), (0, 1),
        (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2),
        (-2, 1),
    ]
    
    # Shape 5: Try different topology
    shape_compact4 = [
        (0, 0), (1, 0), (0, 1),
        (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2),
        (1, 2),
    ]
    
    # Shape 6: Another variant
    shape_compact5 = [
        (0, 0), (1, 0), (0, 1),
        (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2),
        (-1, 3),
    ]
    
    # Generate more shapes by trying all 11-cell polyhexes (limited)
    candidate_shapes = [shape_343, shape_compact1, shape_compact2, shape_compact3, 
                        shape_compact4, shape_compact5]
    
    # Also try to generate some from the polyhex generator if time permits
    if time.time() - start_time < DEADLINE - 60:
        try:
            all_polyhexes = generate_polyhexes(N)
            # Limit to a reasonable number
            for ph in list(all_polyhexes)[:200]:
                cells = list(ph)
                # Check if it's a valid connected shape
                candidate_shapes.append(cells)
        except Exception:
            pass
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_shape_idx = -1
    
    # For each candidate shape, optimize the outer hexagon parameters
    for shape_idx, shape in enumerate(candidate_shapes):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        if len(shape) != N:
            continue
        
        # Convert axial to Cartesian (unit hexagons, angle 0)
        centers = []
        for q, r in shape:
            x = 1.5 * q
            y = SQRT3 * (r + q / 2.0)
            centers.append((x, y))
        
        # Compute centroid
        cx_avg = sum(c[0] for c in centers) / len(centers)
        cy_avg = sum(c[1] for c in centers) / len(centers)
        
        # Translate so centroid is at origin
        centers_centered = [(cx - cx_avg, cy - cy_avg) for cx, cy in centers]
        
        inner_list = [[x, y, 0.0] for x, y in centers_centered]
        
        # Optimize outer angle and center
        # The center optimization is coupled with angle, but for a given angle,
        # the optimal center is the one that minimizes the max projection.
        
        # Use a 1D scan over angle, with center optimization for each angle
        
        best_for_shape = float('inf')
        best_angle_for_shape = 0.0
        best_center_for_shape = (0.0, 0.0)
        
        # Scan angles
        num_angles = 36
        for ai in range(num_angles):
            if time.time() - start_time > DEADLINE - 15:
                break
            
            angle_deg = ai * 180.0 / num_angles
            
            # For this angle, find optimal center
            # The optimal center minimizes the maximum of (max_proj - min_proj) / 2
            # for each of the 6 directions, but since the hexagon is centered,
            # we need to find c such that max_i |c · n_i - midpoint_i| is minimized
            # where midpoint_i = (max_j (p_j · n_i) + min_j (p_j · n_i)) / 2
            
            # Actually, for a centered hexagon, the constraint is:
            # max_j ((p_j - c) · n_i) <= apothem for all i, j
            # This is equivalent to: c · n_i >= max_j (p_j · n_i) - apothem
            # and c · n_i <= min_j (p_j · n_i) + apothem
            
            # The feasible region for c is the intersection of 6 strips.
            # The minimum apothem is when the strips just touch.
            
            # For a given angle, compute the required side by scanning
            # a grid of possible centers near the centroid
            
            cos_a = math.cos(math.radians(angle_deg))
            sin_a = math.sin(math.radians(angle_deg))
            
            normals = []
            for i in range(6):
                theta = math.radians(30 + i * 60)
                nx = math.cos(theta)
                ny = math.sin(theta)
                rnx = nx * cos_a - ny * sin_a
                rny = nx * sin_a + ny * cos_a
                normals.append((rnx, rny))
            
            # Compute projections of all vertices
            all_verts = []
            for cx, cy in centers_centered:
                for i in range(6):
                    theta = i * math.pi / 3.0
                    vx = cx + math.cos(theta)
                    vy = cy + math.sin(theta)
                    all_verts.append((vx, vy))
            
            all_verts = np.array(all_verts)
            
            # For each normal, compute min and max projection
            min_projs = []
            max_projs = []
            for nx, ny in normals:
                projs = all_verts[:, 0] * nx + all_verts[:, 1] * ny
                min_projs.append(np.min(projs))
                max_projs.append(np.max(projs))
            
            # The width in direction i is max_projs[i] - min_projs[i]
            # The minimum side for this angle is max_i (width_i) / sqrt(3)
            widths = [max_projs[i] - min_projs[i] for i in range(6)]
            side_for_angle = max(widths) / SQRT3
            
            # The optimal center is at the midpoint of the projections
            # c · n_i = (max_projs[i] + min_projs[i]) / 2
            # Solve for c from these 6 equations (overdetermined, use least squares)
            
            A = np.array(normals)
            b = np.array([(max_projs[i] + min_projs[i]) / 2.0 for i in range(6)])
            
            # Least squares solution
            c_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            ocx, ocy = c_sol[0], c_sol[1]
            
            if side_for_angle < best_for_shape:
                best_for_shape = side_for_angle
                best_angle_for_shape = angle_deg
                best_center_for_shape = (ocx, ocy)
        
        # Refine with local optimization
        if best_for_shape < best_side:
            # Fine-tune the angle
            def angle_objective(angle):
                return compute_min_enclosing_hex_side(centers_centered, angle)
            
            try:
                res = minimize(
                    angle_objective,
                    [best_angle_for_shape],
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                refined_angle = res.x[0]
                refined_side = res.fun
                
                if refined_side < best_for_shape:
                    best_for_shape = refined_side
                    best_angle_for_shape = refined_angle
                    # Recompute center
                    cos_a = math.cos(math.radians(refined_angle))
                    sin_a = math.sin(math.radians(refined_angle))
                    normals = []
                    for i in range(6):
                        theta = math.radians(30 + i * 60)
                        nx = math.cos(theta)
                        ny = math.sin(theta)
                        rnx = nx * cos_a - ny * sin_a
                        rny = nx * sin_a + ny * cos_a
                        normals.append((rnx, rny))
                    
                    all_verts = []
                    for cx, cy in centers_centered:
                        for i in range(6):
                            theta = i * math.pi / 3.0
                            vx = cx + math.cos(theta)
                            vy = cy + math.sin(theta)
                            all_verts.append((vx, vy))
                    
                    all_verts = np.array(all_verts)
                    min_projs = []
                    max_projs = []
                    for nx, ny in normals:
                        projs = all_verts[:, 0] * nx + all_verts[:, 1] * ny
                        min_projs.append(np.min(projs))
                        max_projs.append(np.max(projs))
                    
                    A = np.array(normals)
                    b = np.array([(max_projs[i] + min_projs[i]) / 2.0 for i in range(6)])
                    c_sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    best_center_for_shape = (c_sol[0], c_sol[1])
            except Exception:
                pass
            
            if best_for_shape < best_side:
                best_side = best_for_shape
                best_angle = best_angle_for_shape
                best_center = best_center_for_shape
                best_inner = inner_list
                best_shape_idx = shape_idx
    
    # If no valid shape found, use the original
    if best_inner is None:
        centers = []
        for q, r in shape_343:
            x = 1.5 * q
            y = SQRT3 * (r + q / 2.0)
            centers.append((x, y))
        
        cx_avg = sum(c[0] for c in centers) / len(centers)
        cy_avg = sum(c[1] for c in centers) / len(centers)
        centers_centered = [(cx - cx_avg, cy - cy_avg) for cx, cy in centers]
        best_inner = [[x, y, 0.0] for x, y in centers_centered]
        best_center = (0.0, 0.0)
        best_angle = 0.0
        best_side = compute_required_side(best_inner, best_center, best_angle)
    
    # Final verification and fallback
    if not check_all_disjoint(best_inner) or not check_inside_outer(best_inner, best_center, best_side, best_angle):
        # Fallback to original with slight scaling
        scale_fallback = 1.05
        fallback_centers = []
        for q, r in shape_343:
            x = 1.5 * scale_fallback * q
            y = SQRT3 * scale_fallback * (r + q / 2.0)
            fallback_centers.append((x, y))
        
        cx_avg = sum(c[0] for c in fallback_centers) / len(fallback_centers)
        cy_avg = sum(c[1] for c in fallback_centers) / len(fallback_centers)
        fallback_centered = [(cx - cx_avg, cy - cy_avg) for cx, cy in fallback_centers]
        best_inner = [[x, y, 0.0] for x, y in fallback_centered]
        best_center = (0.0, 0.0)
        best_angle = 0.0
        best_side = compute_required_side(best_inner, best_center, best_angle)
    
    return best_inner, best_center, best_side, best_angle

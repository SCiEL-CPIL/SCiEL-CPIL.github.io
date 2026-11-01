import numpy as np
import time
from scipy.optimize import differential_evolution

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_rad):
        """Vertices of a regular hexagon (flat-top when angle=0)."""
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def point_in_hexagon(pt, vertices):
        """Check if point is inside or on boundary of convex polygon (hexagon)."""
        n = len(vertices)
        for i in range(n):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % n]
            edge = v2 - v1
            normal = np.array([-edge[1], edge[0]])
            if np.dot(normal, pt - v1) < -1e-9:
                return False
        return True

    def polygons_disjoint_sat(p1, p2, eps=1e-7):
        """
        Returns True if two convex polygons are disjoint (allowing touching with eps tolerance).
        Uses Separating Axis Theorem (SAT).
        """
        for poly in (p1, p2):
            for i in range(len(poly)):
                v1 = poly[i]
                v2 = poly[(i + 1) % len(poly)]
                edge = v2 - v1
                normal = np.array([-edge[1], edge[0]])
                normal = normal / (np.linalg.norm(normal) + 1e-15)
                proj1 = np.dot(p1, normal)
                proj2 = np.dot(p2, normal)
                min1, max1 = proj1.min(), proj1.max()
                min2, max2 = proj2.min(), proj2.max()
                if max1 + eps < min2 or max2 + eps < min1:
                    return True
        return False

    def hexagons_disjoint(verts_list, eps=1e-7):
        """Check all pairs of hexagons are disjoint."""
        n = len(verts_list)
        for i in range(n):
            for j in range(i + 1, n):
                if not polygons_disjoint_sat(verts_list[i], verts_list[j], eps):
                    return False
        return True

    def check_validity(params, outer_side):
        """
        Check if a configuration is valid:
        1. All inner hexagons are disjoint
        2. All inner hexagons are inside the outer hexagon
        """
        # Parse parameters: x0, y0, a0, x1, y1, a1, ...
        verts_list = []
        for k in range(N):
            x = params[3 * k]
            y = params[3 * k + 1]
            angle_deg = params[3 * k + 2]
            angle_rad = np.radians(angle_deg)
            verts_list.append(hexagon_vertices(x, y, 1.0, angle_rad))

        # Check disjointness
        if not hexagons_disjoint(verts_list, eps=1e-7):
            return False

        # Check containment in outer hexagon
        outer_verts = hexagon_vertices(0.0, 0.0, outer_side, 0.0)
        for verts in verts_list:
            for pt in verts:
                if not point_in_hexagon(pt, outer_verts):
                    return False
        return True

    def objective(params):
        """
        Objective: minimize outer side length.
        We use a fixed outer side and penalize violations, or we can search
        over positions and compute the minimal enclosing hexagon.
        
        Strategy: fix outer side, check validity, return 0 if valid else penalty.
        But for DE, we need a continuous objective. Let's use:
        - If valid: return the outer side (we'll binary search on outer_side)
        - If invalid: return a large penalty
        
        Actually, let's make outer_side a parameter too, or use a different approach.
        
        Better: minimize a penalty function that includes:
        1. Overlap penalty (how much hexagons overlap)
        2. Containment penalty (how much hexagons stick out)
        """
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            return 1e10

        # Parse parameters: x0, y0, a0, x1, y1, a1, ...
        verts_list = []
        for k in range(N):
            x = params[3 * k]
            y = params[3 * k + 1]
            angle_deg = params[3 * k + 2]
            angle_rad = np.radians(angle_deg)
            verts_list.append(hexagon_vertices(x, y, 1.0, angle_rad))

        # Compute overlap penalty using SAT (minimum penetration depth approximation)
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                # Check if polygons overlap using SAT
                p1, p2 = verts_list[i], verts_list[j]
                min_gap = np.inf
                for poly in (p1, p2):
                    for idx in range(len(poly)):
                        v1 = poly[idx]
                        v2 = poly[(idx + 1) % len(poly)]
                        edge = v2 - v1
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm < 1e-15:
                            continue
                        normal = normal / norm
                        proj1 = np.dot(p1, normal)
                        proj2 = np.dot(p2, normal)
                        gap = max(proj1.min() - proj2.max(), proj2.min() - proj1.max())
                        if gap < 0:
                            min_gap = min(min_gap, -gap)
                if min_gap < 0:
                    overlap_penalty += min_gap ** 2

        # Containment penalty: compute minimal enclosing hexagon side length
        # For a point to be in a regular hexagon of side S centered at origin (flat-top):
        # The hexagon has vertices at (S*cos(60k), S*sin(60k))
        # A point (x,y) is inside if max over 6 directions of dot product <= S
        # Actually, for a regular hexagon with circumradius R = S, the inradius is R*sin(60) = S*sqrt(3)/2
        # The hexagon is the intersection of 6 half-planes.
        
        # For a flat-top regular hexagon with side length S centered at origin:
        # Vertices at angles 0, 60, 120, 180, 240, 300 degrees, radius S
        # The supporting lines are at distance S*cos(30) = S*sqrt(3)/2 from center
        # in directions 30, 90, 150, 210, 270, 330 degrees
        
        # So point (x,y) is inside hexagon of side S iff:
        # For each of 6 directions theta_k = 30 + 60*k degrees:
        #   x*cos(theta_k) + y*sin(theta_k) <= S * sqrt(3) / 2
        # and
        #   -(x*cos(theta_k) + y*sin(theta_k)) <= S * sqrt(3) / 2
        
        # So max_k |x*cos(theta_k) + y*sin(theta_k)| <= S * sqrt(3) / 2
        # Thus S >= max_k |x*cos(theta_k) + y*sin(theta_k)| * 2 / sqrt(3)
        
        theta_values = np.radians(30 + 60 * np.arange(6))
        cos_th = np.cos(theta_values)
        sin_th = np.sin(theta_values)
        
        max_extent = 0.0
        for verts in verts_list:
            for pt in verts:
                projections = np.abs(pt[0] * cos_th + pt[1] * sin_th)
                required_side = np.max(projections) * 2.0 / np.sqrt(3.0)
                max_extent = max(max_extent, required_side)
        
        # Objective: minimize max_extent (the required outer side) + overlap penalty
        return max_extent + 100.0 * overlap_penalty

    # Define bounds for each parameter
    # Positions: roughly in [-5, 5], angles: [0, 60)
    bounds = []
    for k in range(N):
        bounds.append((-6.0, 6.0))  # x
        bounds.append((-6.0, 6.0))  # y
        bounds.append((0.0, 60.0))  # angle in degrees

    # Initial guess: honeycomb-like arrangement
    def initial_guess():
        params = []
        # Place hexagons in a honeycomb pattern
        # Hexagon centers in honeycomb: spacing 2*sqrt(3) horizontally, sqrt(3) vertically
        dx = 2.0 * np.sqrt(3.0)
        dy = np.sqrt(3.0)
        positions = [
            (0, 0),
            (dx, 0),
            (dx/2, dy),
            (-dx/2, dy),
            (dx*1.5, dy),
            (0, 2*dy),
            (dx, 2*dy),
            (-dx/2, -dy),
            (dx/2, -dy),
            (dx*1.5, -dy),
            (0, -2*dy),
            (dx, -2*dy),
        ]
        for x, y in positions:
            params.append(x)
            params.append(y)
            params.append(0.0)
        return params

    x0 = initial_guess()

    # Run differential evolution
    best_result = None
    best_obj = np.inf

    try:
        result = differential_evolution(
            objective,
            bounds,
            x0=x0,
            maxiter=50,
            popsize=15,
            tol=1e-6,
            mutation=(0.3, 1.0),
            recombination=0.8,
            seed=42,
            polish=False,
            init='latinhypercube'
        )
        if result.success or result.fun < best_obj:
            best_obj = result.fun
            best_result = result.x
    except Exception as e:
        pass

    # If time remains, do a local refinement
    elapsed = time.time() - start_time
    if best_result is not None and elapsed < deadline - 20.0:
        try:
            from scipy.optimize import minimize
            res_local = minimize(
                objective,
                best_result,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if res_local.fun < best_obj:
                best_obj = res_local.fun
                best_result = res_local.x
        except Exception:
            pass

    # Parse best result
    if best_result is not None:
        inner = []
        for k in range(N):
            x = best_result[3 * k]
            y = best_result[3 * k + 1]
            angle_deg = best_result[3 * k + 2]
            inner.append([float(x), float(y), float(angle_deg)])
        
        # Compute the actual outer side length
        verts_list = []
        for k in range(N):
            x = best_result[3 * k]
            y = best_result[3 * k + 1]
            angle_deg = best_result[3 * k + 2]
            angle_rad = np.radians(angle_deg)
            verts_list.append(hexagon_vertices(x, y, 1.0, angle_rad))
        
        theta_values = np.radians(30 + 60 * np.arange(6))
        cos_th = np.cos(theta_values)
        sin_th = np.sin(theta_values)
        
        max_extent = 0.0
        for verts in verts_list:
            for pt in verts:
                projections = np.abs(pt[0] * cos_th + pt[1] * sin_th)
                required_side = np.max(projections) * 2.0 / np.sqrt(3.0)
                max_extent = max(max_extent, required_side)
        
        outer_side = float(max_extent)
        center = (0.0, 0.0)
        angle_outer = 0.0
        
        return inner, center, outer_side, angle_outer
    else:
        # Fallback: simple valid configuration
        inner = []
        for k in range(N):
            x = 2.0 * np.sqrt(3.0) * k
            inner.append([float(x), 0.0, 0.0])
        return inner, (0.0, 0.0), 100.0, 0.0

# EVOLVE-BLOCK-END

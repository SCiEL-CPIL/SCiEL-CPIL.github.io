import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 12
SQRT3 = np.sqrt(3.0)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_rad):
        """Vertices of a regular hexagon (pointy-top when angle=0, i.e., vertex at angle 0)."""
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def hexagon_halfwidths(verts):
        """
        For a regular hexagon, compute the 6 support function values.
        The 6 outward normals are at angles 30, 90, 150, 210, 270, 330 degrees (for flat-top)
        or 0, 60, 120, 180, 240, 300 (for pointy-top).
        Actually, for a regular hexagon with circumradius R (= side length),
        the inradius (apothem) is R * cos(30°) = R * sqrt(3)/2.
        
        For a regular hexagon centered at origin with a vertex at angle theta,
        the 6 outward face normals are at angles theta + 30°, theta + 90°, etc.
        
        The support function in direction n is: max_i dot(v_i, n)
        For a regular hexagon with circumradius R, the support in any direction 
        between two adjacent vertex directions equals R * cos(angle from nearest vertex)
        but it's easier to just compute it.
        
        Actually, for containment in an outer hexagon, I'll use the following approach:
        A point (x,y) is inside a regular hexagon of side S centered at origin (with vertex at angle 0)
        iff for all 6 directions k: dot((x,y), n_k) <= S * sqrt(3)/2
        where n_k are the 6 outward normals at angles 30+60k degrees.
        
        But we can rotate the outer hexagon. Let's fix the outer hexagon orientation to 0
        (vertex at angle 0) and just minimize the side length.
        """
        pass

    def min_enclosing_hex_side(verts_list):
        """
        Compute the minimum side length of a regular hexagon (centered at origin, fixed orientation)
        that contains all vertices of all inner hexagons.
        
        For a regular hexagon of side S centered at origin with vertex at angle 0:
        The 6 outward normals are at angles 30, 90, 150, 210, 270, 330 degrees.
        A point p is inside iff dot(p, n_k) <= S * sqrt(3)/2 for all k.
        Also dot(p, -n_k) <= S * sqrt(3)/2, i.e., |dot(p, n_k)| <= S * sqrt(3)/2.
        
        So S >= max over all points p and all k of |dot(p, n_k)| * 2 / sqrt(3).
        """
        # Normals at 30, 90, 150, 210, 270, 330 degrees
        normal_angles = np.radians(30 + 60 * np.arange(6))
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        max_proj = 0.0
        for verts in verts_list:
            # verts shape (6, 2)
            projs = verts @ normals  # (6, 6)
            max_proj = max(max_proj, np.max(np.abs(projs)))
        
        return max_proj * 2.0 / SQRT3

    def sat_overlap_depth(p1, p2):
        """
        Using SAT, compute the minimum penetration depth between two convex polygons.
        Returns 0 if disjoint, negative if overlapping (magnitude = penetration depth).
        Actually, returns the minimum gap (positive if separated, negative if overlapping).
        """
        min_gap = np.inf
        for poly in (p1, p2):
            for i in range(len(poly)):
                v1 = poly[i]
                v2 = poly[(i + 1) % len(poly)]
                edge = v2 - v1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-15:
                    continue
                normal = normal / norm
                proj1 = p1 @ normal
                proj2 = p2 @ normal
                gap = max(proj1.min() - proj2.max(), proj2.min() - proj1.max())
                if gap < min_gap:
                    min_gap = gap
        return min_gap

    def total_overlap_penalty(verts_list):
        """Sum of squared penetration depths for all overlapping pairs."""
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                gap = sat_overlap_depth(verts_list[i], verts_list[j])
                if gap < 0:
                    penalty += gap * gap
        return penalty

    def get_verts(params):
        """Convert parameter vector to list of hexagon vertex arrays."""
        verts_list = []
        for k in range(N):
            x = params[3 * k]
            y = params[3 * k + 1]
            angle_deg = params[3 * k + 2]
            angle_rad = np.radians(angle_deg)
            verts_list.append(hexagon_vertices(x, y, 1.0, angle_rad))
        return verts_list

    def objective(params):
        """Minimize enclosing hexagon side + overlap penalty."""
        elapsed = time.time() - start_time
        if elapsed > deadline - 3.0:
            return 1e10
        
        verts_list = get_verts(params)
        
        # Compute enclosing side
        enc_side = min_enclosing_hex_side(verts_list)
        
        # Compute overlap penalty
        overlap = total_overlap_penalty(verts_list)
        
        return enc_side + 50.0 * overlap

    # Honeycomb initial arrangement
    # In a honeycomb, adjacent hexagon centers are separated by 2 * apothem = 2 * sqrt(3)
    # for hexagons of side 1.
    # Actually, for unit hexagons (side=1), the distance between centers of adjacent
    # hexagons in a honeycomb is 2 * sqrt(3) ≈ 3.464.
    
    def honeycomb_positions():
        """Generate 12 hexagon center positions in a honeycomb-like cluster."""
        dx = 2.0 * SQRT3  # horizontal spacing
        dy = SQRT3        # vertical spacing for offset rows
        
        # Try a compact arrangement
        positions = [
            (0, 0),
            (dx, 0),
            (-dx, 0),
            (dx/2, dy),
            (-dx/2, dy),
            (3*dx/2, dy),
            (-3*dx/2, dy),
            (dx/2, -dy),
            (-dx/2, -dy),
            (3*dx/2, -dy),
            (-3*dx/2, -dy),
            (0, 2*dy),
        ]
        return positions

    pos = honeycomb_positions()
    x0 = []
    for x, y in pos:
        x0.append(x)
        x0.append(y)
        x0.append(0.0)
    x0 = np.array(x0)

    bounds = []
    for k in range(N):
        bounds.append((-5.0, 5.0))  # x
        bounds.append((-5.0, 5.0))  # y
        bounds.append((0.0, 60.0))  # angle in degrees

    best_params = x0.copy()
    best_obj = objective(x0)

    # Run differential evolution with time check
    try:
        def timed_objective(params):
            if time.time() > deadline - 5.0:
                raise TimeoutError
            return objective(params)

        result = differential_evolution(
            timed_objective,
            bounds,
            x0=x0,
            maxiter=30,
            popsize=12,
            tol=1e-8,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            polish=False,
            init='latinhypercube',
            updating='deferred',
            callback=lambda xk, fk: None
        )
        if result.fun < best_obj:
            best_obj = result.fun
            best_params = result.x
    except (TimeoutError, Exception) as e:
        pass

    # Local refinement with Nelder-Mead
    elapsed = time.time() - start_time
    if elapsed < deadline - 10.0:
        try:
            res_local = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if res_local.fun < best_obj:
                best_obj = res_local.fun
                best_params = res_local.x
        except Exception:
            pass

    # Try a few more local optimizations from perturbed starts if time allows
    elapsed = time.time() - start_time
    if elapsed < deadline - 15.0:
        rng = np.random.default_rng(123)
        for attempt in range(5):
            if time.time() > deadline - 5.0:
                break
            perturbed = best_params + rng.normal(0, 0.5, size=best_params.shape)
            # Wrap angles
            perturbed[2::3] = perturbed[2::3] % 60.0
            try:
                res = minimize(
                    objective,
                    perturbed,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                if res.fun < best_obj:
                    best_obj = res.fun
                    best_params = res.x
            except Exception:
                pass

    # Parse best result
    inner = []
    for k in range(N):
        x = float(best_params[3 * k])
        y = float(best_params[3 * k + 1])
        angle_deg = float(best_params[3 * k + 2]) % 60.0
        inner.append([x, y, angle_deg])

    # Compute actual outer side length
    verts_list = get_verts(best_params)
    outer_side = float(min_enclosing_hex_side(verts_list))
    
    center = (0.0, 0.0)
    angle_outer = 0.0

    return inner, center, outer_side, angle_outer
# EVOLVE-BLOCK-END

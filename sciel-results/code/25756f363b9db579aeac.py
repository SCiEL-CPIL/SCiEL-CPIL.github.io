"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull, distance as scipy_distance

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

    def hex_distance(v1, v2):
        """
        Compute minimum distance between two convex hexagons.
        Returns positive if separated, 0 if touching, negative if overlapping.
        Uses scipy's convex hull distance for accuracy.
        """
        try:
            # Check if one contains the other via point-in-polygon
            # Quick SAT check first
            separated = False
            for poly in [v1, v2]:
                for i in range(6):
                    v1p = poly[i]
                    v2p = poly[(i + 1) % 6]
                    edge = v2p - v1p
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm < 1e-10:
                        continue
                    normal = normal / norm
                    proj1 = np.dot(v1, normal)
                    proj2 = np.dot(v2, normal)
                    if np.max(proj1) < np.min(proj2) - 1e-9 or np.max(proj2) < np.min(proj1) - 1e-9:
                        separated = True
                        break
                if separated:
                    break

            if separated:
                # Use scipy distance for accurate separation distance
                hull1 = ConvexHull(v1)
                hull2 = ConvexHull(v2)
                d, _ = scipy_distance(v1, v2)
                return d
            else:
                # Overlapping: estimate penetration depth via SAT
                max_overlap = 0.0
                for poly in [v1, v2]:
                    for i in range(6):
                        v1p = poly[i]
                        v2p = poly[(i + 1) % 6]
                        edge = v2p - v1p
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm < 1e-10:
                            continue
                        normal = normal / norm
                        proj1 = np.dot(v1, normal)
                        proj2 = np.dot(v2, normal)
                        # Overlap amount: how much they overlap on this axis
                        overlap = min(np.max(proj1), np.max(proj2)) - max(np.min(proj1), np.min(proj2))
                        if overlap > max_overlap:
                            max_overlap = overlap
                return -max_overlap
        except Exception:
            # Fallback: simple center distance heuristic
            c1 = np.mean(v1, axis=0)
            c2 = np.mean(v2, axis=0)
            d = np.linalg.norm(c1 - c2)
            return d - 2.0  # rough estimate

    def compute_outer_side(inner_hexes, outer_cx, outer_cy, outer_angle_rad):
        """
        Compute minimum outer hexagon side length to contain all inner hexagons.
        Uses support function: for a regular hexagon centered at (cx,cy) with angle theta,
        a point (x,y) is inside iff for all 6 edge normals n_k:
            (p - center) . n_k <= side * sqrt(3) / 2
        So required side = max over all points and all 6 directions of:
            (p - center) . n_k * 2 / sqrt(3)
        """
        max_required = 0.0
        theta = outer_angle_rad
        # Precompute the 6 outward normals for the outer hexagon
        normals = []
        for k in range(6):
            edge_angle = theta + np.radians(k * 60)
            nx = np.cos(edge_angle)
            ny = np.sin(edge_angle)
            normals.append(np.array([nx, ny]))
        
        for hex_verts in inner_hexes:
            for v in hex_verts:
                dx = v[0] - outer_cx
                dy = v[1] - outer_cy
                for n in normals:
                    dist = dx * n[0] + dy * n[1]
                    if dist > 0:
                        required_S = dist * 2.0 / np.sqrt(3.0)
                        if required_S > max_required:
                            max_required = required_S
        return max_required

    def objective(params):
        """
        Objective: minimize outer side length + penalty for overlaps.
        params: [x0, y0, a0, ..., x11, y11, a11, cx, cy, ctheta]
        """
        if time.time() > deadline:
            return 1e6
        
        inner_x = params[0:12]
        inner_y = params[12:24]
        inner_a = params[24:36]
        outer_cx = params[36]
        outer_cy = params[37]
        outer_a = params[38]
        
        inner_hexes = []
        for i in range(N):
            verts = hexagon_vertices(inner_x[i], inner_y[i], 1.0, inner_a[i])
            inner_hexes.append(verts)
        
        # Compute smooth overlap penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                d = hex_distance(inner_hexes[i], inner_hexes[j])
                if d < 0:
                    # Overlapping: quadratic penalty
                    penalty += 50.0 * (-d) ** 2
                elif d < 0.01:
                    # Very close: small penalty to encourage separation
                    penalty += 5.0 * (0.01 - d) ** 2
        
        # Compute outer side
        side = compute_outer_side(inner_hexes, outer_cx, outer_cy, outer_a)
        
        return side + penalty

    def objective_fixed_outer(params):
        """
        Objective with outer hexagon fixed at origin, angle 0.
        params: [x0, y0, a0, ..., x11, y11, a11]
        """
        if time.time() > deadline:
            return 1e6
        
        inner_x = params[0:12]
        inner_y = params[12:24]
        inner_a = params[24:36]
        
        inner_hexes = []
        for i in range(N):
            verts = hexagon_vertices(inner_x[i], inner_y[i], 1.0, inner_a[i])
            inner_hexes.append(verts)
        
        # Compute smooth overlap penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                d = hex_distance(inner_hexes[i], inner_hexes[j])
                if d < 0:
                    penalty += 50.0 * (-d) ** 2
                elif d < 0.01:
                    penalty += 5.0 * (0.01 - d) ** 2
        
        # Compute outer side (centered at origin, angle 0)
        side = compute_outer_side(inner_hexes, 0.0, 0.0, 0.0)
        
        return side + penalty

    # Generate multiple initial configurations
    def generate_config_honeycomb_flat():
        """Honeycomb arrangement, flat-top hexagons."""
        hx, hy, ha = [], [], []
        sqrt3 = np.sqrt(3)
        # 4-4-4 arrangement
        for r in range(3):
            for c in range(4):
                x = c * sqrt3
                y = r * 1.5
                if r % 2 == 1:
                    x += sqrt3 / 2
                hx.append(x)
                hy.append(y)
                ha.append(0.0)
        hx = np.array(hx)
        hy = np.array(hy)
        ha = np.array(ha)
        # Center the configuration
        cx = (np.max(hx) + np.min(hx)) / 2
        cy = (np.max(hy) + np.min(hy)) / 2
        hx -= cx
        hy -= cy
        return np.concatenate([hx, hy, ha])

    def generate_config_honeycomb_pointy():
        """Honeycomb arrangement, pointy-top hexagons."""
        hx, hy, ha = [], [], []
        sqrt3 = np.sqrt(3)
        for r in range(4):
            for c in range(3):
                x = c * 1.5
                y = r * sqrt3
                if c % 2 == 1:
                    y += sqrt3 / 2
                hx.append(x)
                hy.append(y)
                ha.append(np.radians(30))
        hx = np.array(hx)
        hy = np.array(hy)
        ha = np.array(ha)
        cx = (np.max(hx) + np.min(hx)) / 2
        cy = (np.max(hy) + np.min(hy)) / 2
        hx -= cx
        hy -= cy
        return np.concatenate([hx, hy, ha])

    def generate_config_ring():
        """6 hexagons in a ring around center + 6 more."""
        hx, hy, ha = [], [], []
        # Inner ring of 6
        for i in range(6):
            angle = np.radians(i * 60)
            r = 1.8
            hx.append(r * np.cos(angle))
            hy.append(r * np.sin(angle))
            ha.append(angle)
        # Outer ring of 6
        for i in range(6):
            angle = np.radians(i * 60 + 30)
            r = 3.5
            hx.append(r * np.cos(angle))
            hy.append(r * np.sin(angle))
            ha.append(angle)
        hx = np.array(hx)
        hy = np.array(hy)
        ha = np.array(ha)
        cx = (np.max(hx) + np.min(hx)) / 2
        cy = (np.max(hy) + np.min(hy)) / 2
        hx -= cx
        hy -= cy
        return np.concatenate([hx, hy, ha])

    def generate_config_random(seed):
        """Random configuration."""
        rng = np.random.RandomState(seed)
        hx = rng.uniform(-3, 3, N)
        hy = rng.uniform(-3, 3, N)
        ha = rng.uniform(-np.pi, np.pi, N)
        hx -= np.mean(hx)
        hy -= np.mean(hy)
        return np.concatenate([hx, hy, ha])

    # Strategy: Use differential evolution with fixed outer hexagon (reduced dimensionality)
    # This is more efficient and the optimal outer hexagon is always centered on the
    # centroid of the inner configuration with optimal angle.
    
    best_params_fixed = None
    best_obj_fixed = np.inf
    
    # Run differential evolution with multiple starting points via bounds
    # Parameters: 36 (12 hexagons * 3 params each)
    bounds = [(-5, 5)] * 24 + [(-np.pi, np.pi)] * 12
    
    # Try different DE strategies
    de_strategies = ['best1bin', 'best1bin', 'rand1bin']
    
    for strategy_idx, strategy in enumerate(de_strategies):
        if time.time() > deadline - 30:
            break
        
        try:
            result = differential_evolution(
                objective_fixed_outer,
                bounds=bounds,
                strategy=strategy,
                maxiter=50,
                popsize=15,
                tol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.8,
                seed=42 + strategy_idx,
                polish=False,
                updating='immediate'
            )
            if result.fun < best_obj_fixed:
                best_obj_fixed = result.fun
                best_params_fixed = result.x
        except Exception:
            pass
    
    # Local polish with L-BFGS-B from best DE result
    if best_params_fixed is not None and time.time() < deadline - 15:
        try:
            result_local = minimize(
                objective_fixed_outer,
                best_params_fixed,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result_local.fun < best_obj_fixed:
                best_obj_fixed = result_local.fun
                best_params_fixed = result_local.x
        except Exception:
            pass
    
    # Extract solution
    inner_x = best_params_fixed[0:12]
    inner_y = best_params_fixed[12:24]
    inner_a = best_params_fixed[24:36]
    
    # Compute optimal outer hexagon parameters
    # Center at centroid, try several angles to find best
    cx = np.mean(inner_x)
    cy = np.mean(inner_y)
    
    inner_hexes = []
    for i in range(N):
        verts = hexagon_vertices(inner_x[i], inner_y[i], 1.0, inner_a[i])
        inner_hexes.append(verts)
    
    # Find best outer angle by sampling
    best_side = np.inf
    best_outer_angle = 0.0
    for angle_deg in range(0, 60, 2):
        angle_rad = np.radians(angle_deg)
        side = compute_outer_side(inner_hexes, cx, cy, angle_rad)
        if side < best_side:
            best_side = side
            best_outer_angle = angle_rad
    
    # Fine-tune outer angle
    for angle_deg in np.arange(best_outer_angle - 0.02, best_outer_angle + 0.02, 0.002):
        angle_rad = np.radians(angle_deg * 57.2958)  # convert
        side = compute_outer_side(inner_hexes, cx, cy, angle_rad)
        if side < best_side:
            best_side = side
            best_outer_angle = angle_rad

    # Verify validity
    valid = True
    for i in range(N):
        for j in range(i + 1, N):
            d = hex_distance(inner_hexes[i], inner_hexes[j])
            if d < -1e-6:  # Significant overlap
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: spread hexagons far apart
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    inner = [[inner_x[i], inner_y[i], np.degrees(inner_a[i])] for i in range(N)]
    outer_side = best_side
    
    return inner, (cx, cy), outer_side, np.degrees(best_outer_angle)
# EVOLVE-BLOCK-END

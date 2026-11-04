"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # --- Geometry helpers (vectorized where possible) ---
    def hex_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
        a = math.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = a + k * math.pi / 3.0
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            verts.append((x, y))
        return verts

    def point_in_convex(p, poly):
        """Check if point p is inside convex polygon poly (CCW)."""
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) < -1e-12:
                return False
        return True

    def polygon_area_signed(verts):
        n = len(verts)
        area = 0.0
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]
            area += x1 * y2 - x2 * y1
        return area / 2.0

    def line_intersect(p1, p2, p3, p4):
        """Intersection of line p1-p2 and line p3-p4."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-15:
            return p1
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    def polygon_overlap_area(v1, v2):
        """Compute overlap area of two convex polygons using Sutherland-Hodgman clipping."""
        def clip_polygon(subject, clip):
            output = list(subject)
            for i in range(len(clip)):
                if not output:
                    break
                inp = output
                output = []
                S = inp[-1]
                for E in inp:
                    if point_in_convex(E, clip):
                        if not point_in_convex(S, clip):
                            output.append(line_intersect(S, E, clip[i], clip[(i + 1) % len(clip)]))
                        output.append(E)
                    elif point_in_convex(S, clip):
                        output.append(line_intersect(S, E, clip[i], clip[(i + 1) % len(clip)]))
                    S = E
            return output

        clipped = clip_polygon(v1, v2)
        if len(clipped) < 3:
            return 0.0
        return abs(polygon_area_signed(clipped))

    def hex_in_hexagon(inner_verts, outer_verts):
        """Check if all vertices of inner hex are inside outer hex."""
        return all(point_in_convex(v, outer_verts) for v in inner_verts)

    # Fast containment check using support functions
    def fast_containment(inner_params, outer_center, outer_side, outer_angle):
        """
        Fast check: for each inner hex, check all 6 vertices against outer hex.
        inner_params: list of (x, y, angle_deg)
        """
        cx, cy = outer_center
        outer_verts = hex_vertices(cx, cy, outer_side, outer_angle)
        for (ix, iy, iang) in inner_params:
            inner_verts = hex_vertices(ix, iy, 1.0, iang)
            if not hex_in_hexagon(inner_verts, outer_verts):
                return False
        return True

    def outer_side_required_smooth(inner_params, outer_center, outer_angle, temperature=0.01):
        """
        Compute minimum outer side length using smooth log-sum-exp surrogate for max.
        Vectorized for speed.
        """
        cx, cy = outer_center
        apothem_factor = math.sqrt(3) / 2.0
        
        # Edge normals: for hexagon with vertex angle offset outer_angle,
        # the edge normals are at angles outer_angle + 30 + k*60 degrees
        normal_angles = np.radians(outer_angle + 30 + np.arange(6) * 60)
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6,2)
        
        # Precompute vertex offsets for each inner hex
        # For each hex, vertices are at angles iang + k*60 for k=0..5
        # Vertex offset: (cos(iang + k*pi/3), sin(iang + k*pi/3))
        
        # Collect all projections vectorized
        n_hex = len(inner_params)
        all_projs = []
        
        for i in range(n_hex):
            ix, iy, iang_deg = inner_params[i]
            iang = math.radians(iang_deg)
            # 6 vertex angle offsets
            v_angles = iang + np.arange(6) * math.pi / 3.0
            v_offsets = np.column_stack([np.cos(v_angles), np.sin(v_angles)])  # (6,2)
            # Vertex positions
            v_pos = v_offsets + np.array([ix, iy])  # (6,2)
            # Translate to outer center
            v_rel = v_pos - np.array([cx, cy])  # (6,2)
            # Projections onto all 6 normals: (6,2) @ (2,6) -> (6,6)
            projs = v_rel @ normals.T  # (6,6)
            all_projs.append(projs.ravel())
        
        if not all_projs:
            return 0.0
        
        projs = np.concatenate(all_projs)
        
        # Use log-sum-exp for smooth max approximation
        max_p = np.max(projs)
        if temperature < 1e-10:
            return float(max_p / apothem_factor)
        
        # Log-sum-exp: max + T * log(sum(exp((p - max)/T)))
        scaled = (projs - max_p) / temperature
        # Clip to avoid overflow
        scaled = np.clip(scaled, -50.0, 50.0)
        log_sum = np.log(np.sum(np.exp(scaled)))
        smooth_max = max_p + temperature * log_sum
        
        return float(smooth_max / apothem_factor)

    def outer_side_required_fast(inner_params, outer_center, outer_angle):
        """
        Compute minimum outer side length using exact max (for final validation).
        Vectorized.
        """
        cx, cy = outer_center
        apothem_factor = math.sqrt(3) / 2.0
        
        # Edge normals
        normal_angles = np.radians(outer_angle + 30 + np.arange(6) * 60)
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # (6,2)
        
        max_val = 0.0
        for (ix, iy, iang_deg) in inner_params:
            iang = math.radians(iang_deg)
            v_angles = iang + np.arange(6) * math.pi / 3.0
            v_offsets = np.column_stack([np.cos(v_angles), np.sin(v_angles)])
            v_pos = v_offsets + np.array([ix, iy])
            v_rel = v_pos - np.array([cx, cy])
            projs = v_rel @ normals.T
            m = np.max(projs)
            if m > max_val:
                max_val = m
        
        return max_val / apothem_factor

    def is_valid(inner_params, outer_center, outer_side, outer_angle):
        """Check if configuration is valid: no overlaps, all inside outer hex."""
        # Check containment
        if not fast_containment(inner_params, outer_center, outer_side, outer_angle):
            return False
        # Check non-overlap
        for i in range(N):
            for j in range(i + 1, N):
                vi = hex_vertices(inner_params[i][0], inner_params[i][1], 1.0, inner_params[i][2])
                vj = hex_vertices(inner_params[j][0], inner_params[j][1], 1.0, inner_params[j][2])
                if polygon_overlap_area(vi, vj) > 1e-8:
                    return False
        return True

    # --- Analytic seed solution ---
    def analytic_solution():
        """
        Construct the concentric ring packing with all hexagons axis-aligned (angle=0).
        - 1 center hex at origin
        - 6 ring-1 hexagons at distance 2 (along axes and diagonals)
        - 4 ring-2 hexagons at distance 4 (along diagonals)
        All hexagons rotated by 0 degrees (flat-top).
        """
        eps = 1e-9
        centers = []
        centers.append((0.0, 0.0))
        
        # Ring 1: 6 hexagons at distance 2, angles 0, 60, 120, 180, 240, 300
        for k in range(6):
            angle = math.radians(k * 60)
            x = (2.0 + eps) * math.cos(angle)
            y = (2.0 + eps) * math.sin(angle)
            centers.append((x, y))
            
        # Ring 2: 4 hexagons at distance 4, angles 30, 90, 150, 210, 270, 330 -> pick 4
        for k in range(4):
            angle = math.radians(30 + k * 90)
            x = (4.0 + 2*eps) * math.cos(angle)
            y = (4.0 + 2*eps) * math.sin(angle)
            centers.append((x, y))
        
        inner = []
        for x, y in centers:
            inner.append((x, y, 0.0))
        
        # Calculate required outer side length analytically
        # For flat-top hexagons, the support function along normal at angle theta
        # For a unit hex at origin with angle 0, vertices at angles 0,60,...,300
        # The outer hexagon with angle 0 has edge normals at 30,90,150,210,270,330
        # We need to find the minimum side length s such that all inner hexes fit.
        
        # Use the fast computation
        side = outer_side_required_fast(inner, (0.0, 0.0), 0.0) + 1e-9
        
        return inner, (0.0, 0.0), side, 0.0

    # Get the analytic solution as baseline
    inner_seed, outer_center_seed, side_seed, angle_seed = analytic_solution()
    best_score = side_seed
    best_result = (inner_seed, outer_center_seed, side_seed, angle_seed)

    # --- Optimization using gradient-based solver ---
    def objective_smooth(params, temperature=0.01):
        """
        Objective: smooth outer side length + smooth overlap penalty.
        params: 34-dimensional vector (t, x, y, angle) for each of 11 hexagons.
        Actually: first element is t (epigraph), then 3*N for positions.
        Angles are in radians.
        """
        t = params[0]
        inner_params = []
        for i in range(N):
            x = params[1 + 3 * i]
            y = params[2 + 3 * i]
            a = params[3 + 3 * i] * 180.0 / math.pi
            inner_params.append((x, y, a))

        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        
        # Use smooth approximation for optimization
        outer_side = outer_side_required_smooth(inner_params, outer_center, outer_angle, temperature)

        # Epigraph constraint: t >= outer_side (penalty if violated)
        epigraph_penalty = max(0.0, outer_side - t) ** 2 * 100.0

        # Smooth overlap penalty
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                vi = hex_vertices(inner_params[i][0], inner_params[i][1], 1.0, inner_params[i][2])
                vj = hex_vertices(inner_params[j][0], inner_params[j][1], 1.0, inner_params[j][2])
                ov = polygon_overlap_area(vi, vj)
                if ov > 1e-12:
                    overlap_penalty += ov * (1.0 + ov)

        return t + epigraph_penalty + 500.0 * overlap_penalty

    def objective(params):
        """
        Objective for exact evaluation (used in final polish).
        params: 34-dimensional vector (t, x, y, angle) for each of 11 hexagons.
        """
        t = params[0]
        inner_params = []
        for i in range(N):
            x = params[1 + 3 * i]
            y = params[2 + 3 * i]
            a = params[3 + 3 * i] * 180.0 / math.pi
            inner_params.append((x, y, a))

        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        
        outer_side = outer_side_required_fast(inner_params, outer_center, outer_angle)

        # Epigraph constraint
        epigraph_penalty = max(0.0, outer_side - t) ** 2 * 1000.0

        # Exact overlap penalty
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                vi = hex_vertices(inner_params[i][0], inner_params[i][1], 1.0, inner_params[i][2])
                vj = hex_vertices(inner_params[j][0], inner_params[j][1], 1.0, inner_params[j][2])
                ov = polygon_overlap_area(vi, vj)
                if ov > 1e-10:
                    overlap_penalty += ov ** 2

        return t + epigraph_penalty + 1000.0 * overlap_penalty

    def initial_guess_from_seed():
        """Generate initial positions from the analytic seed with epigraph variable t."""
        inner, _, _, _ = analytic_solution()
        params = np.zeros(1 + 3 * N)
        # Initial t estimate
        params[0] = 5.0
        for i, (x, y, a) in enumerate(inner):
            params[1 + 3 * i] = x
            params[2 + 3 * i] = y
            params[3 + 3 * i] = a * math.pi / 180.0
        return params

    # Try optimization to see if we can improve
    if time.time() < deadline - 25:
        params0 = initial_guess_from_seed()
        
        try:
            from scipy.optimize import minimize
            
            # Phase 1: Coarse optimization with smooth objective
            result = minimize(
                lambda p: objective_smooth(p, temperature=0.05),
                params0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if time.time() < deadline - 20:
                best_params = result.x
                t_val = best_params[0]
                inner_positions = []
                for i in range(N):
                    x = best_params[1 + 3 * i]
                    y = best_params[2 + 3 * i]
                    a = best_params[3 + 3 * i] * 180.0 / math.pi
                    inner_positions.append((x, y, a))

                outer_side = outer_side_required_fast(inner_positions, (0.0, 0.0), 0.0)

                if is_valid(inner_positions, (0.0, 0.0), outer_side, 0.0):
                    if outer_side < best_score:
                        best_score = outer_side
                        best_result = (inner_positions, (0.0, 0.0), outer_side, 0.0)
                        
                        # Phase 2: Fine optimization with lower temperature
                        if time.time() < deadline - 15:
                            # Update t to current outer_side
                            params_phase2 = best_params.copy()
                            params_phase2[0] = outer_side
                            result2 = minimize(
                                lambda p: objective_smooth(p, temperature=0.01),
                                params_phase2,
                                method='L-BFGS-B',
                                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                            )
                            
                            if time.time() < deadline - 10:
                                best_params2 = result2.x
                                inner_positions2 = []
                                for i in range(N):
                                    x = best_params2[1 + 3 * i]
                                    y = best_params2[2 + 3 * i]
                                    a = best_params2[3 + 3 * i] * 180.0 / math.pi
                                    inner_positions2.append((x, y, a))

                                outer_side2 = outer_side_required_fast(inner_positions2, (0.0, 0.0), 0.0)

                                if is_valid(inner_positions2, (0.0, 0.0), outer_side2, 0.0):
                                    if outer_side2 < best_score:
                                        best_score = outer_side2
                                        best_result = (inner_positions2, (0.0, 0.0), outer_side2, 0.0)
        except Exception:
            pass

    # Also try Nelder-Mead as a fallback or secondary optimization with smooth objective
    if time.time() < deadline - 20:
        params0 = initial_guess_from_seed()
        try:
            from scipy.optimize import minimize
            result = minimize(
                lambda p: objective_smooth(p, temperature=0.02),
                params0,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
            )
            
            if time.time() < deadline - 10:
                best_params = result.x
                inner_positions = []
                for i in range(N):
                    x = best_params[1 + 3 * i]
                    y = best_params[2 + 3 * i]
                    a = best_params[3 + 3 * i] * 180.0 / math.pi
                    inner_positions.append((x, y, a))

                outer_side = outer_side_required_fast(inner_positions, (0.0, 0.0), 0.0)

                if is_valid(inner_positions, (0.0, 0.0), outer_side, 0.0):
                    if outer_side < best_score:
                        best_score = outer_side
                        best_result = (inner_positions, (0.0, 0.0), outer_side, 0.0)
        except Exception:
            pass

    # Final high-precision polish pass with exact objective
    if time.time() < deadline - 5 and best_result is not None:
        try:
            from scipy.optimize import minimize
            # Start from the best found configuration
            best_inner, _, best_side, _ = best_result
            params_best = np.zeros(1 + 3 * N)
            params_best[0] = best_side  # Initialize t
            for i, (x, y, a) in enumerate(best_inner):
                params_best[1 + 3 * i] = x
                params_best[2 + 3 * i] = y
                params_best[3 + 3 * i] = a * math.pi / 180.0
            
            # High-precision polish
            result_polish = minimize(
                objective,
                params_best,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if time.time() < deadline - 1:
                best_params = result_polish.x
                inner_positions = []
                for i in range(N):
                    x = best_params[1 + 3 * i]
                    y = best_params[2 + 3 * i]
                    a = best_params[3 + 3 * i] * 180.0 / math.pi
                    inner_positions.append((x, y, a))

                outer_side = outer_side_required_fast(inner_positions, (0.0, 0.0), 0.0)

                if is_valid(inner_positions, (0.0, 0.0), outer_side, 0.0):
                    if outer_side < best_score:
                        best_score = outer_side
                        best_result = (inner_positions, (0.0, 0.0), outer_side, 0.0)
        except Exception:
            pass

    # Return the best result
    inner_positions, outer_center, outer_side, outer_angle = best_result
    inner = [[p[0], p[1], p[2]] for p in inner_positions]
    return inner, (outer_center[0], outer_center[1]), outer_side, outer_angle
# EVOLVE-BLOCK-END

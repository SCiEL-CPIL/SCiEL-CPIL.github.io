import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin
    sqrt3 = math.sqrt(3)

    # Unit hexagon vertices (pointy-top, centered at origin)
    # For a regular hexagon with side length 1, vertices at angles 0, 60, 120, ...
    # Actually, let's use flat-top: vertices at 0, 60, 120, 180, 240, 300 degrees
    # But the key property: a regular hexagon of side 1 has circumradius = 1
    # and inradius (apothem) = sqrt(3)/2
    # Two unit hexagons with the same orientation don't overlap iff center distance >= sqrt(3)
    # (since the distance between parallel edges is sqrt(3) when aligned)
    
    # Actually for two unit regular hexagons with the SAME orientation, the minimum
    # center distance to avoid overlap is sqrt(3) (the width across flats).
    # With different orientations it can be less.
    
    verts_rel = np.array([[math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] for i in range(6)])
    
    # Precompute edge normals for unit hexagon (6 directions)
    edge_normals_rel = np.array([[-math.sin((i + 0.5) * math.pi / 3), math.cos((i + 0.5) * math.pi / 3)] for i in range(6)])
    
    def get_rotated_verts(theta):
        """Get rotated unit hexagon vertices."""
        c, s = math.cos(theta), math.sin(theta)
        R = np.array([[c, -s], [s, c]])
        return verts_rel @ R.T
    
    def get_rotated_normals(theta):
        """Get rotated edge normals."""
        c, s = math.cos(theta), math.sin(theta)
        R = np.array([[c, -s], [s, c]])
        return edge_normals_rel @ R.T

    def compute_outer_side(centers, angles, outer_angle=0.0):
        """Compute minimum outer hexagon side length containing all inner hexagons.
        
        The outer hexagon has 6 faces. For it to contain a convex polygon,
        every vertex of the polygon must be on the correct side of each face.
        The minimum apothem A is the max over all vertices and all face normals
        of the projection onto that normal.
        """
        # Outer hexagon normals: 6 directions at outer_angle + k*60 degrees
        outer_normals = np.array([[math.cos(outer_angle + k * math.pi / 3), 
                                   math.sin(outer_angle + k * math.pi / 3)] for k in range(6)])
        
        max_proj = 0.0
        for i in range(len(centers)):
            cx, cy = centers[i]
            theta = angles[i]
            rv = get_rotated_verts(theta)
            # All vertices of this hexagon
            vx = rv[:, 0] + cx
            vy = rv[:, 1] + cy
            # Project onto each outer normal
            projs = np.abs(outer_normals[:, 0] * vx + outer_normals[:, 1] * vy)
            # For each normal, we need the max absolute projection (since hexagon is centered)
            # Actually we need max over vertices of dot product, and the constraint is
            # that all vertices satisfy n·v <= A for each normal n (with appropriate sign)
            # Since the hexagon can be anywhere, we need: for each outer normal n,
            # max over all vertices of (n · v) <= A
            # But also the hexagon extends in both directions, so we need the max
            # of |n·v| over all vertices? No - the outer hexagon is centered at origin,
            # so the constraint is that for each of the 6 normals n_k, 
            # all vertices satisfy n_k · v <= A
            # Since the outer hexagon is symmetric, we also need -n_k · v <= A,
            # which is equivalent to n_k · v >= -A.
            # So the constraint is |n_k · v| <= A for all vertices v and all k.
            # Thus A >= max over all vertices and all k of |n_k · v|
            
            # More efficient: for each normal, compute max of |n·v| over 6 vertices
            for k in range(6):
                nx, ny = outer_normals[k]
                proj = np.max(np.abs(nx * vx + ny * vy))
                if proj > max_proj:
                    max_proj = proj
        
        # Apothem = max_proj, side = 2 * apothem / sqrt(3)
        return 2.0 * max_proj / sqrt3

    def overlap_penalty(centers, angles, min_dist_sq=3.0):
        """Penalty for overlaps using center distance (same-orientation approximation).
        For hexagons with similar orientations, min center distance is sqrt(3).
        Use squared distance to avoid sqrt."""
        penalty = 0.0
        n = len(centers)
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist_sq = dx * dx + dy * dy
                if dist_sq < min_dist_sq:
                    # Penalty grows quadratically
                    penalty += (min_dist_sq - dist_sq) ** 2
        return penalty

    def sat_overlap_check(centers, angles):
        """Proper SAT check for hexagon-hexagon overlap. Returns True if any overlap."""
        n = len(centers)
        all_verts = []
        for i in range(n):
            rv = get_rotated_verts(angles[i])
            vx = rv[:, 0] + centers[i][0]
            vy = rv[:, 1] + centers[i][1]
            all_verts.append(np.stack([vx, vy], axis=-1))
        
        for i in range(n):
            for j in range(i + 1, n):
                v1 = all_verts[i]
                v2 = all_verts[j]
                # SAT: check all 12 edge normals (6 from each hexagon)
                sep = False
                for v in (v1, v2):
                    for e in range(6):
                        p1 = v[e]
                        p2 = v[(e + 1) % 6]
                        edge = p2 - p1
                        normal = np.array([-edge[1], edge[0]])
                        proj1 = v1 @ normal
                        proj2 = v2 @ normal
                        if proj1.max() < proj2.min() or proj2.max() < proj1.min():
                            sep = True
                            break
                    if sep:
                        break
                if not sep:
                    return True
        return False

    # Generate good initial packings
    d = sqrt3  # center distance for adjacent hexagons (same orientation)
    
    # Packing 1: 7 hexagons in honeycomb (center + 6 around) + 4 more
    # The 7-hexagon cluster has a "radius" of about 2*sqrt(3) from center to outer vertices
    # Let's compute: center hex at origin, ring 1 at distance sqrt(3)
    # Outer vertices of ring-1 hexagons are at distance sqrt(3) + 1 from origin
    # So the cluster fits in a hexagon with apothem ~ sqrt(3) + 1, side ~ 2(sqrt(3)+1)/sqrt(3)
    
    base7 = [(0.0, 0.0)]
    for i in range(6):
        angle = i * math.pi / 3
        base7.append((d * math.cos(angle), d * math.sin(angle)))
    
    # Ring 2 positions (axial coordinates with |q|,|r|,|s| max = 2)
    ring2 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                x = d * (q + r / 2.0)
                y = d * (sqrt3 / 2.0) * r
                ring2.append((x, y))
    
    # We need 4 more from ring 2 (or nearby positions)
    # Try different subsets and also try "compressed" arrangements
    
    import itertools
    
    best_S = float('inf')
    best_result = None
    
    # Strategy 1: Use 7 base + 4 from ring 2, optimize with Nelder-Mead
    # But to save time, try a few promising combinations
    
    # Actually, let's think about what's optimal.
    # 11 hexagons: the densest known packing of 11 unit circles in a hexagon...
    # But these are hexagons, not circles.
    # 
    # Key insight: if all hexagons have the same orientation, the problem reduces to
    # packing 11 points (centers) such that pairwise distances >= sqrt(3), minimizing
    # the enclosing hexagon.
    #
    # With rotations allowed, we might do slightly better, but let's first solve
    # the axis-aligned case well, then try to improve with rotations.
    
    # For the axis-aligned case, this is equivalent to finding 11 points in a hexagon
    # with mutual distances >= sqrt(3).
    
    # Let's use a different approach: parameterize by the outer hexagon size S,
    # and check feasibility via optimization.
    
    # Actually, let's just do multi-start Nelder-Mead with good initial guesses.
    
    np.random.seed(42)
    
    def objective(params, use_sat=False):
        if time.time() > deadline - 3:
            return 1e10
        
        # params: [x0, y0, x1, y1, ..., x10, y10, outer_angle]
        # All hexagons have angle 0 (same orientation) for simplicity first
        centers = [(params[i], params[i+1]) for i in range(0, 2*N, 2)]
        angles = [0.0] * N
        outer_angle = params[-1]
        
        S = compute_outer_side(centers, angles, outer_angle)
        pen = overlap_penalty(centers, angles)
        
        return S + 50.0 * pen
    
    def objective_with_rot(params):
        """params: [x0, y0, a0, x1, y1, a1, ..., x10, y10, a10, outer_angle]"""
        if time.time() > deadline - 3:
            return 1e10
        
        centers = []
        angles = []
        for i in range(N):
            centers.append((params[3*i], params[3*i+1]))
            angles.append(params[3*i+2])
        outer_angle = params[-1]
        
        S = compute_outer_side(centers, angles, outer_angle)
        pen = overlap_penalty(centers, angles)
        
        return S + 50.0 * pen
    
    # Initial guesses
    initial_guesses = []
    
    # Guess 1: 7 base + 4 ring-2 (first 4)
    g1 = list(base7) + ring2[:4]
    initial_guesses.append(g1)
    
    # Guess 2: 7 base + 4 ring-2 (different subset - opposite sides)
    # Pick ring2 positions that are more spread out
    # ring2 indices: let's pick every other one
    g2 = list(base7) + [ring2[i] for i in [0, 3, 6, 9]]
    initial_guesses.append(g2)
    
    # Guess 3: More compact - shrink the ring
    # Try a "flower" pattern: center + 6 at d + 4 at slightly less than 2d
    g3 = [(0.0, 0.0)]
    for i in range(6):
        angle = i * math.pi / 3
        g3.append((d * math.cos(angle), d * math.sin(angle)))
    for i in range(4):
        angle = i * math.pi / 2 + math.pi / 4  # 45, 135, 225, 315
        r = 1.9 * d
        g3.append((r * math.cos(angle), r * math.sin(angle)))
    initial_guesses.append(g3)
    
    # Guess 4: Two rows of 5-6
    # Hexagons in two staggered rows
    g4 = []
    for i in range(6):
        x = i * d - 2.5 * d
        y = 0
        g4.append((x, y))
    for i in range(5):
        x = i * d - 2 * d
        y = d * sqrt3 / 2 * 2  # offset for staggered
        g4.append((x, y))
    initial_guesses.append(g4)
    
    # Guess 5: Random perturbation of guess 1
    g5 = [(x + np.random.normal(0, 0.1), y + np.random.normal(0, 0.1)) for x, y in g1]
    initial_guesses.append(g5)
    
    # Try each initial guess
    for gi, guess in enumerate(initial_guesses):
        if time.time() > deadline - 15:
            break
        
        # Phase 1: Optimize positions only (angles fixed at 0)
        init_params = []
        for x, y in guess:
            init_params.extend([x, y])
        init_params.append(0.0)  # outer angle
        init_params = np.array(init_params)
        
        try:
            res = minimize(objective, init_params, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5,
                                   'maxfev': 1000})
            
            if res.fun < best_S:
                centers_opt = [(res.x[i], res.x[i+1]) for i in range(0, 2*N, 2)]
                angles_opt = [0.0] * N
                outer_angle_opt = res.x[-1]
                
                # Check validity
                pen = overlap_penalty(centers_opt, angles_opt)
                if pen < 1e-4:
                    S_val = compute_outer_side(centers_opt, angles_opt, outer_angle_opt)
                    if S_val < best_S:
                        best_S = S_val
                        best_result = (centers_opt, angles_opt, outer_angle_opt)
                        
        except Exception:
            continue
        
        # Phase 2: If we have a good result, try adding rotations
        if best_S < float('inf') and time.time() < deadline - 20:
            init_params_rot = []
            for i in range(N):
                init_params_rot.extend([best_result[0][i][0], best_result[0][i][1], 0.0])
            init_params_rot.append(best_result[2])
            init_params_rot = np.array(init_params_rot)
            
            try:
                res2 = minimize(objective_with_rot, init_params_rot, method='Nelder-Mead',
                               options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5,
                                        'maxfev': 800})
                
                if res2.fun < best_S:
                    centers_rot = [(res2.x[3*i], res2.x[3*i+1]) for i in range(N)]
                    angles_rot = [res2.x[3*i+2] for i in range(N)]
                    outer_angle_rot = res2.x[-1]
                    
                    pen = overlap_penalty(centers_rot, angles_rot)
                    if pen < 1e-4:
                        S_val = compute_outer_side(centers_rot, angles_rot, outer_angle_rot)
                        if S_val < best_S:
                            best_S = S_val
                            best_result = (centers_rot, angles_rot, outer_angle_rot)
                            
            except Exception:
                continue
    
    # If no valid result, try a final broad optimization
    if best_S == float('inf') and time.time() < deadline - 25:
        # Start from a very spread out configuration and let optimizer compress
        init_centers = []
        for i in range(N):
            angle = i * 2 * math.pi / N
            r = 5.0
            init_centers.append((r * math.cos(angle), r * math.sin(angle)))
        
        init_params = []
        for x, y in init_centers:
            init_params.extend([x, y])
        init_params.append(0.0)
        init_params = np.array(init_params)
        
        try:
            res = minimize(objective, init_params, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4,
                                   'maxfev': 2000})
            if res.fun < best_S:
                centers_opt = [(res.x[i], res.x[i+1]) for i in range(0, 2*N, 2)]
                angles_opt = [0.0] * N
                outer_angle_opt = res.x[-1]
                pen = overlap_penalty(centers_opt, angles_opt)
                if pen < 1e-4:
                    S_val = compute_outer_side(centers_opt, angles_opt, outer_angle_opt)
                    if S_val < best_S:
                        best_S = S_val
                        best_result = (centers_opt, angles_opt, outer_angle_opt)
        except Exception:
            pass
    
    # Fallback
    if best_S == float('inf') or best_result is None:
        # Line packing - definitely valid but large
        inner = [[d * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 50.0, 0.0
    
    centers_final, angles_final, outer_angle_final = best_result
    
    # Final validation with SAT
    if sat_overlap_check(centers_final, angles_final):
        # SAT says overlap - try to fix by increasing distances slightly
        # Or just return with a note; the penalty should have prevented this
        # For safety, scale up positions
        scale = 1.05
        centers_final = [(x * scale, y * scale) for x, y in centers_final]
        S_final = compute_outer_side(centers_final, angles_final, outer_angle_final)
    else:
        S_final = compute_outer_side(centers_final, angles_final, outer_angle_final)
    
    inner = []
    for i in range(N):
        x, y = centers_final[i]
        ang = math.degrees(angles_final[i])
        inner.append([float(x), float(y), float(ang)])
    
    return inner, (0.0, 0.0), float(S_final), float(math.degrees(outer_angle_final))

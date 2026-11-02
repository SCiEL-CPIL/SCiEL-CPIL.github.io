"""
Packing 11 unit regular hexagons inside a regular hexagon.
Strategy: Use a fixed compact lattice configuration (7 center + 4 ring-2).
Optimize the global rotation of the cluster and the orientation of the outer hexagon
using a high-precision ternary search and linear programming for the minimal container.
"""
import math
import time
import itertools
import numpy as np
from scipy.optimize import linprog, minimize

N = 11

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with side length `side`,
    centered at (cx, cy), rotated by angle_deg degrees."""
    verts = []
    for k in range(6):
        a = math.radians(angle_deg + 60 * k)
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return verts

def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with side `side` rotated by angle_deg."""
    dx = px - cx
    dy = py - cy
    a = -math.radians(angle_deg)
    lx = dx * math.cos(a) - dy * math.sin(a)
    ly = dx * math.sin(a) + dy * math.cos(a)
    apothem = side * math.sqrt(3) / 2
    for k in range(6):
        a_norm = math.radians(30 + 60 * k)
        nx = math.cos(a_norm)
        ny = math.sin(a_norm)
        if lx * nx + ly * ny > apothem + 1e-9:
            return False
    return True

def _hexagons_disjoint(hex1, hex2, eps=1e-9):
    """Check if two hexagons have disjoint interiors using convex polygon separation."""
    cx1, cy1, s1, a1 = hex1
    cx2, cy2, s2, a2 = hex2
    # Quick bounding-circle rejection
    dx = cx2 - cx1
    dy = cy2 - cy1
    if dx * dx + dy * dy > (s1 + s2) ** 2 + 1e-12:
        return True
    verts1 = np.array(_hex_vertices(cx1, cy1, s1, a1))
    verts2 = np.array(_hex_vertices(cx2, cy2, s2, a2))
    # Check if any vertex of poly1 is strictly inside poly2
    for vx, vy in verts1:
        if _point_in_hex(vx, vy, cx2, cy2, s2, a2):
            return False
    for vx, vy in verts2:
        if _point_in_hex(vx, vy, cx1, cy1, s1, a1):
            return False
    # Separating axis theorem: check all 12 edge normals
    for verts, other in [(verts1, verts2), (verts2, verts1)]:
        for i in range(6):
            ax, ay = verts[i]
            bx, by = verts[(i + 1) % 6]
            ex = bx - ax
            ey = by - ay
            length = math.sqrt(ex * ex + ey * ey)
            if length < 1e-15:
                continue
            # Normal perpendicular to edge
            nx = -ey / length
            ny = ex / length
            # Project both polygons onto this normal
            proj1 = verts[:, 0] * nx + verts[:, 1] * ny
            proj2 = other[:, 0] * nx + other[:, 1] * ny
            if proj1.max() <= proj2.min() + eps or proj2.max() <= proj1.min() + eps:
                return True
    return False

def _all_disjoint(inner_hexes, eps=1e-9):
    """Check if all pairs of hexagons are disjoint."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if not _hexagons_disjoint(inner_hexes[i], inner_hexes[j], eps):
                return False
    return True

def _inner_in_outer(inner_hex, outer_hex):
    """Check if inner hexagon is fully inside outer hexagon."""
    cx, cy, s, a = inner_hex
    ox, oy, oside, oangle = outer_hex
    verts = _hex_vertices(cx, cy, s, a)
    for vx, vy in verts:
        if not _point_in_hex(vx, vy, ox, oy, oside, oangle):
            return False
    return True

def _min_outer_side_analytical(hexes, oangle_rad):
    """
    Find minimum side length of outer hexagon with rotation oangle_rad
    that contains all hexagons in `hexes` (centered at origin).
    Returns (min_side, center_x, center_y)
    """
    # Collect all vertices of all inner hexagons
    points = []
    for cx, cy, s, ang in hexes:
        verts = _hex_vertices(cx, cy, s, ang)
        points.extend(verts)
    
    if not points:
        return 0.0, 0.0, 0.0
    
    pts = np.array(points)
    
    # 6 normal directions for the outer hexagon
    u_vectors = []
    for k in range(6):
        theta = oangle_rad + math.radians(30 + 60 * k)
        u_vectors.append((math.cos(theta), math.sin(theta)))
    
    # For each direction, compute max and min projection
    P_vals = np.zeros(6)  # max projection
    Q_vals = np.zeros(6)  # min projection
    for k in range(6):
        ux, uy = u_vectors[k]
        projections = pts[:, 0] * ux + pts[:, 1] * uy
        P_vals[k] = np.max(projections)
        Q_vals[k] = np.min(projections)
    
    # Linear programming: minimize A (apothem)
    # Variables: ox, oy, A (3 variables)
    # Constraints:
    # For each k:
    #   ox*u_kx + oy*uky - A <= Q_k
    #   -ox*u_kx - oy*uky - A <= -P_k
    
    A_mat = np.zeros((12, 3))
    b_vec = np.zeros(12)
    for k in range(6):
        ux, uy = u_vectors[k]
        A_mat[2*k, 0] = ux
        A_mat[2*k, 1] = uy
        A_mat[2*k, 2] = -1
        b_vec[2*k] = Q_vals[k]
        
        A_mat[2*k+1, 0] = -ux
        A_mat[2*k+1, 1] = -uy
        A_mat[2*k+1, 2] = -1
        b_vec[2*k+1] = -P_vals[k]
    
    c_vec = np.array([0.0, 0.0, 1.0])
    
    try:
        result = linprog(c_vec, A_ub=A_mat, b_ub=b_vec, bounds=[(None, None), (None, None), (0, None)], method='highs')
        if result.success:
            A_opt = result.x[2]
            ox_opt = result.x[0]
            oy_opt = result.x[1]
            side = A_opt * 2.0 / math.sqrt(3)
            return side, ox_opt, oy_opt
    except Exception:
        pass
    
    # Fallback if linprog fails (should be rare)
    return float('inf'), 0.0, 0.0

def _soft_max(vals, tau):
    """Log-sum-exp soft max of a 1D array."""
    v = np.asarray(vals, dtype=float)
    if v.size == 0:
        return 0.0
    m = v.max()
    return m + tau * math.log(np.sum(np.exp((v - m) / tau)))

def _soft_min(vals, tau):
    return -_soft_max(-np.asarray(vals, dtype=float), tau)

def _smooth_outer_cost(positions, hexes, angle_rad, tau=0.02):
    """
    Smooth surrogate for the minimal enclosing hexagon side.
    positions: (11,2) array of inner hex centers.
    hexes: list of (cx,cy,s,ang) inner hexagons (used only for vertex extraction).
    Returns (smooth_side, ox, oy).
    """
    # Collect all vertices
    all_pts = []
    for cx, cy, s, ang in hexes:
        all_pts.extend(_hex_vertices(cx, cy, s, ang))
    pts = np.array(all_pts)
    n = len(pts)

    # 6 normal directions
    u = np.zeros((6, 2))
    for k in range(6):
        theta = angle_rad + math.radians(30 + 60 * k)
        u[k, 0] = math.cos(theta)
        u[k, 1] = math.sin(theta)

    # Projections: (6, n)
    proj = pts @ u.T  # proj[k, i] = p_i . u_k

    # We want to choose ox, oy, A to minimize A such that:
    #   (ox*u_k + oy*u_ky) - A <= min_i proj[k,i]   =>  A >= (ox*u_k+oy*uy) - min_k
    #   -(ox*u_k+oy*uy) - A <= -max_i proj[k,i]     =>  A >= max_k - (ox*u_k+oy*uy)
    # For a fixed ox,oy, the required A_k = max( (ox*u_k+oy*uy) - min_k,  max_k - (ox*u_k+oy*uy) )
    # Smooth: A_soft = soft_max over k of those two terms, then A = soft_max of the 12 values? 
    # Actually A must be >= each of 12 values, so A = max of 12 values. Use soft max of 12.
    
    # But ox,oy also variables. To make it a smooth function of (positions) only,
    # we need to optimize ox,oy too. Instead, we return a cost that is the soft-max
    # of the 12 support values AFTER solving ox,oy optimally. That's non-smooth in ox,oy.
    # Alternative: parameterize by ox,oy as part of the optimization.
    # For the surrogate used inside minimize, we'll include ox,oy in the variable vector.
    
    # This function is a helper for the minimize call; it takes (pos_flat, ox, oy).
    # But the signature above is different. Let's restructure: make a separate function.
    # For now, return a placeholder; the real work is in _smooth_cost_full.
    return 0.0

def _smooth_cost_full(x, hexes, angle_rad, tau=0.02):
    """
    x: flat array of length 2*11 + 2 = 24: [x0,y0, ..., x10,y10, ox, oy]
    hexes: list of 11 (cx,cy,s,ang) - only s and ang matter (shape), positions come from x.
    Returns smooth approximation of the enclosing hex side length.
    """
    n = 11
    pos = x[:2*n].reshape(n, 2)
    ox = x[2*n]
    oy = x[2*n+1]

    # Build hexagons with updated centers
    hexes_upd = [(pos[i,0], pos[i,1], hexes[i][2], hexes[i][3]) for i in range(n)]

    # Collect all vertices
    all_pts = []
    for cx, cy, s, ang in hexes_upd:
        all_pts.extend(_hex_vertices(cx, cy, s, ang))
    pts = np.array(all_pts)

    u = np.zeros((6, 2))
    for k in range(6):
        theta = angle_rad + math.radians(30 + 60 * k)
        u[k, 0] = math.cos(theta)
        u[k, 1] = math.sin(theta)

    proj = pts @ u.T  # (6, npts)
    P = proj.max(axis=1)  # (6,)
    Q = proj.min(axis=1)  # (6,)

    # 12 support values
    vals = np.zeros(12)
    for k in range(6):
        vals[2*k]   = ox * u[k,0] + oy * u[k,1] - Q[k]
        vals[2*k+1] = P[k] - (ox * u[k,0] + oy * u[k,1])

    A_soft = _soft_max(vals, tau)
    side = A_soft * 2.0 / math.sqrt(3)
    return side

def _smooth_cost_with_separation(x, hexes, angle_rad, tau=0.02, sep_tau=0.05):
    """
    Smooth cost = enclosing_side + penalty for overlaps.
    Overlap penalty: for each pair, soft-min of separation distance (negative if overlapping).
    """
    n = 11
    pos = x[:2*n].reshape(n, 2)
    ox = x[2*n]
    oy = x[2*n+1]

    hexes_upd = [(pos[i,0], pos[i,1], hexes[i][2], hexes[i][3]) for i in range(n)]

    # Enclosing cost
    all_pts = []
    for cx, cy, s, ang in hexes_upd:
        all_pts.extend(_hex_vertices(cx, cy, s, ang))
    pts = np.array(all_pts)

    u = np.zeros((6, 2))
    for k in range(6):
        theta = angle_rad + math.radians(30 + 60 * k)
        u[k, 0] = math.cos(theta)
        u[k, 1] = math.sin(theta)

    proj = pts @ u.T
    P = proj.max(axis=1)
    Q = proj.min(axis=1)

    vals = np.zeros(12)
    for k in range(6):
        vals[2*k]   = ox * u[k,0] + oy * u[k,1] - Q[k]
        vals[2*k+1] = P[k] - (ox * u[k,0] + oy * u[k,1])

    A_soft = _soft_max(vals, tau)
    side = A_soft * 2.0 / math.sqrt(3)

    # Separation penalty: for each pair, compute minimum distance between their boundaries.
    # Approximate: use center distance minus sum of circumradii as a rough gap.
    # Better: compute actual polygon separation via SAT, but that's non-smooth.
    # Use a smooth surrogate: for each pair, the "gap" is approximated by
    # (center_dist - r1 - r2) where r is circumradius = side.
    # If gap < 0, they overlap. Penalty = soft_min over pairs of gap, clipped.
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            dx = pos[i,0] - pos[j,0]
            dy = pos[i,1] - pos[j,1]
            d = math.sqrt(dx*dx + dy*dy)
            gap = d - 1.0 - 1.0  # side length 1, circumradius = 1
            # Soft penalty for negative gaps
            if gap < 0:
                penalty += (-gap) ** 2

    return side + 10.0 * penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    sqrt3 = math.sqrt(3)

    # Flat-top hexagon mapping
    def axial_to_xy(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return (x, y)

    # Define 11 positions in axial coordinates (q, r)
    # Center
    pos_axial = [(0, 0)]
    # Ring 1 (6 neighbors)
    pos_axial.extend([(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)])
    
    # We need 4 more from Ring 2.
    ring2_candidates = [
        (2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2), 
        (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2), 
        (0, 2), (1, 2)
    ]
    
    best_side = float('inf')
    best_xy = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_inner_hexes = None
    best_outer_center = (0.0, 0.0)
    best_combo = None

    # Pre-compute Cartesian coordinates for all potential positions
    all_axial = pos_axial + ring2_candidates
    all_xy = [axial_to_xy(q, r) for q, r in all_axial]
    
    # Indices for ring 2 in all_xy
    ring2_indices = list(range(7, len(all_xy)))

    # Iterate over combinations of 4 ring-2 positions
    for combo in itertools.combinations(ring2_indices, 4):
        if time.time() > deadline:
            break
            
        # Construct the list of 11 points
        selected_indices = list(range(7)) + list(combo)
        full_xy = [all_xy[i] for i in selected_indices]
        
        # Center the cluster at origin for calculation
        cx_mean = sum(x for x, y in full_xy) / 11.0
        cy_mean = sum(y for x, y in full_xy) / 11.0
        
        # Shift points to origin
        centered_xy = [(x - cx_mean, y - cy_mean) for x, y in full_xy]
        
        # Inner hexagons are aligned with the lattice (angle 0)
        inner_hexes_centered = [(x, y, 1.0, 0.0) for x, y in centered_xy]
        
        # Use ternary search to find the best outer rotation angle in [0, 30] degrees
        lo = 0.0
        hi = 30.0
        gr = (math.sqrt(5) - 1) / 2
        c = hi - gr * (hi - lo)
        d = lo + gr * (hi - lo)
        
        fc = _min_outer_side_analytical(inner_hexes_centered, math.radians(c))[0]
        fd = _min_outer_side_analytical(inner_hexes_centered, math.radians(d))[0]
        
        # Iterate ternary search
        for _ in range(25):  # Increased iterations for precision
            if time.time() > deadline:
                break
            if fc < fd:
                hi = d
                d = c
                fd = fc
                c = hi - gr * (hi - lo)
                fc = _min_outer_side_analytical(inner_hexes_centered, math.radians(c))[0]
            else:
                lo = c
                c = d
                fc = fd
                d = lo + gr * (hi - lo)
                fd = _min_outer_side_analytical(inner_hexes_centered, math.radians(d))[0]
        
        # Final best angle is the midpoint of the final interval
        best_angle_candidate = (lo + hi) / 2.0
        side, ox, oy = _min_outer_side_analytical(inner_hexes_centered, math.radians(best_angle_candidate))
        
        if side < best_side:
            best_side = side
            best_angle = best_angle_candidate
            best_center = (ox + cx_mean, oy + cy_mean)
            best_xy = full_xy
            best_inner_hexes = inner_hexes_centered
            best_outer_center = (ox, oy)
            best_combo = combo

    if best_xy is None:
        # Fallback
        best_xy = [all_xy[i] for i in range(7)]
        cx_mean = sum(x for x, y in best_xy) / 7.0
        cy_mean = sum(y for x, y in best_xy) / 7.0
        centered_xy = [(x - cx_mean, y - cy_mean) for x, y in best_xy]
        inner_hexes_centered = [(x, y, 1.0, 0.0) for x, y in centered_xy]
        best_side, ox, oy = _min_outer_side_analytical(inner_hexes_centered, 0.0)
        best_center = (ox + cx_mean, oy + cy_mean)
        best_angle = 0.0
        best_inner_hexes = inner_hexes_centered
        best_outer_center = (ox, oy)
        best_combo = tuple(range(7, 11))

    # --- High-Precision Polish Phase ---
    # Refine the best angle with a finer search
    if best_inner_hexes is not None and time.time() < deadline - 30.0:
        # Fine angle search around the current best angle
        angle_lo = max(0.0, best_angle - 1.0)
        angle_hi = min(30.0, best_angle + 1.0)
        gr = (math.sqrt(5) - 1) / 2
        c = angle_hi - gr * (angle_hi - angle_lo)
        d = angle_lo + gr * (angle_hi - angle_lo)
        fc = _min_outer_side_analytical(best_inner_hexes, math.radians(c))[0]
        fd = _min_outer_side_analytical(best_inner_hexes, math.radians(d))[0]
        for _ in range(30):
            if time.time() > deadline - 30.0:
                break
            if fc < fd:
                angle_hi = d
                d = c
                fd = fc
                c = angle_hi - gr * (angle_hi - angle_lo)
                fc = _min_outer_side_analytical(best_inner_hexes, math.radians(c))[0]
            else:
                angle_lo = c
                c = d
                fc = fd
                d = angle_lo + gr * (angle_hi - angle_lo)
                fd = _min_outer_side_analytical(best_inner_hexes, math.radians(d))[0]
        
        best_angle_polish = (angle_lo + angle_hi) / 2.0
        side_polish, ox_polish, oy_polish = _min_outer_side_analytical(
            best_inner_hexes, math.radians(best_angle_polish))
        
        if side_polish < best_side - 1e-12:
            best_side = side_polish
            best_angle = best_angle_polish
            # Recalculate center based on the centered inner hexes
            cx_mean = sum(x for x, y, s, a in best_inner_hexes) / 11.0
            cy_mean = sum(y for x, y, s, a in best_inner_hexes) / 11.0
            best_center = (ox_polish + cx_mean, oy_polish + cy_mean)
            best_outer_center = (ox_polish, oy_polish)

    # Build final inner hexagons
    # Note: best_xy contains the centers of the inner hexagons in the global coordinate system
    inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_xy]

    # --- Smooth refinement phase: optimize positions with log-sum-exp surrogate ---
    if best_inner_hexes is not None and time.time() < deadline - 40.0:
        # Start from the current best centered positions
        n = 11
        pos0 = np.array([[h[0], h[1]] for h in best_inner_hexes])
        ox0 = best_outer_center[0]
        oy0 = best_outer_center[1]
        x0 = np.concatenate([pos0.flatten(), [ox0, oy0]])
        angle_rad = math.radians(best_angle)

        # Multi-start with a few tau values and small perturbations
        best_smooth_side = best_side
        best_smooth_x = x0.copy()
        
        for tau in [0.05, 0.02, 0.01]:
            if time.time() > deadline - 40.0:
                break
            for attempt in range(3):
                if time.time() > deadline - 40.0:
                    break
                x_init = x0.copy()
                if attempt > 0:
                    # Small random perturbation
                    rng = np.random.default_rng(hash((tau, attempt)) % (2**32))
                    x_init = x_init + rng.normal(0, 0.01, size=x_init.shape)
                
                try:
                    res = minimize(
                        _smooth_cost_with_separation,
                        x_init,
                        args=(best_inner_hexes, angle_rad, tau, 0.05),
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    if res.success:
                        # Evaluate exact side length at the smooth solution
                        pos_sol = res.x[:2*n].reshape(n, 2)
                        ox_sol = res.x[2*n]
                        oy_sol = res.x[2*n+1]
                        hexes_sol = [(pos_sol[i,0], pos_sol[i,1], 1.0, 0.0) for i in range(n)]
                        # Check disjointness
                        if _all_disjoint(hexes_sol):
                            exact_side, ox_exact, oy_exact = _min_outer_side_analytical(
                                hexes_sol, angle_rad)
                            if exact_side < best_smooth_side - 1e-12:
                                best_smooth_side = exact_side
                                best_smooth_x = res.x.copy()
                except Exception:
                    pass

        if best_smooth_side < best_side - 1e-12:
            # Extract the improved positions
            pos_final = best_smooth_x[:2*n].reshape(n, 2)
            ox_final = best_smooth_x[2*n]
            oy_final = best_smooth_x[2*n+1]
            
            # Recompute exact solution with the new positions
            hexes_final = [(pos_final[i,0], pos_final[i,1], 1.0, 0.0) for i in range(n)]
            # Center the cluster
            cx_mean = sum(h[0] for h in hexes_final) / n
            cy_mean = sum(h[1] for h in hexes_final) / n
            hexes_centered = [(h[0]-cx_mean, h[1]-cy_mean, 1.0, 0.0) for h in hexes_final]
            
            best_side, ox, oy = _min_outer_side_analytical(hexes_centered, math.radians(best_angle))
            best_center = (ox + cx_mean, oy + cy_mean)
            best_outer_center = (ox, oy)
            best_inner_hexes = hexes_centered
            best_xy = [(h[0]+cx_mean, h[1]+cy_mean) for h in hexes_final]
            inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_xy]

    # Verify disjointness
    if not _all_disjoint(inner_hexes):
        # Fallback: scale up slightly to ensure disjointness if local search or lattice logic failed
        # This is a safety net. The lattice configuration should always be disjoint.
        scale = 1.01
        inner_hexes = [(x * scale, y * scale, 1.0, 0.0) for x, y in best_xy]
        cx_mean = sum(x for x, y, s, a in inner_hexes) / 11.0
        cy_mean = sum(y for x, y, s, a in inner_hexes) / 11.0
        centered_xy = [(x - cx_mean, y - cy_mean) for x, y in [(h[0], h[1]) for h in inner_hexes]]
        inner_hexes_centered = [(x, y, 1.0, 0.0) for x, y in centered_xy]
        best_side, ox, oy = _min_outer_side_analytical(inner_hexes_centered, math.radians(best_angle))
        best_center = (ox + cx_mean, oy + cy_mean)

    # Verify all inner hexagons are inside outer
    outer_hex = (best_center[0], best_center[1], best_side, best_angle)
    valid = True
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            valid = False
            break

    if not valid:
        # Increase side slightly to ensure containment
        best_side *= 1.005
        outer_hex = (best_center[0], best_center[1], best_side, best_angle)

    # Format output: inner rows (x, y, angle_degrees)
    inner = [[h[0], h[1], h[3]] for h in inner_hexes]

    return inner, (best_center[0], best_center[1]), best_side, best_angle

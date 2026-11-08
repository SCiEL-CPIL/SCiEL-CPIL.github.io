import numpy as np
import time
from scipy.optimize import minimize

N = 11
DEADLINE = 145.0
SQRT3 = np.sqrt(3.0)

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy)."""
    angle = np.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = angle + np.pi * i / 3.0
        verts[i, 0] = cx + side * np.cos(theta)
        verts[i, 1] = cy + side * np.sin(theta)
    return verts

def _hex_edge_normals(angle_deg):
    """Return 6 outward unit normal vectors of a regular hexagon."""
    angle = np.radians(angle_deg)
    normals = np.empty((6, 2))
    for i in range(6):
        mid_angle = angle + np.pi / 6.0 + np.pi * i / 3.0
        normals[i, 0] = np.cos(mid_angle)
        normals[i, 1] = np.sin(mid_angle)
    return normals

def _min_distance_hex_hex(v1, v2):
    """
    Compute the minimum separation between two hexagons defined by their vertices.
    Returns a negative value if they overlap (approximate penetration depth), 
    or positive if separated.
    Uses SAT: min over axes of the gap.
    """
    axes = []
    for verts in (v1, v2):
        for i in range(6):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % 6]
            ex = x2 - x1
            ey = y2 - y1
            norm = np.sqrt(ex*ex + ey*ey)
            if norm > 1e-12:
                axes.append((-ey / norm, ex / norm))
    
    min_gap = float('inf')
    for ax, ay in axes:
        proj1 = v1[:, 0] * ax + v1[:, 1] * ay
        proj2 = v2[:, 0] * ax + v2[:, 1] * ay
        gap1 = proj2.min() - proj1.max()
        gap2 = proj1.min() - proj2.max()
        gap = min(gap1, gap2)
        if gap < min_gap:
            min_gap = gap
    return min_gap

def _compute_objective(vars, S):
    """
    vars layout: [x1, y1, a1, ..., x11, y11, a11, X, Y, Phi]
    S: fixed outer hexagon side length.
    Returns sum of squared violations.
    """
    # Unpack
    centers = np.array(vars[:3*N]).reshape(N, 3)
    outer_center = np.array(vars[3*N:3*N+2])
    outer_angle = vars[3*N+2]
    
    # Precompute vertices for all inner hexagons
    # v[i] is (6, 2)
    vs = np.empty((N, 6, 2))
    for i in range(N):
        vs[i] = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, centers[i, 2])
    
    penalty = 0.0
    
    # 1. Overlap penalty
    for i in range(N):
        for j in range(i + 1, N):
            dist = _min_distance_hex_hex(vs[i], vs[j])
            if dist < 0:
                penalty += dist * dist
    
    # 2. Containment penalty
    outer_normals = _hex_edge_normals(outer_angle)
    inradius = S * SQRT3 / 2.0
    
    for i in range(N):
        verts = vs[i]
        # Vectorized distance check
        # verts: (6, 2), outer_normals: (6, 2)
        # We need (v - c) . n for all v, n
        # diffs: (6, 1, 2) - (1, 2) -> (6, 2)
        # proj: (6, 2) @ (2, 6) -> (6, 6)
        diffs = verts - outer_center
        projs = diffs @ outer_normals.T # (6, 6)
        violations = projs - inradius
        # Sum of squares of positive violations
        pos_viol = np.maximum(0, violations)
        penalty += np.sum(pos_viol * pos_viol)
        
    return penalty

def _try_fit(S, x0, deadline):
    """
    Try to fit hexagons in hexagon of side S.
    Returns (success, best_vars, best_penalty)
    """
    start_time = time.time()
    
    # Bounds for variables
    # x, y: [-10, 10]
    # angle: [0, 60] (periodicity)
    # Outer center: [-10, 10]
    # Outer angle: [0, 60]
    bounds = [(-10, 10), (-10, 10), (0, 60)] * N
    bounds += [(-10, 10), (-10, 10), (0, 60)]
    
    # Objective wrapper
    def obj(v):
        return _compute_objective(v, S)
    
    # Gradient (numerical)
    # L-BFGS-B can use numerical gradient, but providing one is faster.
    # Let's rely on numerical gradient first for simplicity, as N is small.
    # Actually, 36 variables, numerical gradient is 72 evaluations.
    # Each evaluation is ~55 SAT tests + 66 containment checks.
    # Fast enough.
    
    res = minimize(obj, x0, method='L-BFGS-B', bounds=bounds, 
                   options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False})
    
    if time.time() > deadline:
        return False, x0, obj(x0)
        
    return res.success, res.x, res.fun

def construct():
    start_time = time.time()
    
    # Initial guess: Hexagonal lattice
    # Center at 0,0
    # 6 around at distance sqrt(3)
    # 4 more at distance 2*sqrt(3)
    
    def get_initial_vars(S_guess=5.0):
        centers = []
        centers.append([0.0, 0.0, 0.0])
        for i in range(6):
            ang = np.pi / 3 * i
            centers.append([SQRT3 * np.cos(ang), SQRT3 * np.sin(ang), 0.0])
        for i in range(6):
            ang = np.pi / 3 * i + np.pi / 6
            centers.append([2*SQRT3 * np.cos(ang), 2*SQRT3 * np.sin(ang), 0.0])
        # We need 11. Remove 1 from the second ring?
        # The lattice has 1+6+6=13.
        # Remove 2?
        # Let's just use the first 11.
        centers = centers[:11]
        
        vars = np.array(centers).flatten()
        # Outer center 0,0, angle 0
        vars = np.concatenate([vars, [0.0, 0.0, 0.0]])
        return vars
    
    x0 = get_initial_vars()
    
    # Binary search for minimum S
    S_low = 4.0
    S_high = 8.0
    
    best_valid_vars = None
    best_valid_S = S_high
    
    # First, ensure we have a valid packing at S_high
    # If not, increase S_high
    while time.time() < start_time + 10:
        success, vars, pen = _try_fit(S_high, x0, start_time + 10)
        if pen < 1e-5:
            best_valid_vars = vars
            best_valid_S = S_high
            break
        else:
            S_high *= 1.5
            if S_high > 20:
                # Fallback to a very large S
                S_high = 20.0
                success, vars, pen = _try_fit(S_high, x0, start_time + 10)
                if pen < 1e-3:
                    best_valid_vars = vars
                    best_valid_S = S_high
                break

    if best_valid_vars is None:
        # Emergency fallback
        # Line packing
        centers = [(2 * SQRT3 * k, 0.0, 0.0) for k in range(N)]
        vars = np.array(centers).flatten()
        vars = np.concatenate([vars, [0.0, 0.0, 0.0]])
        best_valid_vars = vars
        # Calculate S for this
        # Just return a large S
        best_valid_S = 20.0

    # Binary search
    x_current = best_valid_vars.copy()
    
    for _ in range(20): # 20 steps of binary search
        if time.time() > DEADLINE - 5:
            break
            
        S_mid = (S_low + S_high) / 2.0
        
        # Warm start with current best
        success, vars, pen = _try_fit(S_mid, x_current, DEADLINE - 5)
        
        if pen < 1e-5:
            S_high = S_mid
            best_valid_S = S_mid
            best_valid_vars = vars
            x_current = vars
        else:
            S_low = S_mid
    
    # Extract best result
    vars = best_valid_vars
    centers = vars[:3*N].reshape(N, 3)
    outer_center = vars[3*N:3*N+2]
    outer_angle = vars[3*N+2]
    
    # Final refinement: Check validity and calculate exact S
    # The S_high from binary search is an upper bound.
    # We can try to minimize S directly now with a very tight penalty?
    # Or just return S_high.
    # To be safe, let's compute the actual minimum enclosing hexagon for this configuration.
    
    def _min_enclosing_hex_side(centers, outer_angle):
        """
        Compute the minimum side length of a regular hexagon with given angle
        that encloses the given hexagons.
        """
        normals = _hex_edge_normals(outer_angle)
        all_points = np.empty((6 * len(centers), 2))
        for i in range(len(centers)):
            cx, cy = centers[i, 0], centers[i, 1]
            all_points[6*i:6*i+6] = _hex_vertices(cx, cy, 1.0, centers[i, 2])
        
        projections = all_points @ normals.T
        A = projections.max(axis=0)
        
        # Minimize max_i (A_i - c·n_i)
        # Simple subgradient step
        extreme_points = np.empty((6, 2))
        for i in range(6):
            idx = projections[:, i].argmax()
            extreme_points[i] = all_points[idx]
        c = extreme_points.mean(axis=0)
        
        for _ in range(50):
            vals = A - c @ normals.T
            max_val = vals.max()
            active = vals >= max_val - 1e-10
            if active.sum() <= 1:
                break
            grad = normals[active].mean(axis=0)
            gnorm = np.linalg.norm(grad)
            if gnorm < 1e-12:
                break
            best_alpha = 0.0
            best_f = max_val
            for alpha in np.linspace(-1.5, 1.5, 31):
                c_try = c + alpha * grad
                f_try = (A - c_try @ normals.T).max()
                if f_try < best_f - 1e-14:
                    best_f = f_try
                    best_alpha = alpha
            if abs(best_alpha) < 1e-11:
                break
            c = c + best_alpha * grad
            
        vals = A - c @ normals.T
        max_proj = vals.max()
        side = max_proj / (SQRT3 / 2.0)
        return side, c

    # Try to optimize the outer angle to minimize the side
    best_side = float('inf')
    best_outer_center = outer_center
    best_outer_angle = outer_angle
    
    # Check a few angles
    for ang in np.linspace(0, 60, 21):
        side, c = _min_enclosing_hex_side(centers, ang)
        if side < best_side:
            best_side = side
            best_outer_center = c
            best_outer_angle = ang
            
    # Verify validity of this final configuration
    # If invalid, return the previous best_valid_S
    
    def _is_valid(centers, outer_center, side, outer_angle):
        # Check containment
        outer_normals = _hex_edge_normals(outer_angle)
        inradius = side * SQRT3 / 2.0
        for i in range(N):
            verts = _hex_vertices(centers[i, 0], centers[i, 1], 1.0, centers[i, 2])
            for vx, vy in verts:
                for nx, ny in outer_normals:
                    dist = (vx - outer_center[0]) * nx + (vy - outer_center[1]) * ny
                    if dist > inradius + 1e-6:
                        return False
        # Check overlap
        vs = [_hex_vertices(centers[i, 0], centers[i, 1], 1.0, centers[i, 2]) for i in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                dist = _min_distance_hex_hex(vs[i], vs[j])
                if dist < -1e-6:
                    return False
        return True

    if _is_valid(centers, best_outer_center, best_side, best_outer_angle):
        final_side = best_side
        final_center = best_outer_center
        final_angle = best_outer_angle
    else:
        # Use the S_high from binary search and the vars that were valid
        # Recompute center/angle for these vars?
        # The vars included outer center/angle.
        # Let's just use the S_high and the center/angle from vars.
        final_side = best_valid_S
        final_center = best_valid_vars[3*N:3*N+2]
        final_angle = best_valid_vars[3*N+2]

    inner = [[float(centers[i, 0]), float(centers[i, 1]), float(centers[i, 2])] for i in range(N)]
    return inner, (float(final_center[0]), float(final_center[1])), float(final_side), float(final_angle)


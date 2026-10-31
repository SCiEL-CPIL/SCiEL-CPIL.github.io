"""Rewritten: compact triangular-lattice seed + smooth objective (outer side + overlap penalty) with scipy.optimize."""
import math
import time
import numpy as np

N = 11

# Time guard
_START = time.time()
_TIMEOUT = 140.0  # seconds

def _time_left():
    return _TIMEOUT - (time.time() - _START)

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon centered at (cx,cy), side length 1, rotated by angle_deg."""
    a = math.radians(angle_deg)
    pts = []
    for k in range(6):
        ang = a + math.radians(60 * k)
        pts.append((cx + side * math.cos(ang), cy + side * math.sin(ang)))
    return pts

def _polygon_overlap_depth(A, B):
    """
    Returns the penetration depth (positive if overlapping, 0 if not).
    Uses separating axis theorem for two convex polygons (hexagons).
    We compute the minimum translation distance to separate; if the polygons
    are already separated, return 0.
    """
    # Combine vertices
    nA = len(A)
    nB = len(B)
    max_depth = 0.0
    # Test axes from A's edges
    for i in range(nA):
        p1 = A[i]
        p2 = A[(i + 1) % nA]
        # Edge vector
        ex = p2[0] - p1[0]
        ey = p2[1] - p1[1]
        # Normal (perpendicular)
        nx = -ey
        ny = ex
        norm = math.hypot(nx, ny)
        if norm < 1e-12:
            continue
        nx /= norm
        ny /= norm
        # Project A and B onto this axis
        minA = maxA = minB = maxB = None
        for v in A:
            proj = v[0]*nx + v[1]*ny
            if minA is None or proj < minA: minA = proj
            if maxA is None or proj > maxA: maxA = proj
        for v in B:
            proj = v[0]*nx + v[1]*ny
            if minB is None or proj < minB: minB = proj
            if maxB is None or proj > maxB: maxB = proj
        # Overlap on this axis?
        if minB >= maxA or minA >= maxB:
            return 0.0  # Separating axis found
        depth = min(maxA - minB, maxB - minA)
        if depth > max_depth:
            max_depth = depth
    # Test axes from B's edges
    for i in range(nB):
        p1 = B[i]
        p2 = B[(i + 1) % nB]
        ex = p2[0] - p1[0]
        ey = p2[1] - p1[1]
        nx = -ey
        ny = ex
        norm = math.hypot(nx, ny)
        if norm < 1e-12:
            continue
        nx /= norm
        ny /= norm
        minA = maxA = minB = maxB = None
        for v in A:
            proj = v[0]*nx + v[1]*ny
            if minA is None or proj < minA: minA = proj
            if maxA is None or proj > maxA: maxA = proj
        for v in B:
            proj = v[0]*nx + v[1]*ny
            if minB is None or proj < minB: minB = proj
            if maxB is None or proj > maxB: maxB = proj
        if minB >= maxA or minA >= maxB:
            return 0.0
        depth = min(maxA - minB, maxB - minA)
        if depth > max_depth:
            max_depth = depth
    return max_depth

def _point_in_hexagon(px, py, cx, cy, angle_deg, side=1.0):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with given angle and side."""
    # Transform to hexagon local frame
    dx = px - cx
    dy = py - cy
    a = math.radians(angle_deg)
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    lx = dx * cos_a + dy * sin_a
    ly = -dx * sin_a + dy * cos_a
    # Regular hexagon with vertices at angles 0,60,...: |x| <= side, |y| <= side*sqrt(3)/2, and |y| <= sqrt(3)*(side - |x|/2) for |x|<=side... 
    # Actually for a "flat-top" hexagon (vertices at 0,60,120,180,240,300), the constraints are:
    # |x| <= side
    # |y| <= side * sqrt(3)/2
    # |y| <= sqrt(3) * (side - |x|/2)  -- this isn't quite right
    # Let me use the 6 half-plane inequalities.
    # For a hexagon centered at origin with vertices at angle k*60 degrees, the edges are perpendicular to the angle k*60+30.
    # Edge i (between vertex i and i+1) has outward normal at angle (i*60 + 30) degrees.
    # The distance from center to edge (apothem) = side * sqrt(3)/2.
    apothem = side * math.sqrt(3) / 2.0
    for k in range(6):
        ang = math.radians(60 * k + 30)
        nx = math.cos(ang)
        ny = math.sin(ang)
        # Half-plane: nx*x + ny*y <= apothem
        if lx * nx + ly * ny > apothem + 1e-12:
            return False
    return True

def _inner_in_outer(inner_verts, ocx, ocy, oside, oangle):
    """Check if all inner vertices are inside the outer hexagon."""
    for (vx, vy) in inner_verts:
        if not _point_in_hexagon(vx, vy, ocx, ocy, oangle, oside):
            return False
    return True

def _compute_objective(params):
    """
    params: flat array of length 3*N + 4:
      First 3*N: for each hexagon k: (x_k, y_k, angle_k_deg)
      Next 4: (ocx, ocy, oside, oangle_deg)
    Returns objective value (outer side length + penalty * overlap).
    """
    Nloc = N
    inner = []
    for k in range(Nloc):
        x = params[3*k]
        y = params[3*k+1]
        ang = params[3*k+2]
        inner.append((x, y, ang))
    ocx = params[3*Nloc]
    ocy = params[3*Nloc+1]
    oside = params[3*Nloc+2]
    oangle = params[3*Nloc+3]
    
    if oside < 1.0:
        return 1e6  # outer side must be at least 1
    
    # Compute vertices for each inner hexagon
    inner_verts_list = []
    for (x, y, ang) in inner:
        verts = _hex_vertices(x, y, ang)
        inner_verts_list.append(verts)
    
    # Check inner-inner overlaps
    total_overlap = 0.0
    for i in range(Nloc):
        for j in range(i+1, Nloc):
            depth = _polygon_overlap_depth(inner_verts_list[i], inner_verts_list[j])
            if depth > 0:
                total_overlap += depth**2
    
    # Check inner-outer containment
    containment_violation = 0.0
    for verts in inner_verts_list:
        for (vx, vy) in verts:
            if not _point_in_hexagon(vx, vy, ocx, ocy, oangle, oside):
                # Compute how far outside: distance to hexagon boundary approx
                # For penalty, use the maximum amount by which the half-plane is violated
                dx = vx - ocx
                dy = vy - ocy
                a = math.radians(oangle)
                cos_a = math.cos(a)
                sin_a = math.sin(a)
                lx = dx * cos_a + dy * sin_a
                ly = -dx * sin_a + dy * cos_a
                apothem = oside * math.sqrt(3) / 2.0
                for k in range(6):
                    ang_edge = math.radians(60 * k + 30)
                    nx = math.cos(ang_edge)
                    ny = math.sin(ang_edge)
                    val = lx * nx + ly * ny
                    if val > apothem:
                        containment_violation += (val - apothem)**2
    
    penalty = 10.0 * total_overlap + 50.0 * containment_violation
    return oside + penalty

def _make_initial_params(scale=1.0):
    """Create initial parameters: triangular lattice cluster of 11 hexagons + outer hexagon."""
    # Triangular lattice basis: spacing sqrt(3)
    sp = math.sqrt(3)
    e1 = (sp, 0.0)
    e2 = (sp/2, 1.5)
    
    # Generate lattice points near origin
    candidates = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * e1[0] + j * e2[0]
            y = i * e1[1] + j * e2[1]
            candidates.append((i*i + j*j + i*j, x, y))
    candidates.sort()
    # Take 11 closest
    lattice_pts = [(x*scale, y*scale) for (_, x, y) in candidates[:11]]
    
    params = []
    for (x, y) in lattice_pts:
        params.append(x)
        params.append(y)
        params.append(0.0)  # angle
    # Outer hexagon: centered at centroid of lattice points
    cx = sum(p[0] for p in lattice_pts) / 11.0
    cy = sum(p[1] for p in lattice_pts) / 11.0
    # Initial outer side: large enough to contain
    max_dist = max(math.hypot(p[0]-cx, p[1]-cy) + 1.0 for p in lattice_pts)
    oside = max_dist * 2.0 / math.sqrt(3) + 1.0
    params.extend([cx, cy, oside, 0.0])
    return np.array(params, dtype=float)

def construct():
    global _START
    _START = time.time()
    
    from scipy.optimize import minimize
    
    best_params = _make_initial_params(1.0)
    best_val = _compute_objective(best_params)
    
    # Multiple restarts with different initial scalings and random perturbations
    rng = np.random.default_rng(42)
    
    restarts = [
        (1.0, 0.0),
        (0.9, 0.0),
        (1.1, 0.0),
        (1.0, 0.5),   # small random perturbation scale
    ]
    
    for scale, pert in restarts:
        if _time_left() < 5.0:
            break
        p0 = _make_initial_params(scale)
        if pert > 0:
            p0[0:3*N] += rng.normal(0, pert, size=3*N)
        
        try:
            res = minimize(
                _compute_objective,
                p0,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'adaptive': True,
                }
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if _time_left() > 3.0:
        try:
            res = minimize(
                _compute_objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'xatol': 1e-9,
                    'fatol': 1e-10,
                    'adaptive': True,
                }
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Extract result
    inner = []
    for k in range(N):
        x = best_params[3*k]
        y = best_params[3*k+1]
        ang = best_params[3*k+2]
        inner.append([x, y, ang])
    
    ocx = best_params[3*N]
    ocy = best_params[3*N+1]
    oside = best_params[3*N+2]
    oangle = best_params[3*N+3]
    
    return inner, (ocx, ocy), oside, oangle


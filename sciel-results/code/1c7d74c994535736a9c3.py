"""Pack N=11 unit hexagons into smallest outer regular hexagon (side length score)."""
import numpy as np
import math
import time

N = 11
DEADLINE = 120.0
SAFETY = 10.0

def _hex_v(cx, cy, s, a):
    """Vertices of regular hexagon, side s, angle a (radians), flat-top at a=0."""
    th = np.arange(6) * np.pi / 3 + a
    return np.column_stack([cx + s * np.cos(th), cy + s * np.sin(th)])

def _separation(A, B):
    """
    Minimum penetration depth between two convex polygons A, B (vertices (6,2)).
    Uses SAT over all edge normals of both polygons.
    Returns max(0, min_gap) where min_gap is the smallest separating-axis gap.
    If 0, polygons overlap or touch.
    """
    axes = []
    for V in (A, B):
        for i in range(6):
            d = V[(i + 1) % 6] - V[i]
            n = np.array([-d[1], d[0]])
            nn = np.linalg.norm(n)
            if nn > 1e-12:
                axes.append(n / nn)
    min_gap = float('inf')
    for n in axes:
        pA = A @ n
        pB = B @ n
        gap = max(pA.min(), pB.min()) - min(pA.max(), pB.max())
        # gap > 0 means separated along this axis
        if gap > min_gap:
            min_gap = gap
        if min_gap > 1e-9:
            return 0.0
    # All axes have negative or zero gap -> overlap
    return -min_gap  # positive value = penetration depth (negative of min gap)

def _outer_side(pts, theta=0.0):
    """
    Minimum side length of outer hexagon (center at origin, rotation theta)
    containing all points pts (n,2).
    """
    th = np.radians(theta)
    best = 0.0
    for k in range(6):
        a = th + k * np.pi / 3
        n = np.array([math.cos(a), math.sin(a)])
        proj = pts @ n
        d = max(abs(proj.max()), abs(proj.min()))
        best = max(best, d)
    return 2.0 * best / math.sqrt(3)

def _objective(params, theta):
    """
    params: array of shape (3N,), [x0,y0,a0, x1,y1,a1, ...]
    theta: outer hexagon angle (radians), treated as fixed during inner optimization
    Returns: (score, penalty)
    """
    n = len(params) // 3
    A = []
    for i in range(n):
        x, y, a = params[3*i], params[3*i+1], params[3*i+2]
        A.append(_hex_v(x, y, 1.0, a))
    A = np.array(A)  # (n, 6, 2)
    
    # Outer side
    all_pts = A.reshape(n * 6, 2)
    S = _outer_side(all_pts, math.degrees(theta))
    
    # Separation penalty
    pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            pen += max(0.0, _separation(A[i], A[j])) ** 2
    return S, pen

def _nm_step(params, theta, budget_left):
    """One Nelder-Mead run from params."""
    n = len(params) // 3
    best_S = float('inf')
    best_pen = float('inf')
    best_params = params.copy()
    
    # Build initial simplex
    alpha = 0.1
    simplex = [params.copy()]
    for d in range(3 * n):
        p = params.copy()
        p[d] += alpha
        simplex.append(p)
    
    fvals = []
    for p in simplex:
        S, pen = _objective(p, theta)
        fvals.append(S + 1e6 * pen)
    
    max_iter = int(min(500, budget_left * 50))
    for it in range(max_iter):
        if time.time() > DEADLINE - SAFETY:
            break
        # Sort
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if fvals[-1] - fvals[0] < 1e-12:
            break
        
        # Reflection
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + (centroid - simplex[-1])
        fr, pr = _objective(xr, theta)
        fr_val = fr + 1e6 * pr
        
        if fr_val < fvals[0]:
            # Expansion
            xe = centroid + 2.0 * (centroid - simplex[-1])
            fe, pe = _objective(xe, theta)
            fe_val = fe + 1e6 * pe
            if fe_val < fr_val:
                simplex[-1] = xe
                fvals[-1] = fe_val
            else:
                simplex[-1] = xr
                fvals[-1] = fr_val
        elif fr_val < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr_val
        else:
            # Contraction
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc, pc = _objective(xc, theta)
            fc_val = fc + 1e6 * pc
            if fc_val < fvals[-1]:
                simplex[-1] = xc
                fvals[-1] = fc_val
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    fvals[i] = _objective(simplex[i], theta)[0] + 1e6 * _objective(simplex[i], theta)[1]
    
    # Return best from this run
    S, pen = _objective(simplex[0], theta)
    if pen < 1e-4:
        if S < best_S:
            best_S = S
            best_params = simplex[0].copy()
    return best_S, best_pen, best_params

def _lattice_seed():
    """Compact hexagonal lattice packing for 11 unit hexagons."""
    # Flat-top hex grid: neighbors at (1.5, 0), (0.75, sqrt(3)/2), etc.
    # Use a cluster that minimizes bounding hexagon.
    # 1 center + 6 ring + 4 ring2 = 11
    s3 = math.sqrt(3)
    positions = [
        (0.0, 0.0),
        (1.5, 0.0), (0.75, s3/2), (-0.75, s3/2),
        (-1.5, 0.0), (-0.75, -s3/2), (0.75, -s3/2),
        (2.25, s3/2), (0.0, s3), (-2.25, s3/2),
        (0.0, -s3),
    ]
    # Center of mass
    cx = np.mean([p[0] for p in positions])
    cy = np.mean([p[1] for p in positions])
    params = np.zeros(3 * N)
    for i, (x, y) in enumerate(positions):
        params[3*i] = x - cx
        params[3*i+1] = y - cy
        params[3*i+2] = 0.0  # angle
    return params

def _staggered_seed():
    """Two-row staggered arrangement."""
    # Row 0: 6 hexagons, Row 1: 5 hexagons
    s3 = math.sqrt(3)
    params = np.zeros(3 * N)
    x0 = -2.25
    y0 = 0.0
    x1 = -1.5
    y1 = s3 / 2
    for i in range(6):
        params[3*i] = x0 + 1.5 * i
        params[3*i+1] = y0
        params[3*i+2] = 0.0
    for i in range(5):
        idx = 6 + i
        params[3*idx] = x1 + 1.5 * i
        params[3*idx+1] = y1
        params[3*idx+2] = 0.0
    # Center
    cx = np.mean(params[0::3])
    cy = np.mean(params[1::3])
    params[0::3] -= cx
    params[1::3] -= cy
    return params

def construct():
    start = time.time()
    
    best_S = float('inf')
    best_params = None
    best_theta = 0.0
    
    seeds = [_lattice_seed(), _staggered_seed()]
    
    # Try a few outer angles
    theta_candidates = [0.0, math.pi/6, math.pi/3, math.pi/2, 2*math.pi/3, 5*math.pi/6]
    
    for theta in theta_candidates:
        if time.time() > DEADLINE - SAFETY:
            break
        for seed in seeds:
            if time.time() > DEADLINE - SAFETY:
                break
            S, pen, params = _nm_step(seed, theta, DEADLINE - time.time())
            if pen < 1e-4 and S < best_S:
                best_S = S
                best_params = params
                best_theta = theta
    
    # If NM didn't find a valid packing, fall back to lattice
    if best_params is None or best_S == float('inf'):
        # Just return the lattice seed as-is with computed outer side
        params = _lattice_seed()
        n = N
        A = []
        for i in range(n):
            x, y, a = params[3*i], params[3*i+1], params[3*i+2]
            A.append(_hex_v(x, y, 1.0, a))
        all_pts = np.array(A).reshape(n * 6, 2)
        best_S = _outer_side(all_pts, 0.0)
        best_params = params
        best_theta = 0.0
    
    # Build output
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        a = math.degrees(best_params[3*i+2])
        inner.append([x, y, a])
    
    return inner, [0.0, 0.0], best_S, math.degrees(best_theta)

"""Hexagon packing: 12 unit hexagons in smallest regular hexagon."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12
_DEADLINE = time.time() + 145.0  # 5s safety margin before 150s kill

def _hex_vertices(cx, cy, s, theta):
    """Vertices of a regular hexagon centered at (cx,cy), side s, rotation theta (radians)."""
    angles = theta + np.arange(6) * np.pi / 3.0
    return cx + s * np.cos(angles), cy + s * np.sin(angles)

def _hexagon_sdf(px, py, cx, cy, s, theta):
    """Approximate signed distance from point (px,py) to hexagon.
    Positive = outside, negative = inside.
    Uses the fact that a regular hexagon with side s has inradius r = s*sqrt(3)/2
    and circumradius R = s.
    We transform to hexagon-local coords and compute distance to the hex boundary.
    """
    dx = px - cx
    dy = py - cy
    # Rotate into hex frame
    ct = np.cos(-theta)
    st = np.sin(-theta)
    lx = ct * dx - st * dy
    ly = st * dx + ct * dy
    
    # Regular hexagon with side s: vertices at angles 0, 60, 120, 180, 240, 300 deg
    # The hexagon is the intersection of 6 half-planes.
    # For a hex with circumradius s (vertices at distance s from center),
    # the inradius is s * sqrt(3)/2.
    # The 6 edge normals point at angles 30, 90, 150, 210, 270, 330 deg (offset by 30 from vertices)
    # Actually for vertices at angles 0,60,..., the edge normals are at 30,90,...
    
    # Half-plane constraints: for each edge, n_i . p <= d_i
    # Vertex i is at angle i*pi/3, edge i is between vertex i and vertex i+1
    # Edge i normal direction: angle (i+0.5)*pi/3
    # Distance from center to edge (inradius): r = s * cos(pi/6) = s * sqrt(3)/2
    
    r = s * np.sqrt(3.0) / 2.0
    normals = np.array([np.cos((i + 0.5) * np.pi / 3.0) for i in range(6)])
    tangents = np.array([np.sin((i + 0.5) * np.pi / 3.0) for i in range(6)])
    
    # Project local point onto each normal
    projections = lx * normals + ly * tangents
    # Signed distance to each edge plane (positive = outside that half-plane)
    dists = projections - r
    # The SDF of a convex polygon is the max of distances to each edge (when outside)
    # but we need to handle the case where the point is "beyond a vertex"
    # For a regular hexagon, a good approximation:
    # sdf = max(dists) when outside, but for points near vertices we need correction.
    
    # Better: compute exact distance to hexagon boundary
    # For each edge, compute the closest point on the infinite line, check if it's on the segment
    # Then take min distance over all edges (and also check if inside)
    
    # Check if inside: all projections <= r
    inside = np.all(projections <= r)
    
    # Compute distance to each edge segment
    min_dist = np.inf
    for i in range(6):
        # Edge from vertex i to vertex i+1
        a_ang = i * np.pi / 3.0
        b_ang = (i + 1) * np.pi / 3.0
        ax = s * np.cos(a_ang)
        ay = s * np.sin(a_ang)
        bx = s * np.cos(b_ang)
        by = s * np.sin(b_ang)
        
        # Point on line: closest point to (lx, ly) on segment (ax,ay)-(bx,by)
        abx = bx - ax
        aby = by - ay
        apx = lx - ax
        apy = ly - ay
        ab_len2 = abx * abx + aby * aby
        t = (apx * abx + apy * aby) / ab_len2
        t = np.clip(t, 0.0, 1.0)
        cx_e = ax + t * abx
        cy_e = ay + t * aby
        d = np.sqrt((lx - cx_e) ** 2 + (ly - cy_e) ** 2)
        if d < min_dist:
            min_dist = d
    
    if inside:
        return -min_dist
    else:
        return min_dist

def _pairwise_min_dist(params_i, params_j):
    """Compute approximate minimum distance between two hexagons.
    params = (cx, cy, theta) for each.
    Returns minimum distance between boundaries (negative if overlapping).
    """
    cx1, cy1, th1 = params_i
    cx2, cy2, th2 = params_j
    
    # Sample points on each hexagon boundary and find min distance
    # Use 24 points per hexagon for reasonable accuracy
    n_pts = 24
    angles1 = th1 + np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    angles2 = th2 + np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    
    x1 = cx1 + np.cos(angles1)  # side=1, circumradius=1
    y1 = cy1 + np.sin(angles1)
    x2 = cx2 + np.cos(angles2)
    y2 = cy2 + np.sin(angles2)
    
    # For each point on hex1, find min distance to hex2 boundary
    # Also check if point is inside hex2 (SDF)
    min_d = np.inf
    for i in range(n_pts):
        # Distance from (x1[i], y1[i]) to hex2 boundary
        d2 = _hexagon_sdf(x1[i], y1[i], cx2, cy2, 1.0, th2)
        # If outside hex2, distance is d2; if inside, distance is -d2 (penalty)
        if d2 > 0:
            # Outside: actual Euclidean distance to boundary
            # Find closest point on hex2 boundary
            min_d_pt = np.inf
            for j in range(n_pts):
                dd = (x1[i] - x2[j]) ** 2 + (y1[i] - y2[j]) ** 2
                if dd < min_d_pt:
                    min_d_pt = dd
            min_d = min(min_d, np.sqrt(min_d_pt))
        else:
            # Inside hex2: overlapping, penalty
            min_d = min(min_d, d2)  # negative value
    
    # Also check reverse: points on hex2 vs hex1
    for j in range(n_pts):
        d1 = _hexagon_sdf(x2[j], y2[j], cx1, cy1, 1.0, th1)
        if d1 > 0:
            min_d_pt = np.inf
            for i in range(n_pts):
                dd = (x2[j] - x1[i]) ** 2 + (y2[j] - y1[i]) ** 2
                if dd < min_d_pt:
                    min_d_pt = dd
            min_d = min(min_d, np.sqrt(min_d_pt))
        else:
            min_d = min(min_d, d1)
    
    return min_d

def _outer_hex_sdf(px, py, ox, oy, S, otheta):
    """SDF from point to outer hexagon boundary. Positive = outside (violation)."""
    dx = px - ox
    dy = py - oy
    ct = np.cos(-otheta)
    st = np.sin(-otheta)
    lx = ct * dx - st * dy
    ly = st * dx + ct * dy
    
    r = S * np.sqrt(3.0) / 2.0
    normals = np.array([np.cos((i + 0.5) * np.pi / 3.0) for i in range(6)])
    tangents = np.array([np.sin((i + 0.5) * np.pi / 3.0) for i in range(6)])
    
    projections = lx * normals + ly * tangents
    inside = np.all(projections <= r)
    
    min_dist = np.inf
    for i in range(6):
        a_ang = i * np.pi / 3.0
        b_ang = (i + 1) * np.pi / 3.0
        ax = S * np.cos(a_ang)
        ay = S * np.sin(a_ang)
        bx = S * np.cos(b_ang)
        by = S * np.sin(b_ang)
        
        abx = bx - ax
        aby = by - ay
        apx = lx - ax
        apy = ly - ay
        ab_len2 = abx * abx + aby * aby
        t = (apx * abx + apy * aby) / ab_len2
        t = np.clip(t, 0.0, 1.0)
        cx_e = ax + t * abx
        cy_e = ay + t * aby
        d = np.sqrt((lx - cx_e) ** 2 + (ly - cy_e) ** 2)
        if d < min_dist:
            min_dist = d
    
    if inside:
        return -min_dist
    else:
        return min_dist

def _objective(params):
    """Objective: minimize outer side S with penalty for overlaps and containment violations."""
    n = N
    # params layout: [cx1,cy1,th1, cx2,cy2,th2, ..., ox,oy,S,otheta]
    inner_params = params[:3 * n]
    ox, oy, S, otheta = params[3 * n:3 * n + 4]
    
    if S < 2.0:
        return 1e6
    
    penalty = 0.0
    
    # Containment: all inner hex vertices must be inside outer hex
    for i in range(n):
        cx, cy, th = inner_params[3 * i:3 * i + 3]
        angles = th + np.arange(6) * np.pi / 3.0
        vx = cx + np.cos(angles)
        vy = cy + np.sin(angles)
        for k in range(6):
            d = _outer_hex_sdf(vx[k], vy[k], ox, oy, S, otheta)
            if d > 0:
                penalty += d * d * 100.0
    
    # Pairwise non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            pi = inner_params[3 * i:3 * i + 3]
            pj = inner_params[3 * j:3 * j + 3]
            d = _pairwise_min_dist(pi, pj)
            if d < 0:
                penalty += d * d * 50.0
    
    return S + penalty

def construct():
    global _DEADLINE
    start_time = time.time()
    
    # Initial guess: hexagonal lattice arrangement
    # Place hexagons in a pattern that fits reasonably
    # For 12 hexagons, try a 3x4 or hexagonal arrangement
    
    # Hexagonal lattice positions (side-to-side touching, distance = sqrt(3) between centers)
    # Actually for unit hexagons, center-to-center distance for side-touching is 2*sqrt(3)/2 = sqrt(3)
    # Wait: inradius = sqrt(3)/2, so center-to-center for edge-touching = 2 * sqrt(3)/2 = sqrt(3)
    
    # Let's try a compact arrangement:
    # Row-based: 4-4-4 or 3-4-3 or 2-3-4-3-2 etc.
    # For hexagonal packing, offset rows
    
    # Start with a reasonable initial configuration
    # 12 hexagons in a roughly circular/hexagonal cluster
    
    # Use hexagonal coordinates
    # Center at origin, place in hex grid
    hex_positions = []
    # 1 center + 6 ring + 5 more = 12
    # Or: 2-3-4-3-2 = 12 (5 rows)
    
    # Let's use: rows of 2,3,4,3,2
    row_counts = [2, 3, 4, 3, 2]
    dx = np.sqrt(3)  # horizontal spacing for edge-touching
    dy = 1.5  # vertical spacing (3/2 of side for hex grid)
    
    positions = []
    y_offset = 0
    for row_idx, count in enumerate(row_counts):
        x_start = -(count - 1) * dx / 2
        for k in range(count):
            x = x_start + k * dx
            y = y_offset
            positions.append((x, y, 0.0))
        y_offset += dy
    
    # Scale down a bit for initial guess
    scale = 0.95
    positions = [(x * scale, y * scale, th) for x, y, th in positions]
    
    # Initial outer hex: large enough to contain
    ox, oy = 0.0, 0.0
    # Estimate required S
    max_r = max(np.sqrt(x**2 + y**2) + 1.0 for x, y, _ in positions)
    S_init = max_r * 2 / np.sqrt(3) + 0.5
    otheta = 0.0
    
    # Build initial params vector
    x0 = np.zeros(3 * N + 4)
    for i, (px, py, pt) in enumerate(positions):
        x0[3*i] = px
        x0[3*i+1] = py
        x0[3*i+2] = pt
    x0[3*N] = ox
    x0[3*N+1] = oy
    x0[3*N+2] = S_init
    x0[3*N+3] = otheta
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((-5.0, 5.0))   # cx
        bounds.append((-5.0, 5.0))   # cy
        bounds.append((0.0, np.pi/3))  # theta (hex symmetry: 60 deg period)
    bounds.append((-3.0, 3.0))   # ox
    bounds.append((-3.0, 3.0))   # oy
    bounds.append((3.0, 15.0))   # S
    bounds.append((0.0, np.pi/3))  # otheta
    
    best_result = None
    best_score = np.inf
    
    # Multiple restarts with perturbation
    rng = np.random.default_rng(42)
    
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > _DEADLINE - 20:
            break
        
        if restart == 0:
            x_start = x0.copy()
        else:
            x_start = x0.copy()
            # Perturb inner positions
            for i in range(N):
                x_start[3*i] += rng.uniform(-0.5, 0.5)
                x_start[3*i+1] += rng.uniform(-0.5, 0.5)
                x_start[3*i+2] = rng.uniform(0, np.pi/6)
            x_start[3*N+2] = S_init + rng.uniform(0, 1)
        
        try:
            result = minimize(
                _objective,
                x_start,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6, 'maxfun': 5000}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_result = result
        
        except Exception:
            continue
        
        if time.time() > _DEADLINE - 15:
            break
    
    if best_result is None:
        # Fallback: return initial guess
        inner = [[positions[i][0], positions[i][1], np.degrees(positions[i][2])] for i in range(N)]
        return inner, (ox, oy), S_init, 0.0
    
    # Extract best solution
    params = best_result.x
    inner = []
    for i in range(N):
        cx = params[3*i]
        cy = params[3*i+1]
        th_deg = np.degrees(params[3*i+2])
        inner.append([cx, cy, th_deg])
    
    ox = params[3*N]
    oy = params[3*N+1]
    S = params[3*N+2]
    otheta_deg = np.degrees(params[3*N+3])
    
    # Polish: try to reduce S slightly
    if time.time() < _DEADLINE - 10:
        # Try smaller S
        for trial_S in [S - 0.05, S - 0.02, S - 0.01]:
            if trial_S < 3.0:
                break
            params_trial = params.copy()
            params_trial[3*N+2] = trial_S
            try:
                result2 = minimize(
                    _objective,
                    params_trial,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-9, 'gtol': 1e-7, 'maxfun': 2000}
                )
                if result2.fun < best_score:
                    best_score = result2.fun
                    best_result = result2
                    params = result2.x
            except Exception:
                break
        
        # Re-extract
        inner = []
        for i in range(N):
            cx = params[3*i]
            cy = params[3*i+1]
            th_deg = np.degrees(params[3*i+2])
            inner.append([cx, cy, th_deg])
        ox = params[3*N]
        oy = params[3*N+1]
        S = params[3*N+2]
        otheta_deg = np.degrees(params[3*N+3])
    
    return inner, (ox, oy), S, otheta_deg

import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 8.0
EPS = 1e-9

# Unit hexagon vertices (flat-top orientation, angle=0)
# Vertices at angles 0, 60, 120, 180, 240, 300 degrees
HEX_VERTS = np.array([
    [1.0, 0.0],
    [0.5, math.sqrt(3)/2],
    [-0.5, math.sqrt(3)/2],
    [-1.0, 0.0],
    [-0.5, -math.sqrt(3)/2],
    [0.5, -math.sqrt(3)/2],
])

def time_ok():
    return time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN

def rotate_pts(pts, angle):
    c = math.cos(angle)
    s = math.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T

def hexagon_vertices(cx, cy, angle):
    """Return 6 vertices of unit hexagon centered at (cx,cy) with rotation angle."""
    return rotate_pts(HEX_VERTS, angle) + np.array([cx, cy])

def smooth_max(vals, temp=0.05):
    """Numerically stable smooth maximum."""
    v = np.asarray(vals, dtype=float)
    m = np.max(v)
    if m > 1e10:
        return m
    shifted = (v - m) / temp
    shifted = np.clip(shifted, -50, 50)
    logsum = np.log(np.sum(np.exp(shifted)))
    return m + temp * logsum

def minkowski_smooth_dist(v1, v2, temp=0.05):
    """
    Smooth measure of how far apart two convex hexagons (as vertex sets) are.
    Uses support function difference: for each edge-normal direction of hex1,
    project both; the gap in that direction is h1 - h2. If all gaps <= 0 (with margin),
    they overlap. We return a smooth positive value when separated, ~0 when touching.
    Actually we want a constraint g <= 0 where g>0 means overlap.
    Define g = max over directions d of (h1(d) - h2(d) + 0) ... 
    Let's use: for each of 6 edge normals of hex1 (directions pointing outward),
    the support of hex1 in direction d minus support of hex2 in direction d.
    If hex1 is entirely to the positive side, and hex2 too, overlap if
    max_d (h1(d) - h2(d)) < 0 is NOT the right test.
    
    Correct separation test: two convex sets A,B are disjoint iff there exists
    direction d such that max_{a in A} a.d < min_{b in B} b.d, i.e.,
    h_A(d) + h_A(-d) ... simpler: min_{a in A, b in B} |a-b| > 0.
    Use smooth min distance between the two vertex sets as proxy? Not exact but works
    well for hexagons when close. We'll use smooth minimum Euclidean distance between
    all vertex pairs as the separation measure; enforce > 0.
    """
    diffs = v1[:, None, :] - v2[None, :, :]
    d2 = np.sum(diffs**2, axis=2)  # (6,6)
    d = np.sqrt(d2 + 1e-12)
    # smooth minimum distance
    m = np.min(d)
    shifted = (d - m) / temp
    shifted = np.clip(shifted, -50, 50)
    logsum = np.log(np.sum(np.exp(shifted)))
    return m - temp * logsum

def pack_objective(params, temp=0.05):
    """
    params: [cx, cy, S, beta, x1,y1,a1, ..., x11,y11,a11]
    Minimize S + penalty for overlaps + penalty for containment.
    """
    if len(params) != 4 + 3*N:
        return 1e10
    cx, cy, S, beta = params[0], params[1], params[2], params[3]
    if S < 0.5:
        return 1e10
    S = max(S, 0.5)
    inner = []
    for i in range(N):
        x = params[4 + 3*i]
        y = params[5 + 3*i]
        a = params[6 + 3*i]
        inner.append((x, y, a))
    
    # Containment: each inner hexagon must be inside outer hexagon.
    # Outer hexagon: center (cx,cy), side S, rotation beta.
    # Edge normals of outer hexagon (pointing inward):
    # For a regular hexagon, the 6 inward normals are at angles beta + 30 + k*60.
    # A point p is inside iff for all k: (p - c) . n_k <= S * APOTHEM where n_k are unit inward normals.
    # Actually support: max over vertices of inner of (v - c).n_k <= S*APOTHEM for all k.
    D = S * APOTHEM
    normals = []
    for k in range(6):
        ang = beta + math.pi/6 + k * math.pi/3
        # inward normal direction
        normals.append((math.cos(ang), math.sin(ang)))
    normals = np.array(normals)
    
    contain_penalty = 0.0
    for (x, y, a) in inner:
        verts = hexagon_vertices(x, y, a)
        # project each vertex onto each normal
        projs = (verts - np.array([cx, cy])) @ normals.T  # (6,6)
        # need max over vertices of proj <= D for each normal
        maxprojs = np.max(projs, axis=0)  # (6,)
        violations = np.maximum(0, maxprojs - D)
        contain_penalty += np.sum(violations**2)
    
    # Overlap penalty between all pairs
    overlap_penalty = 0.0
    verts_list = [hexagon_vertices(x, y, a) for (x, y, a) in inner]
    for i in range(N):
        for j in range(i+1, N):
            d = minkowski_smooth_dist(verts_list[i], verts_list[j], temp=temp)
            # want d >= 0 (separated). Penalize if d < 0.
            viol = max(0, -d)
            overlap_penalty += viol**2
    
    # Regularization: keep S reasonable
    obj = S + 1000.0 * contain_penalty + 1000.0 * overlap_penalty
    return obj

def pack_objective_grad(params, temp=0.05):
    return pack_objective(params, temp)

def make_initial(cx=0.0, cy=0.0, S=2.5, beta=0.0, spread=0.3):
    """Create initial params: S, center, and N inner hexagons in a loose pattern."""
    params = [cx, cy, S, beta]
    # Place N hexagons in a rough grid/hex pattern
    positions = []
    # Use a hexagonal lattice arrangement
    # For N=11, try 3-4-3-1 or similar
    rows = [3, 4, 3, 1]
    idx = 0
    row_spacing = 1.5 * spread
    col_spacing = math.sqrt(3) * spread
    for ri, count in enumerate(rows):
        y = cy + (ri - 1.5) * row_spacing
        for ci in range(count):
            x = cx + (ci - (count-1)/2.0) * col_spacing
            if idx < N:
                positions.append((x, y))
                idx += 1
    # Fill remaining if any
    while len(positions) < N:
        positions.append((cx + len(positions)*0.1, cy + len(positions)*0.1))
    for (x, y) in positions[:N]:
        params.extend([x, y, 0.0])
    return np.array(params)

def construct():
    if not time_ok():
        # Fallback: trivial valid-ish answer
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    best_S = float('inf')
    best_params = None
    
    # Multiple random restarts
    rng = np.random.default_rng(42)
    
    # Try several initial configurations
    initial_configs = []
    
    # Config 1: hexagonal cluster
    initial_configs.append(make_initial(cx=0, cy=0, S=2.8, beta=0.0, spread=0.45))
    # Config 2: linear-ish
    initial_configs.append(make_initial(cx=0, cy=0, S=3.5, beta=0.0, spread=0.5))
    # Config 3: random
    for _ in range(4):
        cx = rng.uniform(-1, 1)
        cy = rng.uniform(-1, 1)
        S = rng.uniform(2.5, 4.0)
        beta = rng.uniform(0, math.pi/6)
        params = [cx, cy, S, beta]
        for i in range(N):
            x = cx + rng.uniform(-1.5, 1.5)
            y = cy + rng.uniform(-1.5, 1.5)
            a = rng.uniform(0, math.pi/3)
            params.extend([x, y, a])
        initial_configs.append(np.array(params))
    
    temps = [0.1, 0.05, 0.02, 0.01]
    
    for init in initial_configs:
        if not time_ok():
            break
        current = init.copy()
        for temp in temps:
            if not time_ok():
                break
            try:
                res = minimize(pack_objective, current, args=(temp,), method='L-BFGS-B',
                              options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
                if res.fun < 1e8:
                    current = res.x
            except Exception:
                try:
                    res = minimize(pack_objective, current, args=(temp,), method='Nelder-Mead',
                                  options={'maxiter': 1000, 'xatol': 1e-9, 'fatol': 1e-9})
                    if res.fun < 1e8:
                        current = res.x
                except Exception:
                    pass
        
        # Evaluate final with small temp
        if time_ok():
            try:
                res = minimize(pack_objective, current, args=(0.01,), method='L-BFGS-B',
                              options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
                if res.fun < 1e8:
                    current = res.x
            except Exception:
                pass
        
        # Extract S and check validity
        S_val = max(0.5, current[2])
        # Verify containment and non-overlap strictly
        cx, cy = current[0], current[1]
        beta = current[3]
        inner = []
        valid = True
        verts_list = []
        for i in range(N):
            x = current[4 + 3*i]
            y = current[5 + 3*i]
            a = current[6 + 3*i]
            inner.append((x, y, a))
            verts_list.append(hexagon_vertices(x, y, a))
        
        # Check containment
        D = S_val * APOTHEM
        normals = np.array([
            (math.cos(beta + math.pi/6 + k*math.pi/3),
             math.sin(beta + math.pi/6 + k*math.pi/3))
            for k in range(6)
        ])
        for verts in verts_list:
            projs = (verts - np.array([cx, cy])) @ normals.T
            maxprojs = np.max(projs, axis=0)
            if np.any(maxprojs > D + 1e-7):
                valid = False
                break
        
        # Check non-overlap: min distance between all vertex pairs > 0
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    diffs = verts_list[i][:, None, :] - verts_list[j][None, :, :]
                    d2 = np.sum(diffs**2, axis=2)
                    if np.min(d2) < 1e-10:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid and S_val < best_S:
            best_S = S_val
            best_params = current
    
    if best_params is None or not time_ok():
        # Fallback: use a known-good lattice packing
        # Simple hexagonal packing of 11 unit hexagons
        # Use axial coordinates for a compact shape
        axial = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (-2, 0), (0, 2), (0, -2)
        ]
        pts = []
        for q, r in axial:
            x = math.sqrt(3) * (q + r/2.0)
            y = 1.5 * r
            pts.append((x, y))
        pts = np.array(pts)
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        # Compute required S
        max_proj = 0
        for k in range(6):
            ang = math.pi/6 + k*math.pi/3
            nx, ny = math.cos(ang), math.sin(ang)
            projs = pts[:, 0]*nx + pts[:, 1]*ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        S = (max_proj + APOTHEM) / APOTHEM
        inner = [[float(x), float(y), 0.0] for x, y in pts]
        return inner, [float(cx), float(cy)], float(S), 0.0
    
    cx, cy = float(best_params[0]), float(best_params[1])
    beta = float(best_params[3])
    S_val = float(max(0.5, best_params[2]))
    inner = []
    for i in range(N):
        x = float(best_params[4 + 3*i])
        y = float(best_params[5 + 3*i])
        a = float(best_params[6 + 3*i])
        inner.append([x, y, math.degrees(a)])
    
    return inner, [cx, cy], S_val, math.degrees(beta)

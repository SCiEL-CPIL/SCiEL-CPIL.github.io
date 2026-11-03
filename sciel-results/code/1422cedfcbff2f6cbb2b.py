import numpy as np
import time
import math
import random

N = 12
START_TIME = time.time()
DEADLINE = 145.0  # Safety margin before 150s

# Precompute unit hexagon vertices (side=1, angle=0)
_UNIT_HEX_VERTS = np.array([
    [math.cos(k * math.pi / 3), math.sin(k * math.pi / 3)] for k in range(6)
])

def _hex_verts_fast(cx, cy, angle_rad):
    """Fast: return 6 vertices for hexagon at (cx,cy) with rotation angle_rad, side=1."""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return _UNIT_HEX_VERTS @ R.T + np.array([cx, cy])

def _get_all_vertices(params):
    """Vectorized: compute all 72 vertices."""
    n = N
    inner = params[:n * 3]
    all_v = np.zeros((n * 6, 2))
    for i in range(n):
        x, y, a = inner[i*3], inner[i*3+1], inner[i*3+2]
        c = math.cos(a)
        s = math.sin(a)
        R = np.array([[c, -s], [s, c]])
        all_v[i*6:(i+1)*6] = _UNIT_HEX_VERTS @ R.T + np.array([x, y])
    return all_v

def _outer_side_fast(params):
    """
    Fast vectorized outer hexagon side length.
    params: [x1,y1,a1, ..., x12,y12,a12, cx, cy, outer_angle]
    Returns the minimum side length of outer hexagon (center cx,cy, rotation outer_angle)
    containing all inner vertices.
    """
    n = N
    inner = params[:n * 3]
    cx, cy, outer_ang = params[n*3], params[n*3+1], params[n*3+2]
    
    # All 72 vertices
    all_v = np.zeros((n * 6, 2))
    for i in range(n):
        x, y, a = inner[i*3], inner[i*3+1], inner[i*3+2]
        c = math.cos(a)
        s = math.sin(a)
        R = np.array([[c, -s], [s, c]])
        all_v[i*6:(i+1)*6] = _UNIT_HEX_VERTS @ R.T + np.array([x, y])
    
    # Outer hexagon normals: 6 directions
    theta = outer_ang
    normals = np.array([[math.cos(theta + math.pi/6 + k*math.pi/3),
                         math.sin(theta + math.pi/6 + k*math.pi/3)] for k in range(6)])
    
    rel = all_v - np.array([cx, cy])
    proj = rel @ normals.T  # (72, 6)
    max_proj = proj.max(axis=0)  # (6,)
    # Side length = max over normals of (max_proj / (sqrt(3)/2))
    # Actually for a regular hexagon with side s, the distance from center to each edge
    # is s * sqrt(3)/2. So s = max_proj / (sqrt(3)/2)
    return max_proj.max() / (math.sqrt(3) / 2)

def _polygon_distance(v1, v2):
    """
    Approximate minimum distance between two convex polygons (each 6 vertices).
    Returns negative if overlapping (approx), positive if separated.
    Uses support function approach: for each edge of poly1, compute distance to poly2,
    and vice versa. Returns the minimum signed distance.
    """
    min_dist = float('inf')
    n1, n2 = len(v1), len(v2)
    
    for i in range(n1):
        p1 = v1[i]
        p2 = v1[(i+1) % n1]
        edge = p2 - p1
        edge_len = np.linalg.norm(edge)
        if edge_len < 1e-12:
            continue
        n_edge = np.array([-edge[1], edge[0]]) / edge_len  # outward normal
        
        # For each vertex of poly2, compute signed distance to this edge
        # If any vertex of poly2 is on the wrong side (inside poly1), polygons overlap
        for j in range(n2):
            d = np.dot(v2[j] - p1, n_edge)
            if d < min_dist:
                min_dist = d
    
    for i in range(n2):
        p1 = v2[i]
        p2 = v2[(i+1) % n2]
        edge = p2 - p1
        edge_len = np.linalg.norm(edge)
        if edge_len < 1e-12:
            continue
        n_edge = np.array([-edge[1], edge[0]]) / edge_len
        for j in range(n1):
            d = np.dot(v1[j] - p1, n_edge)
            if d < min_dist:
                min_dist = d
    
    return min_dist

def _overlap_penalty(params, beta=15.0):
    """
    True polygon-polygon overlap penalty using minimum separation distance.
    Returns sum of exp(beta * max(0, -dist)) for all pairs.
    """
    n = N
    inner = params[:n * 3]
    
    # Precompute all vertices
    verts = []
    for i in range(n):
        x, y, a = inner[i*3], inner[i*3+1], inner[i*3+2]
        c = math.cos(a)
        s = math.sin(a)
        R = np.array([[c, -s], [s, c]])
        verts.append(_UNIT_HEX_VERTS @ R.T + np.array([x, y]))
    
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            # Quick reject: center distance
            dx = inner[j*3] - inner[i*3]
            dy = inner[j*3+1] - inner[i*3+1]
            cd = math.sqrt(dx*dx + dy*dy)
            if cd > 4.0:  # too far apart
                continue
            dist = _polygon_distance(verts[i], verts[j])
            if dist < 0:
                penalty += math.exp(beta * (-dist))
            elif dist < 0.01:
                penalty += 0.5 * math.exp(-beta * dist)
    
    return penalty

def objective(params):
    """
    params: [x1,y1,a1, ..., x12,y12,a12, cx, cy, outer_angle]
    Returns outer hexagon side length.
    """
    return _outer_side_fast(params)

def smooth_objective(params, beta=8.0):
    """
    Smooth surrogate: log-sum-exp on projections + overlap penalty.
    """
    n = N
    inner = params[:n * 3]
    cx, cy, outer_ang = params[n*3], params[n*3+1], params[n*3+2]
    
    # All vertices
    all_v = np.zeros((n * 6, 2))
    for i in range(n):
        x, y, a = inner[i*3], inner[i*3+1], inner[i*3+2]
        c = math.cos(a)
        s = math.sin(a)
        R = np.array([[c, -s], [s, c]])
        all_v[i*6:(i+1)*6] = _UNIT_HEX_VERTS @ R.T + np.array([x, y])
    
    theta = outer_ang
    normals = np.array([[math.cos(theta + math.pi/6 + k*math.pi/3),
                         math.sin(theta + math.pi/6 + k*math.pi/3)] for k in range(6)])
    
    rel = all_v - np.array([cx, cy])
    proj = rel @ normals.T  # (72, 6)
    
    # Smooth max via log-sum-exp
    shifted = proj * beta
    max_shifted = shifted.max(axis=0)
    lse = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted), axis=0))
    smooth_max = lse / beta
    
    side = smooth_max.max() / (math.sqrt(3) / 2)
    
    # Overlap penalty
    op = _overlap_penalty(params, beta=12.0)
    
    return side + 0.5 * op

def _initial_grid():
    """Hexagonal close-packed grid: 3 rows of 4, staggered."""
    params = []
    dx = math.sqrt(3)
    dy = 1.5
    positions = []
    for row in range(3):
        for col in range(4):
            x = col * dx + (row % 2) * dx / 2
            y = row * dy
            positions.append((x, y))
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    for x, y in positions:
        params.extend([x - cx, y - cy, 0.0])
    params.extend([0.0, 0.0, 0.0])
    return np.array(params)

def _initial_circular():
    """12 hexagons on a circle of radius 3."""
    params = []
    for i in range(N):
        ang = 2 * math.pi * i / N
        params.extend([3.0 * math.cos(ang), 3.0 * math.sin(ang), 0.0])
    params.extend([0.0, 0.0, 0.0])
    return np.array(params)

def _initial_rings():
    """6 inner ring + 6 outer ring."""
    params = []
    for i in range(6):
        ang = 2 * math.pi * i / 6
        params.extend([math.sqrt(3) * math.cos(ang), math.sqrt(3) * math.sin(ang), 0.0])
    for i in range(6):
        ang = 2 * math.pi * i / 6 + math.pi / 6
        params.extend([2 * math.sqrt(3) * math.cos(ang), 2 * math.sqrt(3) * math.sin(ang), 0.0])
    params.extend([0.0, 0.0, 0.0])
    return np.array(params)

def _initial_flower():
    """1 center + 6 ring + 5 outer."""
    params = [0.0, 0.0, 0.0]
    for i in range(6):
        ang = 2 * math.pi * i / 6
        params.extend([math.sqrt(3) * math.cos(ang), math.sqrt(3) * math.sin(ang), 0.0])
    for i in range(5):
        ang = 2 * math.pi * i / 5 + math.pi / 5
        params.extend([2 * math.sqrt(3) * math.cos(ang), 2 * math.sqrt(3) * math.sin(ang), 0.0])
    params.extend([0.0, 0.0, 0.0])
    return np.array(params)

def _nelder_mead_fast(func, x0, max_iter=200, tol=1e-8):
    """
    Fast Nelder-Mead with time check. Returns (x_best, f_best).
    """
    n = len(x0)
    # Build initial simplex
    simplex = [x0.copy()]
    for i in range(n):
        xi = x0.copy()
        step = 0.1 if abs(x0[i]) < 1.0 else 0.1 * abs(x0[i])
        xi[i] += step
        simplex.append(xi)
    
    fvals = [func(s) for s in simplex]
    best_idx = int(np.argmin(fvals))
    x_best = simplex[best_idx].copy()
    f_best = fvals[best_idx]
    
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    for it in range(max_iter):
        if time.time() > DEADLINE:
            break
        # Sort simplex
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        if fvals[0] < f_best:
            f_best = fvals[0]
            x_best = simplex[0].copy()
        
        # Check convergence
        if abs(fvals[-1] - fvals[0]) < tol:
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        x_r = centroid + alpha * (centroid - simplex[-1])
        f_r = func(x_r)
        
        if f_r < fvals[0]:
            # Expansion
            x_e = centroid + gamma * (x_r - centroid)
            f_e = func(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                fvals[-1] = f_e
            else:
                simplex[-1] = x_r
                fvals[-1] = f_r
        elif f_r < fvals[-2]:
            simplex[-1] = x_r
            fvals[-1] = f_r
        else:
            # Contraction
            x_c = centroid + rho * (simplex[-1] - centroid)
            f_c = func(x_c)
            if f_c < fvals[-1]:
                simplex[-1] = x_c
                fvals[-1] = f_c
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    fvals[i] = func(simplex[i])
    
    return x_best, f_best

def construct():
    from scipy.optimize import minimize
    
    best_params = None
    best_score = float('inf')
    
    configs = [
        _initial_grid(),
        _initial_circular(),
        _initial_rings(),
        _initial_flower(),
    ]
    
    # Add random perturbations
    rng = np.random.default_rng(42)
    for base in configs[:3]:
        for _ in range(4):
            p = base + rng.standard_normal(len(base)) * 0.8
            configs.append(p)
    
    # Stage 1: Nelder-Mead on smooth objective (fast, many iterations)
    for config in configs:
        if time.time() > DEADLINE - 20:
            break
        try:
            x_nm, f_nm = _nelder_mead_fast(smooth_objective, config, max_iter=300)
            # Evaluate with true objective
            f_true = objective(x_nm)
            if f_true < best_score:
                best_score = f_true
                best_params = x_nm.copy()
        except Exception:
            pass
    
    # Stage 2: L-BFGS-B polish on smooth objective
    if best_params is not None and time.time() < DEADLINE - 10:
        try:
            res = minimize(
                smooth_objective, best_params,
                method='L-BFGS-B',
                options={'maxiter': 400, 'maxfun': 2000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            f_true = objective(res.x)
            if f_true < best_score:
                best_score = f_true
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Stage 3: Simulated Annealing on true objective (vectorized-friendly)
    if best_params is not None and time.time() < DEADLINE - 5:
        x = best_params.copy()
        f_current = best_score
        T = 0.5
        T_min = 0.001
        cooling = 0.995
        n_params = len(x)
        
        sa_start = time.time()
        sa_budget = DEADLINE - 5 - sa_start
        it_count = 0
        
        while time.time() < DEADLINE - 5:
            it_count += 1
            # Propose move: perturb 1-3 random parameters
            n_perturb = rng.integers(1, 4)
            idx = rng.integers(0, n_params, size=n_perturb)
            x_new = x.copy()
            scale = T * 2.0
            for i in idx:
                x_new[i] += rng.standard_normal() * scale
            
            f_new = objective(x_new)
            
            if f_new < f_current or rng.random() < math.exp(-(f_new - f_current) / T):
                x = x_new
                f_current = f_new
                if f_new < best_score:
                    best_score = f_new
                    best_params = x.copy()
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        # Final NM polish from SA result
        if time.time() < DEADLINE - 2:
            try:
                x_nm2, f_nm2 = _nelder_mead_fast(objective, x, max_iter=200)
                f_true2 = objective(x_nm2)
                if f_true2 < best_score:
                    best_score = f_true2
                    best_params = x_nm2.copy()
            except Exception:
                pass
    
    # Stage 4: Final L-BFGS-B on true objective
    if best_params is not None and time.time() < DEADLINE - 1:
        try:
            res2 = minimize(
                objective, best_params,
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 1500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        return [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)], (0.0, 0.0), 40.0, 0.0
    
    inner = []
    for i in range(N):
        x = best_params[i * 3]
        y = best_params[i * 3 + 1]
        a = math.degrees(best_params[i * 3 + 2])
        inner.append([x, y, a])
    
    cx = best_params[N * 3]
    cy = best_params[N * 3 + 1]
    outer_angle = math.degrees(best_params[N * 3 + 2])
    
    side = best_score
    if side < 1.0:
        side = 1.0
    
    return inner, (cx, cy), side, outer_angle

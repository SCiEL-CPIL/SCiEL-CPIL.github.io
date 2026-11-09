import numpy as np
import time

def _normalize(pts):
    """Subtract centroid, divide by mean distance to centroid."""
    c = pts.mean(axis=0)
    p = pts - c
    d = np.linalg.norm(p, axis=1)
    m = d.mean()
    if m <= 0.01:
        return p, m
    return p / m, m

def _is_strictly_convex(pts):
    """Check strictly convex: no repeated points, all turns same sign nonzero, total turn ±2π."""
    n = len(pts)
    for i in range(n):
        if np.linalg.norm(pts[i] - pts[(i+1) % n]) < 1e-10:
            return False
    turns = []
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i+1) % n]
        p2 = pts[(i+2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        if abs(cross) < 1e-12:
            return False
        turns.append(cross)
    if not (all(t > 0 for t in turns) or all(t < 0 for t in turns)):
        return False
    # Total turning
    angles = []
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i+1) % n]
        p2 = pts[(i+2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        a1 = np.arctan2(v1[1], v1[0])
        a2 = np.arctan2(v2[1], v2[0])
        da = a2 - a1
        while da > np.pi: da -= 2*np.pi
        while da < -np.pi: da += 2*np.pi
        angles.append(da)
    total = sum(angles)
    if abs(abs(total) - 2*np.pi) > 1e-6:
        return False
    return True

def _vertex_score(pts, i):
    """Compute score for vertex i."""
    n = len(pts)
    v = pts[i]
    dists = []
    for j in range(n):
        if j != i:
            dists.append(np.linalg.norm(v - pts[j]))
    dists = np.sort(dists)
    # 19 distances
    if any(d < 0.05 for d in dists):
        return 0.0
    # Find k: first index minimizing d_{k+1} - d_k
    gaps = dists[1:] - dists[:-1]
    k = np.argmin(gaps)
    D = (dists[k] + dists[k+1]) / 2.0
    if D < 0.1:
        return 0.0
    if k == 0:
        idx = [0, 1, 2, 3]
    elif k == 17:
        idx = [15, 16, 17, 18]
    else:
        idx = [k-1, k, k+1, k+2]
    four = dists[idx]
    raw = np.mean([max(d/D, D/d) for d in four])
    return min(1.0, 1.0 / raw) if raw > 0 else 0.0

def _polygon_score(pts):
    """Compute min vertex score."""
    n = len(pts)
    scores = [_vertex_score(pts, i) for i in range(n)]
    return min(scores)

def _build_polygon(params):
    """
    Build a 20-gon from parameters.
    params: array of shape (5,) representing angular offsets for a period-5 symmetric pattern.
    We use: angle_i = 2πi/20 + offset_{i mod 5}
    And radius_i = r0 + r1 * cos(2π*(i mod 5)/5)
    Actually let's use a simpler parameterization:
    Each of the 5 "groups" of 4 consecutive vertices gets an angular offset and radial modulation.
    
    Let me use: for vertex i, 
      base_angle = 2πi/20
      offset = params[i % 5]  (small angular perturbation)
      angle = base_angle + offset
      radius = 1.0 + params2[i % 5]  (radial perturbation)
    """
    n = 20
    # params shape (10,): 5 angular offsets, 5 radial offsets
    ang_off = params[:5]
    rad_off = params[5:]
    angles = 2.0 * np.pi * np.arange(n) / n
    angles = angles + np.tile(ang_off, 4)
    radii = 1.0 + np.tile(rad_off, 4)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.stack([x, y], axis=1)

def _score_params(params):
    """Score a parameter vector."""
    pts = _build_polygon(params)
    pts_norm, m = _normalize(pts)
    if m <= 0.01:
        return 0.0
    if not _is_strictly_convex(pts_norm):
        return 0.0
    return _polygon_score(pts_norm)

def _nelder_mead(func, x0, max_iter=200, tol=1e-10):
    """Simple Nelder-Mead implementation."""
    n = len(x0)
    simplex = [x0.astype(float)]
    f0 = func(x0)
    simplex.append(f0)  # store f values separately
    
    # Build simplex
    simplex_pts = [x0.astype(float)]
    simplex_f = [f0]
    for i in range(n):
        xi = x0.copy()
        step = 0.05 if abs(x0[i]) > 1e-10 else 0.05
        xi[i] += step
        simplex_pts.append(xi)
        simplex_f.append(func(xi))
    
    for it in range(max_iter):
        # Sort
        order = np.argsort(simplex_f)
        simplex_pts = [simplex_pts[i] for i in order]
        simplex_f = [simplex_f[i] for i in order]
        
        if abs(simplex_f[-1] - simplex_f[0]) < tol:
            break
        
        centroid = np.mean(simplex_pts[:-1], axis=0)
        worst = simplex_pts[-1]
        f_worst = simplex_f[-1]
        
        # Reflect
        alpha = 1.0
        x_r = centroid + alpha * (centroid - worst)
        f_r = func(x_r)
        
        if f_r < simplex_f[0]:
            # Expand
            gamma = 2.0
            x_e = centroid + gamma * (centroid - worst)
            f_e = func(x_e)
            if f_e < f_r:
                simplex_pts[-1] = x_e
                simplex_f[-1] = f_e
            else:
                simplex_pts[-1] = x_r
                simplex_f[-1] = f_r
        elif f_r < simplex_f[-2]:
            simplex_pts[-1] = x_r
            simplex_f[-1] = f_r
        else:
            # Contract
            rho = 0.5
            x_c = centroid + rho * (worst - centroid)
            f_c = func(x_c)
            if f_c < f_worst:
                simplex_pts[-1] = x_c
                simplex_f[-1] = f_c
            else:
                # Shrink
                for i in range(1, n+1):
                    simplex_pts[i] = simplex_pts[0] + 0.5 * (simplex_pts[i] - simplex_pts[0])
                    simplex_f[i] = func(simplex_pts[i])
    
    return simplex_pts[0], simplex_f[0]

def _coordinate_descent(func, x0, max_iter=100, tol=1e-10):
    """Simple coordinate descent."""
    x = x0.copy()
    f = func(x)
    n = len(x)
    for it in range(max_iter):
        improved = False
        for i in range(n):
            best_x = x.copy()
            best_f = f
            for delta in [0.01, -0.01, 0.005, -0.005, 0.001, -0.001, 0.02, -0.02]:
                xi = x.copy()
                xi[i] += delta
                fi = func(xi)
                if fi < best_f:
                    best_f = fi
                    best_x = xi
            if best_x is not x:
                x = best_x
                f = best_f
                improved = True
        if not improved:
            break
    return x, f

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex 20-gon optimized for the vertex equidistance score.
    Uses a period-5 symmetric parameterization with coordinate descent + Nelder-Mead.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    
    # Initial parameters: small perturbations
    np.random.seed(42)
    best_params = np.zeros(10)
    best_score = _score_params(best_params)
    
    # Try several random initializations
    for trial in range(20):
        if time.time() > deadline - 60.0:
            break
        params = np.random.randn(10) * 0.05
        # Coordinate descent
        params, f = _coordinate_descent(_score_params, params, max_iter=50)
        if f > best_score:
            best_score = f
            best_params = params.copy()
    
    # Nelder-Mead fine-tuning from best
    if time.time() < deadline - 30.0:
        best_params, best_score = _nelder_mead(_score_params, best_params, max_iter=300)
    
    # Build final polygon
    pts = _build_polygon(best_params)
    pts_norm, m = _normalize(pts)
    
    # Verify
    if m <= 0.01 or not _is_strictly_convex(pts_norm):
        # Fallback to regular polygon
        n = 20
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        x = np.cos(angles)
        y = np.sin(angles)
        return np.stack([x, y], axis=1)
    
    return pts_norm

# EVOLVE-BLOCK-END

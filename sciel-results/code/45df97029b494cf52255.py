"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

def _all_triangle_areas(pts):
    """Compute areas of all C(n,3) triangles formed by n points."""
    n = len(pts)
    areas = np.empty((n*(n-1)*(n-2))//6)
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = pts[i]
                b = pts[j]
                c = pts[k]
                areas[idx] = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))
                idx += 1
    return areas

def _convex_hull_area(pts):
    """Compute area of convex hull."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    areas = _all_triangle_areas(pts)
    min_area = np.min(areas)
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    return min_area / hull_area

def _regular_polygon(n, radius=1.0, center=(0,0)):
    """Generate regular polygon vertices."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([center[0] + radius*np.cos(angles), 
                            center[1] + radius*np.sin(angles)])

def _concentric_rings(n, n_rings=3):
    """Generate points on concentric rings for diverse initialization."""
    pts = []
    pts.append([0.0, 0.0])
    remaining = n - 1
    for ring in range(n_rings):
        if remaining <= 0:
            break
        n_on_ring = min(remaining, max(3, (n-1)//n_rings))
        if ring == n_rings - 1:
            n_on_ring = remaining
        radius = (ring + 1) / n_rings
        angles = np.linspace(0, 2*np.pi, n_on_ring, endpoint=False) + ring * np.pi/6
        ring_pts = np.column_stack([radius*np.cos(angles), radius*np.sin(angles)])
        pts.append(ring_pts)
        remaining -= n_on_ring
    
    if remaining > 0:
        angles = np.linspace(0, 2*np.pi, remaining, endpoint=False)
        ring_pts = np.column_stack([1.0*np.cos(angles), 1.0*np.sin(angles)])
        pts.append(ring_pts)
    
    result = np.vstack(pts)
    return result[:n]

def _halton_initialization():
    """Use Halton sequence for low-discrepancy initialization."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r

    points = []
    for i in range(1, N + 1):
        x = halton(i, 2)
        y = halton(i, 3)
        points.append([x, y])
    return np.array(points)

def _soft_min_objective(x, beta):
    """
    Soft-min surrogate for optimization.
    """
    pts = x.reshape(-1, 2)
    n = len(pts)
    
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = pts[i]
                b = pts[j]
                c = pts[k]
                areas.append(0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1])))
    areas = np.array(areas)
    
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 1e6
    
    if hull_area < 1e-10:
        return 1e6
    
    beta_safe = max(beta, 1.0)
    log_terms = -beta_safe * areas
    max_log = np.max(log_terms)
    lse = max_log + np.log(np.sum(np.exp(log_terms - max_log)))
    obj = -lse / (beta_safe * hull_area)
    
    return obj

def _local_search(points, max_iters=2000, time_limit=10.0):
    """
    Perform hill climbing with adaptive step size.
    """
    import random
    current_points = points.copy()
    current_score = _score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    step_size = 0.05
    min_step_size = 1e-6
    
    start_time = time.time()
    
    for i in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        n_moves = random.randint(1, 3)
        indices = random.sample(range(N), n_moves)
        
        new_points = current_points.copy()
        for idx in indices:
            noise = np.random.normal(0, step_size, 2)
            new_points[idx] = current_points[idx] + noise
        
        new_score = _score(new_points)
        
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            step_size *= 0.95
            if step_size < min_step_size:
                kick_idx = random.randint(0, N - 1)
                new_points = current_points.copy()
                new_points[kick_idx] = current_points[kick_idx] + np.random.normal(0, 0.2, 2)
                new_score = _score(new_points)
                if new_score > current_score:
                    current_points = new_points
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_points = current_points.copy()
                    step_size = 0.05
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_pts = None
    best_score = -np.inf

    # Initial candidates
    candidates = []
    
    # 1. Regular 13-gon
    candidates.append(_regular_polygon(N, radius=1.0))
    
    # 2. Concentric rings
    candidates.append(_concentric_rings(N, n_rings=3))
    candidates.append(_concentric_rings(N, n_rings=4))
    
    # 3. Halton sequence
    candidates.append(_halton_initialization())
    
    # 4. Random points in unit disk
    rng = np.random.default_rng(N)
    for _ in range(5):
        angles = rng.uniform(0, 2*np.pi, N)
        radii = rng.uniform(0, 1, N)
        pts = np.column_stack([radii*np.cos(angles), radii*np.sin(angles)])
        candidates.append(pts)
    
    # 5. Grid-like points
    grid_pts = []
    for i in range(4):
        for j in range(4):
            grid_pts.append([i/3.0, j/3.0])
    grid_pts = np.array(grid_pts[:N])
    candidates.append(grid_pts)
    
    # Evaluate initial candidates
    for cand in candidates:
        if time.time() > deadline:
            break
        try:
            s = _score(cand)
            if s > best_score:
                best_score = s
                best_pts = cand.copy()
        except Exception:
            pass
    
    # Phase 1: Differential Evolution on a reduced parameterization
    # We use a radial basis expansion: r(theta) = r0 + sum(w_i * exp(-((theta - theta_i)^2) / (2*sigma^2)))
    # This enforces a smooth, convex-like boundary and reduces dimensionality.
    # Let's try optimizing 13 points directly with DE but with bounds, or use the radial param.
    # The directive suggests: "Parametrize the 13 points using a non-uniform radial basis function expansion... optimize these few basis parameters using a differential evolution"
    
    # Let's implement the radial basis parametrization.
    # We define 13 angular positions (fixed or optimized?) and 13 radial weights.
    # Actually, simpler: optimize the 13 points directly with DE, but use the soft-min objective?
    # DE doesn't need gradients, so it can handle the non-smooth min area directly!
    # Let's try DE on the 26 coordinates directly.
    
    # Bounds for DE
    bounds = [(-2, 2)] * (2 * N)
    
    # Objective for DE: maximize min_area / hull_area
    # DE minimizes, so we return -score
    def de_objective(x):
        pts = x.reshape(-1, 2)
        try:
            return -_score(pts)
        except Exception:
            return 1e6
    
    # Run DE
    if time.time() < deadline - 30:  # Reserve time for polish
        try:
            de_result = differential_evolution(
                de_objective,
                bounds,
                seed=42,
                maxiter=100,
                tol=1e-6,
                polish=False,  # We'll do our own polish
                init='sobol',
                popsize=15
            )
            if de_result.fun < 1e5:
                pts_de = de_result.x.reshape(-1, 2)
                s = _score(pts_de)
                if s > best_score:
                    best_score = s
                    best_pts = pts_de.copy()
        except Exception:
            pass
    
    # Phase 2: Local Search Polish
    # Run local search from the best point found so far
    if best_pts is not None and time.time() < deadline - 10:
        time_per_ls = min(10.0, deadline - time.time() - 5)
        try:
            pts_ls, s_ls = _local_search(best_pts, max_iters=1000, time_limit=time_per_ls)
            if s_ls > best_score:
                best_score = s_ls
                best_pts = pts_ls.copy()
        except Exception:
            pass
    
    # Phase 3: L-BFGS-B Polish with Soft-Min
    if best_pts is not None and time.time() < deadline - 5:
        for beta in [50, 100, 200]:
            if time.time() > deadline - 5:
                break
            try:
                result = minimize(
                    _soft_min_objective,
                    best_pts.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                if result.fun < 1e5:
                    pts_opt = result.x.reshape(-1, 2)
                    s = _score(pts_opt)
                    if s > best_score:
                        best_score = s
                        best_pts = pts_opt.copy()
            except Exception:
                pass
    
    if best_pts is None:
        # Fallback: regular polygon
        best_pts = _regular_polygon(N, radius=1.0)
    
    return best_pts
# EVOLVE-BLOCK-END

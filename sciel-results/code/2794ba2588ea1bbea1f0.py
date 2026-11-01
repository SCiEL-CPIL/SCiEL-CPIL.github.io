"""Pack 200 equal circles in a unit square (centred at origin) to maximise common radius."""
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

N = 200
SIDE = 1.0
HALF = 0.5
DEADLINE = 115.0  # safety margin under 120s kill (evaluator wall-clock)


def _compute_radius(pts):
    """Return the common radius for a valid configuration, or -1 if invalid."""
    if pts is None or pts.shape != (N, 2):
        return -1.0
    if not np.all(np.isfinite(pts)):
        return -1.0
    # Boundary margin
    margin = HALF - np.max(np.abs(pts), axis=1)
    r_boundary = float(np.min(margin))
    if r_boundary <= 0:
        return -1.0
    # Pairwise distances
    tree = cKDTree(pts)
    # query_pairs returns unique pairs; distance is the pairwise distance
    pairs = tree.query_pairs(2.0 * r_boundary + 1e-12, output_type='ndarray')
    if len(pairs) == 0:
        # no pairs within 2*r_boundary? then min pairwise distance > 2*r_boundary
        # compute global min pairwise distance
        d, _ = tree.query(pts, k=2)  # k=2 includes self (0) and nearest
        min_pair = float(np.min(d[:, 1]))
        r_pair = 0.5 * min_pair
    else:
        diffs = pts[pairs[:, 0]] - pts[pairs[:, 1]]
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        min_pair = float(np.min(dists))
        r_pair = 0.5 * min_pair
    return min(r_boundary, r_pair)


def _soft_min(x, beta=100.0):
    """
    Compute the soft minimum of array x using log-sum-exp.
    soft_min(x) = min(x) - (1/beta) * log(sum(exp(-beta * (x - min(x)))))
    """
    min_val = np.min(x)
    shifted = x - min_val
    exp_terms = np.exp(-beta * shifted)
    log_sum = np.log(np.sum(exp_terms))
    return min_val - (1.0 / beta) * log_sum


def _smooth_objective(params, beta=50.0):
    """
    Objective function for L-BFGS-B.
    Minimizes -soft_min(radius_candidates).
    """
    pts = params.reshape(N, 2)
    
    # 1. Boundary margins
    margins = HALF - np.max(np.abs(pts), axis=1)
    
    # 2. Pairwise distances / 2
    # Use pdist for efficiency
    dists = pdist(pts, 'sqeuclidean')
    dists = np.sqrt(dists)
    r_pairs = 0.5 * dists
    
    # Combine all candidates
    candidates = np.concatenate([margins, r_pairs])
    
    # Compute soft min
    r_soft = _soft_min(candidates, beta=beta)
    
    # We want to maximize r_soft, so minimize -r_soft
    return -r_soft


def _initialize_hexagonal():
    """Generate a hexagonal lattice packing initialised inside the square, trimmed to N."""
    s = 0.08
    dx = s
    dy = s * np.sqrt(3) / 2.0
    extent = 1.0
    nx = int(2 * extent / dx) + 3
    ny = int(2 * extent / dy) + 3
    xs = np.arange(-nx // 2, nx // 2 + 1) * dx
    ys = np.arange(-ny // 2, ny // 2 + 1) * dy
    pts_list = []
    for i, y in enumerate(ys):
        offset = (dx / 2.0) if (i % 2 == 1) else 0.0
        for x in xs:
            px = x + offset
            py = y
            if -HALF <= px <= HALF and -HALF <= py <= HALF:
                pts_list.append([px, py])
    pts = np.array(pts_list, dtype=float)
    if len(pts) < N:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N].copy()
    return pts[:N].copy()


def _optimize(pts, time_limit=50.0):
    """
    Optimize the packing using L-BFGS-B on a smooth surrogate.
    """
    start = time.time()
    best_pts = pts.copy()
    best_score = _compute_radius(best_pts)
    
    # Flatten points for scipy
    x0 = best_pts.flatten()
    
    # Constraints: points must be inside the square.
    # L-BFGS-B supports bounds.
    bounds = [(-HALF + 1e-9, HALF - 1e-9)] * (2 * N)
    
    # We will run L-BFGS-B. 
    # To handle the non-smoothness of the true objective, we use a soft-min.
    # We might run multiple passes with increasing beta to sharpen the minimum.
    
    betas = [20.0, 50.0, 100.0, 200.0]
    
    current_pts = best_pts.copy()
    
    for beta in betas:
        if time.time() - start > time_limit:
            break
            
        # Define objective for this beta
        def obj(params):
            return _smooth_objective(params, beta=beta)
        
        try:
            res = minimize(obj, current_pts.flatten(), method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.success or res.fun < -best_score:
                new_pts = res.x.reshape(N, 2)
                new_score = _compute_radius(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
                    current_pts = new_pts.copy()
        except Exception as e:
            print(f"Optimization error: {e}")
            continue
            
    return best_pts, best_score


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE

    # Start with hexagonal init
    pts = _initialize_hexagonal()
    s = _compute_radius(pts)
    
    best_pts = pts.copy()
    best_score = s

    # Optimization
    if time.time() < deadline - 10:
        pts, s = _optimize(pts, time_limit=min(50.0, deadline - time.time() - 5))
        if s > best_score:
            best_score = s
            best_pts = pts.copy()

    # Final safety check
    final_s = _compute_radius(best_pts)
    if final_s <= 0:
        # Fallback to square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    return best_pts

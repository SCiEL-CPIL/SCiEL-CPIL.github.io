import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist
import time

N = 50

def _score(points):
    """Compute the objective: minimum of half-min-pair-distance and min-margin."""
    if len(points) != N:
        return -1.0
    if not np.all(np.isfinite(points)):
        return -1.0
    # Use cdist for speed
    dist = cdist(points, points)
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    norms = np.hypot(points[:, 0], points[:, 1])
    min_margin = np.min(1.0 - norms)
    return min(min_pair_dist / 2.0, min_margin)

def _softmin_logsumexp(dists, beta):
    """Soft minimum via log-sum-exp."""
    neg_beta = -beta * dists
    max_val = np.max(neg_beta)
    if not np.isfinite(max_val):
        return -np.inf
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta - max_val)))
    return -log_sum_exp / beta

def _smooth_objective_and_grad(params, beta):
    """
    params: flat array of length 2*N (x0,y0,x1,y1,...)
    Returns (objective, gradient) for smooth approximation of -min_radius.
    Uses vectorized operations where possible.
    """
    pts = params.reshape(N, 2)
    
    # Pairwise distances
    # cdist is much faster than manual loop for N=50
    dist_matrix = cdist(pts, pts)
    iu = np.triu_indices(N, k=1)
    pair_dists = dist_matrix[iu]
    
    # Boundary distances
    radii = np.hypot(pts[:, 0], pts[:, 1])
    boundary_dists = 1.0 - radii
    
    all_dists = np.concatenate([pair_dists, boundary_dists])
    scaled = all_dists / 2.0
    
    # Soft min
    soft_min_val = _softmin_logsumexp(scaled, beta)
    objective = -soft_min_val
    
    # Weights for gradient
    neg_beta_scaled = -beta * scaled
    max_val = np.max(neg_beta_scaled)
    weights = np.exp(neg_beta_scaled - max_val)
    w_sum = np.sum(weights)
    if w_sum < 1e-300:
        return objective, np.zeros_like(params)
    weights /= w_sum
    
    grad_scaled = -beta * weights
    n_pairs = len(pair_dists)
    
    grad_pts = np.zeros_like(pts)
    
    # Pair contributions
    # Vectorized gradient accumulation is tricky with scatter.
    # Loop is O(N^2) = 1225. This is fast enough.
    for k in range(n_pairs):
        w = grad_scaled[k] / 2.0
        d = pair_dists[k]
        if d > 1e-14:
            i, j = iu[0][k], iu[1][k]
            d_dij = (pts[i] - pts[j]) / d
            grad_pts[i] += w * d_dij
            grad_pts[j] -= w * d_dij
    
    # Boundary contributions
    for i in range(N):
        w = grad_scaled[n_pairs + i] / 2.0
        r = radii[i]
        if r > 1e-14:
            d_dr = pts[i] / r
            grad_pts[i] += w * d_dr
    
    return objective, grad_pts.flatten()

def _ring_initialization(ring_radii, ring_counts, angular_offsets=None):
    """
    Generate points from ring specification.
    """
    points = []
    n_rings = len(ring_radii)
    if angular_offsets is None:
        angular_offsets = np.zeros(n_rings)
    
    for i in range(n_rings):
        r = ring_radii[i]
        count = ring_counts[i]
        offset = angular_offsets[i]
        if count == 0:
            continue
        angles = np.linspace(0, 2*np.pi, count, endpoint=False) + offset
        for a in angles:
            points.append([r * np.cos(a), r * np.sin(a)])
    
    return np.array(points[:N])

def _ring_params_to_points(params):
    """
    Convert DE parameters to points.
    """
    ring_counts = np.array([1, 6, 12, 18, 13])
    ring_radii = params[:5]
    angular_offsets = params[5:10]
    
    ring_radii[0] = 0.0
    
    return _ring_initialization(ring_radii, ring_counts, angular_offsets)

def _optimize_ring_with_de(deadline):
    """Use differential evolution to find good ring parameters."""
    bounds = [(0.0, 0.95)] * 5 + [(0.0, 2*np.pi)] * 5
    
    def neg_score(params):
        pts = _ring_params_to_points(params)
        return -_score(pts)
    
    try:
        result = differential_evolution(
            neg_score, bounds,
            seed=42, maxiter=100, popsize=15, # Reduced maxiter to save time for polish
            tol=1e-8, mutation=(0.5, 1.0), recombination=0.8,
            polish=False, init='latinhypercube'
        )
        best_params = result.x
        best_pts = _ring_params_to_points(best_params)
        return best_pts, -result.fun
    except Exception:
        ring_radii = np.array([0.0, 0.18, 0.36, 0.54, 0.72])
        ring_counts = np.array([1, 6, 12, 18, 13])
        return _ring_initialization(ring_radii, ring_counts), _score(_ring_initialization(ring_radii, ring_counts))

def _multi_stage_smooth_polish(points, deadline):
    """
    Multi-stage beta annealing.
    """
    current_pts = points.copy()
    current_score = _score(current_pts)
    
    # Start with lower beta to escape local minima, increase for precision
    beta_stages = [5.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    
    for beta in beta_stages:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                lambda x, b=beta: _smooth_objective_and_grad(x, b)[0],
                lambda x, b=beta: _smooth_objective_and_grad(x, b)[1],
                current_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            
            if time.time() > deadline:
                break
            
            new_pts = result.x.reshape(N, 2)
            
            if not np.all(np.isfinite(new_pts)):
                continue
            
            # Clamp to unit circle
            norms = np.hypot(new_pts[:, 0], new_pts[:, 1])
            mask = norms > 1.0
            if np.any(mask):
                new_pts[mask] *= (1.0 - 1e-12) / norms[mask]
            
            new_score = _score(new_pts)
            
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
        
        except Exception:
            continue
    
    return current_pts, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    np.random.seed(42)
    
    best_points = None
    best_score = -1.0
    
    # Phase 1: DE over ring parameters
    # We run DE to get a good starting point.
    if time.time() < deadline - 80.0:
        ring_pts, ring_score = _optimize_ring_with_de(deadline - 80.0)
        if ring_score > best_score:
            best_score = ring_score
            best_points = ring_pts.copy()
    
    # Phase 2: Multi-stage smooth polish from ring solution
    if best_points is not None and time.time() < deadline - 20.0:
        polished, score = _multi_stage_smooth_polish(best_points, deadline - 20.0)
        if score > best_score:
            best_score = score
            best_points = polished.copy()
    
    # Phase 3: Try a few random perturbations of the best solution and polish
    # This helps escape local minima in the coordinate space.
    if best_points is not None and time.time() < deadline - 10.0:
        rng = np.random.default_rng(123)
        for attempt in range(3):
            if time.time() > deadline - 5.0:
                break
            # Perturb slightly
            perturbed = best_points + rng.normal(0, 0.01, size=(N, 2))
            # Clamp
            norms = np.hypot(perturbed[:, 0], perturbed[:, 1])
            mask = norms > 1.0
            if np.any(mask):
                perturbed[mask] *= (1.0 - 1e-12) / norms[mask]
            
            polished, score = _multi_stage_smooth_polish(perturbed, deadline - 5.0)
            if score > best_score:
                best_score = score
                best_points = polished.copy()

    if best_points is None:
        # Fallback: simple ring
        ring_radii = np.array([0.0, 0.18, 0.36, 0.54, 0.72])
        ring_counts = np.array([1, 6, 12, 18, 13])
        best_points = _ring_initialization(ring_radii, ring_counts)
        best_score = _score(best_points)
    
    best_points = np.array(best_points, dtype=np.float64)
    norms = np.linalg.norm(best_points, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_points[mask] *= (1.0 - 1e-12) / norms[mask]
    
    return best_points

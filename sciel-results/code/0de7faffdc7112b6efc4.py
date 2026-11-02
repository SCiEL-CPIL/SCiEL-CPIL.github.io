"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools

def _all_triangle_areas(pts):
    """Vectorized computation of all triangle areas."""
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            v1 = pts[j] - pts[i]
            for k in range(j + 1, n):
                v2 = pts[k] - pts[i]
                areas.append(0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0]))
    return np.array(areas)

def _convex_hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score_points(pts):
    """Compute min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 0.0
    
    areas = _all_triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    
    min_area = np.min(areas)
    return min_area / hull_area

def _soft_min_triangle_area(pts, beta=50.0):
    """Smooth approximation of min triangle area using log-sum-exp."""
    areas = _all_triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    # Use log-sum-exp for numerical stability
    max_a = np.max(areas)
    if max_a <= 0:
        return 0.0
    # -log(sum(exp(-beta * a))) / beta approximates min(a)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - max_a)))) + (-beta * max_a)
    soft_min = -log_sum / beta
    return soft_min

def _smooth_score(pts, beta=50.0):
    """Smooth score for optimization."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 0.0
    soft_min = _soft_min_triangle_area(pts, beta)
    return soft_min / hull_area

def _neg_smooth_score(params, beta=50.0):
    """Negative smooth score for minimization."""
    pts = params.reshape(-1, 2)
    return -_smooth_score(pts, beta)

def _sa_optimize(init_pts, rng, deadline, beta=50.0):
    """Simulated Annealing with smooth surrogate."""
    n = len(init_pts)
    current = init_pts.copy().flatten()
    current_score = _smooth_score(init_pts, beta)
    best = init_pts.copy()
    best_score = current_score
    
    T0 = 0.1
    T_end = 0.001
    cooling = 0.999
    
    T = T0
    step = 0.05
    no_improve = 0
    max_no_improve = 500
    
    while time.time() < deadline - 2.0 and T > T_end and no_improve < max_no_improve:
        # Generate perturbation
        delta = rng.randn(2 * n) * step
        candidate = current + delta
        candidate_pts = candidate.reshape(-1, 2)
        
        # Keep points in reasonable bounds
        candidate_pts = np.clip(candidate_pts, -5, 5)
        
        cand_score = _smooth_score(candidate_pts, beta)
        
        if cand_score > current_score:
            current = candidate
            current_score = cand_score
            no_improve = 0
            
            if cand_score > best_score:
                best_score = cand_score
                best = candidate_pts.copy()
        else:
            # Accept with probability exp((cand - curr) / T)
            delta_score = cand_score - current_score
            if delta_score < 0:
                accept_prob = np.exp(delta_score / T)
                if rng.random() < accept_prob:
                    current = candidate
                    current_score = cand_score
                else:
                    no_improve += 1
            
            no_improve += 1
        
        T *= cooling
        # Adapt step size
        if no_improve > 100:
            step *= 0.9
        elif no_improve == 0:
            step = min(step * 1.05, 0.2)
    
    return best, best_score

def _local_refine(pts, deadline):
    """Local refinement using Nelder-Mead on smooth objective."""
    try:
        res = minimize(
            lambda p: _neg_smooth_score(p, beta=100.0),
            pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        candidate = res.x.reshape(-1, 2)
        return candidate
    except Exception:
        return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_score = 0.0
    best_pts = np.zeros((N, 2))
    rng = np.random.RandomState(42)
    
    # Generate diverse initializations
    inits = []
    
    # Regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # Regular polygon with radial jitter
    for jitter in [0.02, 0.05, 0.1, 0.15]:
        r = 1.0 + jitter * rng.randn(N)
        inits.append(np.column_stack([r * np.cos(angles), r * np.sin(angles)]))
    
    # Two concentric rings
    n_outer = 7
    n_inner = N - n_outer
    angles_out = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    angles_in = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + np.pi / n_inner
    outer = np.column_stack([np.cos(angles_out), np.sin(angles_out)])
    inner = np.column_stack([0.5 * np.cos(angles_in), 0.5 * np.sin(angles_in)])
    inits.append(np.vstack([outer, inner]))
    
    # Random points in unit disk
    for _ in range(15):
        theta = rng.uniform(0, 2 * np.pi, N)
        r = np.sqrt(rng.uniform(0, 1, N))
        inits.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))
    
    # Random points in unit square
    for _ in range(10):
        inits.append(rng.uniform(-0.5, 0.5, (N, 2)))
    
    # Run SA from each initialization
    for idx, init_pts in enumerate(inits):
        if time.time() > deadline - 5.0:
            break
        
        sa_pts, sa_score = _sa_optimize(init_pts, rng, deadline)
        sa_true_score = _score_points(sa_pts)
        
        if sa_true_score > best_score:
            best_score = sa_true_score
            best_pts = sa_pts.copy()
        
        # Local refinement if time permits
        if time.time() < deadline - 3.0:
            refined = _local_refine(best_pts, deadline)
            refined_score = _score_points(refined)
            if refined_score > best_score:
                best_score = refined_score
                best_pts = refined.copy()
    
    return best_pts
# EVOLVE-BLOCK-END

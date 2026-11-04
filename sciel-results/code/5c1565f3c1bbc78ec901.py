"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import itertools

# Precompute triple indices for N=14 for vectorized area computation
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

def _calculate_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    
    # Calculate Minimum Triangle Area
    # points shape (N, 2)
    # TRIPLE_INDICES shape (M, 3)
    
    # Extract points for all triples
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    
    # Compute cross product (pj - pi) x (pk - pi)
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    min_area = 0.5 * np.min(np.abs(cross))
    
    return min_area / hull_area

def _soft_min_score(points, beta=50.0):
    """
    Smooth surrogate: soft_min of triangle areas / hull area.
    Returns a value that approximates min_area/hull_area smoothly.
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]
    
    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])
    
    areas = 0.5 * np.abs(cross)
    
    # Soft-min via log-sum-exp: soft_min(a) ≈ (1/beta) * log(sum(exp(-beta * a)))
    # We want to maximize this, so we minimize the negative.
    # But here we return the positive soft-min value to be used in -objective.
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    soft_min_val = (1.0 / beta) * (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled))))
    
    return soft_min_val / hull_area

def _generate_ring_config(n_out, n_in, r, theta):
    """Generate points for a two-ring configuration."""
    angles_out = 2 * np.pi * np.arange(n_out) / n_out
    pts_out = np.column_stack([np.cos(angles_out), np.sin(angles_out)])
    
    angles_in = 2 * np.pi * np.arange(n_in) / n_in + theta
    pts_in = r * np.column_stack([np.cos(angles_in), np.sin(angles_in)])
    
    return np.vstack([pts_out, pts_in])

def _score_ring_params(params, n_out, n_in):
    """
    Objective function for ring optimization (minimization).
    params: [r, theta]
    """
    r, theta = params
    if r <= 0 or r >= 1.0:
        return 1e9
        
    pts = _generate_ring_config(n_out, n_in, r, theta)
    score = _calculate_score(pts)
    return -score

def _optimize_ring(n_out, n_in, deadline):
    """Optimize a specific ring configuration using DE and Nelder-Mead."""
    best_score = 0.0
    best_params = None
    
    g = np.gcd(n_out, n_in)
    theta_max = 2 * np.pi / g
    
    bounds = [(0.01, 0.99), (0.0, theta_max)]
    
    # 1. Differential Evolution for global search
    if time.time() < deadline - 30:
        try:
            res = differential_evolution(
                lambda p: _score_ring_params(p, n_out, n_in),
                bounds,
                seed=42,
                maxiter=20,
                popsize=10,
                tol=1e-5,
                polish=False # We do our own polish
            )
            if res.fun < 1e8:
                best_score = -res.fun
                best_params = res.x
        except Exception:
            pass

    # 2. Local polishing with Nelder-Mead from multiple starts
    if time.time() < deadline - 10:
        # Generate diverse starts
        starts = []
        if best_params is not None:
            starts.append(best_params)
        
        # Heuristic starts
        for r_start in [0.3, 0.5, 0.7, 0.9]:
            for theta_start in [0.0, theta_max/2, theta_max/3]:
                starts.append([r_start, theta_start])
        
        for start in starts:
            if time.time() > deadline - 10:
                break
            try:
                res_nm = minimize(
                    lambda p: _score_ring_params(p, n_out, n_in),
                    start,
                    method='Nelder-Mead',
                    options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 200}
                )
                if res_nm.fun < 1e8:
                    score = -res_nm.fun
                    if score > best_score:
                        best_score = score
                        best_params = res_nm.x
            except Exception:
                continue

    if best_params is None:
        return 0.0, None
        
    pts = _generate_ring_config(n_out, n_in, best_params[0], best_params[1])
    return best_score, pts

def _local_polish(points, deadline):
    """
    General local search using coordinate ascent with random perturbations.
    This is the structural refinement from the current program, but applied to general points.
    """
    current_pts = points.copy()
    current_score = _calculate_score(current_pts)
    best_pts = current_pts.copy()
    best_score = current_score
    
    step = 0.05
    no_improve_count = 0
    rng = np.random.RandomState(42)
    
    # Time budget for local polish
    polish_deadline = deadline - 5.0
    
    while time.time() < polish_deadline and no_improve_count < 300:
        # Pick a random point
        idx = rng.randint(len(current_pts))
        
        # Random direction
        direction = rng.randn(2)
        direction /= (np.linalg.norm(direction) + 1e-12)
        
        new_pt = current_pts[idx].copy()
        new_pt += step * direction
        
        old_pt = current_pts[idx].copy()
        current_pts[idx] = new_pt
        
        new_score = _calculate_score(current_pts)
        
        if new_score > current_score:
            current_score = new_score
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
        else:
            current_pts[idx] = old_pt
            no_improve_count += 1
            # Adaptive step size
            if no_improve_count % 50 == 0:
                step *= 0.8
                if step < 1e-4:
                    break
            else:
                # Occasionally try a larger jump
                if no_improve_count % 10 == 0:
                    step *= 1.2
    
    return best_pts, best_score

def construct():
    start = time.time()
    deadline = start + 115.0  # Leave 5s margin for safety
    
    best_pts = None
    best_score = 0.0
    
    # Strategy 1: Optimize 7-7 Ring Configuration
    # This is the symmetric case which often performs well.
    score_77, pts_77 = _optimize_ring(7, 7, deadline - 20)
    if score_77 > best_score:
        best_score = score_77
        best_pts = pts_77
    
    # Strategy 2: Optimize 6-8 Ring Configuration (Asymmetric)
    if time.time() < deadline - 20:
        score_68, pts_68 = _optimize_ring(6, 8, deadline - 20)
        if score_68 > best_score:
            best_score = score_68
            best_pts = pts_68

    # Strategy 3: Optimize 5-9 Ring Configuration
    if time.time() < deadline - 10:
        score_59, pts_59 = _optimize_ring(5, 9, deadline - 10)
        if score_59 > best_score:
            best_score = score_59
            best_pts = pts_59

    # Fallback: Regular 14-gon
    if best_pts is None:
        angles = 2 * np.pi * np.arange(N) / N
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = _calculate_score(best_pts)

    # Strategy 4: Local Polish on the best candidate
    if time.time() < deadline - 5:
        polished_pts, polished_score = _local_polish(best_pts, deadline)
        if polished_score > best_score:
            best_score = polished_score
            best_pts = polished_pts

    return best_pts
# EVOLVE-BLOCK-END

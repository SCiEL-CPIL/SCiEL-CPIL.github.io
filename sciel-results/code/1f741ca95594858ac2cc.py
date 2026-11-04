import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize, minimize_scalar

N = 14
SOFT_BETA = 200.0

# Precompute indices for all triples
def _get_triple_indices(n):
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=int)

_TRIPLE_INDICES = _get_triple_indices(N)

def _calculate_score(points):
    """
    Compute the score: min triangle area / convex hull area.
    """
    if len(points) < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    i_idx = _TRIPLE_INDICES[:, 0]
    j_idx = _TRIPLE_INDICES[:, 1]
    k_idx = _TRIPLE_INDICES[:, 2]
    
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def _soft_min_score(points, beta=SOFT_BETA):
    """
    Soft-min approximation of the minimum triangle area.
    Uses log-sum-exp trick for numerical stability.
    """
    i_idx = _TRIPLE_INDICES[:, 0]
    j_idx = _TRIPLE_INDICES[:, 1]
    k_idx = _TRIPLE_INDICES[:, 2]
    
    p_i = points[i_idx]
    p_j = points[j_idx]
    p_k = points[k_idx]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min: -log(sum(exp(-beta * areas))) / beta
    # Use log-sum-exp trick for stability
    shifted = -beta * areas
    max_shifted = np.max(shifted)
    log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    soft_min = -log_sum_exp / beta
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    
    if hull_area < 1e-12:
        return 0.0
    
    return soft_min / hull_area

def _score_ring_params(params):
    """
    Score for ring configuration with parameters [r, theta].
    Outer ring: 7 points on unit circle.
    Inner ring: 7 points on circle of radius r, rotated by theta.
    """
    r, theta = params
    angles_out = np.linspace(0, 2*np.pi, 7, endpoint=False)
    outer_pts = np.column_stack([np.cos(angles_out), np.sin(angles_out)])
    
    angles_in = theta + np.linspace(0, 2*np.pi, 7, endpoint=False)
    inner_pts = r * np.column_stack([np.cos(angles_in), np.sin(angles_in)])
    
    points = np.vstack([outer_pts, inner_pts])
    return _calculate_score(points)

def _soft_min_ring_score(params, beta=SOFT_BETA):
    """
    Soft-min score for ring configuration.
    """
    r, theta = params
    angles_out = np.linspace(0, 2*np.pi, 7, endpoint=False)
    outer_pts = np.column_stack([np.cos(angles_out), np.sin(angles_out)])
    
    angles_in = theta + np.linspace(0, 2*np.pi, 7, endpoint=False)
    inner_pts = r * np.column_stack([np.cos(angles_in), np.sin(angles_in)])
    
    points = np.vstack([outer_pts, inner_pts])
    return _soft_min_score(points, beta)

def _optimize_ring(deadline):
    """
    Optimize ring parameters using differential evolution.
    """
    def neg_soft_ring(params):
        if time.time() > deadline - 5:
            raise TimeoutError()
        try:
            return -_soft_min_ring_score(params)
        except Exception:
            return 0.0
    
    bounds = [(0.3, 0.8), (0.0, 2*np.pi/7)]
    
    try:
        res = differential_evolution(
            neg_soft_ring,
            bounds,
            seed=42,
            maxiter=80,
            popsize=15,
            tol=1e-6,
            polish=False,
            time_limit=max(1, deadline - time.time() - 15)
        )
        
        best_params = res.x
        points = np.vstack([
            np.column_stack([np.cos(np.linspace(0, 2*np.pi, 7, endpoint=False)), 
                           np.sin(np.linspace(0, 2*np.pi, 7, endpoint=False))]),
            best_params[0] * np.column_stack([np.cos(best_params[1] + np.linspace(0, 2*np.pi, 7, endpoint=False)), 
                                             np.sin(best_params[1] + np.linspace(0, 2*np.pi, 7, endpoint=False))])
        ])
        score = _calculate_score(points)
        return points, score
    except Exception:
        # Fallback: regular 14-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        score = _calculate_score(points)
        return points, score

def _coordinate_polish(points, deadline):
    """
    Coordinate descent with annealed soft-beta for local refinement.
    """
    best_points = points.copy()
    best_score = _calculate_score(best_points)
    
    # Anneal beta from high to low for progressive sharpening
    betas = [SOFT_BETA, SOFT_BETA * 0.5, SOFT_BETA * 0.25, SOFT_BETA * 0.125]
    
    for beta in betas:
        if time.time() > deadline - 3:
            break
            
        # Try moving each coordinate
        for _ in range(50):
            if time.time() > deadline - 3:
                break
            
            improved = False
            for i in range(N):
                for dim in range(2):
                    # Try small perturbations
                    for delta in [0.01, -0.01, 0.005, -0.005, 0.02, -0.02]:
                        candidate = best_points.copy()
                        candidate[i, dim] += delta
                        
                        # Check if still valid (convex hull)
                        try:
                            ConvexHull(candidate)
                        except Exception:
                            continue
                        
                        # Use soft-min score for comparison
                        soft_score = _soft_min_score(candidate, beta)
                        if soft_score > best_score * 0.999:  # Allow small tolerance
                            # Verify with true score
                            true_score = _calculate_score(candidate)
                            if true_score > best_score:
                                best_points = candidate
                                best_score = true_score
                                improved = True
                                break
                    if improved:
                        break
            if not improved:
                break
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Phase 1: Global search with differential evolution on ring parameters
    best_points, best_score = _optimize_ring(deadline)
    
    # Phase 2: Local refinement with coordinate polish
    if time.time() < deadline - 5:
        polished_points, polished_score = _coordinate_polish(best_points, deadline)
        if polished_score > best_score:
            best_points = polished_points
            best_score = polished_score
    
    # Phase 3: Final Nelder-Mead refinement if time permits
    if time.time() < deadline - 10:
        def objective_full(x):
            if time.time() > deadline - 3:
                raise TimeoutError()
            p = x.reshape(N, 2)
            return -_soft_min_score(p, beta=SOFT_BETA * 0.5)
        
        try:
            x0 = best_points.flatten()
            res_nm = minimize(
                objective_full,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if res_nm.fun < -best_score * 0.999:
                new_pts = res_nm.x.reshape(N, 2)
                s = _calculate_score(new_pts)
                if s > best_score:
                    best_points = new_pts
                    best_score = s
                    
        except TimeoutError:
            pass
        except Exception:
            pass
    
    # Final fallback
    if best_points is None or best_score <= 0:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = _calculate_score(best_points)
    
    return best_points

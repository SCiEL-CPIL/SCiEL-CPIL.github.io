import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
import time
import itertools

N = 14
TRIS = list(itertools.combinations(range(N), 3))
TRI_INDICES = np.array(TRIS)

def calc_score(pts):
    """
    Calculate the minimum triangle area divided by the convex hull area.
    pts: (N, 2) array
    Returns: score (higher is better)
    """
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return -1e10
    except Exception:
        return -1e10
        
    p_i = pts[TRI_INDICES[:, 0]]
    p_j = pts[TRI_INDICES[:, 1]]
    p_k = pts[TRI_INDICES[:, 2]]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    return min_area / hull_area

def polar_to_cartesian(params):
    """
    Convert polar parameters (radius, angle) to Cartesian coordinates.
    params: (N, 2) array where column 0 is radius, column 1 is angle
    Returns: (N, 2) Cartesian coordinates
    """
    radii = params[:, 0]
    angles = params[:, 1]
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack([x, y])

def soft_min_score(pts, temperature=0.01):
    """
    Compute a smooth approximation of the minimum triangle score.
    Uses soft-min to make the objective differentiable.
    """
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 1e10
    except Exception:
        return 1e10
        
    p_i = pts[TRI_INDICES[:, 0]]
    p_j = pts[TRI_INDICES[:, 1]]
    p_k = pts[TRI_INDICES[:, 2]]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min: -temperature * log(sum(exp(-area/temperature)))
    # This is smooth and differentiable
    min_val = np.min(areas)
    scaled = (min_val - areas) / temperature
    # Clip to avoid overflow
    scaled = np.clip(scaled, -50, 50)
    soft_min = min_val - temperature * np.log(np.sum(np.exp(scaled)))
    
    return soft_min / hull_area

def objective_polar(params_flat):
    """
    Objective function for polar parameter optimization.
    Minimizes negative soft-min score.
    """
    params = params_flat.reshape(N, 2)
    pts = polar_to_cartesian(params)
    
    # Check for degeneracy
    if np.any(np.isinf(pts)) or np.any(np.isnan(pts)):
        return 1e10
    
    score = soft_min_score(pts, temperature=0.005)
    return -score

def run_differential_evolution(time_limit, seed):
    """
    Run differential evolution on polar coordinate space.
    """
    start_time = time.time()
    rng = np.random.RandomState(seed)
    
    # Bounds: radius in [0.1, 2.0], angle in [0, 2*pi)
    # We'll sort angles later to enforce order
    bounds = [(0.1, 2.0)] * N + [(0, 2*np.pi)] * N
    
    # Initial population: mix of structured and random
    def init_pop():
        pop = []
        # Regular polygon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        radii = np.ones(N)
        pop.append(np.concatenate([radii, angles]))
        
        # Two concentric circles
        angles_outer = np.linspace(0, 2*np.pi, N//2, endpoint=False)
        angles_inner = angles_outer + np.pi/(N//2)
        radii_outer = np.ones(N//2) * 1.5
        radii_inner = np.ones(N - N//2) * 0.7
        pop.append(np.concatenate([radii_outer, radii_inner, angles_outer, angles_inner]))
        
        # Random
        for _ in range(10):
            radii = rng.uniform(0.2, 1.8, N)
            angles = rng.uniform(0, 2*np.pi, N)
            pop.append(np.concatenate([radii, angles]))
        
        return np.array(pop)
    
    try:
        res = differential_evolution(
            objective_polar,
            bounds,
            seed=seed,
            maxiter=100,
            tol=1e-6,
            popsize=15,
            init='sobol',
            polish=False,
            updating='immediate',
            callback=lambda x, convergence: None
        )
        
        if time.time() - start_time > time_limit:
            return None, -1e10
        
        best_params = res.x.reshape(N, 2)
        best_pts = polar_to_cartesian(best_params)
        best_score = calc_score(best_pts)
        
        return best_pts, best_score
    except Exception:
        return None, -1e10

def polish_with_lm(pts, time_limit):
    """
    Polish using Levenberg-Marquardt style local search in Cartesian space.
    """
    start_time = time.time()
    
    def objective_cartesian(pts_flat):
        pts = pts_flat.reshape(N, 2)
        score = soft_min_score(pts, temperature=0.003)
        return -score
    
    x0 = pts.flatten()
    
    try:
        res = minimize(
            objective_cartesian,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if time.time() - start_time > time_limit:
            return pts, calc_score(pts)
        
        polished_pts = res.x.reshape(N, 2)
        polished_score = calc_score(polished_pts)
        
        if polished_score > calc_score(pts):
            return polished_pts, polished_score
        else:
            return pts, calc_score(pts)
    except Exception:
        return pts, calc_score(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_pts = None
    best_score = -1e10
    
    # Run differential evolution with different seeds
    de_configs = [
        {'seed': 42, 'time_limit': 30},
        {'seed': 123, 'time_limit': 30},
        {'seed': 456, 'time_limit': 30},
    ]
    
    for config in de_configs:
        if time.time() > deadline - 20.0:
            break
        
        de_pts, de_score = run_differential_evolution(config['time_limit'], config['seed'])
        
        if de_pts is not None and de_score > best_score:
            best_score = de_score
            best_pts = de_pts.copy()
    
    # Polish the best result
    if best_pts is not None:
        remaining_time = deadline - time.time()
        if remaining_time > 5.0:
            polished_pts, polished_score = polish_with_lm(best_pts, remaining_time - 5.0)
            if polished_score > best_score:
                best_score = polished_score
                best_pts = polished_pts
    else:
        # Fallback: regular 14-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = calc_score(best_pts)
        
    return best_pts

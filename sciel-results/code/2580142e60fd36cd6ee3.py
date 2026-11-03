import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 14
_START_TIME = time.time()
DEADLINE = 140.0  # safety margin before 150 s kill

def _time_left():
    return DEADLINE - (time.time() - _START_TIME)

# Precompute triangle index triples for vectorized area computation
_idx = np.arange(N)
_II, _JJ, _KK = np.meshgrid(_idx, _idx, _idx, indexing='ij')
_mask = (_II < _JJ) & (_JJ < _KK)
_T_I = _II[_mask].ravel()
_T_J = _JJ[_mask].ravel()
_T_K = _KK[_mask].ravel()
N_TRI = len(_T_I)

def _all_areas(points):
    """Return array of all C(N,3) triangle areas (vectorized)."""
    pi = points[_T_I]
    pj = points[_T_J]
    pk = points[_T_K]
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    return 0.5 * np.abs(v1x * v2y - v1y * v2x)

def _score(points):
    """min triangle area / convex hull area."""
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    if hull_area < 1e-12:
        return 0.0
    areas = _all_areas(points)
    min_a = np.min(areas)
    if min_a < 1e-14:
        return 0.0
    return min_a / hull_area

def _objective_hard(p):
    """Hard objective: -score."""
    points = p.reshape(N, 2)
    return -_score(points)

def _generate_initials():
    inits = []
    rng = np.random.RandomState(42)

    # 1. Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))

    # 2. Perturbed regular polygons
    for _ in range(5):
        noise = rng.uniform(-0.1, 0.1, (N, 2))
        inits.append(np.column_stack([np.cos(angles), np.sin(angles)]) + noise)

    # 3. Two-ring: inner 4 + outer 10
    inner_a = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    outer_a = np.linspace(0, 2 * np.pi, 10, endpoint=False) + np.pi / 10
    inner = 0.5 * np.column_stack([np.cos(inner_a), np.sin(inner_a)])
    outer = np.column_stack([np.cos(outer_a), np.sin(outer_a)])
    inits.append(np.vstack((inner, outer)))
    
    # 4. Random points in unit disk
    for _ in range(5):
        r = np.sqrt(rng.rand(N))
        theta = rng.rand(N) * 2 * np.pi
        inits.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))

    return inits

def _polish(points, time_budget=10.0):
    """High-precision polish using Nelder-Mead and Powell."""
    start = time.time()
    best_params = points.flatten().copy()
    best_score = _score(points)
    
    # Nelder-Mead
    if _time_left() > time_budget + 5:
        try:
            res_nm = minimize(
                _objective_hard,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            cand_score = _score(res_nm.x.reshape(N, 2))
            if cand_score > best_score:
                best_score = cand_score
                best_params = res_nm.x.copy()
        except Exception:
            pass
            
    # Powell
    if _time_left() > time_budget:
        try:
            res_pow = minimize(
                _objective_hard,
                best_params,
                method='Powell',
                options={'maxiter': 500, 'xtol': 1e-12, 'ftol': 1e-14, 'adaptive': True}
            )
            cand_score = _score(res_pow.x.reshape(N, 2))
            if cand_score > best_score:
                best_score = cand_score
                best_params = res_pow.x.copy()
        except Exception:
            pass
            
    return best_params.reshape(N, 2), best_score

def construct():
    start = time.time()
    inits = _generate_initials()

    best_params = inits[0].flatten().copy()
    best_score = _score(inits[0])
    
    # Initial polish of the best start
    if _time_left() > 120:
        best_params, best_score = _polish(best_params.reshape(N, 2), time_budget=10)

    # Simulated Annealing
    # We focus on breaking the "bottleneck" triangle (the one with min area)
    # by moving its vertices away from each other.
    
    T = 0.01
    T_min = 1e-6
    cooling = 0.9995
    rng = np.random.RandomState(123)
    
    current_points = best_params.reshape(N, 2).copy()
    current_score = best_score
    
    # Track the best bottleneck triangle found
    best_bottleneck_tri = None
    
    while _time_left() > 25:
        areas = _all_areas(current_points)
        min_idx = np.argmin(areas)
        min_area = areas[min_idx]
        
        # Identify the triangle causing the bottleneck
        tri_idx = (min_idx // 14 + 1) * 13 - (min_idx // 14) * (min_idx // 14 + 1) / 2 # This is complex, easier to use precomputed arrays
        # Actually, we precomputed _T_I, _T_J, _T_K
        i, j, k = _T_I[min_idx], _T_J[min_idx], _T_K[min_idx]
        
        # Generate a move: perturb one of the vertices of the bottleneck triangle
        # or a random vertex if we are "stuck"
        if rng.rand() < 0.7:
            # Targeted move: move one of i, j, k
            target_vertex = rng.choice([i, j, k])
        else:
            # Random move
            target_vertex = rng.randint(N)
            
        # Move size: larger when T is high
        move_size = 0.05 + 0.1 * (T / 0.01)
        delta = rng.normal(0, move_size, 2)
        
        new_points = current_points.copy()
        new_points[target_vertex] += delta
        
        new_score = _score(new_points)
        
        # Acceptance criterion
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_params = current_points.flatten().copy()
                best_bottleneck_tri = (i, j, k)
        else:
            # Metropolis criterion
            delta_score = current_score - new_score
            if delta_score < 0:
                accept = True
            else:
                # Probability of accepting a worse move
                # Note: score is positive, so delta_score > 0 means new is worse
                # We want to accept with prob exp(-delta_score / T)
                # But score scales are small, so we might need to scale T
                prob = np.exp(-delta_score / (T * 0.01)) # Scaling T to match score scale
                accept = rng.rand() < prob
                
            if accept:
                current_points = new_points
                current_score = new_score
                
        T *= cooling
        if T < T_min:
            T = T_min
            
        # Occasional restart if stuck? 
        # For now, let it cool.

    # Final Polish
    final_points, final_score = _polish(best_params.reshape(N, 2), time_budget=15)
    
    # Normalize to unit hull area
    points = final_points
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        hull = ConvexHull(points)
        hull_area = hull.volume

    if hull_area > 1e-12:
        scale = np.sqrt(1.0 / hull_area)
        points = points * scale

    return points

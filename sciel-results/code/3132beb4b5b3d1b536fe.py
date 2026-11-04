import numpy as np
from scipy.spatial import ConvexHull
import time

N = 14

def _get_points(r, theta):
    # Outer heptagon
    angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False)
    x_out = np.cos(angles_outer)
    y_out = np.sin(angles_outer)
    
    # Inner heptagon
    angles_inner = angles_outer + theta
    x_in = r * np.cos(angles_inner)
    y_in = r * np.sin(angles_inner)
    
    points = np.vstack([
        np.column_stack((x_out, y_out)),
        np.column_stack((x_in, y_in))
    ])
    return points

def _score(params):
    r, theta = params
    if r <= 0 or r >= 1:
        return 0.0
        
    points = _get_points(r, theta)
    
    # Convex Hull Area
    # Since inner points are strictly inside the outer heptagon (for r<1),
    # the hull is the outer heptagon.
    # Area of regular heptagon with circumradius R=1:
    # A = 0.5 * n * R^2 * sin(2*pi/n)
    hull_area = 0.5 * 7 * np.sin(2*np.pi/7)
    
    # Min Triangle Area
    # Vectorized calculation of all triangle areas
    # points shape (14, 2)
    # We need all triples i, j, k with i < j < k
    
    # Precomputed indices for triples
    # This is done once in construct or globally
    
    # Area = 0.5 * |det(p_j - p_i, p_k - p_i)|
    # Using cross product in 2D: (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
    
    # Efficient way:
    # p_i, p_j, p_k
    # We can compute all pairwise cross products? No, just iterate triples.
    # 364 triples is small.
    
    min_area = float('inf')
    # Using precomputed indices
    for i, j, k in TRIPLE_INDICES:
        pi = points[i]
        pj = points[j]
        pk = points[k]
        # Cross product magnitude
        # (pj - pi) x (pk - pi)
        # (xj - xi)(yk - yi) - (yj - yi)(xk - xi)
        val = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0])
        area = 0.5 * abs(val)
        if area < min_area:
            min_area = area
            
    return min_area / hull_area

# Precompute triple indices
TRIPLE_INDICES = []
for i in range(N):
    for j in range(i+1, N):
        for k in range(j+1, N):
            TRIPLE_INDICES.append((i, j, k))
TRIPLE_INDICES = np.array(TRIPLE_INDICES)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Strategy:
    # 1. Global search for r, theta
    # 2. Local polish
    
    best_score = 0.0
    best_params = None
    
    # Differential Evolution
    from scipy.optimize import differential_evolution
    
    bounds = [(0.1, 0.9), (0, 2*np.pi/7)]
    
    try:
        res = differential_evolution(
            lambda p: -_score(p), # Minimize negative score
            bounds,
            seed=42,
            maxiter=200,
            popsize=20,
            tol=1e-6,
            callback=lambda xk, convergence: None # No callback needed for time check inside DE easily, 
            # but we can wrap the objective to check time? 
            # Actually, DE runs in C, so it's fast. 
            # 200 iters * 20 pop * 2 dims is very fast.
        )
        
        if res.success:
            best_params = res.x
            best_score = -res.fun
            
    except Exception as e:
        print(f"DE failed: {e}")
        # Fallback to grid or simple search
        
    # Local Polish with Nelder-Mead
    if best_params is not None:
        from scipy.optimize import minimize
        try:
            res_nm = minimize(
                lambda p: -_score(p),
                best_params,
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000}
            )
            if -res_nm.fun > best_score:
                best_score = -res_nm.fun
                best_params = res_nm.x
        except Exception as e:
            pass
    
    # Multi-start local search if DE failed or to ensure robustness
    # Let's add a few manual starts just in case
    starts = [
        [0.5, 0.0],
        [0.7, np.pi/7],
        [0.3, np.pi/7],
        [0.8, 0.0],
        [0.6, 2*np.pi/7]
    ]
    
    for start in starts:
        if time.time() > deadline:
            break
        try:
            res = minimize(
                lambda p: -_score(p),
                start,
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 500}
            )
            if -res.fun > best_score:
                best_score = -res.fun
                best_params = res.x
        except Exception:
            pass
    
    if best_params is None:
        # Fallback to random
        return np.random.rand(N, 2)
        
    points = _get_points(best_params[0], best_params[1])
    return points

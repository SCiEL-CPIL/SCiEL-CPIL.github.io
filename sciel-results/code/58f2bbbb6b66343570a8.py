import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 14

# Pre-compute indices for all triangles i < j < k
TRI_INDICES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])

def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area using pre-computed indices."""
    if len(points) < 3:
        return 0.0
    i, j, k = TRI_INDICES.T
    pi = points[i]
    pj = points[j]
    pk = points[k]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _convex_hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-9:
        return 0.0
    min_area = _min_triangle_area_vec(points)
    return min_area / hull_area

def _two_heptagons(r, theta):
    """
    Generate 14 points: 7 on outer unit circle, 7 on inner circle of radius r,
    with inner ring rotated by theta relative to outer ring.
    """
    n_out = 7
    n_in = 7
    angles_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    angles_in = np.linspace(0, 2 * np.pi, n_in, endpoint=False) + theta
    
    x_out = np.cos(angles_out)
    y_out = np.sin(angles_out)
    x_in = r * np.cos(angles_in)
    y_in = r * np.sin(angles_in)
    
    return np.vstack([
        np.column_stack((x_out, y_out)),
        np.column_stack((x_in, y_in))
    ])

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Initial configuration: Two concentric heptagons
    # Based on previous best practices, r ~ 0.5-0.7, theta ~ pi/7
    best_points = _two_heptagons(0.6, np.pi / 7)
    best_score = _score(best_points)
    
    # Strategy 1: Simulated Annealing
    # We will run SA to explore the neighborhood of the initial guess.
    # SA helps escape local minima that simple local search might get stuck in.
    
    current_points = best_points.copy()
    current_score = best_score
    
    # Temperature schedule
    T_start = 0.01
    T_end = 1e-6
    n_steps = 3000  # Adjust based on time budget
    
    # Estimate time per step to adjust n_steps if necessary
    # We'll just run a fixed number of steps and check time periodically.
    
    rng = np.random.default_rng(42)
    
    # Adaptive step size
    step_scale = 0.05
    
    for i in range(n_steps):
        if time.time() > deadline - 20: # Leave 20s for final polish
            break
            
        # Generate perturbation
        # Perturb all points
        noise = rng.normal(0, step_scale, size=(N, 2))
        new_points = current_points + noise
        
        # Evaluate
        new_score = _score(new_points)
        
        # Acceptance criterion
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = new_points.copy()
        else:
            # Metropolis criterion
            delta = current_score - new_score
            if delta > 0:
                T = T_start * (T_end / T_start) ** (i / n_steps)
                if T > 1e-10:
                    prob = np.exp(-delta / T)
                    if rng.random() < prob:
                        current_points = new_points
                        current_score = new_score
        
        # Adapt step size: if we've been stuck (not improving best) for a while, shrink?
        # Or if we are improving, maybe keep it.
        # Simple heuristic: shrink step size slowly
        step_scale *= 0.999
        
    # Strategy 2: Local Optimization with Powell's Method
    # Powell is robust for this kind of problem.
    # We run it from the best point found by SA.
    
    def obj(x):
        pts = x.reshape(-1, 2)
        return -_score(pts)
    
    x0 = best_points.flatten()
    
    try:
        # Powell's method
        res = minimize(
            obj, x0, method='Powell',
            options={
                'maxiter': 200, 
                'xtol': 1e-8, 
                'ftol': 1e-10
            }
        )
        
        if time.time() < deadline - 5:
            final_score = -res.fun
            if final_score > best_score:
                best_score = final_score
                best_points = res.x.reshape(-1, 2).copy()
                
    except Exception:
        pass
    
    # Fallback if something went wrong
    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points

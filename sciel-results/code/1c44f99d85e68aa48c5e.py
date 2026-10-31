"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

# The equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
H = np.sqrt(3) / 2

# Precompute triangle indices for vectorized area computation
# There are C(11, 3) = 165 triangles
_TRI_IDX = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])

def _all_areas(pts):
    """Fully vectorized computation of all triangle areas using broadcasting."""
    pi = pts[_TRI_IDX[:, 0]]
    pj = pts[_TRI_IDX[:, 1]]
    pk = pts[_TRI_IDX[:, 2]]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    areas = 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)
    return areas

def _min_area(pts):
    """Compute the minimum area of any triangle formed by three of the points."""
    return float(np.min(_all_areas(pts)))

def _clamp_to_triangle(p):
    """Clamp a point to be inside the equilateral triangle using barycentric coordinates."""
    x, y = p
    # Convert to barycentric coordinates (a, b, c)
    # p = a*V0 + b*V1 + c*V2
    # x = b + 0.5c
    # y = c * H  => c = y / H = 2y/sqrt(3)
    c = 2 * y / np.sqrt(3)
    b = x - 0.5 * c
    a = 1 - b - c
    
    eps = 1e-9
    # Clamp barycentric coordinates to [eps, 1-2*eps] to ensure strict interior
    a = max(eps, min(1 - 2*eps, a))
    b = max(eps, min(1 - 2*eps, b))
    c = max(eps, min(1 - 2*eps, c))
    
    # Renormalize
    s = a + b + c
    a /= s
    b /= s
    c /= s
    
    return a * V0 + b * V1 + c * V2

def _clamp_all(pts):
    """Clamp all points to triangle."""
    return np.array([_clamp_to_triangle(p) for p in pts])

def _generate_initial_config(seed):
    """Generate a random initial configuration inside the triangle."""
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(N):
        r1 = rng.random()
        r2 = rng.random()
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2
        p = r1 * V1 + r2 * V2
        pts.append(p)
    return np.array(pts)

def _simulated_annealing(start_points, time_limit, seed):
    """
    Perform Simulated Annealing to maximize the minimum triangle area.
    This is the primary global search strategy.
    """
    rng = np.random.default_rng(seed)
    
    points = np.array(start_points, dtype=float)
    points = _clamp_all(points)
    current_score = _min_area(points)
    
    best_points = points.copy()
    best_score = current_score
    
    # SA Parameters
    # Initial temperature should be comparable to the scale of the objective function changes
    T = 0.01 
    T_min = 1e-8
    alpha = 0.9995  # Cooling rate
    
    # Initial step size for perturbation
    # The triangle has side length 1. A step of 0.05 is reasonable for global moves.
    step_size = 0.05 
    
    start_time = time.time()
    
    # Pre-allocate for speed if possible, but simple loop is fine with vectorized eval
    while time.time() - start_time < time_limit:
        if T < T_min:
            # If we cool down too fast, reheat slightly or break? 
            # Reheating can help escape local minima.
            T = T_min * 10 
            step_size = max(0.01, step_size * 0.5)
            
        # Pick a random point to perturb
        idx = rng.integers(N)
        
        # Perturb with Gaussian noise
        noise = rng.normal(0, step_size, size=2)
        new_point = points[idx] + noise
        
        # Clamp the new point
        new_point = _clamp_to_triangle(new_point)
        
        # Update points
        old_point = points[idx].copy()
        points[idx] = new_point
        
        # Evaluate
        new_score = _min_area(points)
        
        # Acceptance criterion: Metropolis
        delta = new_score - current_score
        if delta > 0 or (T > 0 and rng.random() < np.exp(delta / T)):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        else:
            # Revert
            points[idx] = old_point
            
        # Cool down
        T *= alpha
        # Slightly reduce step size to allow fine tuning later, but keep it from vanishing
        step_size *= 0.9999
        if step_size < 0.005:
            step_size = 0.005

    return best_points, best_score

def _objective_hard(flat_pts):
    """Objective to minimize: negative min area, with penalty if outside (though we clamp)."""
    pts = flat_pts.reshape(-1, 2)
    pts = _clamp_all(pts)
    return -_min_area(pts)

def construct():
    start_time = time.time()
    # Evaluator kills at 150s. We use 140s as a hard deadline to be safe.
    deadline = start_time + 140.0
    
    def time_left():
        return deadline - time.time()
        
    best_pts = None
    best_score = -1.0
    
    # Phase 1: Global Search using Simulated Annealing
    # We run SA from a few diverse random starts.
    # SA is computationally expensive per iteration due to _min_area, but _min_area is vectorized.
    # Let's estimate iterations: 140s / ~100us per iter = ~1.4M iterations.
    # We split this among a few runs.
    
    seeds = [42, 123, 456, 789, 101]
    
    for i, seed in enumerate(seeds):
        if time_left() < 15.0: # Leave 15s for refinement
            break
            
        # Allocate time to this SA run. 
        # First run gets more time to establish a baseline.
        time_budget = 25.0
        if i == 0:
            time_budget = 40.0
            
        init_pts = _generate_initial_config(seed)
        
        try:
            sa_pts, sa_score = _simulated_annealing(init_pts, time_budget, seed=seed + 1000)
            if sa_score > best_score:
                best_score = sa_score
                best_pts = sa_pts
        except Exception as e:
            pass

    # If no points found (should not happen), use a symmetric fallback
    if best_pts is None:
        best_pts = _generate_initial_config(0)
        best_score = _min_area(best_pts)

    # Phase 2: Local Refinement
    # SA leaves points in a good basin but not optimized. 
    # Nelder-Mead is robust for non-smooth objectives like min-area.
    
    if best_pts is not None and time_left() > 5.0:
        # Run NM on the best point
        # We might run NM on a few of the top SA results if we kept them, 
        # but for simplicity and time safety, we just polish the global best.
        
        try:
            res_nm = minimize(
                _objective_hard,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 2000, 
                    'xatol': 1e-10, 
                    'fatol': 1e-12, 
                    'adaptive': True
                }
            )
            nm_pts = _clamp_all(res_nm.x.reshape(-1, 2))
            nm_score = _min_area(nm_pts)
            if nm_score > best_score:
                best_score = nm_score
                best_pts = nm_pts
        except Exception:
            pass

    # Phase 3: Final Polish / Verification
    # Sometimes NM gets stuck in a slightly suboptimal spot due to numerical noise.
    # A very short, tight NM run can help.
    if best_pts is not None and time_left() > 2.0:
        try:
            res_final = minimize(
                _objective_hard,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 500, 
                    'xatol': 1e-12, 
                    'fatol': 1e-14, 
                    'adaptive': True
                }
            )
            final_pts = _clamp_all(res_final.x.reshape(-1, 2))
            final_score = _min_area(final_pts)
            if final_score > best_score:
                best_score = final_score
                best_pts = final_pts
        except Exception:
            pass

    return _clamp_all(best_pts)

# EVOLVE-BLOCK-END

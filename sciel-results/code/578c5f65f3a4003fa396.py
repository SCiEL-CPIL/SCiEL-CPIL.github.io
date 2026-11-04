"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import itertools

N = 13
TIME_LIMIT = 140  # seconds, safety margin before 150s kill

# Precompute triangle indices for vectorized area computation
def _precompute_triangles(n):
    """Precompute all (i,j,k) triangle index triples."""
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=np.int32)

TRIANGLE_INDICES = _precompute_triangles(N)

def _compute_hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _triangle_areas(points):
    """Vectorized computation of all triangle areas."""
    n = len(points)
    idx = TRIANGLE_INDICES
    pi = points[idx[:, 0]]
    pj = points[idx[:, 1]]
    pk = points[idx[:, 2]]
    # Area = 0.5 * |cross(pj-pi, pk-pi)|
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return 0.5 * np.abs(cross)

def _compute_min_ratio(points):
    """Compute min triangle area / convex hull area."""
    points = np.asarray(points, dtype=np.float64)
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas(points)
    return np.min(areas) / hull_area

def _smooth_min(values, k=50.0):
    """Smooth minimum using log-sum-exp trick."""
    # -log-sum-exp(-k * values) / k approximates min(values)
    scaled = -k * values
    max_scaled = np.max(scaled)
    if max_scaled < -1e10:
        return np.min(values)
    lse = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return -lse / k

def _objective_from_params(params):
    """Objective function for optimization: minimize negative score.
    
    params: array of shape (N,) representing radial distances.
    Angles are fixed at uniform spacing.
    """
    radii = np.asarray(params, dtype=np.float64)
    # Ensure positive radii
    radii = np.maximum(radii, 0.01)
    
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-12:
        return 1e10
    
    areas = _triangle_areas(points)
    # Use smooth min for differentiability
    sm_min = _smooth_min(areas, k=50.0)
    
    return -(sm_min / hull_area)

def _exact_objective(params):
    """Exact objective for final evaluation."""
    radii = np.asarray(params, dtype=np.float64)
    radii = np.maximum(radii, 0.01)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return -_compute_min_ratio(points)

def _transform_to_fixed(params):
    """Transform radial params to Cartesian points."""
    radii = np.asarray(params, dtype=np.float64)
    radii = np.maximum(radii, 0.01)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _generate_initial_params(rng, n=N):
    """Generate initial radial parameters."""
    # Regular polygon
    if rng.random() < 0.3:
        return np.ones(n)
    # Slightly perturbed
    elif rng.random() < 0.5:
        return 1.0 + rng.normal(0, 0.05, n)
    # More varied
    elif rng.random() < 0.7:
        return 1.0 + rng.normal(0, 0.1, n)
    # Random convex-ish
    else:
        return rng.uniform(0.5, 1.2, n)

def _local_optimize_radii(radii0, time_budget, rng):
    """Local optimization of radial parameters using L-BFGS-B."""
    start_time = time.time()
    
    def obj(x):
        return _objective_from_params(x)
    
    def grad(x):
        # Numerical gradient
        eps = 1e-7
        g = np.zeros_like(x)
        for i in range(len(x)):
            xp = x.copy()
            xp[i] += eps
            xm = x.copy()
            xm[i] -= eps
            g[i] = (obj(xp) - obj(xm)) / (2 * eps)
        return g
    
    # Bounds: radii in [0.01, 2.0]
    bounds = [(0.01, 2.0)] * N
    
    result = minimize(
        obj, radii0, method='L-BFGS-B',
        jac=grad, bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 20000}
    )
    
    elapsed = time.time() - start_time
    best_radii = result.x
    best_score = -_exact_objective(best_radii)
    
    return best_radii, best_score, elapsed

def construct():
    start_time = time.time()
    rng = np.random.default_rng(N)
    
    best_radii = None
    best_score = -float('inf')
    
    # Phase 1: Differential evolution for global search
    de_time_budget = min(40, TIME_LIMIT - 60)
    
    def de_obj(x):
        if time.time() - start_time > TIME_LIMIT - 20:
            return 1e10
        return _objective_from_params(x)
    
    try:
        de_result = differential_evolution(
            de_obj,
            bounds=[(0.01, 2.0)] * N,
            seed=N,
            maxiter=30,
            tol=1e-6,
            popsize=15,
            polish=True,
            init='sobol',
            workers=-1
        )
        if de_result.fun < 1e9:
            de_radii = de_result.x
            de_score = -_exact_objective(de_radii)
            if de_score > best_score:
                best_score = de_score
                best_radii = de_radii.copy()
    except Exception:
        pass
    
    # Phase 2: Local optimization from multiple starting points
    n_starts = 12
    for i in range(n_starts):
        if time.time() - start_time > TIME_LIMIT - 15:
            break
        
        # Generate initial radii
        if i == 0:
            radii0 = np.ones(N)
        elif i == 1:
            radii0 = 1.0 + rng.normal(0, 0.05, N)
        elif i == 2:
            radii0 = 1.0 + rng.normal(0, 0.1, N)
        elif i == 3:
            radii0 = rng.uniform(0.6, 1.1, N)
        elif i == 4:
            # Alternating pattern
            radii0 = np.where(np.arange(N) % 2 == 0, 1.0, 0.92)
        elif i == 5:
            radii0 = np.where(np.arange(N) % 3 == 0, 1.05, 0.95)
        elif i == 6 and best_radii is not None:
            radii0 = best_radii + rng.normal(0, 0.02, N)
        elif i == 7:
            radii0 = 1.0 + 0.08 * np.sin(2 * np.pi * np.arange(N) / N)
        elif i == 8:
            radii0 = 1.0 + 0.06 * np.cos(2 * np.pi * np.arange(N) / N)
        elif i == 9:
            radii0 = rng.uniform(0.7, 1.0, N)
        elif i == 10:
            radii0 = 1.0 + rng.normal(0, 0.03, N)
        else:
            radii0 = rng.uniform(0.8, 1.1, N)
        
        # Clamp to valid range
        radii0 = np.clip(radii0, 0.01, 2.0)
        
        try:
            opt_radii, opt_score, elapsed = _local_optimize_radii(radii0, 10, rng)
            if opt_score > best_score:
                best_score = opt_score
                best_radii = opt_radii.copy()
        except Exception:
            pass
    
    # Phase 3: Final polish with more iterations on best solution
    if best_radii is not None and time.time() - start_time < TIME_LIMIT - 10:
        try:
            def obj(x):
                return _objective_from_params(x)
            
            def grad(x):
                eps = 1e-8
                g = np.zeros_like(x)
                for i in range(len(x)):
                    xp = x.copy()
                    xp[i] += eps
                    xm = x.copy()
                    xm[i] -= eps
                    g[i] = (obj(xp) - obj(xm)) / (2 * eps)
                return g
            
            bounds = [(0.01, 2.0)] * N
            result = minimize(
                obj, best_radii, method='L-BFGS-B',
                jac=grad, bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 50000}
            )
            polish_score = -_exact_objective(result.x)
            if polish_score > best_score:
                best_score = polish_score
                best_radii = result.x.copy()
        except Exception:
            pass
    
    # Convert to Cartesian points
    if best_radii is None:
        best_radii = np.ones(N)
    
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    best_points = np.column_stack([best_radii * np.cos(angles), best_radii * np.sin(angles)])
    
    return best_points

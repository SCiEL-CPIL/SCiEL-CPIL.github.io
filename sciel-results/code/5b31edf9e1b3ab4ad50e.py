"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

def _triangle_area(p, i, j, k):
    """Return 2*area of triangle (p[i], p[j], p[k]) (unsigned)."""
    return abs(np.cross(p[j] - p[i], p[k] - p[i]))

def _min_triangle_area(p):
    """Return minimum triangle area over all triples."""
    n = len(p)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(p, i, j, k)
                if a < min_area:
                    min_area = a
    return min_area

def _convex_hull_area(p):
    """Return area of the convex hull of point set p."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(p)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: use shapely if available, else approximate
        # For robustness, if hull fails (e.g., collinear points), return a small value
        return 1e-10

def _score(p):
    """Return the score: min_triangle_area / convex_hull_area."""
    hull_area = _convex_hull_area(p)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(p)
    return min_area / hull_area

def _objective(x, T=0.01):
    """
    Smooth lower bound on the minimum triangle area using log-sum-exp.
    We want to maximize the min area, so we minimize -logsumexp(-areas/T).
    However, logsumexp of areas approximates the max. To approximate min,
    we can use -logsumexp(-areas/T).
    
    Let A be the vector of all triangle areas (2*area).
    min(A) ≈ -T * log(sum(exp(-A/T)))
    So we want to maximize this, i.e., minimize -min(A) ≈ T * log(sum(exp(-A/T))).
    
    But we also need to normalize by hull area. The hull area is not smooth.
    Instead, we can fix the hull area by constraining the points to lie in a unit area convex hull,
    or we can just maximize the raw min area and then normalize at the end.
    Since the problem is scale-invariant (score is ratio), we can fix the scale.
    Let's fix the convex hull area to be 1.0 by scaling the points after optimization,
    or by constraining the optimization to a specific domain.
    
    A simpler approach: Optimize the points to maximize the minimum triangle area,
    subject to the convex hull area being 1.
    This is a non-convex constraint.
    
    Alternative: Use the fact that the score is scale-invariant.
    We can optimize the shape of the point set, and then scale it so the hull area is 1.
    To make the optimization well-behaved, we can constrain the points to lie within a unit square or circle,
    and then normalize.
    
    Let's try: Maximize the log-sum-exp approximation of the min triangle area,
    with points constrained to a unit square [0,1]x[0,1].
    Then, at the end, compute the actual score by normalizing by the hull area.
    This might not be optimal because the optimal shape might not fit in a square,
    but it's a reasonable starting point.
    
    Even better: Use a smooth penalty for points leaving a convex region, or just use the square.
    
    Let's define the objective as:
    f(x) = T * log(sum(exp(-A_i / T)))
    where A_i are the 2*areas of all triangles.
    We minimize f(x).
    """
    p = x.reshape(-1, 2)
    n = len(p)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = _triangle_area(p, i, j, k)
                areas.append(a)
    areas = np.array(areas)
    
    # Avoid numerical issues with very small areas
    areas = np.maximum(areas, 1e-10)
    
    # logsumexp approximation of min
    # min(A) ≈ -T * log(sum(exp(-A/T)))
    # So we want to maximize min(A), which is equivalent to minimizing -min(A).
    # -min(A) ≈ T * log(sum(exp(-A/T)))
    log_sum_exp = T * np.log(np.sum(np.exp(-areas / T)))
    return log_sum_exp

def _constraints(x):
    """
    Constraints: Points must be in [0, 1] x [0, 1].
    This is handled by bounds in L-BFGS-B.
    """
    return []

def _bounds():
    """
    Bounds: Each coordinate in [0, 1].
    """
    return [(0.0, 1.0)] * (2 * N)

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Leave 20s margin for the 120s limit
    
    best_score = 0.0
    best_points = None
    
    # Multiple random initializations
    n_restarts = 10
    rng = np.random.default_rng(N)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Random initialization in [0, 1] x [0, 1]
        x0 = rng.uniform(0.1, 0.9, size=(2 * N,))
        
        # L-BFGS-B optimization
        try:
            result = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                bounds=_bounds(),
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success:
                p_opt = result.x.reshape(-1, 2)
                score = _score(p_opt)
                
                if score > best_score:
                    best_score = score
                    best_points = p_opt.copy()
        except Exception:
            continue
    
    # If no valid points found, return a default configuration
    if best_points is None:
        # Regular 14-gon
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([0.5 + 0.4 * np.cos(angles), 0.5 + 0.4 * np.sin(angles)])
    
    return best_points
# EVOLVE-BLOCK-END

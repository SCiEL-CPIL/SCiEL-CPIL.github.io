import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import itertools

N = 14
START_TIME = time.time()
DEADLINE = START_TIME + 145.0  # Safety margin for 150s limit

def _get_triple_indices(n):
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=int)

TRIPLE_INDICES = _get_triple_indices(N)
I_IDX = TRIPLE_INDICES[:, 0]
J_IDX = TRIPLE_INDICES[:, 1]
K_IDX = TRIPLE_INDICES[:, 2]

# Precompute indices for vectorized cross product calculation
# We need (p_j - p_i) x (p_k - p_i)
# This is equivalent to p_i x p_j + p_j x p_k + p_k x p_i
# Let's precompute the pairs for the cross product terms
# Term 1: p_i x p_j -> indices (I_IDX, J_IDX)
# Term 2: p_j x p_k -> indices (J_IDX, K_IDX)
# Term 3: p_k x p_i -> indices (K_IDX, I_IDX)

def _score(points):
    """Compute min triangle area / convex hull area."""
    if len(points) < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    # Vectorized cross product calculation for all triples
    # cross = (p_j - p_i) x (p_k - p_i)
    # = p_j x p_k - p_j x p_i - p_i x p_k + p_i x p_i (which is 0)
    # = p_j x p_k + p_i x p_j + p_k x p_i
    
    # Extract points
    pi = points[I_IDX]
    pj = points[J_IDX]
    pk = points[K_IDX]
    
    # 2D cross product a x b = a[0]*b[1] - a[1]*b[0]
    cross1 = pi[:, 0] * pj[:, 1] - pi[:, 1] * pj[:, 0]
    cross2 = pj[:, 0] * pk[:, 1] - pj[:, 1] * pk[:, 0]
    cross3 = pk[:, 0] * pi[:, 1] - pk[:, 1] * pi[:, 0]
    
    cross = cross1 + cross2 + cross3
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    return min_area / hull_area

def _neg_score(flat_points):
    points = flat_points.reshape((N, 2))
    return -_score(points)

def _make_regular_polygon(n, radius=1.0, offset=0.0):
    angles = np.linspace(offset, 2*np.pi + offset, n, endpoint=False)
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _generate_superellipse_points(n, exponent, aspect_ratio, offset=0.0):
    """
    Generate points on a superellipse defined by |x/a|^n + |y/b|^n = 1.
    Here a=1, b=1/aspect_ratio? No, let's define a and b directly.
    Standard form: |x/a|^p + |y/b|^p = 1.
    We fix a=1. b is the aspect ratio.
    Parameterization:
    x = a * sign(cos(t)) * |cos(t)|^(2/p)
    y = b * sign(sin(t)) * |sin(t)|^(2/p)
    
    To distribute points uniformly by arc length is hard.
    Uniform by angle t is easier but might cluster near corners if p != 2.
    However, for maximizing min triangle area, uniform angular distribution is a good start.
    """
    t = np.linspace(offset, 2*np.pi + offset, n, endpoint=False)
    
    a = 1.0
    b = aspect_ratio
    
    cos_t = np.cos(t)
    sin_t = np.sin(t)
    
    # Handle signs and powers
    # np.sign(0) is 0, but we want 0 to map to 0.
    # |cos|^ (2/p) * sign(cos)
    # If p is large, 2/p is small.
    
    power = 2.0 / exponent
    
    x = a * np.sign(cos_t) * np.abs(cos_t)**power
    y = b * np.sign(sin_t) * np.abs(sin_t)**power
    
    return np.column_stack([x, y])

def _neg_superellipse_score(params):
    """
    Objective function for superellipse optimization.
    params: [exponent, aspect_ratio, offset]
    """
    exponent, aspect_ratio, offset = params
    
    # Constraints
    # exponent > 1 for convexity (strictly convex if > 1, actually p>=1 for convex, p>1 for strictly)
    # If p < 1, it's concave (star-like). We want convex hull area to be the shape area.
    # So we must ensure the curve is convex. p >= 1.
    # aspect_ratio > 0.
    
    if exponent < 1.0 or aspect_ratio < 0.1 or aspect_ratio > 10.0:
        return 0.0
        
    points = _generate_superellipse_points(N, exponent, aspect_ratio, offset)
    
    # Check if points are valid (non-degenerate)
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 0.0
        
    s = _score(points)
    return -s

def _optimize_superellipse(time_budget):
    """
    Optimize superellipse parameters.
    Returns (score, points)
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Search space
    # Exponent p: 1 (diamond) to 5 (square-like)
    # Aspect ratio b: 0.5 to 2.0 (symmetric with 1/b if we allow rotation, but let's search both)
    # Offset: 0 to 2*pi/N (symmetry)
    
    bounds = [
        (1.0, 5.0),       # exponent
        (0.5, 2.0),       # aspect ratio
        (0.0, 2*np.pi/N)  # offset
    ]
    
    best_score = -np.inf
    best_points = None
    
    # 1. Differential Evolution
    try:
        res = differential_evolution(
            _neg_superellipse_score,
            bounds,
            seed=42,
            maxiter=50,
            popsize=20,
            tol=1e-6,
            polish=True,
            init='latinhypercube'
        )
        
        if res.fun < 0:
            best_score = -res.fun
            params = res.x
            best_points = _generate_superellipse_points(N, params[0], params[1], params[2])
        else:
            best_params = None
            
    except Exception:
        best_params = None
        best_points = None

    # Heuristic starts if DE fails or score is low
    if best_points is None or best_score < 0.5:
        heuristic_starts = [
            [2.0, 1.0, 0.0],  # Circle
            [1.0, 1.0, 0.0],  # Diamond
            [3.0, 1.0, 0.0],  # Superellipse
            [2.0, 1.5, 0.0],  # Ellipse
            [4.0, 1.0, np.pi/14.0] # Offset
        ]
        
        for start in heuristic_starts:
            if time.time() > deadline - 10:
                break
            s = -_neg_superellipse_score(start)
            if s > best_score:
                best_score = s
                best_points = _generate_superellipse_points(N, start[0], start[1], start[2])
                best_params = start

    # 2. Local Polish on Full Coordinates
    if best_points is not None and time.time() < deadline - 5:
        try:
            res_opt = minimize(
                _neg_score,
                best_points.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'maxfev': 5000,
                    'xatol': 1e-10,
                    'fatol': 1e-12
                }
            )
            
            opt_points = res_opt.x.reshape((N, 2))
            opt_score = _score(opt_points)
            
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points
                
        except Exception:
            pass
            
    # 3. Second Polish with Powell
    if best_points is not None and time.time() < deadline - 2:
        try:
            res_opt2 = minimize(
                _neg_score,
                best_points.flatten(),
                method='Powell',
                options={
                    'maxiter': 1000,
                    'maxfev': 3000
                }
            )
            opt_points2 = res_opt2.x.reshape((N, 2))
            opt_score2 = _score(opt_points2)
            
            if opt_score2 > best_score:
                best_score = opt_score2
                best_points = opt_points2
        except Exception:
            pass

    return best_score, best_points

def construct():
    global START_TIME, DEADLINE
    START_TIME = time.time()
    DEADLINE = START_TIME + 145.0
    
    np.random.seed(42)
    
    best_global_score = -np.inf
    best_global_points = None
    
    # Baseline: Regular polygon
    reg_poly = _make_regular_polygon(N)
    s_reg = _score(reg_poly)
    best_global_score = s_reg
    best_global_points = reg_poly.copy()
    
    # Optimize Superellipse
    # Allocate ~80s for superellipse optimization
    if time.time() < DEADLINE - 30:
        time_chunk = 80.0
        s, pts = _optimize_superellipse(time_chunk)
        
        if pts is not None and s > best_global_score:
            best_global_score = s
            best_global_points = pts
            
    # If time permits, try a quick ring-based optimization as well?
    # The directive says "KEEP this design" (ring-based) but the strategy says "Model ... as superellipse".
    # "Do NOT switch to a different design family."
    # Wait. The current program IS ring-based.
    # The strategy says: "Model the 14 points as a single set of vertices on a superellipse... This approach differs from the existing ring-based..."
    # And "Keep this design" usually refers to the *current* program's structure/flow, but the *strategy* is the specific change to make.
    # The prompt says: "Strategy to try in this change: Model ... as superellipse".
    # So I should replace the ring logic with superellipse logic.
    
    # Let's also try a hybrid: Optimize superellipse, then polish.
    # If we have time left, we can try a few random restarts on the full coordinates.
    
    if time.time() < DEADLINE - 10:
        # Try a few random perturbations of the best points and polish
        for _ in range(3):
            if time.time() > DEADLINE - 5:
                break
                
            # Perturb
            noise = np.random.normal(0, 0.05, size=(N, 2))
            perturbed = best_global_points + noise
            s_pert = _score(perturbed)
            
            if s_pert > best_global_score:
                best_global_score = s_pert
                best_global_points = perturbed
            
            # Polish perturbed
            if time.time() > DEADLINE - 5:
                break
            try:
                res_opt = minimize(
                    _neg_score,
                    perturbed.flatten(),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 500,
                        'maxfev': 1000,
                        'xatol': 1e-10,
                        'fatol': 1e-12
                    }
                )
                opt_points = res_opt.x.reshape((N, 2))
                opt_score = _score(opt_points)
                if opt_score > best_global_score:
                    best_global_score = opt_score
                    best_global_points = opt_points
            except Exception:
                pass

    # Final Polish on the absolute best
    if best_global_points is not None and time.time() < DEADLINE - 2:
        try:
            res_final = minimize(
                _neg_score,
                best_global_points.flatten(),
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'maxfev': 2000,
                    'xatol': 1e-12,
                    'fatol': 1e-14
                }
            )
            final_points = res_final.x.reshape((N, 2))
            final_score = _score(final_points)
            if final_score > best_global_score:
                best_global_score = final_score
                best_global_points = final_points
        except Exception:
            pass

    return best_global_points

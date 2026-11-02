"""Heilbronn 11 triangles: Barycentric soft-min optimization with DE exploration."""
import numpy as np
import time
from itertools import combinations
from scipy.optimize import differential_evolution, minimize

N = 11
S3H = np.sqrt(3) / 2
TRI_AREA = S3H / 2.0
DEADLINE = 145  # seconds, safety margin from 150s limit

# Precompute all C(11,3) = 165 triples
_TRIPLES = list(combinations(range(N), 3))
_TRIPLES_ARR = np.array(_TRIPLES, dtype=np.int64)  # shape (165, 3)


def _bary_to_cart(bary):
    """Convert barycentric coordinates to Cartesian.
    bary has shape (N, 3) with columns (alpha, beta, gamma).
    Vertex 0 = (0,0), Vertex 1 = (1,0), Vertex 2 = (0.5, S3H).
    """
    cart = np.zeros((bary.shape[0], 2))
    cart[:, 0] = bary[:, 1] + 0.5 * bary[:, 2]
    cart[:, 1] = S3H * bary[:, 2]
    return cart


def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 of the N points.
    points: (N, 2) array in Cartesian coordinates.
    Returns the raw (unnormalized) minimum area.
    """
    n = points.shape[0]
    # Area of triangle (i,j,k) = 0.5 * |cross(p_j - p_i, p_k - p_i)|
    areas = np.zeros(len(_TRIPLES_ARR))
    for idx, (i, j, k) in enumerate(_TRIPLES):
        dx1 = points[j, 0] - points[i, 0]
        dy1 = points[j, 1] - points[i, 1]
        dx2 = points[k, 0] - points[i, 0]
        dy2 = points[k, 1] - points[i, 1]
        areas[idx] = 0.5 * abs(dx1 * dy2 - dy1 * dx2)
    return np.min(areas)


def _soft_min_area(points, temperature):
    """Smooth approximation of the minimum area using log-sum-exp.
    Lower temperature -> closer to true min.
    points: (N, 2) Cartesian.
    """
    n = points.shape[0]
    areas = np.zeros(len(_TRIPLES_ARR))
    for idx, (i, j, k) in enumerate(_TRIPLES):
        dx1 = points[j, 0] - points[i, 0]
        dy1 = points[j, 1] - points[i, 1]
        dx2 = points[k, 0] - points[i, 0]
        dy2 = points[k, 1] - points[i, 1]
        areas[idx] = 0.5 * abs(dx1 * dy2 - dy1 * dx2)
    
    # Log-sum-exp for soft min: -T * log(sum(exp(-areas/T)))
    # For numerical stability, subtract max
    scaled = -areas / temperature
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return -temperature * log_sum


def _project_to_triangle(bary):
    """Project barycentric coordinates to the valid simplex.
    bary: (N, 3) or (3,) array.
    Ensures non-negative and sums to 1.
    """
    bary = bary.copy()
    # Clamp negatives to 0
    bary[bary < 0] = 0
    # Normalize to sum to 1
    sums = bary.sum(axis=-1, keepdims=True)
    sums[sums == 0] = 1.0
    bary = bary / sums
    return bary


def _bary_objective(x_flat, temperature):
    """Objective in barycentric space. Minimize negative soft-min area.
    x_flat: 1D array of length N*3 (flattened barycentric coords).
    Returns a scalar to minimize.
    """
    bary = x_flat.reshape(N, 3)
    bary = _project_to_triangle(bary)
    cart = _bary_to_cart(bary)
    soft_min = _soft_min_area(cart, temperature)
    return -soft_min


def _bary_hard_objective(x_flat):
    """Hard objective: minimize negative true min area in barycentric space."""
    bary = x_flat.reshape(N, 3)
    bary = _project_to_triangle(bary)
    cart = _bary_to_cart(bary)
    min_area = _min_triangle_area(cart)
    return -min_area


def _penalized_hard_min(x_flat):
    """For DE: minimize negative min area with penalty for invalid barycentric coords."""
    bary = x_flat.reshape(N, 3)
    # Penalty for negative barycentric coords
    neg = np.maximum(0, -bary)
    penalty = 100.0 * np.sum(neg ** 2)
    # Also penalize if sum deviates from 1
    sums = bary.sum(axis=1)
    sum_penalty = 10.0 * np.sum((sums - 1.0) ** 2)
    
    bary_clamped = _project_to_triangle(bary)
    cart = _bary_to_cart(bary_clamped)
    min_area = _min_triangle_area(cart)
    
    return -min_area + penalty + sum_penalty


def _annealed_optimize(bary_init, start_time, deadline_margin=8):
    """Annealed soft-min optimization using L-BFGS-B.
    Start with high temperature (smooth), gradually reduce.
    """
    best_bary = bary_init.copy()
    best_score = _min_triangle_area(_bary_to_cart(_project_to_triangle(bary_init)))
    
    # Temperature schedule: start high, reduce geometrically
    temperatures = [0.05, 0.02, 0.008, 0.003, 0.001, 0.0004]
    
    current_bary = bary_init.copy()
    
    for temp in temperatures:
        if time.time() - start_time > DEADLINE - deadline_margin:
            break
        
        try:
            res = minimize(
                _bary_objective,
                current_bary.flatten(),
                args=(temp,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 5000}
            )
            if time.time() - start_time > DEADLINE - deadline_margin:
                break
            new_bary = _project_to_triangle(res.x.reshape(N, 3))
            new_cart = _bary_to_cart(new_bary)
            new_score = _min_triangle_area(new_cart)
            if new_score > best_score:
                best_score = new_score
                best_bary = new_bary
            current_bary = new_bary
        except Exception:
            pass
    
    # Final hard refinement with Powell
    if time.time() - start_time < DEADLINE - deadline_margin:
        try:
            res = minimize(
                _bary_hard_objective,
                best_bary.flatten(),
                method='Powell',
                options={'maxiter': 200, 'maxfev': 2000, 'ftol': 1e-14, 'xtol': 1e-12}
            )
            if time.time() - start_time > DEADLINE - deadline_margin:
                pass
            else:
                new_bary = _project_to_triangle(res.x.reshape(N, 3))
                new_cart = _bary_to_cart(new_bary)
                new_score = _min_triangle_area(new_cart)
                if new_score > best_score:
                    best_score = new_score
                    best_bary = new_bary
        except Exception:
            pass
    
    return best_bary, best_score


def _random_barycentric(rng):
    """Generate a random valid barycentric configuration."""
    bary = np.zeros((N, 3))
    for i in range(N):
        # Random point in simplex
        u = rng.random(2)
        u.sort()
        bary[i] = [u[0], u[1] - u[0], 1.0 - u[1]]
    return bary


def _symmetric_init():
    """Create a symmetric initial configuration.
    Place points at vertices, edge midpoints, and center-ish positions.
    """
    bary = np.zeros((N, 3))
    # 3 vertices
    bary[0] = [1.0, 0.0, 0.0]
    bary[1] = [0.0, 1.0, 0.0]
    bary[2] = [0.0, 0.0, 1.0]
    # 3 edge midpoints
    bary[3] = [0.5, 0.5, 0.0]
    bary[4] = [0.5, 0.0, 0.5]
    bary[5] = [0.0, 0.5, 0.5]
    # Center
    bary[6] = [1.0/3, 1.0/3, 1.0/3]
    # 4 more points: perturb center slightly and add near edges
    bary[7] = [0.3, 0.3, 0.4]
    bary[8] = [0.4, 0.3, 0.3]
    bary[9] = [0.3, 0.4, 0.3]
    bary[10] = [0.35, 0.35, 0.3]
    return bary


def construct():
    start_time = time.time()
    rng = np.random.RandomState(123)
    
    best_bary = None
    best_score = -float('inf')
    
    # Phase 1: Differential Evolution in barycentric space
    # Bounded: each barycentric coord in [0, 1]
    bounds = [(0.0, 1.0)] * (N * 3)
    
    de_time_limit = 50.0
    
    def de_callback(xk, convergence=None):
        return time.time() - start_time > start_time + de_time_limit
    
    try:
        result = differential_evolution(
            _penalized_hard_min,
            bounds,
            maxiter=80,
            popsize=12,
            tol=1e-5,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            callback=de_callback,
            polish=False,
            init='latinhypercube'
        )
        
        if time.time() - start_time < DEADLINE - 15:
            bary_de = _project_to_triangle(result.x.reshape(N, 3))
            cart_de = _bary_to_cart(bary_de)
            score_de = _min_triangle_area(cart_de)
            if score_de > best_score:
                best_score = score_de
                best_bary = bary_de.copy()
    except Exception:
        pass
    
    # Phase 2: Annealed optimization on DE result
    if best_bary is not None:
        if time.time() - start_time < DEADLINE - 10:
            best_bary, best_score = _annealed_optimize(best_bary, start_time, deadline_margin=10)
    
    # Phase 3: Try symmetric init + annealing
    if time.time() - start_time < DEADLINE - 12:
        bary_sym = _symmetric_init()
        bary_sym, score_sym = _annealed_optimize(bary_sym, start_time, deadline_margin=10)
        if score_sym > best_score:
            best_score = score_sym
            best_bary = bary_sym
    
    # Phase 4: Random restarts with annealing
    n_restarts = 8
    for r in range(n_restarts):
        if time.time() - start_time > DEADLINE - 15:
            break
        bary_rand = _random_barycentric(rng)
        bary_rand, score_rand = _annealed_optimize(bary_rand, start_time, deadline_margin=10)
        if score_rand > best_score:
            best_score = score_rand
            best_bary = bary_rand
    
    # Fallback
    if best_bary is None:
        best_bary = _symmetric_init()
        best_score = _min_triangle_area(_bary_to_cart(best_bary))
    
    # Convert to Cartesian and return
    cart = _bary_to_cart(best_bary)
    
    # Final safety check: ensure points are inside the triangle
    for i in range(N):
        x, y = cart[i]
        if y < 0:
            cart[i, 1] = 1e-9
        if y > S3H:
            cart[i, 1] = S3H - 1e-9
        if x < 0:
            cart[i, 0] = 1e-9
        if x > 1:
            cart[i, 0] = 1.0 - 1e-9
        # Check slant constraints
        if y > S3H * x + 1e-9:
            # Project onto the line y = S3H * x
            cart[i, 1] = S3H * cart[i, 0] - 1e-9
        if y > S3H * (1 - x) + 1e-9:
            cart[i, 1] = S3H * (1 - cart[i, 0]) - 1e-9
    
    return cart

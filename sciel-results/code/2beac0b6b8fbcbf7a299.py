"""Heilbronn 11-point problem: maximize minimum triangle area in unit equilateral triangle."""
import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

DEADLINE_MARGIN = 140.0

# Precompute all triplets
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
NUM_TRIP = len(_TRIPLETS)  # 165

# Triangle vertices
_VERTS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, S3H]])


def _bary_to_cart(bary):
    """Convert barycentric (N,3) to Cartesian (N,2)."""
    return bary[:, 0:1] * _VERTS[0] + bary[:, 1:2] * _VERTS[1] + bary[:, 2:3] * _VERTS[2]


def _cart_to_bary(cart):
    """Convert Cartesian (N,2) to barycentric (N,3)."""
    x = cart[:, 0]
    y = cart[:, 1]
    b2 = y / S3
    b1 = x - 0.5 * b2
    b0 = 1.0 - b1 - b2
    return np.column_stack([b0, b1, b2])


def _min_area_cart(points):
    """Minimum area of all triangles formed by points (Cartesian)."""
    p1 = points[_T_I]
    p2 = points[_T_J]
    p3 = points[_T_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _all_areas_cart(points):
    """All triangle areas (Cartesian)."""
    p1 = points[_T_I]
    p2 = points[_T_J]
    p3 = points[_T_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _soft_min_cart(points, alpha):
    """Soft minimum of triangle areas via log-sum-exp."""
    areas = _all_areas_cart(points)
    min_a = np.min(areas)
    return min_a - (1.0 / alpha) * np.log(np.sum(np.exp(-alpha * (areas - min_a))))


def _project_to_triangle(points):
    """Project points into the equilateral triangle."""
    x = points[:, 0].copy()
    y = points[:, 1].copy()
    for _ in range(3):
        y = np.maximum(0.0, y)
        y = np.minimum(y, S3 * x)
        y = np.minimum(y, S3 - S3 * x)
        x = np.maximum(0.0, x)
        x = np.minimum(1.0, x)
        y = np.maximum(0.0, y)
        y = np.minimum(y, S3 * x)
        y = np.minimum(y, S3 - S3 * x)
    return np.column_stack([x, y])


def _is_valid(points, tol=1e-8):
    x = points[:, 0]
    y = points[:, 1]
    return (np.all(y >= -tol) and
            np.all(y <= S3 * x + tol) and
            np.all(S3 * x + y <= S3 + tol))


def _random_bary():
    r = np.random.random((N, 3))
    r /= r.sum(axis=1, keepdims=True)
    return r


def _bary_obj(flat, alpha):
    """Objective in barycentric space: minimize -soft_min + penalties."""
    bary = flat.reshape(N, 3)
    bary = np.maximum(0.0, bary)
    s = bary.sum(axis=1, keepdims=True)
    s = np.where(s < 1e-12, 1.0, s)
    bary = bary / s
    cart = _bary_to_cart(bary)
    cart = _project_to_triangle(cart)
    soft = _soft_min_cart(cart, alpha)
    x = cart[:, 0]
    y = cart[:, 1]
    pen = (np.sum(np.maximum(0.0, -y) ** 2) +
           np.sum(np.maximum(0.0, y - S3 * x) ** 2) +
           np.sum(np.maximum(0.0, S3 * x + y - S3) ** 2))
    return -soft + 50.0 * pen


def _cart_obj(flat, alpha):
    """Objective in Cartesian space: minimize -soft_min + boundary penalties."""
    cart = flat.reshape(N, 2)
    cart = _project_to_triangle(cart)
    soft = _soft_min_cart(cart, alpha)
    x = cart[:, 0]
    y = cart[:, 1]
    pen = (np.sum(np.maximum(0.0, -y) ** 2) +
           np.sum(np.maximum(0.0, y - S3 * x) ** 2) +
           np.sum(np.maximum(0.0, S3 * x + y - S3) ** 2))
    return -soft + 50.0 * pen


def _refine_nm(cart, alpha, maxiter=3000):
    """Nelder-Mead refinement in Cartesian space."""
    flat0 = cart.flatten()
    res = minimize(_cart_obj, flat0, method='Nelder-Mead', args=(alpha,),
                   options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8})
    new_cart = _project_to_triangle(res.x.reshape(N, 2))
    return new_cart


def _refine_bfgs(cart, alpha, maxiter=2000):
    """BFGS refinement in Cartesian space."""
    flat0 = cart.flatten()
    res = minimize(_cart_obj, flat0, method='BFGS', args=(alpha,),
                   options={'maxiter': maxiter, 'gtol': 1e-6})
    new_cart = _project_to_triangle(res.x.reshape(N, 2))
    return new_cart


def _refine_lbfgsb(cart, alpha, maxiter=2000):
    """L-BFGS-B refinement in Cartesian space."""
    flat0 = cart.flatten()
    res = minimize(_cart_obj, flat0, method='L-BFGS-B', args=(alpha,),
                   options={'maxiter': maxiter, 'gtol': 1e-6})
    new_cart = _project_to_triangle(res.x.reshape(N, 2))
    return new_cart


def _construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN

    np.random.seed(42)

    # --- Phase 1: Multi-start with annealing in barycentric space ---
    best_cart = None
    best_score = 0.0

    # Generate diverse initial configurations
    initial_configs = []

    # Random barycentric starts
    for i in range(20):
        b = _random_bary()
        c = _project_to_triangle(_bary_to_cart(b))
        initial_configs.append(c)

    # Grid-like starts (approximate lattice)
    for i in range(5):
        # Create a rough grid perturbation
        base = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, S3H],
                         [0.25, S3H / 2], [0.75, S3H / 2], [0.5, S3H],
                         [0.5, 0.0], [0.25, 0.0], [0.75, 0.0],
                         [0.375, S3H / 4], [0.625, S3H / 4]])
        noise = np.random.normal(0, 0.05, size=(N, 2))
        c = _project_to_triangle(base + noise)
        initial_configs.append(c)

    # Symmetric quasi-lattice starts
    for i in range(5):
        # 3x4 grid with perturbation
        xs = np.linspace(0.1, 0.9, 4)
        ys = np.linspace(0.05, 0.4, 3)
        pts = []
        for yy in ys:
            for xx in xs:
                pts.append([xx, yy])
        # Add corner-ish points
        pts = pts[:11]
        pts = np.array(pts)
        noise = np.random.normal(0, 0.03, size=(N, 2))
        c = _project_to_triangle(pts + noise)
        initial_configs.append(c)

    # Annealing schedule
    alphas = [5, 20, 50, 100, 300, 1000, 3000]

    for alpha in alphas:
        if time.time() > deadline:
            break

        for config in initial_configs:
            if time.time() > deadline:
                break

            # Optimize from this config
            cart = config.copy()
            bary = _cart_to_bary(cart)

            # Nelder-Mead in barycentric
            res = minimize(_bary_obj, bary.flatten(), method='Nelder-Mead',
                           args=(alpha,),
                           options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-7})
            new_bary = res.x.reshape(N, 3)
            new_bary = np.maximum(0, new_bary)
            new_bary /= new_bary.sum(axis=1, keepdims=True)
            new_cart = _project_to_triangle(_bary_to_cart(new_bary))
            score = _min_area_cart(new_cart)

            if score > best_score:
                best_score = score
                best_cart = new_cart

    # --- Phase 2: Refinement of best solution ---
    if best_cart is not None and time.time() < deadline - 20:
        # Progressive alpha refinement
        for alpha in [100, 300, 1000, 3000, 10000]:
            if time.time() > deadline - 5:
                break
            best_cart = _refine_nm(best_cart, alpha, maxiter=3000)
            s = _min_area_cart(best_cart)
            if s > best_score:
                best_score = s

    # --- Phase 3: High-precision polish ---
    if best_cart is not None and time.time() < deadline - 5:
        # Very tight NM polish
        for alpha in [5000, 20000]:
            if time.time() > deadline - 2:
                break
            best_cart = _refine_nm(best_cart, alpha, maxiter=5000)
            s = _min_area_cart(best_cart)
            if s > best_score:
                best_score = s

    # --- Phase 4: Try L-BFGS-B polish ---
    if best_cart is not None and time.time() < deadline - 3:
        best_cart = _refine_lbfgsb(best_cart, 5000, maxiter=3000)
        s = _min_area_cart(best_cart)
        if s > best_score:
            best_score = s

    # Final validity check
    if best_cart is None:
        # Fallback: return a valid configuration
        b = _random_bary()
        best_cart = _project_to_triangle(_bary_to_cart(b))

    if not _is_valid(best_cart):
        best_cart = _project_to_triangle(best_cart)

    return best_cart


# Entry point
def construct():
    return _construct()

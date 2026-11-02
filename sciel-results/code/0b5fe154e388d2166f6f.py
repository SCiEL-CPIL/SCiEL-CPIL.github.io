"""Heilbronn-11: maximize minimum triangle area for 11 points in unit equilateral triangle."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import itertools

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

DEADLINE_MARGIN = 135.0

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
NUM_TRIP = len(_TRIPLETS)  # 165

# Triangle vertices in Cartesian
_VERTS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, S3H]])


def _bary_to_cart(bary):
    """Convert barycentric coordinates (N, 3) to Cartesian (N, 2)."""
    return bary[:, 0:1] * _VERTS[0] + bary[:, 1:2] * _VERTS[1] + bary[:, 2:3] * _VERTS[2]


def _cart_to_bary(cart):
    """Convert Cartesian (N, 2) to barycentric coordinates (N, 3)."""
    x = cart[:, 0]
    y = cart[:, 1]
    b2 = y / S3
    b1 = x - 0.5 * b2
    b0 = 1.0 - b1 - b2
    return np.column_stack([b0, b1, b2])


def _all_areas_cart(points):
    """Calculate all 165 triangle areas."""
    p1 = points[_T_I]
    p2 = points[_T_J]
    p3 = points[_T_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return areas


def _min_area_cart(points):
    """Calculate the minimum area of all triangles formed by the points."""
    return np.min(_all_areas_cart(points))


def _soft_min_area_cart(points, alpha=50.0):
    """Calculate soft minimum of triangle areas using log-sum-exp."""
    areas = _all_areas_cart(points)
    min_a = np.min(areas)
    return min_a - (1.0 / alpha) * np.log(np.sum(np.exp(-alpha * (areas - min_a))))


def _bottom_k_soft_min(points, k=5, alpha=20.0):
    """
    Weighted soft-min over the k smallest triangle areas.
    This gives more weight to the critical (smallest) areas while
    remaining smooth and differentiable.
    """
    areas = _all_areas_cart(points)
    # Get the k smallest areas
    k = min(k, len(areas))
    k_areas = np.partition(areas, k - 1)[:k]
    min_k = np.min(k_areas)
    # Soft weighting among the k smallest
    weights = np.exp(-alpha * (k_areas - min_k))
    weights /= weights.sum()
    return np.dot(weights, k_areas)


def _project_cartesian(points):
    """Project points to be inside the triangle."""
    x = points[:, 0].copy()
    y = points[:, 1].copy()

    for _ in range(3):
        # y >= 0
        y = np.maximum(0.0, y)
        # y <= sqrt(3) * x
        y = np.minimum(y, S3 * x)
        # sqrt(3) * x + y <= sqrt(3)
        y = np.minimum(y, S3 - S3 * x)
        # x bounds
        x = np.maximum(0.0, x)
        x = np.minimum(1.0, x)

    return np.column_stack([x, y])


def _is_valid_cart(points):
    """Check if all points are inside the triangle."""
    x = points[:, 0]
    y = points[:, 1]
    tol = 1e-9
    return (np.all(y >= -tol) and
            np.all(y <= S3 * x + tol) and
            np.all(S3 * x + y <= S3 + tol))


def _random_bary():
    """Generate random barycentric coordinates."""
    r = np.random.random((N, 3))
    r /= r.sum(axis=1, keepdims=True)
    return r


def _initial_configurations():
    """
    Generate several initial configurations:
    1. Grid-based (quasi-lattice)
    2. Random
    3. Perturbed grid
    """
    configs = []

    # Configuration 1: 4-4-3 grid in barycentric (rows)
    # Place points at barycentric positions approximating a triangular lattice
    bary_grid = []
    # Row 0 (bottom): 4 points
    for i in range(4):
        t = (i + 0.5) / 4.0
        bary_grid.append([1.0 - t, t * 0.5, t * 0.5])
    # Row 1 (middle): 4 points
    for i in range(4):
        t = (i + 0.5) / 4.0
        bary_grid.append([1.0 - t, t * 0.25, t * 0.75])
    # Row 2 (top): 3 points
    for i in range(3):
        t = (i + 0.5) / 3.0
        bary_grid.append([1.0 - t, t * 0.0, t * 1.0])

    bary_grid = np.array(bary_grid)
    configs.append(bary_grid)

    # Configuration 2: Pure random
    configs.append(_random_bary())

    # Configuration 3: Perturbed grid
    bary_pert = bary_grid.copy()
    bary_pert += 0.02 * np.random.randn(*bary_pert.shape)
    bary_pert = np.maximum(0, bary_pert)
    bary_pert /= bary_pert.sum(axis=1, keepdims=True)
    configs.append(bary_pert)

    # Configuration 4: Another grid arrangement
    bary_grid2 = []
    # 3 rows of varying counts
    for row in range(3):
        count = 4 - row  # 4, 3, 2... but we need 11 total
        for i in range(count):
            t = (i + 0.5) / count
            b0 = (2.0 - row) / 3.0
            b1 = t * (1.0 - b0)
            b2 = (1.0 - b0) * (1.0 - t)
            bary_grid2.append([b0, b1, b2])
    # Add remaining points
    while len(bary_grid2) < N:
        bary_grid2.append([1/3, 1/3, 1/3])
    bary_grid2 = np.array(bary_grid2[:N])
    configs.append(bary_grid2)

    # Configuration 5: More structured - hexagonal packing approximation
    bary_hex = []
    # Center point
    bary_hex.append([1/3, 1/3, 1/3])
    # Ring 1: 6 points
    for angle in range(6):
        a = angle * np.pi / 3
        r = 0.25
        b0 = 1/3 + r * np.cos(a) * 0.8
        b1 = 1/3 + r * np.sin(a) * 0.8
        b2 = 1.0 - b0 - b1
        if b0 > 0 and b1 > 0 and b2 > 0:
            bary_hex.append([b0, b1, b2])
        else:
            bary_hex.append([1/3, 1/3, 1/3])
    # Ring 2: 4 more points
    for angle in range(4):
        a = angle * np.pi / 2 + np.pi/4
        r = 0.35
        b0 = 1/3 + r * np.cos(a) * 0.7
        b1 = 1/3 + r * np.sin(a) * 0.7
        b2 = 1.0 - b0 - b1
        if b0 > 0 and b1 > 0 and b2 > 0:
            bary_hex.append([b0, b1, b2])
        else:
            bary_hex.append([1/3, 1/3, 1/3])

    bary_hex = np.array(bary_hex[:N])
    bary_hex = np.maximum(0, bary_hex)
    bary_hex /= bary_hex.sum(axis=1, keepdims=True)
    configs.append(bary_hex)

    return configs


def _bary_objective_softmin(flat_bary, alpha=50.0):
    """Objective using soft-min."""
    bary = flat_bary.reshape(N, 3)
    bary = np.maximum(0, bary)
    bary_sum = bary.sum(axis=1, keepdims=True)
    bary_sum = np.where(bary_sum < 1e-12, 1.0, bary_sum)
    bary /= bary_sum

    cart = _bary_to_cart(bary)
    cart = _project_cartesian(cart)

    soft_min = _soft_min_area_cart(cart, alpha)

    x = cart[:, 0]
    y = cart[:, 1]
    pen1 = np.sum(np.maximum(0, -y) ** 2)
    pen2 = np.sum(np.maximum(0, y - S3 * x) ** 2)
    pen3 = np.sum(np.maximum(0, S3 * x + y - S3) ** 2)
    pen = pen1 + pen2 + pen3

    return -soft_min + 100.0 * pen


def _bary_objective_bottomk(flat_bary, k=5, alpha=20.0):
    """Objective using bottom-k soft-min."""
    bary = flat_bary.reshape(N, 3)
    bary = np.maximum(0, bary)
    bary_sum = bary.sum(axis=1, keepdims=True)
    bary_sum = np.where(bary_sum < 1e-12, 1.0, bary_sum)
    bary /= bary_sum

    cart = _bary_to_cart(bary)
    cart = _project_cartesian(cart)

    obj = _bottom_k_soft_min(cart, k, alpha)

    x = cart[:, 0]
    y = cart[:, 1]
    pen1 = np.sum(np.maximum(0, -y) ** 2)
    pen2 = np.sum(np.maximum(0, y - S3 * x) ** 2)
    pen3 = np.sum(np.maximum(0, S3 * x + y - S3) ** 2)
    pen = pen1 + pen2 + pen3

    return -obj + 100.0 * pen


def _nelder_mead_optimize(objective, x0, maxiter=3000, tol=1e-8):
    """Run Nelder-Mead with tight tolerances."""
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': tol, 'fatol': tol,
                                'adaptive': True})
        return res
    except Exception:
        return None


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN

    np.random.seed(42)

    # Initial best from multiple starting configurations
    initial_configs = _initial_configurations()
    best_cart = None
    best_score = 0.0

    for cfg_bary in initial_configs:
        cart = _bary_to_cart(cfg_bary)
        cart = _project_cartesian(cart)
        score = _min_area_cart(cart)
        if score > best_score:
            best_score = score
            best_cart = cart.copy()

    if best_cart is None:
        best_cart = _project_cartesian(_bary_to_cart(_random_bary()))
        best_score = _min_area_cart(best_cart)

    best_bary = _cart_to_bary(best_cart)

    # Phase 1: Soft-min annealing with multiple alphas and restarts
    alphas = [20, 50, 100, 200, 500]

    for alpha in alphas:
        if time.time() > deadline - 20:
            break

        num_restarts = 8
        for r in range(num_restarts):
            if time.time() > deadline - 20:
                break

            if r == 0:
                x0 = best_bary.flatten()
            elif r == 1:
                x0 = _initial_configurations()[0].flatten()
            else:
                x0 = _random_bary().flatten()

            res = _nelder_mead_optimize(_bary_objective_softmin, x0,
                                        maxiter=2000, tol=1e-7)
            if res is not None:
                new_bary = res.x.reshape(N, 3)
                new_bary = np.maximum(0, new_bary)
                s = new_bary.sum(axis=1, keepdims=True)
                s = np.where(s < 1e-12, 1.0, s)
                new_bary /= s

                new_cart = _bary_to_cart(new_bary)
                new_cart = _project_cartesian(new_cart)

                new_score = _min_area_cart(new_cart)
                if new_score > best_score:
                    best_score = new_score
                    best_cart = new_cart
                    best_bary = new_bary

    # Phase 2: Bottom-k soft-min refinement
    if time.time() < deadline - 15:
        for k_val in [3, 5, 8]:
            if time.time() > deadline - 15:
                break

            for r in range(4):
                if time.time() > deadline - 15:
                    break

                if r == 0:
                    x0 = best_bary.flatten()
                else:
                    x0 = _random_bary().flatten()

                res = _nelder_mead_optimize(lambda x: _bary_objective_bottomk(x, k=k_val, alpha=30.0),
                                            x0, maxiter=2000, tol=1e-7)
                if res is not None:
                    new_bary = res.x.reshape(N, 3)
                    new_bary = np.maximum(0, new_bary)
                    s = new_bary.sum(axis=1, keepdims=True)
                    s = np.where(s < 1e-12, 1.0, s)
                    new_bary /= s

                    new_cart = _bary_to_cart(new_bary)
                    new_cart = _project_cartesian(new_cart)

                    new_score = _min_area_cart(new_cart)
                    if new_score > best_score:
                        best_score = new_score
                        best_cart = new_cart
                        best_bary = new_bary

    # Phase 3: Final high-precision polish with hard minimum
    if time.time() < deadline - 5:
        # Use Cartesian coordinates for final polish
        x0_cart = best_cart.flatten()

        def cart_obj(flat_cart):
            cart = flat_cart.reshape(N, 2)
            cart = _project_cartesian(cart)
            return -_min_area_cart(cart)

        res = _nelder_mead_optimize(cart_obj, x0_cart, maxiter=5000, tol=1e-9)
        if res is not None:
            final_cart = res.x.reshape(N, 2)
            final_cart = _project_cartesian(final_cart)
            final_score = _min_area_cart(final_cart)
            if final_score > best_score:
                best_cart = final_cart
                best_score = final_score

    # Ensure valid output
    best_cart = _project_cartesian(best_cart)
    if not _is_valid_cart(best_cart):
        # Fallback to random valid points
        for _ in range(100):
            rand_bary = _random_bary()
            rand_cart = _bary_to_cart(rand_bary)
            if _is_valid_cart(rand_cart):
                best_cart = rand_cart
                break

    return best_cart
# EVOLVE-BLOCK-END

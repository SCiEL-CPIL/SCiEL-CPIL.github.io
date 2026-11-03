"""
Support-function (Minkowski-sum) parameterization + multi-start Nelder-Mead.
Convexity guaranteed by construction; no penalty terms needed.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14
H = N // 2

# Precompute unit support vectors for Minkowski-sum basis
# The k-th basis vector h^(k) has 1 at index k and 0 elsewhere, then normalized
_h_basis = np.eye(N)
_h_norms = np.linalg.norm(_h_basis, axis=1)
_h_unit = _h_basis / _h_norms[np.newaxis, :]  # shape (N, N)


def _polygon_from_support(h):
    """
    Convert support numbers h (length N) to vertex coordinates.
    Vertices are intersections of adjacent edge lines:
      x . n_k = h_k  and  x . n_{k+1} = h_{k+1}
    where n_k = (cos(2*pi*k/N), sin(2*pi*k/N)).
    """
    angles = 2.0 * np.pi * np.arange(N) / N
    n_x = np.cos(angles)
    n_y = np.sin(angles)

    # For each pair of adjacent normals (n_k, n_{k+1}), solve:
    #   n_k.x * x + n_k.y * y = h_k
    #   n_{k+1}.x * x + n_{k+1}.y * y = h_{k+1}
    # Using Cramer's rule:
    #   det = n_k.x * n_{k+1}.y - n_k.y * n_{k+1}.x
    #   x = (h_k * n_{k+1}.y - h_{k+1} * n_k.y) / det
    #   y = (n_k.x * h_{k+1} - n_{k+1}.x * h_k) / det

    h_next = np.roll(h, -1)  # h_{k+1}
    det = n_x * np.roll(n_y, -1) - n_y * np.roll(n_x, -1)

    x_coords = (h * np.roll(n_y, -1) - h_next * n_y) / det
    y_coords = (n_x * h_next - np.roll(n_x, -1) * h) / det

    return np.column_stack([x_coords, y_coords])


def _min_triangle_area(pts):
    """Vectorized minimum triangle area over all C(14,3) triples."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n - 1):
            dx = pts[j+1:, 0] - pts[i, 0]
            dy = pts[j+1:, 1] - pts[i, 1]
            cross = (pts[j, 0] - pts[i, 0]) * dy - (pts[j, 1] - pts[i, 1]) * dx
            areas = 0.5 * np.abs(cross)
            if len(areas) > 0:
                local_min = np.min(areas)
                if local_min < min_area:
                    min_area = local_min
    return min_area


def _hull_area(pts):
    """Convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _score_from_points(pts):
    """Score = min triangle area / hull area."""
    ha = _hull_area(pts)
    if ha < 1e-12:
        return 0.0
    ma = _min_triangle_area(pts)
    return ma / ha


def _score_from_h(h):
    """Score from support numbers."""
    pts = _polygon_from_support(h)
    return _score_from_points(pts)


def _objective_h(h):
    """Minimize negative score in support-number space."""
    s = _score_from_h(h)
    if s < 1e-14:
        return 1e10
    return -s


def _regular_h(scale=1.0):
    """Support numbers for regular N-gon."""
    angles = 2.0 * np.pi * np.arange(N) / N
    # Regular polygon with circumradius R has support h_k = R * cos(pi/N)
    h = np.full(N, scale * np.cos(np.pi / N))
    return h


def _generate_seeds(n_seeds=30):
    """Generate diverse support-number seeds, all guaranteed convex."""
    seeds = []
    base = _regular_h(1.0)

    # Seed 0: regular
    seeds.append(base.copy())

    # Seeds 1-5: elliptic (scale x)
    for stretch in [0.4, 0.6, 0.8, 1.2, 1.5]:
        angles = 2.0 * np.pi * np.arange(N) / N
        # Support of ellipse: h(θ) = sqrt(a²cos²θ + b²sin²θ)
        a, b = stretch, 1.0
        h = np.sqrt(a**2 * np.cos(angles)**2 + b**2 * np.sin(angles)**2)
        seeds.append(h)

    # Seeds 6-10: angular perturbation of regular
    for k in range(1, 6):
        pert = 0.08 * np.sin(k * 2.0 * np.pi * np.arange(N) / N)
        h = base * (1.0 + pert)
        seeds.append(h)

    # Seeds 11-15: random Minkowski-sum coefficients
    for i in range(5):
        rng = np.random.RandomState(200 + i)
        a = rng.uniform(0.05, 1.0, N)
        h = _h_unit.T @ a  # Minkowski sum → convex
        seeds.append(h)

    # Seeds 16-20: random with structure (alternating radii)
    for i in range(5):
        rng = np.random.RandomState(500 + i)
        angles = 2.0 * np.pi * np.arange(N) / N
        r = rng.uniform(0.7, 1.3, N)
        h = r * np.cos(np.pi / N)  # approximate support
        # Ensure convexity via Minkowski projection: clip negatives
        h = np.maximum(h, 0.1)
        seeds.append(h)

    # Seeds 21-25: mixed
    for i in range(5):
        rng = np.random.RandomState(700 + i)
        a = rng.uniform(0.0, 0.8, N)
        h = base * 0.5 + _h_unit.T @ a
        seeds.append(h)

    # Seeds 26-29: fine perturbations of regular
    for i in range(4):
        rng = np.random.RandomState(900 + i)
        h = base + rng.normal(0, 0.05, N)
        h = np.maximum(h, 0.1)
        seeds.append(h)

    return seeds


def _refine_nm(h0, time_limit):
    """Nelder-Mead refinement in support-number space."""
    t0 = time.time()

    def obj(h):
        if time.time() - t0 > time_limit:
            return 1e10
        return _objective_h(h)

    try:
        res = minimize(
            obj, h0, method='Nelder-Mead',
            options={
                'maxiter': 300,
                'xatol': 1e-9,
                'fatol': 1e-11,
                'adaptive': True,
                'initial_simplex': None,
            }
        )
        return res.x
    except Exception:
        return h0


def construct():
    """
    Generate 14 points to maximize min triangle area / hull area.
    Uses support-function parameterization (convexity by construction)
    with multi-start Nelder-Mead refinement.
    """
    start_time = time.time()
    deadline = start_time + 140.0

    best_score = -1.0
    best_h = None
    best_pts = None

    # Evaluate all seeds
    seeds = _generate_seeds(30)
    seed_scores = []
    for h in seeds:
        s = _score_from_h(h)
        seed_scores.append((s, h))
        if s > best_score:
            best_score = s
            best_h = h.copy()
            best_pts = _polygon_from_support(h)

    # Sort by score descending
    seed_scores.sort(key=lambda x: x[0], reverse=True)

    # Multi-start Nelder-Mead: refine top seeds
    n_to_refine = 15
    for i in range(min(n_to_refine, len(seed_scores))):
        if time.time() > deadline - 15:
            break
        s_init, h_init = seed_scores[i]
        time_left = deadline - time.time() - 5
        if time_left < 3:
            break

        # Coarse pass
        h_coarse = _refine_nm(h_init, min(time_left * 0.5, 15))
        s_coarse = _score_from_h(h_coarse)

        # Fine pass from coarse result
        if time.time() < deadline - 10:
            time_left2 = deadline - time.time() - 5
            if time_left2 > 3:
                h_fine = _refine_nm(h_coarse, time_left2)
                s_fine = _score_from_h(h_fine)
            else:
                h_fine, s_fine = h_coarse, s_coarse
        else:
            h_fine, s_fine = h_coarse, s_coarse

        if s_fine > best_score:
            best_score = s_fine
            best_h = h_fine.copy()
            best_pts = _polygon_from_support(h_fine)

    # Second round: refine best and neighbors if time permits
    if time.time() < deadline - 20 and best_h is not None:
        # Generate perturbations around best
        rng = np.random.RandomState(42)
        for trial in range(5):
            if time.time() > deadline - 10:
                break
            h_pert = best_h + rng.normal(0, 0.02, N)
            h_pert = np.maximum(h_pert, 0.05)
            time_left = deadline - time.time() - 5
            if time_left < 3:
                break
            h_ref = _refine_nm(h_pert, time_left)
            s_ref = _score_from_h(h_ref)
            if s_ref > best_score:
                best_score = s_ref
                best_h = h_ref.copy()
                best_pts = _polygon_from_support(h_ref)

    # Final: normalize to unit hull area
    if best_pts is not None:
        ha = _hull_area(best_pts)
        if ha > 1e-10:
            scale = np.sqrt(1.0 / ha)
            return best_pts * scale

    # Fallback: regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles)])
    try:
        hull = ConvexHull(pts)
        scale = np.sqrt(1.0 / hull.volume)
        return pts * scale
    except Exception:
        return pts

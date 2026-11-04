"""Two concentric regular heptagons, Nelder-Mead over 6 ring parameters."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull


def _min_triangle_area(pts):
    """Minimum area over all C(n,3) triangles."""
    n = len(pts)
    best = np.inf
    for i in range(n):
        pi = pts[i]
        for j in range(i + 1, n):
            pi_j = pts[j] - pi
            for k in range(j + 1, n):
                pi_k = pts[k] - pi
                a = 0.5 * abs(pi_j[0] * pi_k[1] - pi_j[1] * pi_k[0])
                if a < best:
                    best = a
    return best


def _score(pts):
    """min triangle area / convex hull area."""
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
    except Exception:
        return 0.0
    if hull_area <= 1e-12:
        return 0.0
    return _min_triangle_area(pts) / hull_area


def _build_points(params):
    """Build 14 points from 6 ring parameters: [R1, phi1, R2, phi2, c_x, c_y]."""
    R1, phi1, R2, phi2, cx, cy = params
    angles1 = phi1 + 2.0 * np.pi * np.arange(7) / 7.0
    angles2 = phi2 + 2.0 * np.pi * np.arange(7) / 7.0
    x1 = cx + R1 * np.cos(angles1)
    y1 = cy + R1 * np.sin(angles1)
    x2 = cx + R2 * np.cos(angles2)
    y2 = cy + R2 * np.sin(angles2)
    return np.column_stack([np.concatenate([x1, x2]), np.concatenate([y1, y2])])


def _neg_score(params):
    pts = _build_points(params)
    return -_score(pts)


def construct():
    deadline = time.time() + 120.0
    best_score = 0.0
    best_pts = None

    # Analytic initial guess: outer unit circle, inner at r that balances mixed triangles
    # Mixed triangle (2 outer, 1 inner) area ~ 0.5 * r * sin(2*pi/7) * (chord factor)
    # Pure outer ~ 0.5 * sin(2*pi/7)^2 * cot(pi/7) ... just start with r=0.5
    initial = np.array([1.0, 0.0, 0.5, np.pi / 7.0, 0.0, 0.0])

    # Multi-start Nelder-Mead over the 6D symmetry-reduced space
    rng = np.random.default_rng(14)
    n_starts = 0
    max_starts = 12

    for start_idx in range(max_starts):
        if time.time() > deadline - 5.0:
            break
        if start_idx == 0:
            x0 = initial.copy()
        else:
            # Perturb around best found so far, or random
            if best_pts is not None and start_idx % 2 == 0:
                # Recover approximate params from best? Just perturb initial
                x0 = initial + 0.1 * rng.standard_normal(6)
            else:
                x0 = np.array([
                    0.8 + 0.3 * rng.random(),
                    2.0 * np.pi * rng.random(),
                    0.2 + 0.5 * rng.random(),
                    2.0 * np.pi * rng.random(),
                    0.1 * (rng.random() - 0.5),
                    0.1 * (rng.random() - 0.5),
                ])
            n_starts += 1

        res = minimize(
            _neg_score, x0, method='Nelder-Mead',
            options={'maxiter': 800, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        if time.time() > deadline - 3.0:
            break

        s = -res.fun
        if s > best_score:
            best_score = s
            best_pts = _build_points(res.x)

    if best_pts is None:
        best_pts = _build_points(initial)

    return best_pts
# EVOLVE-BLOCK-END

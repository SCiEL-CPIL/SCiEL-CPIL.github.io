"""Heilbronn triangles: place 11 points in a unit-area equilateral triangle to maximize the minimum triangle area."""
import time
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

N = 11
H = np.sqrt(3) / 2
AREA_EQ = 0.5 * H  # area of the equilateral triangle with side 1

# Precompute all triplets
TRIPLETS = list(combinations(range(N), 3))
N_TRIP = len(TRIPLETS)


def _bary_to_xy(b):
    """Convert barycentric coordinates (n,3) to Cartesian (n,2).
    Vertex 0: (0,0), Vertex 1: (1,0), Vertex 2: (0.5, H)
    """
    x = b[:, 1] + 0.5 * b[:, 2]
    y = H * b[:, 2]
    return np.column_stack([x, y])


def _min_triangle_area(xy):
    """Compute the minimum area over all triplets of points."""
    # xy has shape (N, 2)
    # For each triplet (i,j,k), area = 0.5 * |cross(P_j - P_i, P_k - P_i)|
    areas = np.empty(N_TRIP)
    for idx, (i, j, k) in enumerate(TRIPLETS):
        v1 = xy[j] - xy[i]
        v2 = xy[k] - xy[i]
        areas[idx] = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
    return np.min(areas)


def _soft_min_objective(bary, beta=50.0):
    """Objective: negative of soft-minimum of triangle areas (to maximize min area).
    bary: shape (N, 3) with rows summing to 1, all >= 0.
    """
    xy = _bary_to_xy(bary)
    areas = np.empty(N_TRIP)
    for idx, (i, j, k) in enumerate(TRIPLETS):
        v1 = xy[j] - xy[i]
        v2 = xy[k] - xy[i]
        areas[idx] = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
    # Soft minimum: -log(sum(exp(-beta * areas))) / beta
    # We want to maximize min, so minimize -min ≈ -softmin
    m = np.min(areas)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - m))))
    softmin = m - np.log(np.sum(np.exp(-beta * (areas - m)))) / beta
    return -softmin


def _objective_and_grad(bary, beta=50.0):
    """Return objective and gradient for L-BFGS-B."""
    # Project to valid barycentric: clip and renormalize
    b = np.maximum(bary, 0)
    row_sums = b.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-12)
    b = b / row_sums

    xy = _bary_to_xy(b)
    areas = np.empty(N_TRIP)
    for idx, (i, j, k) in enumerate(TRIPLETS):
        v1 = xy[j] - xy[i]
        v2 = xy[k] - xy[i]
        areas[idx] = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])

    # Soft minimum
    m = np.min(areas)
    exp_vals = np.exp(-beta * (areas - m))
    sum_exp = np.sum(exp_vals)
    softmin = m - np.log(sum_exp) / beta

    # Gradient of softmin w.r.t. areas: -exp(-beta*(areas-m)) / sum_exp
    grad_areas = -exp_vals / sum_exp

    # Gradient w.r.t. barycentric coordinates
    # d(area)/d(xy) then d(xy)/d(bary)
    grad_b = np.zeros_like(b)
    for idx, (i, j, k) in enumerate(TRIPLETS):
        ga = grad_areas[idx]
        if abs(ga) < 1e-15:
            continue
        v1 = xy[j] - xy[i]
        v2 = xy[k] - xy[i]
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        sign = 1.0 if cross >= 0 else -1.0
        # d(0.5*|cross|)/d(xy_i) = -0.5 * sign * (v2[1], -v2[0]) ... let's be careful
        # area = 0.5 * sign * (v1[0]*v2[1] - v1[1]*v2[0])
        # d(area)/d(xy_i) = -0.5*sign * [v2[1], -v2[0]]
        # d(area)/d(xy_j) =  0.5*sign * [v2[1], -v2[0]]
        # d(area)/d(xy_k) = -0.5*sign * [v1[1], -v1[0]]
        # Wait, let me re-derive:
        # cross = (xj-xi)(yk-yi) - (yj-yi)(xk-xi)
        # d(cross)/d(xi) = -(yk-yi) = -v2[1]
        # d(cross)/d(yi) = (xk-xi) = v2[0]
        # d(cross)/d(xj) = (yk-yi) = v2[1]
        # d(cross)/d(yj) = -(xk-xi) = -v2[0]
        # d(cross)/d(xk) = -(yj-yi) = -v1[1]
        # d(cross)/d(yk) = (xj-xi) = v1[0]
        # area = 0.5 * sign * cross
        # d(area)/d(xy_i) = 0.5*sign * [-v2[1], v2[0]]
        # d(area)/d(xy_j) = 0.5*sign * [v2[1], -v2[0]]
        # d(area)/d(xy_k) = 0.5*sign * [-v1[1], v1[0]]

        darea_dxyi = 0.5 * sign * np.array([-v2[1], v2[0]])
        darea_dxyj = 0.5 * sign * np.array([v2[1], -v2[0]])
        darea_dxyk = 0.5 * sign * np.array([-v1[1], v1[0]])

        # d(xy)/d(bary): xy[:,0] = b[:,1] + 0.5*b[:,2], xy[:,1] = H*b[:,2]
        # d(x)/d(b0)=0, d(x)/d(b1)=1, d(x)/d(b2)=0.5
        # d(y)/d(b0)=0, d(y)/d(b1)=0, d(y)/d(b2)=H
        # So d(xy)/d(b_p) = [1, 0] if p=1, [0.5, H] if p=2, [0,0] if p=0

        for p, dxy in [(0, np.array([0.0, 0.0])),
                       (1, np.array([1.0, 0.0])),
                       (2, np.array([0.5, H]))]:
            grad_b[i, p] += ga * np.dot(darea_dxyi, dxy)
            grad_b[j, p] += ga * np.dot(darea_dxyj, dxy)
            grad_b[k, p] += ga * np.dot(darea_dxyk, dxy)

    obj = -softmin
    grad = -grad_b
    return obj, grad


def _initial_guesses():
    """Generate diverse initial guesses for barycentric coordinates."""
    guesses = []

    # 1. Grid-like arrangement
    # Place points in a rough 4x3 grid within the triangle
    bary = np.zeros((N, 3))
    positions = [
        (0.1, 0.1, 0.8),
        (0.3, 0.1, 0.6),
        (0.5, 0.1, 0.4),
        (0.7, 0.1, 0.2),
        (0.05, 0.3, 0.65),
        (0.25, 0.3, 0.45),
        (0.45, 0.3, 0.25),
        (0.65, 0.3, 0.05),
        (0.15, 0.55, 0.3),
        (0.35, 0.55, 0.1),
        (0.5, 0.45, 0.05),
    ]
    for i, (b0, b1, b2) in enumerate(positions):
        s = b0 + b1 + b2
        bary[i] = [b0 / s, b1 / s, b2 / s]
    guesses.append(bary.copy())

    # 2. Symmetric arrangement: 1 center, 6 around, 4 more
    bary = np.zeros((N, 3))
    bary[0] = [1/3, 1/3, 1/3]  # center
    # 6 points around center
    for i in range(1, 7):
        angle = 2 * np.pi * (i - 1) / 6
        r = 0.3
        # Convert polar to barycentric approximately
        bary[i] = [1/3 + r * np.cos(angle) * 0.5,
                   1/3 + r * np.cos(angle + np.pi/3) * 0.5,
                   1/3 + r * np.cos(angle - np.pi/3) * 0.5]
        bary[i] = np.maximum(bary[i], 0)
        bary[i] /= bary[i].sum()
    # 4 more near edges
    for i, (b0, b1, b2) in enumerate([(0.05, 0.45, 0.5), (0.5, 0.05, 0.45),
                                       (0.45, 0.5, 0.05), (0.15, 0.15, 0.7)]):
        bary[7 + i] = [b0, b1, b2]
        bary[7 + i] /= bary[7 + i].sum()
    guesses.append(bary.copy())

    # 3. Random
    np.random.seed(42)
    for _ in range(5):
        bary = np.random.dirichlet(np.ones(3), size=N)
        guesses.append(bary)

    return guesses


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s

    best_bary = None
    best_score = -np.inf

    guesses = _initial_guesses()

    for guess in guesses:
        if time.time() > deadline:
            break

        bary = guess.copy()

        # Use L-BFGS-B with the soft-min objective
        try:
            result = minimize(
                _objective_and_grad,
                bary,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8,
                         'maxls': 50, 'maxfun': 1000}
            )
            if result.success or result.fun < best_score:
                bary_opt = np.maximum(result.x, 0)
                row_sums = bary_opt.sum(axis=1, keepdims=True)
                row_sums = np.maximum(row_sums, 1e-12)
                bary_opt = bary_opt / row_sums
                xy = _bary_to_xy(bary_opt)
                score = _min_triangle_area(xy)
                if score > best_score:
                    best_score = score
                    best_bary = bary_opt.copy()
        except Exception:
            pass

        # Also try with higher beta for sharper min
        if time.time() > deadline:
            break
        try:
            def obj_high_beta(b):
                return _soft_min_objective(b, beta=100.0)
            result2 = minimize(
                obj_high_beta,
                bary,
                method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8,
                         'maxls': 50, 'maxfun': 800}
            )
            bary_opt2 = np.maximum(result2.x, 0)
            row_sums = bary_opt2.sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-12)
            bary_opt2 = bary_opt2 / row_sums
            xy2 = _bary_to_xy(bary_opt2)
            score2 = _min_triangle_area(xy2)
            if score2 > best_score:
                best_score = score2
                best_bary = bary_opt2.copy()
        except Exception:
            pass

    # Fallback if no valid result
    if best_bary is None:
        bary = np.zeros((N, 3))
        for i in range(N):
            bary[i] = [1/3, 1/3, 1/3]
        # Perturb slightly
        np.random.seed(0)
        for i in range(N):
            bary[i] += np.random.normal(0, 0.05, 3)
            bary[i] = np.maximum(bary[i], 0)
            bary[i] /= bary[i].sum()
        best_bary = bary

    xy = _bary_to_xy(best_bary)
    return xy


if __name__ == '__main__':
    pts = construct()
    print("Points:")
    for p in pts:
        print(f"  ({p[0]:.6f}, {p[1]:.6f})")
    min_area = _min_triangle_area(pts)
    print(f"Min triangle area: {min_area:.8f}")
    print(f"Normalized by equilateral area: {min_area / AREA_EQ:.8f}")

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 110.0  # seconds, safety margin before 120s kill


def _time_remaining():
    return DEADLINE - time.time()


def _check_validity(circles, W, H):
    """Check if circles are valid: inside rectangle, no overlaps."""
    if len(circles) != N:
        return False
    for i in range(N):
        x, y, r = circles[i]
        if r <= 0:
            return False
        if x - r < -1e-9 or y - r < -1e-9 or x + r > W + 1e-9 or y + r > H + 1e-9:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - 1e-9:
                return False
    return True


def _grid_init(W, H):
    """Create a grid-based initial layout."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) * W / side
            cy = (j + 0.5) * H / side
            r = min(W, H) / side * 0.49
            cells.append([cx, cy, r])
    return np.array(cells[:N])


def _objective(params, W, H):
    """
    Minimize negative sum of radii + smooth penalties for overlaps and boundary violations.
    This is a smooth, differentiable objective for L-BFGS-B.
    """
    n = N
    circles = params.reshape(n, 3)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]

    obj = -np.sum(r)

    # Boundary penalties (smooth)
    eps = 1e-4

    def _smooth_max(v):
        return 0.5 * (v + np.sqrt(v ** 2 + eps ** 2))

    # x - r >= 0  => violation = r - x
    v = r - x
    obj += 100.0 * np.sum(_smooth_max(v) ** 2)

    # y - r >= 0
    v = r - y
    obj += 100.0 * np.sum(_smooth_max(v) ** 2)

    # x + r <= W  => violation = x + r - W
    v = x + r - W
    obj += 100.0 * np.sum(_smooth_max(v) ** 2)

    # y + r <= H
    v = y + r - H
    obj += 100.0 * np.sum(_smooth_max(v) ** 2)

    # r >= 0
    v = -r
    obj += 100.0 * np.sum(_smooth_max(v) ** 2)

    # Overlap penalties: for each pair, violation = r_i + r_j - dist_ij
    overlap_weight = 100.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < 1e-12:
                dist = 1e-12
            v = r[i] + r[j] - dist
            obj += overlap_weight * _smooth_max(v) ** 2

    return obj


def _local_optimize(circles, W, H, max_iter=3000):
    """Use L-BFGS-B to locally optimize the packing."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))  # x
        bounds.append((1e-6, H - 1e-6))  # y
        bounds.append((1e-6, r_max))     # r

    result = minimize(
        _objective, params, args=(W, H),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
    )
    return result.x.reshape(N, 3)


def _polish(circles, W, H, max_iter=5000):
    """High-precision polish pass."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    result = minimize(
        _objective, params, args=(W, H),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
    )
    return result.x.reshape(N, 3)


def _separate_and_clamp(circles, W, H, max_iter=200):
    """Resolve any remaining overlaps or boundary violations by direct separation."""
    n = len(circles)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()

    for _ in range(max_iter):
        moved = False

        # Clamp to boundary
        for i in range(n):
            if r[i] <= 0:
                r[i] = 1e-4
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])

        # Resolve overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist - 1e-10:
                    if dist < 1e-12:
                        angle = np.random.uniform(0, 2 * np.pi)
                        dx = np.cos(angle) * 1e-6
                        dy = np.sin(angle) * 1e-6
                        dist = 1e-6
                    overlap = min_dist - dist
                    total_r = r[i] + r[j]
                    wi = r[i] / total_r if total_r > 0 else 0.5
                    wj = r[j] / total_r if total_r > 0 else 0.5
                    push = overlap + 1e-8
                    x[i] += wi * push * dx / dist
                    y[i] += wi * push * dy / dist
                    x[j] -= wj * push * dx / dist
                    y[j] -= wj * push * dy / dist
                    moved = True

        # Re-clamp and shrink if needed
        for i in range(n):
            max_r_x = min(x[i], W - x[i])
            max_r_y = min(y[i], H - y[i])
            max_r = max(1e-4, min(max_r_x, max_r_y))
            if r[i] > max_r:
                r[i] = max_r
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])

        if not moved:
            break

    return np.column_stack([x, y, r])


def _try_aspect_ratio(W, H, start_time):
    """Try to optimize packing in a rectangle of size W x H."""
    if _time_remaining() < 5.0:
        return None

    # Initial grid layout
    circles = _grid_init(W, H)

    # Local optimization
    circles = _local_optimize(circles, W, H)

    # Polish
    if _time_remaining() > 10.0:
        circles = _polish(circles, W, H)

    # Separate and clamp
    circles = _separate_and_clamp(circles, W, H)

    # Check validity
    if not _check_validity(circles, W, H):
        return None

    return circles


def _evaluate(circles, W, H):
    """Return the sum of radii if valid, else -inf."""
    if _check_validity(circles, W, H):
        return np.sum(circles[:, 2])
    return -np.inf


def construct():
    """
    Main entry point. Packs 21 circles in a variable rectangle of perimeter 4
    to maximize the sum of radii.
    """
    start_time = time.time()
    best_circles = None
    best_score = -np.inf
    best_W = 1.0
    best_H = 1.0

    # Perimeter constraint: 2*(W+H) = 4 => W+H = 2
    # Try various aspect ratios
    # W ranges from small to large, H = 2 - W
    # W must be at least enough to fit some circles, and H too

    # We'll try a range of aspect ratios
    # W in [0.3, 1.7], H = 2 - W
    # But we need W and H both positive and reasonable

    aspect_ratios = []
    # Square
    aspect_ratios.append((1.0, 1.0))
    # Various ratios
    for W in np.linspace(0.4, 1.6, 13):
        H = 2.0 - W
        if H > 0.1 and W > 0.1:
            aspect_ratios.append((W, H))

    # Sort by some heuristic - try square first, then others
    # Actually, let's try a diverse set

    for W, H in aspect_ratios:
        if _time_remaining() < 15.0:
            break

        circles = _try_aspect_ratio(W, H, start_time)
        if circles is not None:
            score = _evaluate(circles, W, H)
            if score > best_score:
                best_score = score
                best_circles = circles.copy()
                best_W = W
                best_H = H

    # If we found a good solution, try to improve it with more restarts
    if best_circles is not None and _time_remaining() > 20.0:
        # Try perturbed versions of the best solution
        for _ in range(5):
            if _time_remaining() < 10.0:
                break
            # Perturb the best solution
            perturbed = best_circles.copy()
            perturbed[:, 0] += np.random.normal(0, 0.01, N)
            perturbed[:, 1] += np.random.normal(0, 0.01, N)
            # Clamp radii to positive
            perturbed[:, 2] = np.maximum(perturbed[:, 2], 0.01)

            # Optimize
            perturbed = _local_optimize(perturbed, best_W, best_H)
            if _time_remaining() > 10.0:
                perturbed = _polish(perturbed, best_W, best_H)
            perturbed = _separate_and_clamp(perturbed, best_W, best_H)

            if _check_validity(perturbed, best_W, best_H):
                score = _evaluate(perturbed, best_W, best_H)
                if score > best_score:
                    best_score = score
                    best_circles = perturbed.copy()

    # Final fallback: if we still don't have a valid solution, use the grid
    if best_circles is None:
        W, H = 1.0, 1.0
        best_circles = _grid_init(W, H)
        best_circles = _separate_and_clamp(best_circles, W, H)
        if not _check_validity(best_circles, W, H):
            # Last resort: tiny circles in a grid
            side = int(np.ceil(np.sqrt(N)))
            cells = []
            for i in range(side):
                for j in range(side):
                    cx = (i + 0.5) * W / side
                    cy = (j + 0.5) * H / side
                    r = min(W, H) / side * 0.4
                    cells.append([cx, cy, r])
            best_circles = np.array(cells[:N])

    return best_circles
# EVOLVE-BLOCK-END

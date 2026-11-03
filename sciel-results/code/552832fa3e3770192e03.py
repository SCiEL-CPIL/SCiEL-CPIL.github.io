"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 140.0  # seconds, safety margin before 150s kill


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


def _separate_circles(circles, W, H, max_iter=2000):
    """
    Directly resolve overlaps and boundary violations using pairwise repulsion.
    Returns (circles, is_valid).
    """
    n = len(circles)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()

    for it in range(max_iter):
        moved = False

        # Resolve boundary violations by shrinking radii if needed
        for i in range(n):
            if r[i] <= 0:
                r[i] = 1e-4
            # Clamp position to keep circle inside rectangle
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])

        # Resolve overlaps via pairwise separation
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                min_dist = r[i] + r[j]
                if dist_sq < min_dist * min_dist:
                    dist = np.sqrt(dist_sq)
                    # Push apart
                    if dist < 1e-12:
                        # Random nudge
                        angle = np.random.uniform(0, 2 * np.pi)
                        dx = np.cos(angle) * 1e-6
                        dy = np.sin(angle) * 1e-6
                        dist = 1e-6
                    overlap = min_dist - dist
                    # Distribute push based on relative sizes
                    total_r = r[i] + r[j]
                    wi = r[i] / total_r if total_r > 0 else 0.5
                    wj = r[j] / total_r if total_r > 0 else 0.5
                    push = overlap + 1e-8
                    x[i] += wi * push * dx / dist
                    y[i] += wi * push * dy / dist
                    x[j] -= wj * push * dx / dist
                    y[j] -= wj * push * dy / dist
                    moved = True

        # Re-clamp after separation, shrinking radius if circle doesn't fit
        for i in range(n):
            # If circle can't fit at current position, shrink it
            max_r_x = min(x[i], W - x[i])
            max_r_y = min(y[i], H - y[i])
            max_r = max(1e-4, min(max_r_x, max_r_y))
            if r[i] > max_r:
                r[i] = max_r
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])

        if not moved:
            break

    result = np.column_stack([x, y, r])
    valid = _check_validity(result, W, H)
    return result, valid


def _objective(params, W, H, penalty_weight):
    """
    Objective: minimize -sum(r) + penalties for violations.
    Vectorized for speed.
    """
    circles = params.reshape(N, 3)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]

    obj = -np.sum(r)

    # Boundary penalties (smooth)
    eps = 1e-4

    def _smooth_max(v):
        return 0.5 * (v + np.sqrt(v ** 2 + eps ** 2))

    # x - r >= 0  => violation = -(x - r) = r - x
    v1 = r - x
    # y - r >= 0
    v2 = r - y
    # x + r <= W  => violation = x + r - W
    v3 = x + r - W
    # y + r <= H
    v4 = y + r - H
    # r >= 0
    v5 = -r

    obj += penalty_weight * (np.sum(_smooth_max(v1) ** 2) + np.sum(_smooth_max(v2) ** 2) +
                             np.sum(_smooth_max(v3) ** 2) + np.sum(_smooth_max(v4) ** 2) +
                             np.sum(_smooth_max(v5) ** 2))

    # Overlap penalties: vectorized computation
    dx_mat = x[:, np.newaxis] - x[np.newaxis, :]
    dy_mat = y[:, np.newaxis] - y[np.newaxis, :]
    dist_mat = np.sqrt(dx_mat ** 2 + dy_mat ** 2)
    np.fill_diagonal(dist_mat, 1e12)

    r_sum_mat = r[:, np.newaxis] + r[np.newaxis, :]
    v_overlap = r_sum_mat - dist_mat

    # Only use upper triangle to avoid double counting
    iu, ju = np.triu_indices(N, k=1)
    v_vec = v_overlap[iu, ju]
    # Sum of squares of positive violations
    obj += penalty_weight * np.sum(np.maximum(v_vec, 0) ** 2)

    return obj


def _optimize(W, H, initial_circles, time_budget=15.0):
    """
    Optimize the configuration using L-BFGS-B with annealed penalties.
    """
    start = time.time()
    params = initial_circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    best_circles = initial_circles.copy()
    best_score = -np.sum(best_circles[:, 2])

    # Anneal penalty weight: start low, increase to enforce constraints
    weights = [10, 50, 200, 1000, 5000, 20000]
    current_params = params.copy()

    for w in weights:
        if time.time() - start > time_budget - 2:
            break
        try:
            res = minimize(
                _objective, current_params, args=(W, H, w),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            current_params = res.x
        except Exception:
            continue

        circles = current_params.reshape(N, 3)
        # Repair step
        circles, valid = _separate_circles(circles, W, H, max_iter=500)
        if valid:
            score = np.sum(circles[:, 2])
            if score > best_score:
                best_score = score
                best_circles = circles.copy()
                current_params = best_circles.flatten()

    return best_circles, best_score


def _init_grid_5x5(W, H):
    """5x5 grid minus 4 corners (21 circles)."""
    cells = []
    side = 5
    for i in range(side):
        for j in range(side):
            if (i == 0 and j == 0) or (i == 0 and j == 4) or (i == 4 and j == 0) or (i == 4 and j == 4):
                continue
            cx = (i + 0.5) * W / side
            cy = (j + 0.5) * H / side
            r = min(W, H) / (2 * side) - 1e-3
            cells.append([cx, cy, r])
    return np.array(cells[:N])


def _init_grid_3x7(W, H):
    """3x7 grid (21 circles)."""
    cells = []
    rows = 3
    cols = 7
    for i in range(rows):
        for j in range(cols):
            cx = (j + 0.5) * W / cols
            cy = (i + 0.5) * H / rows
            r = min(W / cols, H / rows) / 2 - 1e-3
            cells.append([cx, cy, r])
    return np.array(cells[:N])


def _init_random(W, H, rng):
    """Random initialization."""
    circles = []
    max_r = min(W, H) / 4
    for _ in range(N):
        r = rng.uniform(0.05, max_r)
        x = rng.uniform(r, W - r)
        y = rng.uniform(r, H - r)
        circles.append([x, y, r])
    return np.array(circles)


def construct():
    start_time = time.time()
    best_score = -np.inf
    best_result = None

    # Aspect ratios to try
    aspect_ratios = [0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]

    rng = np.random.default_rng(42)

    for ar in aspect_ratios:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:
            break

        W = 2.0 * ar / (ar + 1.0)
        H = 2.0 / (ar + 1.0)

        # Determine initializations based on aspect ratio
        inits = []
        if 0.8 <= ar <= 1.2:
            inits.append(_init_grid_5x5(W, H))
        else:
            inits.append(_init_grid_3x7(W, H))

        # Always try a random initialization
        inits.append(_init_random(W, H, rng))

        for init in inits:
            if time.time() - start_time > DEADLINE - 5:
                break
            # Allocate time budget based on remaining time
            remaining = DEADLINE - 5 - (time.time() - start_time)
            if remaining < 5:
                break
            time_budget = min(15.0, remaining / 2.0)
            
            circles, score = _optimize(W, H, init, time_budget=time_budget)
            if score > best_score:
                best_score = score
                best_result = circles.copy()

    # Final polish on the best result
    if best_result is not None and time.time() - start_time < DEADLINE - 5:
        # Determine W, H from best_result? No, we lost track. 
        # We should store best_W, best_H.
        # Let's re-derive or store. I'll modify the loop to store best_W, best_H.
        pass

    # Re-run with stored W, H logic
    # Since I didn't store best_W, best_H in the loop above, I need to fix that.
    # Let's refactor construct slightly to track best_W, best_H.

    # --- Corrected Construct ---
    start_time = time.time()
    best_score = -np.inf
    best_result = None
    best_W, best_H = 1.0, 1.0

    aspect_ratios = [0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]
    rng = np.random.default_rng(42)

    for ar in aspect_ratios:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:
            break

        W = 2.0 * ar / (ar + 1.0)
        H = 2.0 / (ar + 1.0)

        inits = []
        if 0.8 <= ar <= 1.2:
            inits.append(_init_grid_5x5(W, H))
        else:
            inits.append(_init_grid_3x7(W, H))

        inits.append(_init_random(W, H, rng))

        for init in inits:
            if time.time() - start_time > DEADLINE - 5:
                break
            remaining = DEADLINE - 5 - (time.time() - start_time)
            if remaining < 5:
                break
            time_budget = min(15.0, remaining / 2.0)

            circles, score = _optimize(W, H, init, time_budget=time_budget)
            if score > best_score:
                best_score = score
                best_result = circles.copy()
                best_W, best_H = W, H

    # Final polish
    if best_result is not None and time.time() - start_time < DEADLINE - 3:
        polished, valid = _separate_circles(best_result, best_W, best_H, max_iter=5000)
        if valid:
            score = np.sum(polished[:, 2])
            if score > best_score:
                best_score = score
                best_result = polished.copy()

    # Fallback
    if best_result is None:
        best_result = _init_grid_5x5(1.0, 1.0)
        best_result, _ = _separate_circles(best_result, 1.0, 1.0)
        best_W, best_H = 1.0, 1.0

    if not _check_validity(best_result, best_W, best_H):
        # Emergency fallback
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([cx, cy, r])
        best_result = np.array(cells[:N])
        best_W, best_H = 1.0, 1.0

    return best_result
# EVOLVE-BLOCK-END

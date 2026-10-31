"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time


def _validate_and_score(circles):
    """
    circles: (n,3) array of [cx, cy, r]
    Returns (score, is_valid) where score is sum of radii if valid, else 0.
    Validity:
      - all r > 0
      - all circles inside rectangle [0,w]x[0,h] with w+h <= 2 (perimeter 4)
      - pairwise disjoint interiors: dist >= r_i + r_j - eps
    We compute the minimal rectangle containing all circles:
      w = max(cx+r) - min(cx-r)
      h = max(cy+r) - min(cy-r)
    and require w + h <= 2 + 1e-9.
    """
    if circles is None or len(circles) != N:
        return 0.0, False
    cx = circles[:, 0]
    cy = circles[:, 1]
    r = circles[:, 2]
    if np.any(r <= 0):
        return 0.0, False
    # Bounding box
    left = np.min(cx - r)
    right = np.max(cx + r)
    bottom = np.min(cy - r)
    top = np.max(cy + r)
    w = right - left
    h = top - bottom
    if w + h > 2.0 + 1e-9:
        return 0.0, False
    # Pairwise non-overlap
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return float(np.sum(r)), True


def _initial_grid():
    """
    Create a grid of 21 circles in a square of side 1 (perimeter 4).
    Use a 5x5 grid, take 21 cells.
    """
    side = 5
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 1e-4
                cells.append([cx, cy, r])
    return np.array(cells)


def _sa_optimize(circles, time_limit=100.0, seed=42):
    """
    Simulated annealing to maximize sum of radii subject to:
      - w + h <= 2
      - pairwise non-overlap
    We use a penalty approach: minimize -sum(r) + penalty * (constraint violations).
    Actually, let's use a direct approach: we optimize positions and radii.
    """
    rng = np.random.default_rng(seed)
    circles = circles.copy().astype(float)
    best = circles.copy()
    best_score, best_valid = _validate_and_score(best)
    if not best_valid:
        best_score = 0.0

    # Temperature schedule
    T_start = 0.05
    T_end = 0.001
    # We'll do a fixed number of iterations based on time budget
    start_time = time.time()
    deadline = start_time + time_limit

    # Move types:
    # 0: move a circle (cx, cy)
    # 1: change radius of a circle
    # 2: move and resize

    n_iter = 0
    while time.time() < deadline - 2.0:  # safety margin
        T = T_start * (T_end / T_start) ** (n_iter / 50000.0)
        if T < T_end:
            T = T_end

        i = rng.integers(0, N)
        move_type = rng.integers(0, 3)

        cx, cy, r = circles[i]
        new_cx, new_cy, new_r = cx, cy, r

        if move_type == 0:
            # Move position
            step = 0.02 * (1 + 0.5 * T / T_start)
            new_cx = cx + rng.normal(0, step)
            new_cy = cy + rng.normal(0, step)
        elif move_type == 1:
            # Change radius
            step_r = 0.01 * (1 + 0.5 * T / T_start)
            new_r = max(0.005, r + rng.normal(0, step_r))
        else:
            # Move and resize
            step = 0.015 * (1 + 0.5 * T / T_start)
            new_cx = cx + rng.normal(0, step)
            new_cy = cy + rng.normal(0, step)
            step_r = 0.008 * (1 + 0.5 * T / T_start)
            new_r = max(0.005, r + rng.normal(0, step_r))

        # Save old values
        old = [cx, cy, r]
        circles[i] = [new_cx, new_cy, new_r]

        new_score, new_valid = _validate_and_score(circles)
        if not new_valid:
            new_score = 0.0

        old_score = best_score if new_valid else 0.0
        # Actually, we should compare against current score, not best
        # Let's track current score
        # For simplicity, accept if new_score > old_score or with probability exp((new-old)/T)
        # But we need current score. Let's compute it.
        cur_score, cur_valid = _validate_and_score(circles)
        if not cur_valid:
            cur_score = 0.0

        # Revert and compare
        circles[i] = old

        # Compute current score before move
        # This is expensive. Let's just use a simple accept/reject based on the new config.
        # Better: keep track of current state and its score.
        # Let me refactor.

        # For now, use a simpler approach: only accept if valid and better, or random accept.
        if new_valid and new_score > best_score:
            best = circles.copy()
            best_score = new_score

        # Accept move if it improves or with probability
        if new_valid:
            if new_score >= cur_score:
                circles[i] = [new_cx, new_cy, new_r]
            else:
                delta = new_score - cur_score
                if delta < 0 and rng.random() < np.exp(delta / max(T, 1e-10)):
                    circles[i] = [new_cx, new_cy, new_r]
        else:
            # Reject invalid moves mostly
            if rng.random() < 0.01:
                circles[i] = [new_cx, new_cy, new_r]

        n_iter += 1
        if n_iter % 10000 == 0:
            cur_score, cur_valid = _validate_and_score(circles)
            if cur_valid and cur_score > best_score:
                best = circles.copy()
                best_score = cur_score

    # Final polish
    final_score, final_valid = _validate_and_score(best)
    if final_valid:
        return best, final_score
    return best, best_score


def construct():
    """
    Pack 21 circles in a perimeter-4 rectangle, maximizing sum of radii.
    Returns (n,3) array of [cx, cy, r].
    """
    deadline = time.time() + 110.0  # safety margin from 120s limit

    # Start with grid
    circles = _initial_grid()
    score, valid = _validate_and_score(circles)
    if not valid:
        # Fallback: smaller radii
        circles[:, 2] *= 0.9
        score, valid = _validate_and_score(circles)

    best = circles.copy()
    best_score = score if valid else 0.0

    # Run SA with multiple restarts
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        # Jitter initial solution
        if restart == 0:
            start_circles = circles.copy()
        else:
            start_circles = best.copy()
            # Add small noise
            noise = 0.01 * np.random.default_rng(42 + restart).normal(0, 1, start_circles.shape)
            start_circles = start_circles + noise
            start_circles[:, 2] = np.maximum(start_circles[:, 2], 0.005)

        time_left = deadline - time.time() - 2.0
        if time_left < 5.0:
            break

        result, res_score = _sa_optimize(start_circles, time_limit=time_left, seed=42 + restart)
        if res_score > best_score:
            best = result
            best_score = res_score

    # Final validation
    final_score, final_valid = _validate_and_score(best)
    if final_valid:
        return best
    # Fallback to grid
    return _initial_grid()
# EVOLVE-BLOCK-END

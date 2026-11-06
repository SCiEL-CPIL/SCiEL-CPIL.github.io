"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from collections import defaultdict

def _line_key(x1, y1, x2, y2):
    """Compute canonical line key from two points."""
    dx = x2 - x1
    dy = y2 - y1
    g = gcd(abs(dx), abs(dy))
    if g == 0:
        return None
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    c = dy * x1 - dx * y1
    return (dx, dy, c)

def _compute_lines(points):
    """Compute all lines and their point index sets. Returns dict of line_key -> set of indices."""
    n = len(points)
    line_points = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i][0], points[i][1], points[j][0], points[j][1])
            if key is not None:
                line_points[key].add(i)
                line_points[key].add(j)
    return line_points

def _evaluate(points):
    """Compute score and validity. Returns (is_valid, score)."""
    n = len(points)
    if n != 16:
        return False, 0
    if len(set(tuple(p) for p in points)) != 16:
        return False, 0
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return False, 0
    line_points = _compute_lines(points)
    for indices in line_points.values():
        if len(indices) >= 5:
            return False, 0
    score = sum(1 for indices in line_points.values() if len(indices) == 4)
    return True, score

def _evaluate_incremental(points, line_points, moved_idx):
    """
    After moving point at moved_idx, update line_points incrementally.
    Returns (is_valid, score).
    """
    n = len(points)
    if n != 16:
        return False, 0
    if len(set(tuple(p) for p in points)) != 16:
        return False, 0
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return False, 0

    # Remove old contributions of moved_idx
    for key in list(line_points.keys()):
        if moved_idx in line_points[key]:
            line_points[key].discard(moved_idx)
            if not line_points[key]:
                del line_points[key]

    # Add new contributions of moved_idx
    x_m, y_m = points[moved_idx]
    for j in range(n):
        if j == moved_idx:
            continue
        key = _line_key(x_m, y_m, points[j][0], points[j][1])
        if key is not None:
            line_points[key].add(moved_idx)
            line_points[key].add(j)

    # Check validity and compute score
    for indices in line_points.values():
        if len(indices) >= 5:
            return False, 0
    score = sum(1 for indices in line_points.values() if len(indices) == 4)
    return True, score

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    random.seed(12345)

    # Generate diverse initial configurations
    initial_configs = []

    # 4x4 grid
    initial_configs.append([[x, y] for x in range(4) for y in range(4)])

    # Various grid offsets
    for ox in range(-2, 3):
        for oy in range(-2, 3):
            initial_configs.append([[x + ox, y + oy] for x in range(4) for y in range(4)])

    # Perturbed grids
    for trial in range(40):
        pts = [[x, y] for x in range(4) for y in range(4)]
        for _ in range(random.randint(2, 6)):
            idx = random.randint(0, 15)
            pts[idx] = [random.randint(-15, 15), random.randint(-15, 15)]
        initial_configs.append(pts)

    # Random configurations with some structure
    for trial in range(60):
        pts = []
        base = random.randint(-20, 20)
        for i in range(16):
            pts.append([base + random.randint(-10, 10), base + random.randint(-10, 10)])
        initial_configs.append(pts)

    # Pure random
    for trial in range(60):
        pts = []
        for _ in range(16):
            pts.append([random.randint(-50, 50), random.randint(-50, 50)])
        initial_configs.append(pts)

    # Evaluate all initial configs, keep best
    best_points = None
    best_score = 0
    best_line_points = None

    for cfg in initial_configs:
        valid, score = _evaluate(cfg)
        if valid and score > best_score:
            best_score = score
            best_points = [p[:] for p in cfg]
            best_line_points = _compute_lines(best_points)

    if best_points is None:
        # Fallback: use 4x4 grid
        best_points = [[x, y] for x in range(4) for y in range(4)]
        valid, best_score = _evaluate(best_points)
        best_line_points = _compute_lines(best_points)

    # Local search with incremental evaluation
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(16):
            if time.time() > deadline:
                break
            old_x, old_y = best_points[i]

            # Try a range of moves: small, medium, large
            move_ranges = [
                (range(-6, 7), range(-6, 7)),       # small
                (range(-15, 16), range(-15, 16)),   # medium
                (range(-40, 41, 7), range(-40, 41, 7)),  # large sparse
            ]

            for dx_range, dy_range in move_ranges:
                for dx in dx_range:
                    for dy in dy_range:
                        if dx == 0 and dy == 0:
                            continue
                        new_x = old_x + dx
                        new_y = old_y + dy
                        if abs(new_x) > 1000 or abs(new_y) > 1000:
                            continue

                        # Save old position
                        best_points[i] = [new_x, new_y]
                        valid, score = _evaluate_incremental(best_points, best_line_points, i)

                        if valid and score > best_score:
                            best_score = score
                            improved = True
                        else:
                            # Revert
                            best_points[i] = [old_x, old_y]
                            # Revert line_points
                            for key in list(best_line_points.keys()):
                                if i in best_line_points[key]:
                                    best_line_points[key].discard(i)
                                    if not best_line_points[key]:
                                        del best_line_points[key]
                            for j in range(16):
                                if j == i:
                                    continue
                                key = _line_key(old_x, old_y, best_points[j][0], best_points[j][1])
                                if key is not None:
                                    best_line_points[key].add(i)
                                    best_line_points[key].add(j)

    # Second phase: random larger moves
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(16):
            if time.time() > deadline:
                break
            old_x, old_y = best_points[i]
            for _ in range(100):
                new_x = random.randint(max(-1000, old_x - 60), min(1000, old_x + 60))
                new_y = random.randint(max(-1000, old_y - 60), min(1000, old_y + 60))

                best_points[i] = [new_x, new_y]
                valid, score = _evaluate_incremental(best_points, best_line_points, i)

                if valid and score > best_score:
                    best_score = score
                    improved = True
                else:
                    best_points[i] = [old_x, old_y]
                    for key in list(best_line_points.keys()):
                        if i in best_line_points[key]:
                            best_line_points[key].discard(i)
                            if not best_line_points[key]:
                                del best_line_points[key]
                    for j in range(16):
                        if j == i:
                            continue
                        key = _line_key(old_x, old_y, best_points[j][0], best_points[j][1])
                        if key is not None:
                            best_line_points[key].add(i)
                            best_line_points[key].add(j)

    # Third phase: move two points simultaneously
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(16):
            if time.time() > deadline:
                break
            for j in range(i + 1, 16):
                if time.time() > deadline:
                    break
                old_xi, old_yi = best_points[i]
                old_xj, old_yj = best_points[j]
                for _ in range(30):
                    new_xi = random.randint(max(-1000, old_xi - 30), min(1000, old_xi + 30))
                    new_yi = random.randint(max(-1000, old_yi - 30), min(1000, old_yi + 30))
                    new_xj = random.randint(max(-1000, old_xj - 30), min(1000, old_xj + 30))
                    new_yj = random.randint(max(-1000, old_yj - 30), min(1000, old_yj + 30))

                    best_points[i] = [new_xi, new_yi]
                    best_points[j] = [new_xj, new_yj]
                    valid, score = _evaluate_incremental(best_points, best_line_points, i)
                    valid, score = _evaluate_incremental(best_points, best_line_points, j)

                    if valid and score > best_score:
                        best_score = score
                        improved = True
                    else:
                        best_points[i] = [old_xi, old_yi]
                        best_points[j] = [old_xj, old_yj]
                        # Revert both
                        for key in list(best_line_points.keys()):
                            if i in best_line_points[key]:
                                best_line_points[key].discard(i)
                            if j in best_line_points[key]:
                                best_line_points[key].discard(j)
                            if not best_line_points[key]:
                                del best_line_points[key]
                        for k in range(16):
                            if k == i or k == j:
                                continue
                            key_i = _line_key(old_xi, old_yi, best_points[k][0], best_points[k][1])
                            if key_i is not None:
                                best_line_points[key_i].add(i)
                                best_line_points[key_i].add(k)
                            key_j = _line_key(old_xj, old_yj, best_points[k][0], best_points[k][1])
                            if key_j is not None:
                                best_line_points[key_j].add(j)
                                best_line_points[key_j].add(k)
                        # Also handle line between i and j
                        key_ij = _line_key(old_xi, old_yi, old_xj, old_yj)
                        if key_ij is not None:
                            best_line_points[key_ij].add(i)
                            best_line_points[key_ij].add(j)

    # Final verification
    valid, final_score = _evaluate(best_points)
    if not valid:
        # Fallback
        best_points = [[x, y] for x in range(4) for y in range(4)]
        valid, final_score = _evaluate(best_points)

    return best_points
# EVOLVE-BLOCK-END

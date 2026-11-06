"""Combined program: structured candidate generation + fast counting + multi-strategy local search."""
import time
import math
import random
from collections import defaultdict
from math import gcd

# EVOLVE-BLOCK-START

def count_ordinary_lines(points):
    """Count lines containing exactly 2 of the given points. Fast O(n^2) with dict."""
    n = len(points)
    if n < 3:
        return 0
    lines = defaultdict(set)
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            if dx < 0:
                dx = -dx
                dy = -dy
            elif dx == 0 and dy < 0:
                dy = -dy
            # Line key: (a, b, c) where a*x + b*y = c
            # a = dy, b = -dx, c = dy*x1 - dx*y1
            a = dy
            b = -dx
            c = a * x1 + b * y1
            key = (a, b, c)
            lines[key].add(i)
            lines[key].add(j)
    ordinary = 0
    for pts in lines.values():
        if len(pts) == 2:
            ordinary += 1
    return ordinary


def _is_valid(pts):
    """Check: 12 distinct integer points, |x|,|y|<=10^9, not all collinear."""
    if len(pts) != 12:
        return False
    if len(set(tuple(p) for p in pts)) != 12:
        return False
    for p in pts:
        if abs(p[0]) > 10**9 or abs(p[1]) > 10**9:
            return False
    if len(pts) >= 3:
        (x0, y0), (x1, y1), (x2, y2) = pts[0], pts[1], pts[2]
        all_col = True
        for i in range(3, 12):
            xi, yi = pts[i]
            if (x1 - x0) * (yi - y0) != (y1 - y0) * (xi - x0):
                all_col = False
                break
        if all_col:
            return False
    return True


def _make_initial_configs():
    """Generate a diverse set of structured starting configurations."""
    configs = []

    # 1. 3x4 grid
    configs.append([[x, y] for x in range(4) for y in range(3)])

    # 2. 2x6 grid
    configs.append([[x, y] for x in range(6) for y in range(2)])

    # 3. 6x2 grid
    configs.append([[x, y] for x in range(2) for y in range(6)])

    # 4. Diagonal-shifted grid: y in 0..3, x in 0..2, point=(x+y, y)
    pts_diag = []
    for y in range(4):
        for x in range(3):
            pts_diag.append([x + y, y])
    configs.append(pts_diag)

    # 5. 4x4 minus corners
    pts = [[x, y] for x in range(4) for y in range(4)]
    for corner in [[0, 0], [3, 0], [0, 3], [3, 3]]:
        if corner in pts:
            pts.remove(corner)
    configs.append(pts)

    # 6. 5x3 grid (first 12)
    configs.append([[x, y] for x in range(5) for y in range(3)][:12])

    # 7. Hexagonal-like: two rows with offset
    pts_hex = []
    for x in range(6):
        pts_hex.append([2 * x, 0])
    for x in range(6):
        pts_hex.append([2 * x + 1, 2])
    configs.append(pts_hex)

    # 8. Regular hexagon vertices + doubled
    R = 10**6
    hex_pts = [
        [R, 0],
        [R // 2, int(R * 0.866025)],
        [-R // 2, int(R * 0.866025)],
        [-R, 0],
        [-R // 2, -int(R * 0.866025)],
        [R // 2, -int(R * 0.866025)],
    ]
    hex_pts2 = list(hex_pts)
    for p in hex_pts:
        hex_pts2.append([2 * p[0], 2 * p[1]])
    configs.append(hex_pts2)

    # 9. Near-pencil: 3 points on each of 4 lines through origin
    near_pencil = []
    for d in [1, 2, 3]:
        near_pencil.append([d, 0])
        near_pencil.append([d, d])
        near_pencil.append([-d, -d])
        near_pencil.append([0, d])
    configs.append(near_pencil)

    # 10. Staggered rows
    pts_stag = []
    for y in range(4):
        offset = y % 2
        for x in range(3):
            pts_stag.append([x + offset * 2, y])
    configs.append(pts_stag)

    # 11. 3 points on y=0, 3 on y=1, 3 on y=2, 3 on y=3 with increasing offset
    pts_off = []
    for y in range(4):
        for x in range(3):
            pts_off.append([x + y, y])
    configs.append(pts_off)

    # 12. Two parallel lines of 6 with interleave
    pts_par = []
    for x in range(6):
        pts_par.append([x, 0])
        pts_par.append([x, 1])
    configs.append(pts_par)

    # 13. Triangle-based: 4 on each side of a triangle (with shared vertices)
    # Triangle vertices: (0,0), (6,0), (3,6)
    # Side 1 (bottom): (0,0),(2,0),(4,0),(6,0)
    # Side 2 (left): (0,0),(1,2),(2,4),(3,6)
    # Side 3 (right): (6,0),(5,2),(4,4),(3,6)
    # That's 4+4+4 - 3 shared = 9... need 12. Add 3 interior or adjust.
    # Let's do: 4 on each of 3 lines, no shared vertices (parallel lines)
    # y=0: x=0,1,2,3
    # y=2: x=1,2,3,4
    # y=4: x=2,3,4,5
    pts_tri = []
    for y, ox in [(0, 0), (2, 1), (4, 2)]:
        for x in range(4):
            pts_tri.append([x + ox, y])
    configs.append(pts_tri)

    # 14. Random structured configs
    rng = random.Random(12345)
    for _ in range(30):
        pts = []
        for _ in range(12):
            pts.append([rng.randint(-10, 10), rng.randint(-10, 10)])
        if len(set(map(tuple, pts))) == 12:
            configs.append(pts)

    # 15. 3x4 grid with one row shifted
    pts_shift = []
    for x in range(4):
        pts_shift.append([x, 0])
    for x in range(4):
        pts_shift.append([x + 1, 1])
    for x in range(4):
        pts_shift.append([x, 2])
    configs.append(pts_shift)

    # 16. 4 points on y=0, 4 on y=2, 4 on y=4 with different x ranges
    pts_3row = []
    for x in range(4):
        pts_3row.append([x, 0])
    for x in range(1, 5):
        pts_3row.append([x, 2])
    for x in range(2, 6):
        pts_3row.append([x, 4])
    configs.append(pts_3row)

    # Filter valid configs
    valid_configs = []
    for cfg in configs:
        if _is_valid(cfg):
            valid_configs.append(cfg)

    return valid_configs


def _find_collinear_targets(points, max_k=5):
    """Find positions where a new point would be collinear with 2 existing points."""
    n = len(points)
    targets = set()
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            g = gcd(abs(dx), abs(dy))
            if g == 0:
                continue
            dx //= g
            dy //= g
            for k in range(-max_k, max_k + 1):
                if k == 0:
                    continue
                nx = p1[0] + k * dx
                ny = p1[1] + k * dy
                if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                    if [nx, ny] not in points:
                        targets.add((nx, ny))
    return list(targets)


def _evaluate_and_update(points, best_points, best_score):
    """Evaluate a configuration and update best if improved."""
    if not _is_valid(points):
        return best_points, best_score
    score = count_ordinary_lines(points)
    if score < best_score:
        return [p[:] for p in points], score
    return best_points, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin (evaluator kills at 150s)

    # Generate and evaluate initial configs
    initial_configs = _make_initial_configs()

    best_points = None
    best_score = float('inf')

    for cfg in initial_configs:
        if time.time() > deadline:
            break
        score = count_ordinary_lines(cfg)
        if score < best_score:
            best_score = score
            best_points = [p[:] for p in cfg]
        if best_score == 6:
            break

    if best_points is None:
        best_points = [[x, y] for x in range(4) for y in range(3)]
        best_score = count_ordinary_lines(best_points)

    # Local search with multiple strategies
    rng = random.Random(42)
    no_improve_count = 0
    max_no_improve = 20000
    step_counter = 0

    while time.time() < deadline:
        step_counter += 1
        strategy = rng.random()

        if strategy < 0.30:
            # Single point random move
            idx = rng.randint(0, 11)
            old_p = best_points[idx][:]

            step_size = rng.choice([1, 2, 3, 5, 8, 12, 16, 24, 32, 48, 64])
            dx = rng.randint(-step_size, step_size)
            dy = rng.randint(-step_size, step_size)
            if dx == 0 and dy == 0:
                continue

            new_p = [old_p[0] + dx, old_p[1] + dy]

            new_points = [p[:] for p in best_points]
            new_points[idx] = new_p
            old_best_score = best_score
            best_points, best_score = _evaluate_and_update(new_points, best_points, best_score)

            if best_score == 6:
                break
            if best_score < old_best_score:
                no_improve_count = 0
            else:
                no_improve_count += 1

        elif strategy < 0.55:
            # Move a point to a collinear target position
            targets = _find_collinear_targets(best_points)
            if targets:
                idx = rng.randint(0, 11)
                target = rng.choice(targets)
                if [target[0], target[1]] == best_points[idx]:
                    continue

                new_points = [p[:] for p in best_points]
                new_points[idx] = [target[0], target[1]]
                old_best_score = best_score
                best_points, best_score = _evaluate_and_update(new_points, best_points, best_score)

                if best_score == 6:
                    break
                if best_score < old_best_score:
                    no_improve_count = 0
                else:
                    no_improve_count += 1

        elif strategy < 0.75:
            # Swap two points' positions (perturb both)
            i, j = rng.sample(range(12), 2)
            p1 = best_points[i][:]
            p2 = best_points[j][:]

            step_size = rng.choice([1, 2, 3, 5, 8, 12])
            dx1 = rng.randint(-step_size, step_size)
            dy1 = rng.randint(-step_size, step_size)
            dx2 = rng.randint(-step_size, step_size)
            dy2 = rng.randint(-step_size, step_size)

            if dx1 == 0 and dy1 == 0 and dx2 == 0 and dy2 == 0:
                continue

            new_points = [p[:] for p in best_points]
            new_points[i] = [p1[0] + dx1, p1[1] + dy1]
            new_points[j] = [p2[0] + dx2, p2[1] + dy2]
            old_best_score = best_score
            best_points, best_score = _evaluate_and_update(new_points, best_points, best_score)

            if best_score == 6:
                break
            if best_score < old_best_score:
                no_improve_count = 0
            else:
                no_improve_count += 1

        elif strategy < 0.85:
            # Replace a point with a random point that lies on a line through 2 existing points
            # (more targeted than collinear targets)
            n = 12
            targets = []
            for i in range(n):
                for j in range(i + 1, n):
                    x1, y1 = best_points[i]
                    x2, y2 = best_points[j]
                    dx = x2 - x1
                    dy = y2 - y1
                    g = gcd(abs(dx), abs(dy))
                    if g == 0:
                        continue
                    dx //= g
                    dy //= g
                    for k in range(-8, 9):
                        if k == 0:
                            continue
                        nx = x1 + k * dx
                        ny = y1 + k * dy
                        if abs(nx) <= 10**9 and abs(ny) <= 10**9:
                            if [nx, ny] not in best_points:
                                targets.append([nx, ny])
            if targets:
                idx = rng.randint(0, 11)
                target = rng.choice(targets)
                if target == best_points[idx]:
                    continue
                new_points = [p[:] for p in best_points]
                new_points[idx] = target
                old_best_score = best_score
                best_points, best_score = _evaluate_and_update(new_points, best_points, best_score)
                if best_score == 6:
                    break
                if best_score < old_best_score:
                    no_improve_count = 0
                else:
                    no_improve_count += 1

        else:
            # Occasional larger restart: perturb 3 random points significantly
            indices = rng.sample(range(12), 3)
            new_points = [p[:] for p in best_points]
            for idx in indices:
                step_size = rng.choice([10, 20, 30, 50])
                dx = rng.randint(-step_size, step_size)
                dy = rng.randint(-step_size, step_size)
                new_points[idx] = [new_points[idx][0] + dx, new_points[idx][1] + dy]
            old_best_score = best_score
            best_points, best_score = _evaluate_and_update(new_points, best_points, best_score)
            if best_score == 6:
                break
            if best_score < old_best_score:
                no_improve_count = 0
            else:
                no_improve_count += 1

        # Check for stagnation and do a mini-restart
        if no_improve_count > max_no_improve:
            no_improve_count = 0
            # Try a few random restarts from scratch
            for _ in range(5):
                if time.time() > deadline:
                    break
                pts = []
                for _ in range(12):
                    pts.append([rng.randint(-20, 20), rng.randint(-20, 20)])
                if _is_valid(pts):
                    s = count_ordinary_lines(pts)
                    if s < best_score:
                        best_score = s
                        best_points = [p[:] for p in pts]
                        if best_score == 6:
                            break

    return best_points

# EVOLVE-BLOCK-END

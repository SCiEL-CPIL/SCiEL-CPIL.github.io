"""
Optimized solver for the 16-point integer collinearity problem.
Uses a fast incremental scorer, a line-completion heuristic, and simulated annealing
with multiple restarts to maximize the number of lines with exactly 4 points.
"""
import time
import random
import math
from math import gcd

_DEADLINE = 140.0  # Safety margin before 150s limit


def _canonical_line_key(p1, p2):
    """
    Returns a canonical tuple (A, B, C) representing the line Ax + By = C.
    Normalization:
    1. A, B are integers.
    2. gcd(A, B) = 1.
    3. A > 0, or (A == 0 and B > 0).
    """
    x1, y1 = p1
    x2, y2 = p2

    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        return None

    A = dy
    B = -dx
    C = A * x1 + B * y1

    g = gcd(abs(A), abs(B))
    if g > 0:
        A //= g
        B //= g
        C //= g

    if A < 0 or (A == 0 and B < 0):
        A = -A
        B = -B
        C = -C

    return (A, B, C)


def _score_and_lines(points):
    """
    Evaluates the configuration.
    Returns (score, valid, line_map).
    Score: number of lines with exactly 4 points.
    Valid: True if no 5 collinear, distinct, and within bounds.
    line_map: dict of line_key -> set of point indices.
    """
    n = len(points)
    if n != 16:
        return -1, False, {}

    seen = set()
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return -1, False, {}
        pt = (x, y)
        if pt in seen:
            return -1, False, {}
        seen.add(pt)

    line_map = {}

    for i in range(n):
        for j in range(i + 1, n):
            key = _canonical_line_key(points[i], points[j])
            if key is None:
                return -1, False, {}
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)

    exact_4_count = 0
    for key, pts_in_line in line_map.items():
        if len(pts_in_line) > 4:
            return -1, False, line_map
        if len(pts_in_line) == 4:
            exact_4_count += 1

    return exact_4_count, True, line_map


def _fast_score(points):
    """
    Fast scorer that only returns the score (number of exact-4 lines) or -1 if invalid.
    Used for quick checks during search.
    """
    n = len(points)
    if n != 16:
        return -1

    seen = set()
    for x, y in points:
        if abs(x) > 1000 or abs(y) > 1000:
            return -1
        pt = (x, y)
        if pt in seen:
            return -1
        seen.add(pt)

    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _canonical_line_key(points[i], points[j])
            if key is None:
                return -1
            if key not in line_map:
                line_map[key] = 0
            line_map[key] += 1

    exact_4_count = 0
    for cnt in line_map.values():
        if cnt > 6:  # More than 6 pairs means more than 4 points
            return -1
        if cnt == 6:
            exact_4_count += 1

    return exact_4_count


def _find_completion_candidates(points, line_map):
    """
    Find lines with exactly 3 points and return candidate positions to complete them to 4.
    Returns a list of (idx_to_move, new_position, line_key).
    """
    candidates = []
    occupied = set((p[0], p[1]) for p in points)

    for key, pts_in_line in line_map.items():
        if len(pts_in_line) == 3:
            A, B, C = key
            line_pts_idx = list(pts_in_line)

            # Find points NOT on this line
            others = [i for i in range(16) if i not in pts_in_line]
            if not others:
                continue

            # Generate a few candidate positions on the line
            # We want integer points (x, y) such that Ax + By = C
            # and (x, y) not in occupied.

            # Strategy: pick a point from 'others' and try to move it to the line.
            # To keep it local, we look for positions near that point.
            # But for simplicity and speed, let's just generate a few random valid points on the line.

            # If A == 0: y = C/B (if B != 0)
            # If B == 0: x = C/A (if A != 0)
            # If both non-zero: x can be any integer such that (C - Ax) is divisible by B.

            # Let's try to find points on the line that are "reasonable" (within bounds).
            # We'll pick a few x values and check if y is integer and in bounds.

            if A == 0:
                if B == 0:
                    continue
                if C % B != 0:
                    continue
                y_val = C // B
                if abs(y_val) > 1000:
                    continue
                # x can be anything. Let's pick a few random x values.
                for _ in range(5):
                    x_val = random.randint(-1000, 1000)
                    if (x_val, y_val) in occupied:
                        continue
                    # Check if moving any point to (x_val, y_val) is valid.
                    # We'll just return the position and let the caller decide which point to move.
                    # Actually, we need to return the index to move too.
                    # Let's pick a random point from 'others'.
                    idx = random.choice(others)
                    candidates.append((idx, (x_val, y_val), key))
            elif B == 0:
                if C % A != 0:
                    continue
                x_val = C // A
                if abs(x_val) > 1000:
                    continue
                for _ in range(5):
                    y_val = random.randint(-1000, 1000)
                    if (x_val, y_val) in occupied:
                        continue
                    idx = random.choice(others)
                    candidates.append((idx, (x_val, y_val), key))
            else:
                # Both A and B non-zero.
                # We need x such that (C - Ax) % B == 0.
                # Let's try a few random x values.
                for _ in range(10):
                    x_val = random.randint(-1000, 1000)
                    rem = C - A * x_val
                    if rem % B != 0:
                        continue
                    y_val = rem // B
                    if abs(y_val) > 1000:
                        continue
                    if (x_val, y_val) in occupied:
                        continue
                    idx = random.choice(others)
                    candidates.append((idx, (x_val, y_val), key))

    return candidates


def _local_search_heuristic(points, time_limit):
    """
    Greedy local search using line-completion heuristic.
    """
    current = [list(p) for p in points]
    current_score, valid, current_line_map = _score_and_lines(current)

    if not valid:
        return current, 0

    best = [list(p) for p in current]
    best_score = current_score

    start_time = time.time()

    while time.time() < time_limit:
        candidates = _find_completion_candidates(current, current_line_map)

        if not candidates:
            break

        # Pick the best candidate
        best_candidate = None
        best_candidate_score = current_score

        for idx, new_pos, key in candidates:
            trial = [list(p) for p in current]
            trial[idx] = [new_pos[0], new_pos[1]]

            score = _fast_score(trial)
            if score > best_candidate_score:
                best_candidate_score = score
                best_candidate = (idx, new_pos, trial)

        if best_candidate is None:
            break

        idx, new_pos, trial = best_candidate
        current = trial
        current_score = best_candidate_score
        current_score, valid, current_line_map = _score_and_lines(current)

        if not valid:
            # Revert
            current = [list(p) for p in best]
            current_score = best_score
            current_score, valid, current_line_map = _score_and_lines(current)
            break

        if current_score > best_score:
            best_score = current_score
            best = [list(p) for p in current]

    return best, best_score


def _make_grid_4x4():
    """4x4 grid centered at origin."""
    pts = []
    for x in range(-1, 3):
        for y in range(-1, 3):
            pts.append([x, y])
    return pts


def _make_perturbed_grid(seed):
    """
    Generates a perturbed grid to provide a better starting point.
    """
    random.seed(seed)
    base = _make_grid_4x4()
    s = random.randint(1, 3)
    scale = random.randint(2, 5)
    offset_x = random.randint(-50, 50)
    offset_y = random.randint(-50, 50)

    pts = []
    for x, y in base:
        nx = (x * scale) + (y * s) + offset_x
        ny = (y * scale) + offset_y
        pts.append([nx, ny])
    return pts


def _make_random_points():
    """Random points within bounds."""
    return [[random.randint(-1000, 1000), random.randint(-1000, 1000)] for _ in range(16)]


def _anneal(points, time_limit, seed=None):
    """
    Runs Simulated Annealing with structure-aware moves.
    """
    if seed is not None:
        random.seed(seed)

    current = [list(p) for p in points]
    current_score, valid, current_line_map = _score_and_lines(current)

    if not valid:
        # Try to fix initial state
        for _ in range(100):
            idx = random.randint(0, 15)
            current[idx] = [current[idx][0] + random.randint(-2, 2),
                            current[idx][1] + random.randint(-2, 2)]
            current_score, valid, current_line_map = _score_and_lines(current)
            if valid:
                break
        if not valid:
            current = _make_grid_4x4()
            current_score, valid, current_line_map = _score_and_lines(current)
            if not valid:
                return current, 0

    best = [list(p) for p in current]
    best_score = current_score

    T = 5.0
    T_min = 0.01
    cooling = 0.99995

    start_time = time.time()

    # Precompute occupied set for fast lookup
    occupied = set((p[0], p[1]) for p in current)

    while time.time() < time_limit:
        move_type = random.random()

        if move_type < 0.6:
            # Move Type 1: Pull to Line (Complete a 3-point line)
            three_point_lines = [key for key, pts in current_line_map.items() if len(pts) == 3]

            if three_point_lines:
                key = random.choice(three_point_lines)
                A, B, C = key
                line_pts_idx = current_line_map[key]

                # Pick a point P to move (not on this line)
                others = [i for i in range(16) if i not in line_pts_idx]
                if others:
                    idx = random.choice(others)
                    P = current[idx]

                    # Find candidate positions on the line Ax+By=C
                    # We want to move P to a point D on the line.
                    # D should be "close" to P or just any valid point.
                    # Let's try to find integer solutions (x,y) to Ax+By=C
                    # that are not in occupied.

                    found_target = False
                    # Search range around P
                    # If A != 0, x = (C - By)/A.
                    # If B != 0, y = (C - Ax)/B.

                    # Iterate x in a range around P[0]
                    # We need to be careful with the range.
                    # Since we are moving P, D can be anywhere, but closer is better for "local" moves.
                    # However, to create new lines, we might need to jump.
                    # Let's try a few candidates.

                    # If A == 0: y is fixed.
                    if A == 0:
                        if C % B != 0:
                            continue
                        y_val = C // B
                        if abs(y_val) > 1000:
                            continue
                        # x can be anything. Let's pick a few x values.
                        # Maybe x close to P[0]?
                        for dx in [-5, -2, 0, 2, 5, 10, -10]:
                            x_val = P[0] + dx
                            if abs(x_val) > 1000:
                                continue
                            if (x_val, y_val) in occupied:
                                continue
                            # Trial move
                            trial = [list(p) for p in current]
                            trial[idx] = [x_val, y_val]

                            # Update occupied
                            occupied.remove(tuple(P))
                            occupied.add((x_val, y_val))

                            score = _fast_score(trial)

                            occupied.remove((x_val, y_val))
                            occupied.add(tuple(P))

                            if score >= 0:
                                found_target = True
                                delta = score - current_score
                                if delta > 0 or (delta < 0 and random.random() < math.exp(delta / T)):
                                    current = trial
                                    current_score = score
                                    current_score, valid, current_line_map = _score_and_lines(current)
                                    if score > best_score:
                                        best_score = score
                                        best = [list(p) for p in current]
                                    break
                    else:
                        # A != 0.
                        # x = (C - By) / A.
                        # We can iterate y in a range around P[1] or x in a range around P[0].
                        # Let's iterate x.
                        # y = (C - Ax) / B.
                        # If B == 0, x is fixed.
                        if B == 0:
                            if C % A != 0:
                                continue
                            x_val = C // A
                            if abs(x_val) > 1000:
                                continue
                            for dy in [-5, -2, 0, 2, 5, 10, -10]:
                                y_val = P[1] + dy
                                if abs(y_val) > 1000:
                                    continue
                                if (x_val, y_val) in occupied:
                                    continue
                                trial = [list(p) for p in current]
                                trial[idx] = [x_val, y_val]

                                occupied.remove(tuple(P))
                                occupied.add((x_val, y_val))

                                score = _fast_score(trial)

                                occupied.remove((x_val, y_val))
                                occupied.add(tuple(P))

                                if score >= 0:
                                    found_target = True
                                    delta = score - current_score
                                    if delta > 0 or (delta < 0 and random.random() < math.exp(delta / T)):
                                        current = trial
                                        current_score = score
                                        current_score, valid, current_line_map = _score_and_lines(current)
                                        if score > best_score:
                                            best_score = score
                                            best = [list(p) for p in current]
                                        break
                        else:
                            # Both A and B non-zero.
                            # Iterate x around P[0]
                            for dx in [-10, -5, -2, -1, 1, 2, 5, 10]:
                                x_val = P[0] + dx
                                if abs(x_val) > 1000:
                                    continue
                                rem = C - A * x_val
                                if rem % B != 0:
                                    continue
                                y_val = rem // B
                                if abs(y_val) > 1000:
                                    continue
                                if (x_val, y_val) in occupied:
                                    continue

                                trial = [list(p) for p in current]
                                trial[idx] = [x_val, y_val]

                                occupied.remove(tuple(P))
                                occupied.add((x_val, y_val))

                                score = _fast_score(trial)

                                occupied.remove((x_val, y_val))
                                occupied.add(tuple(P))

                                if score >= 0:
                                    found_target = True
                                    delta = score - current_score
                                    if delta > 0 or (delta < 0 and random.random() < math.exp(delta / T)):
                                        current = trial
                                        current_score = score
                                        current_score, valid, current_line_map = _score_and_lines(current)
                                        if score > best_score:
                                            best_score = score
                                            best = [list(p) for p in current]
                                        break

        elif move_type < 0.9:
            # Move Type 2: Small Jitter
            idx = random.randint(0, 15)
            delta_x = random.randint(-3, 3)
            delta_y = random.randint(-3, 3)
            if delta_x == 0 and delta_y == 0:
                continue
            new_pt = [current[idx][0] + delta_x, current[idx][1] + delta_y]
            if abs(new_pt[0]) > 1000 or abs(new_pt[1]) > 1000:
                continue
            if (new_pt[0], new_pt[1]) in occupied:
                continue

            trial = [list(p) for p in current]
            trial[idx] = new_pt

            occupied.remove(tuple(current[idx]))
            occupied.add(tuple(new_pt))

            score = _fast_score(trial)

            occupied.remove(tuple(new_pt))
            occupied.add(tuple(current[idx]))

            if score >= 0:
                delta = score - current_score
                if delta > 0 or (delta < 0 and random.random() < math.exp(delta / T)):
                    current = trial
                    current_score = score
                    current_score, valid, current_line_map = _score_and_lines(current)
                    if current_score > best_score:
                        best_score = current_score
                        best = [list(p) for p in current]
        else:
            # Move Type 3: Large Jump
            idx = random.randint(0, 15)
            new_pt = [random.randint(-1000, 1000), random.randint(-1000, 1000)]
            if (new_pt[0], new_pt[1]) in occupied:
                continue

            trial = [list(p) for p in current]
            trial[idx] = new_pt

            occupied.remove(tuple(current[idx]))
            occupied.add(tuple(new_pt))

            score = _fast_score(trial)

            occupied.remove(tuple(new_pt))
            occupied.add(tuple(current[idx]))

            if score >= 0:
                delta = score - current_score
                if delta > 0 or (delta < 0 and random.random() < math.exp(delta / T)):
                    current = trial
                    current_score = score
                    current_score, valid, current_line_map = _score_and_lines(current)
                    if current_score > best_score:
                        best_score = current_score
                        best = [list(p) for p in current]

        T = max(T * cooling, T_min)

    return best, best_score


def construct():
    start_time = time.time()
    remaining = _DEADLINE - start_time

    best_points = None
    best_score = -1

    # Define multiple starting configurations
    starts = [
        _make_grid_4x4(),
        _make_perturbed_grid(1),
        _make_perturbed_grid(2),
        _make_perturbed_grid(3),
        _make_random_points(),
    ]

    num_starts = len(starts)
    time_per_start = remaining / (num_starts + 1)

    for i, start in enumerate(starts):
        if time.time() > _DEADLINE - 5:
            break

        s, v, _ = _score_and_lines(start)
        if not v:
            for _ in range(20):
                idx = random.randint(0, 15)
                start[idx] = [start[idx][0] + random.randint(-5, 5),
                              start[idx][1] + random.randint(-5, 5)]
                s, v, _ = _score_and_lines(start)
                if v:
                    break
            if not v:
                continue

        # First, try local search heuristic
        local_best, local_score = _local_search_heuristic(start, time.time() + time_per_start / 2)

        # Then, try annealing
        if time.time() < _DEADLINE - 5:
            anneal_best, anneal_score = _anneal(local_best, time.time() + time_per_start / 2, seed=100 + i)
            if anneal_score > local_score:
                local_best = anneal_best
                local_score = anneal_score

        if local_score > best_score:
            best_score = local_score
            best_points = local_best

    if best_points is not None and time.time() < _DEADLINE - 2:
        time_left = _DEADLINE - 2 - time.time()
        if time_left > 5:
            final_best, final_score = _anneal(best_points, time.time() + time_left, seed=999)
            if final_score > best_score:
                best_score = final_score
                best_points = final_best
    else:
        best_points = _make_grid_4x4()
        best_score, _, _ = _score_and_lines(best_points)

    return [list(p) for p in best_points]

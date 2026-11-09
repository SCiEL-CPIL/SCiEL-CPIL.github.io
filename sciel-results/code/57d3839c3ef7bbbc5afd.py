# EVOLVE-BLOCK-START
import numpy as np
import math
import time


def _normalize_line(A, B, C):
    """Normalize a line so it's unique: divide by gcd, make first nonzero positive."""
    g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)


def _count_incidences(points, lines):
    """Count point-line incidences exactly."""
    n, m = len(points), len(lines)
    count = 0
    for i in range(n):
        x, y = points[i]
        for j in range(m):
            A, B, C = lines[j]
            if A * x + B * y + C == 0:
                count += 1
    return count


def _try_concentric_lines(p=59, n_lines=27):
    """
    Lines all through origin in F_p^2. Pick 54 best intersection points.
    (0,0) is on all n_lines; other intersections are on exactly 2 lines.
    """
    chosen_lines = [(1, 0, 0)]
    for m in range(1, n_lines):
        chosen_lines.append((-m % p, 1, 0))
    
    intersection_points = set()
    for i in range(n_lines):
        A1, B1, C1 = chosen_lines[i]
        for j in range(i + 1, n_lines):
            A2, B2, C2 = chosen_lines[j]
            D = (A1 * B2 - A2 * B1) % p
            if D == 0:
                continue
            D_inv = pow(D, p - 2, p)
            x = ((C2 * B1 - C1 * B2) * D_inv) % p
            y = ((A1 * C2 - A2 * C1) * D_inv) % p
            intersection_points.add((x, y))
    
    points_list = [(0, 0)]
    for pt in intersection_points:
        if pt != (0, 0):
            points_list.append(pt)
        if len(points_list) >= 54:
            break
    
    if len(points_list) < 54:
        for x in range(p):
            for y in range(p):
                if (x, y) not in points_list:
                    points_list.append((x, y))
                if len(points_list) >= 54:
                    break
            if len(points_list) >= 54:
                break
    
    points_arr = np.array(points_list[:54], dtype=np.int64)
    lines_arr = np.array(chosen_lines, dtype=np.int64)
    
    total = 0
    for (px, py) in points_list[:54]:
        for (A, B, C) in chosen_lines:
            if (A * px + B * py + C) % p == 0:
                total += 1
    
    return points_arr, lines_arr, total


def _try_two_centers(p=59, n1=14, n2=14):
    """
    n1 lines through P=(0,0), n2 lines through Q=(1,0), sharing the x-axis.
    Total distinct lines = n1 + n2 - 1 = 27.
    """
    lines_p = [(1, 0, 0)]
    lines_p.append((0, 1, 0))
    for m in range(1, n1 - 1):
        lines_p.append((-m % p, 1, 0))
    
    lines_q = [(1, 0, -1 % p)]
    for m in range(1, n2 - 1):
        lines_q.append((m, -1, -m % p))
    
    all_lines = lines_p + lines_q
    unique_lines = []
    seen = set()
    for line in all_lines:
        norm = _normalize_line(*line)
        if norm not in seen:
            seen.add(norm)
            unique_lines.append(line)
    
    unique_lines = unique_lines[:27]
    
    intersection_points = set()
    n_lines = len(unique_lines)
    for i in range(n_lines):
        A1, B1, C1 = unique_lines[i]
        for j in range(i + 1, n_lines):
            A2, B2, C2 = unique_lines[j]
            D = (A1 * B2 - A2 * B1) % p
            if D == 0:
                continue
            D_inv = pow(D, p - 2, p)
            x = ((C2 * B1 - C1 * B2) * D_inv) % p
            y = ((A1 * C2 - A2 * C1) * D_inv) % p
            intersection_points.add((x, y))
    
    point_incs = {}
    for pt in intersection_points:
        px, py = pt
        count = 0
        for (A, B, C) in unique_lines:
            if (A * px + B * py + C) % p == 0:
                count += 1
        point_incs[pt] = count
    
    sorted_points = sorted(point_incs.items(), key=lambda x: -x[1])
    selected_points = [pt for pt, _ in sorted_points[:54]]
    
    if len(selected_points) < 54:
        for x in range(p):
            for y in range(p):
                if (x, y) not in selected_points:
                    selected_points.append((x, y))
                if len(selected_points) >= 54:
                    break
            if len(selected_points) >= 54:
                break
    
    points_arr = np.array(selected_points[:54], dtype=np.int64)
    lines_arr = np.array(unique_lines[:27], dtype=np.int64)
    
    total = 0
    for (px, py) in selected_points[:54]:
        for (A, B, C) in unique_lines[:27]:
            if (A * px + B * py + C) % p == 0:
                total += 1
    
    return points_arr, lines_arr, total


def _try_grid_configuration(a, b, p=59):
    """Try an a x b grid in F_p^2 and find best 27 lines."""
    xs = list(range(a))
    ys = list(range(b))
    points = [(x, y) for x in xs for y in ys]
    
    if len(points) != 54:
        return None, None, -1
    
    line_inc_counts = {}
    
    for c in range(p):
        A, B, C = 1, 0, -c % p
        count = 0
        for (px, py) in points:
            if (A * px + B * py + C) % p == 0:
                count += 1
        if count > 0:
            key = _normalize_line(A, B, C)
            line_inc_counts[key] = max(line_inc_counts.get(key, 0), count)
    
    for m in range(p):
        for b in range(p):
            A, B, C = -m % p, 1, -b % p
            count = 0
            for (px, py) in points:
                if (A * px + B * py + C) % p == 0:
                    count += 1
            if count > 0:
                key = _normalize_line(A, B, C)
                line_inc_counts[key] = max(line_inc_counts.get(key, 0), count)
    
    sorted_lines = sorted(line_inc_counts.items(), key=lambda x: -x[1])
    top_27 = sorted_lines[:27]
    total_inc = sum(count for _, count in top_27)
    
    points_arr = np.array(points, dtype=np.int64)
    lines_arr = np.array([line for line, _ in top_27], dtype=np.int64)
    
    return points_arr, lines_arr, total_inc


def _local_search_points(points, lines, max_iterations=200, time_limit=30):
    """Local search: move points to improve incidences."""
    start_time = time.time()
    current_points = points.copy()
    current_lines = lines.copy()
    current_score = _count_incidences(current_points, current_lines)
    n = len(current_points)
    
    for iteration in range(max_iterations):
        if time.time() > start_time + time_limit:
            break
        
        improved = False
        for pi in range(n):
            if time.time() > start_time + time_limit:
                break
            
            orig_x, orig_y = int(current_points[pi][0]), int(current_points[pi][1])
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    new_x = orig_x + dx
                    new_y = orig_y + dy
                    if new_x < 0 or new_y < 0:
                        continue
                    
                    occupied = False
                    for i in range(n):
                        if i != pi and current_points[i][0] == new_x and current_points[i][1] == new_y:
                            occupied = True
                            break
                    if occupied:
                        continue
                    
                    saved = current_points[pi].copy()
                    current_points[pi] = np.array([new_x, new_y], dtype=np.int64)
                    new_score = _count_incidences(current_points, current_lines)
                    
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                    else:
                        current_points[pi] = saved
            
            if not improved:
                break
        
        if not improved:
            break
    
    return current_points, current_lines, current_score


def _refine_lines_for_points(points, p=59, max_time=30):
    """Given fixed points, find the best 27 lines in F_p."""
    start_time = time.time()
    n = len(points)
    
    line_inc_counts = {}
    
    for c in range(p):
        A, B, C = 1, 0, -c % p
        count = 0
        for i in range(n):
            px, py = int(points[i][0]), int(points[i][1])
            if (A * px + B * py + C) % p == 0:
                count += 1
        if count > 0:
            key = _normalize_line(A, B, C)
            line_inc_counts[key] = max(line_inc_counts.get(key, 0), count)
    
    for m in range(p):
        for b in range(p):
            if time.time() > start_time + max_time:
                break
            A, B, C = -m % p, 1, -b % p
            count = 0
            for i in range(n):
                px, py = int(points[i][0]), int(points[i][1])
                if (A * px + B * py + C) % p == 0:
                    count += 1
            if count > 0:
                key = _normalize_line(A, B, C)
                line_inc_counts[key] = max(line_inc_counts.get(key, 0), count)
    
    sorted_lines = sorted(line_inc_counts.items(), key=lambda x: -x[1])
    top_27 = sorted_lines[:27]
    
    if len(top_27) < 27:
        for c in range(p):
            A, B, C = 1, 0, -c % p
            key = _normalize_line(A, B, C)
            if key not in [l for l, _ in top_27]:
                top_27.append((key, 0))
            if len(top_27) >= 27:
                break
    
    selected_lines = np.array([line for line, _ in top_27[:27]], dtype=np.int64)
    score = sum(count for _, count in top_27[:27])
    
    return selected_lines, score


def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences with 54 points and 27 lines.
    Uses finite field F_59 construction.
    """
    start_time = time.time()
    deadline = start_time + 570
    
    best_points = None
    best_lines = None
    best_score = -1
    
    strategies = [
        ("concentric", lambda: _try_concentric_lines(59, 27)),
        ("two_centers", lambda: _try_two_centers(59, 14, 14)),
        ("grid_6x9", lambda: _try_grid_configuration(6, 9, 59)),
        ("grid_9x6", lambda: _try_grid_configuration(9, 6, 59)),
        ("grid_3x18", lambda: _try_grid_configuration(3, 18, 59)),
        ("grid_18x3", lambda: _try_grid_configuration(18, 3, 59)),
        ("grid_2x27", lambda: _try_grid_configuration(2, 27, 59)),
        ("grid_27x2", lambda: _try_grid_configuration(27, 2, 59)),
    ]
    
    for name, func in strategies:
        if time.time() > deadline - 60:
            break
        try:
            points, lines, score = func()
            if points is not None and score > best_score:
                best_score = score
                best_points = points.copy()
                best_lines = lines.copy()
        except Exception:
            continue
    
    if best_points is not None and time.time() < deadline - 90:
        refined_points, refined_lines, refined_score = _local_search_points(
            best_points, best_lines, max_iterations=300, time_limit=45
        )
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_points
            best_lines = refined_lines
        
        if time.time() < deadline - 30:
            final_lines, final_score = _refine_lines_for_points(best_points, p=59, max_time=20)
            if final_score > best_score:
                best_score = final_score
                best_lines = final_lines
    
    if best_points is None:
        points, lines, score = _try_grid_configuration(6, 9, 59)
        best_points = points
        best_lines = lines
        best_score = score
    
    assert best_points.shape == (54, 2)
    assert best_lines.shape == (27, 3)
    
    pts_set = set()
    for i in range(54):
        pt = (int(best_points[i][0]), int(best_points[i][1]))
        assert pt not in pts_set
        pts_set.add(pt)
    
    lines_set = set()
    for i in range(27):
        norm = _normalize_line(int(best_lines[i][0]), int(best_lines[i][1]), int(best_lines[i][2]))
        assert norm not in lines_set
        lines_set.add(norm)
        assert norm != (0, 0, 0)
    
    return best_points, best_lines


# EVOLVE-BLOCK-END

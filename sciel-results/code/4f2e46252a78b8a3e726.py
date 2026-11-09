# EVOLVE-BLOCK-START
import numpy as np
import time
from math import gcd
import random
import sys
from itertools import combinations

def _normalize_line(A, B, C):
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        return (0, 0, 0)
    A, B, C = A // g, B // g, C // g
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def _count_incidents(points, lines):
    """Fast count using numpy vectorization."""
    pts = points.astype(np.int64)
    lns = lines.astype(np.int64)
    # A*x + B*y + C
    # pts shape (N, 2), lns shape (M, 3)
    x = pts[:, 0:1]  # (N, 1)
    y = pts[:, 1:2]  # (N, 1)
    A = lns[:, 0:1].T  # (1, M)
    B = lns[:, 1:2].T  # (1, M)
    C = lns[:, 2:3].T  # (1, M)
    val = A * x + B * y + C
    return int(np.sum(val == 0))

def _is_valid(points, lines):
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return False
    pt_set = set()
    for i in range(54):
        x, y = points[i]
        if not np.isfinite(x) or not np.isfinite(y):
            return False
        if abs(x - round(x)) > 1e-6 or abs(y - round(y)) > 1e-6:
            return False
        xi, yi = int(round(x)), int(round(y))
        if (xi, yi) in pt_set:
            return False
        pt_set.add((xi, yi))
    line_set = set()
    for i in range(27):
        A, B, C = lines[i]
        if not np.isfinite(A) or not np.isfinite(B) or not np.isfinite(C):
            return False
        if abs(A - round(A)) > 1e-6 or abs(B - round(B)) > 1e-6 or abs(C - round(C)) > 1e-6:
            return False
        Ai, Bi, Ci = int(round(A)), int(round(B)), int(round(C))
        if Ai == 0 and Bi == 0:
            return False
        key = _normalize_line(Ai, Bi, Ci)
        if key in line_set:
            return False
        line_set.add(key)
    return True

def _generate_grid_points(cols, rows, offset_x=0, offset_y=0):
    pts = []
    for i in range(cols):
        for j in range(rows):
            pts.append((i + offset_x, j + offset_y))
    return np.array(pts[:54], dtype=np.int64)

def _candidate_lines_for_points(points):
    """Generate unique lines through all pairs of points, plus axis-aligned lines."""
    n = len(points)
    lines_dict = {}
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            if A == 0 and B == 0:
                continue
            key = _normalize_line(A, B, C)
            if key[0] == 0 and key[1] == 0:
                continue
            lines_dict[key] = 0

    # Add vertical and horizontal lines passing through at least one point
    xs = points[:, 0]
    ys = points[:, 1]
    for x in np.unique(xs):
        key = _normalize_line(1, 0, -x)
        if key[0] != 0 or key[1] != 0:
            lines_dict[key] = 0
    for y in np.unique(ys):
        key = _normalize_line(0, 1, -y)
        if key[0] != 0 or key[1] != 0:
            lines_dict[key] = 0

    # Add diagonal lines (slope 1 and -1) passing through points
    for d in range(-20, 21):
        key = _normalize_line(1, -1, d)
        if key[0] != 0 or key[1] != 0:
            lines_dict[key] = 0
        key = _normalize_line(1, 1, d)
        if key[0] != 0 or key[1] != 0:
            lines_dict[key] = 0

    # Add some "near" lines for robustness
    for x in np.unique(xs):
        for dx in [-1, 1]:
            key = _normalize_line(1, 0, -(x + dx))
            if key[0] != 0 or key[1] != 0:
                lines_dict[key] = 0
    for y in np.unique(ys):
        for dy in [-1, 1]:
            key = _normalize_line(0, 1, -(y + dy))
            if key[0] != 0 or key[1] != 0:
                lines_dict[key] = 0

    return list(lines_dict.keys())

def _score_lines(points, lines):
    """Return count of incidences."""
    return _count_incidents(points, lines)

def _select_top_k_lines(points, candidates, k=27):
    """Select top k lines by number of incidents."""
    if len(candidates) <= k:
        return np.array(candidates, dtype=np.int64)

    pts = points.astype(np.int64)
    x = pts[:, 0:1]
    y = pts[:, 1:2]

    # Score each candidate
    scored = []
    for line in candidates:
        A, B, C = line
        val = A * x + B * y + C
        cnt = int(np.sum(val == 0))
        scored.append((cnt, line))

    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [line for cnt, line in scored[:k]]
    return np.array(selected, dtype=np.int64)

def _local_search_lines(points, lines, deadline, safety_margin=10):
    """Improve line selection via local search."""
    current_lines = lines.copy()
    current_score = _score_lines(points, current_lines)
    best_lines = current_lines.copy()
    best_score = current_score

    candidates = _candidate_lines_for_points(points)
    # Pre-compute scores for all candidates
    cand_scores = {}
    pts = points.astype(np.int64)
    x = pts[:, 0:1]
    y = pts[:, 1:2]
    for line in candidates:
        A, B, C = line
        val = A * x + B * y + val
        cnt = int(np.sum(val == 0))
        cand_scores[line] = cnt

    current_set = set(tuple(l) for l in current_lines)

    improved = True
    while improved and time.time() < deadline - safety_margin:
        improved = False
        # Try to swap each line with a better candidate
        for j in range(len(current_lines)):
            current_line = tuple(current_lines[j])
            # Find best replacement
            best_swap_gain = 0
            best_swap_line = None
            for line in candidates:
                if line in current_set:
                    continue
                # Estimate gain: score(line) - score(current_line)
                # But we need to be careful: removing current_line might affect other lines? No, incidences are independent per line.
                # So gain = cand_scores[line] - cand_scores[current_line]
                if cand_scores[line] > cand_scores[current_line]:
                    gain = cand_scores[line] - cand_scores[current_line]
                    if gain > best_swap_gain:
                        best_swap_gain = gain
                        best_swap_line = line

            if best_swap_line is not None:
                current_lines[j] = np.array(best_swap_line, dtype=np.int64)
                current_set.remove(current_line)
                current_set.add(best_swap_line)
                current_score += best_swap_gain
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_lines = current_lines.copy()
                break  # Restart search

    return best_lines, best_score

def _perturb_points(points, num_perturb=2, max_delta=2):
    """Randomly perturb some points."""
    new_pts = points.copy()
    n = len(points)
    indices = random.sample(range(n), num_perturb)
    for idx in indices:
        dx = random.randint(-max_delta, max_delta)
        dy = random.randint(-max_delta, max_delta)
        new_x = new_pts[idx, 0] + dx
        new_y = new_pts[idx, 1] + dy
        # Check for collision
        if any(new_pts[i, 0] == new_x and new_pts[i, 1] == new_y for i in range(n) if i != idx):
            continue
        new_pts[idx, 0] = new_x
        new_pts[idx, 1] = new_y
    return new_pts

def _is_valid_points(points):
    pt_set = set()
    for i in range(54):
        x, y = points[i]
        if not np.isfinite(x) or not np.isfinite(y):
            return False
        if abs(x - round(x)) > 1e-6 or abs(y - round(y)) > 1e-6:
            return False
        xi, yi = int(round(x)), int(round(y))
        if (xi, yi) in pt_set:
            return False
        pt_set.add((xi, yi))
    return True

def szemeredi_trotter_incidences():
    deadline = time.time() + 630
    safety_margin = 30
    best_points = None
    best_lines = None
    best_score = -1

    # Initial configurations
    configs = []
    configs.append(_generate_grid_points(9, 6))
    configs.append(_generate_grid_points(6, 9))
    configs.append(_generate_grid_points(3, 18))
    configs.append(_generate_grid_points(18, 3))
    configs.append(_generate_grid_points(2, 27))
    configs.append(_generate_grid_points(27, 2))

    # Add some random clusters
    for seed in [42, 123, 999]:
        random.seed(seed)
        pts = []
        while len(pts) < 54:
            x = random.randint(1, 15)
            y = random.randint(1, 15)
            if (x, y) not in pts:
                pts.append((x, y))
        configs.append(np.array(pts, dtype=np.int64))

    for pts in configs:
        if time.time() > deadline - safety_margin:
            break

        if not _is_valid_points(pts):
            continue

        candidates = _candidate_lines_for_points(pts)
        lines = _select_top_k_lines(pts, candidates, 27)

        if not _is_valid(pts, lines):
            continue

        score = _score_lines(pts, lines)

        # Local search on lines
        lines, score = _local_search_lines(pts, lines, deadline, safety_margin)

        if score > best_score:
            best_score = score
            best_points = pts.copy()
            best_lines = lines.copy()

        # Try to improve points via perturbation
        # Only do this if we have time
        if time.time() < deadline - 60:
            for _ in range(5):
                if time.time() > deadline - safety_margin:
                    break
                new_pts = _perturb_points(pts, 2, 2)
                if not _is_valid_points(new_pts):
                    continue
                new_candidates = _candidate_lines_for_points(new_pts)
                new_lines = _select_top_k_lines(new_pts, new_candidates, 27)
                if not _is_valid(new_pts, new_lines):
                    continue
                new_score = _score_lines(new_pts, new_lines)
                new_lines, new_score = _local_search_lines(new_pts, new_lines, deadline, safety_margin)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_pts.copy()
                    best_lines = new_lines.copy()

    if best_points is None:
        best_points = _generate_grid_points(9, 6)
        candidates = _candidate_lines_for_points(best_points)
        best_lines = _select_top_k_lines(best_points, candidates, 27)

    return best_points, best_lines

# EVOLVE-BLOCK-END

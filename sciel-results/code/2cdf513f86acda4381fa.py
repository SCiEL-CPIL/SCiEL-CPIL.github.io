# EVOLVE-BUDGET-START
import numpy as np
import time
import math
from itertools import combinations
from collections import Counter

def _canonical_line(a, b, c):
    """Return canonical form of line (a,b,c): divide by gcd, make first nonzero positive."""
    if a == 0 and b == 0 and c == 0:
        return (0, 0, 0)
    g = math.gcd(math.gcd(abs(a), abs(b)), abs(c))
    a //= g
    b //= g
    c //= g
    if a < 0:
        a, b, c = -a, -b, -c
    elif a == 0 and b < 0:
        a, b, c = -a, -b, -c
    return (a, b, c)

def _count_incidences(points, lines):
    """Count point-line incidences exactly."""
    count = 0
    for i in range(len(points)):
        x, y = points[i]
        for j in range(len(lines)):
            A, B, C = lines[j]
            if A * x + B * y + C == 0:
                count += 1
    return count

def _all_lines_through_pairs(points):
    """Find all distinct lines passing through at least 2 points."""
    n = len(points)
    line_counts = Counter()
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            # Line: (y1-y2)*x + (x2-x1)*y + (y2*x1 - y1*x2) = 0
            a = y1 - y2
            b = x2 - x1
            c = y2 * x1 - y1 * x2
            canon = _canonical_line(a, b, c)
            if canon != (0, 0, 0):
                line_counts[canon] += 1
    return line_counts

def _top_lines_for_points(points, max_lines=27):
    """Greedy: pick lines that maximize total incidences."""
    n = len(points)
    line_counts = _all_lines_through_pairs(points)
    
    # Sort lines by number of points they pass through (descending)
    sorted_lines = sorted(line_counts.items(), key=lambda x: -x[1])
    
    # Greedy selection: pick lines that add the most new incidences
    selected = []
    covered = [0] * n  # count of lines already covering each point
    
    # For each point, track which lines cover it
    point_lines = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            a = y1 - y2
            b = x2 - x1
            c = y2 * x1 - y1 * x2
            canon = _canonical_line(a, b, c)
            if canon != (0, 0, 0):
                point_lines[i].append(canon)
    
    # Remove duplicates in point_lines
    for i in range(n):
        point_lines[i] = list(set(point_lines[i]))
    
    # Greedy: repeatedly pick the line that covers the most uncovered points
    selected_lines = []
    covered_points = [False] * n
    
    for _ in range(min(max_lines, len(sorted_lines))):
        best_line = None
        best_count = 0
        for line, cnt in sorted_lines:
            if line in selected_lines:
                continue
            # Count how many points on this line are not yet covered
            count = 0
            for i in range(n):
                if not covered_points[i] and line in point_lines[i]:
                    count += 1
            if count > best_count:
                best_count = count
                best_line = line
        if best_line is not None and best_count > 0:
            selected_lines.append(best_line)
            # Mark points on this line as covered
            for i in range(n):
                if best_line in point_lines[i]:
                    covered_points[i] = True
    
    # Convert to array
    lines_arr = np.array(selected_lines, dtype=np.int64)
    return lines_arr

def _evaluate(points, lines):
    """Evaluate the configuration."""
    return _count_incidences(points, lines)

def _search_random_configs(n_points=54, n_lines=27, budget_seconds=550, seed=42):
    """Search for good configurations using random point sets and greedy line selection."""
    rng = np.random.default_rng(seed)
    start_time = time.time()
    deadline = start_time + budget_seconds
    
    best_score = 0
    best_points = None
    best_lines = None
    
    # Try various grid-like and random configurations
    configs_tried = 0
    
    while time.time() < deadline:
        configs_tried += 1
        
        # Strategy 1: Random integer points in a bounded box
        # Strategy 2: Grid-like structures
        # Strategy 3: Points on a few lines (concentrated)
        
        strategy = rng.integers(0, 3)
        
        if strategy == 0:
            # Random points in a box
            size = rng.integers(5, 20)
            points = rng.integers(-size, size + 1, size=(n_points, 2))
            # Remove duplicates
            unique_points = np.unique(points, axis=0)
            if len(unique_points) < n_points:
                continue
            points = unique_points[:n_points]
            
        elif strategy == 1:
            # Grid-like: k x m grid
            k = rng.integers(2, 8)
            m = rng.integers(2, 20)
            if k * m < n_points:
                continue
            xs = np.arange(1, k + 1)
            ys = np.arange(1, m + 1)
            px, py = np.meshgrid(xs, ys, indexing="ij")
            all_points = np.stack([px.ravel(), py.ravel()], axis=1)
            # Randomly select n_points
            idx = rng.choice(len(all_points), size=n_points, replace=False)
            points = all_points[idx]
            
        else:
            # Points concentrated on a few lines
            n_lines_for_points = rng.integers(2, 5)
            points_list = []
            for _ in range(n_lines_for_points):
                # Random line
                a = rng.integers(-3, 4)
                b = rng.integers(-3, 4)
                c = rng.integers(-10, 11)
                if a == 0 and b == 0:
                    continue
                # Find integer points on this line
                line_points = []
                for x in range(-15, 16):
                    if b != 0:
                        val = -(a * x + c)
                        if val % b == 0:
                            y = val // b
                            line_points.append((x, y))
                    elif a != 0:
                        val = -c
                        if val % a == 0:
                            x_val = val // a
                            if -15 <= x_val <= 15:
                                line_points.append((x_val, rng.integers(-15, 16)))
                if len(line_points) > 0:
                    points_list.extend(line_points[:n_points // n_lines_for_points + 1])
            
            if len(points_list) < n_points:
                continue
            points = np.array(points_list[:n_points], dtype=np.int64)
            # Remove duplicates
            unique_points = np.unique(points, axis=0)
            if len(unique_points) < n_points:
                continue
            points = unique_points[:n_points]
        
        # Select best lines
        lines = _top_lines_for_points(points, max_lines=n_lines)
        if len(lines) < n_lines:
            continue
        
        score = _evaluate(points, lines)
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
            best_lines = lines.copy()
    
    return best_points, best_lines, best_score

def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization.
    Returns (points, lines) where points is (54,2) and lines is (27,3).
    """
    # Start with the known good seed construction
    k = 3
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    seed_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    rows = []
    for a in range(1, k + 1):
        for b in range(1, k * k + 1):
            rows.append((a, -1, b))
    seed_lines = np.array(rows, dtype=np.int64)
    
    seed_score = _count_incidences(seed_points, seed_lines)
    
    # Try to improve via search
    best_points, best_lines, best_score = _search_random_configs(
        n_points=54, n_lines=27, budget_seconds=550, seed=42
    )
    
    if best_points is not None and best_score > seed_score:
        return best_points, best_lines
    else:
        return seed_points, seed_lines

# EVOLVE-BUDGET-END

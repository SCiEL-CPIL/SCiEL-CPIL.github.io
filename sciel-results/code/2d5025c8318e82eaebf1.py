# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from itertools import combinations
from collections import Counter

DEADLINE = 630.0
SAFETY_MARGIN = 10.0


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
    """Count point-line incidences exactly using numpy."""
    # points: (n, 2), lines: (m, 3)
    # incidence if A*x + B*y + C == 0
    n = points.shape[0]
    m = lines.shape[0]
    # Use broadcasting: (n,1,3) * (1,m,3) won't work directly
    # Instead: for each line, compute A*x + B*y + C for all points
    # points @ [A, B] + C
    # lines[:, 0:1, 0] is (m,1,1), points[:, 0:1] is (n,1)
    # Let's do: (points @ lines[:, :2].T) + lines[:, 2:3].T
    # points: (n,2), lines[:, :2].T: (2, m) -> result (n, m)
    # lines[:, 2:3].T: (1, m) -> broadcast to (n, m)
    vals = points @ lines[:, :2].T + lines[:, 2:3].T
    count = np.sum(vals == 0)
    return int(count)


def _all_lines_through_pairs(points):
    """
    Generate all distinct lines through pairs of points.
    Returns a list of canonical (A, B, C) tuples.
    """
    n = len(points)
    line_set = set()
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            # Line through (x1,y1) and (x2,y2): (y1-y2)x + (x2-x1)y + (x1*y2 - x2*y1) = 0
            a = y1 - y2
            b = x2 - x1
            c = x1 * y2 - x2 * y1
            line_set.add(_canonical_line(a, b, c))
    return list(line_set)


def _top_lines(points, num_lines):
    """
    Find the top num_lines lines (by number of point incidences) from all pair-defined lines.
    Returns a list of (A, B, C) tuples.
    """
    lines = _all_lines_through_pairs(points)
    if not lines:
        return []
    # Count incidences for each line
    line_counts = []
    for line in lines:
        a, b, c = line
        # Count points on this line
        # Use numpy for speed
        vals = points[:, 0] * a + points[:, 1] * b + c
        cnt = int(np.sum(vals == 0))
        line_counts.append((cnt, line))
    
    # Sort by count descending
    line_counts.sort(key=lambda x: -x[0])
    top = [lc[1] for lc in line_counts[:num_lines]]
    return top


def _evaluate(points, lines):
    """
    Evaluate the configuration. Returns the number of incidences.
    Also performs validity checks.
    """
    # Check shapes
    if points.shape != (54, 2) or lines.shape != (27, 3):
        return -1
    
    # Check for NaN/inf
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(lines)):
        return -1
    
    # Check integer-ness
    if not np.allclose(points, np.round(points), atol=1e-6):
        return -1
    if not np.allclose(lines, np.round(lines), atol=1e-6):
        return -1
    
    # Convert to int64
    points_int = np.round(points).astype(np.int64)
    lines_int = np.round(lines).astype(np.int64)
    
    # Check repeated points
    if len(set(map(tuple, points_int.tolist()))) != 54:
        return -1
    
    # Check line validity
    seen_lines = set()
    for i in range(27):
        a, b, c = lines_int[i]
        if a == 0 and b == 0:
            return -1
        canon = _canonical_line(int(a), int(b), int(c))
        if canon in seen_lines:
            return -1
        seen_lines.add(canon)
    
    # Count incidences
    count = _count_incidences(points_int, lines_int)
    return count


def _seed_program():
    """The original Szemeredi-Trotter construction."""
    k = 3
    xs = np.arange(1, k + 1)
    ys = np.arange(1, 2 * k * k + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    rows = []
    for a in range(1, k + 1):
        for b in range(1, k * k + 1):
            rows.append((a, -1, b))
    lines = np.array(rows, dtype=np.int64)
    
    return points, lines


def _search_grid_points(w, h, num_points=54, num_lines=27, time_limit=300.0):
    """
    Search for a subset of num_points from a w x h grid that maximizes the sum of the
    top num_lines line incidences.
    Uses a greedy/local search approach.
    """
    start_time = time.time()
    
    # Create grid
    grid_points = []
    for x in range(w):
        for y in range(h):
            grid_points.append((x, y))
    total_grid = len(grid_points)
    
    if total_grid < num_points:
        return None, 0
    
    # Start with a random subset
    np.random.seed(42)
    selected_indices = set(np.random.choice(total_grid, size=num_points, replace=False).tolist())
    current_points = [grid_points[i] for i in sorted(selected_indices)]
    current_points = np.array(current_points, dtype=np.int64)
    
    # Get top lines
    top = _top_lines(current_points, num_lines)
    current_score = _count_incidences(current_points, np.array(top, dtype=np.int64))
    
    best_score = current_score
    best_points = current_points.copy()
    best_lines = np.array(top, dtype=np.int64)
    
    # Local search: try swapping points
    iteration = 0
    while time.time() - start_time < time_limit:
        iteration += 1
        improved = False
        
        # Try adding a random unselected point and removing a random selected point
        unselected = [i for i in range(total_grid) if i not in selected_indices]
        selected = list(selected_indices)
        
        if not unselected or not selected:
            break
        
        # Try multiple swaps
        for _ in range(200):
            if time.time() - start_time > time_limit:
                break
            
            add_idx = np.random.choice(unselected)
            rem_idx = np.random.choice(selected)
            
            new_selected = (selected_indices - {rem_idx}) | {add_idx}
            new_points = np.array([grid_points[i] for i in sorted(new_selected)], dtype=np.int64)
            
            top_new = _top_lines(new_points, num_lines)
            new_score = _count_incidences(new_points, np.array(top_new, dtype=np.int64))
            
            if new_score > current_score:
                current_score = new_score
                current_points = new_points
                selected_indices = new_selected
                improved = True
                
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
                    best_lines = np.array(top_new, dtype=np.int64)
            
            # Reset unselected and selected
            unselected = [i for i in range(total_grid) if i not in selected_indices]
            selected = list(selected_indices)
        
        if not improved and iteration > 50:
            # Try a different random restart
            if iteration % 100 == 0:
                selected_indices = set(np.random.choice(total_grid, size=num_points, replace=False).tolist())
                current_points = [grid_points[i] for i in sorted(selected_indices)]
                current_points = np.array(current_points, dtype=np.int64)
                top = _top_lines(current_points, num_lines)
                current_score = _count_incidences(current_points, np.array(top, dtype=np.int64))
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
                    best_lines = np.array(top, dtype=np.int64)
    
    return best_points, best_lines, best_score


def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization. Construct a set of n points and a set of
    m lines in the plane (on the integer lattice) so as to maximize the number of
    point-line incidences (pairs (p, l) with p lying on l). Here the instance is fixed
    at n = 54 points and m = 27 lines.
    """
    start_time = time.time()
    
    # Start with the seed program's solution
    seed_points, seed_lines = _seed_program()
    seed_score = _count_incidences(seed_points, seed_lines)
    
    best_points = seed_points.copy()
    best_lines = seed_lines.copy()
    best_score = seed_score
    
    # Try different grid sizes
    # The knowledge suggests a 7x8 grid works well
    grid_configs = [
        (7, 8),   # 56 cells, choose 54
        (8, 7),   # 56 cells
        (6, 9),   # 54 cells, choose 54 (all)
        (9, 6),   # 54 cells
        (7, 9),   # 63 cells
        (9, 7),   # 63 cells
        (8, 8),   # 64 cells
    ]
    
    for w, h in grid_configs:
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        # Allocate time based on remaining budget
        remaining = DEADLINE - SAFETY_MARGIN - (time.time() - start_time)
        time_per_config = remaining / len(grid_configs)
        # Cap time per config
        time_per_config = min(time_per_config, 120.0)
        
        result = _search_grid_points(w, h, 54, 27, time_limit=time_per_config)
        if result is not None:
            pts, lns, score = result
            if score > best_score:
                best_score = score
                best_points = pts.copy()
                best_lines = lns.copy()
    
    # Final validation
    final_score = _evaluate(best_points, best_lines)
    if final_score < 0:
        # Fallback to seed
        return seed_points, seed_lines
    
    return best_points, best_lines


# EVOLVE-BLOCK-END

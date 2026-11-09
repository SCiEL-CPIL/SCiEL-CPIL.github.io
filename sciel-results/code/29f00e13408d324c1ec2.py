import time
import math
import random
import itertools
from math import gcd as _gcd

_START = time.time()
_DEADLINE = _START + 115.0
_BOUND = 1000


def _line_key(p, q):
    x1, y1 = p
    x2, y2 = q
    dx = x2 - x1
    dy = y2 - y1
    g = _gcd(abs(dx), abs(dy))
    if g:
        dx //= g
        dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx = -dx
        dy = -dy
    return (dx, dy, x1 * dy - y1 * dx)


def _score_points(points):
    """Return (num_exact4, num_5plus)."""
    counts = {}
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(points[i], points[j])
            counts[k] = counts.get(k, 0) + 1
    n4 = n5 = 0
    for c in counts.values():
        if c == 6:
            n4 += 1
        elif c >= 10:
            n5 += 1
    return n4, n5


def _valid(points):
    n4, n5 = _score_points(points)
    return n5 == 0


def _generate_candidate_lines():
    """Generate a pool of candidate lines that could contain 4+ integer points
    within bounds. Each line is represented by (A, B, C) with Ax+By=C, normalized.
    We include small-slope rational lines and a few random ones."""
    lines = set()
    # Horizontal and vertical lines
    for c in range(-_BOUND, _BOUND + 1, 1):
        if c % 3 == 0 or abs(c) <= 50:
            lines.add((0, 1, c))  # y = c
            lines.add((1, 0, c))  # x = c
    # Small-slope rational lines: y = (p/q)x + c, with |p|,|q| small
    for p in range(-6, 7):
        for q in range(1, 7):
            if p == 0:
                continue
            g = _gcd(abs(p), q)
            p //= g
            q //= g
            # Line: p*x - q*y = c  =>  A=p, B=-q, C=c
            # We want integer points within bounds. c range depends on p,q.
            # Just add a few c values.
            for c in range(-20, 21):
                lines.add((p, -q, c))
            # Also some larger offsets
            for c in range(-100, 101, 10):
                lines.add((p, -q, c))
    # Add a few random lines
    for _ in range(200):
        x1 = random.randint(-_BOUND, _BOUND)
        y1 = random.randint(-_BOUND, _BOUND)
        x2 = random.randint(-_BOUND, _BOUND)
        y2 = random.randint(-_BOUND, _BOUND)
        if x1 == x2 and y1 == y2:
            continue
        dx = x2 - x1
        dy = y2 - y1
        g = _gcd(abs(dx), abs(dy))
        if g:
            dx //= g
            dy //= g
        if dx < 0 or (dx == 0 and dy < 0):
            dx = -dx
            dy = -dy
        # A = dy, B = -dx, C = A*x1 + B*y1
        A = dy
        B = -dx
        C = A * x1 + B * y1
        lines.add((A, B, C))
    return list(lines)


def _points_on_line(line, points):
    """Return list of indices of points that lie on the given line (A,B,C)."""
    A, B, C = line
    result = []
    for i, (x, y) in enumerate(points):
        if A * x + B * y == C:
            result.append(i)
    return result


def _max_independent_set(covered_indices, points):
    """Given a set of indices that are 'covered' by at least one selected line,
    find the maximum subset of these indices such that no 5 are collinear.
    Use a greedy approach: sort by degree (number of lines they're on) and
    greedily add if it doesn't create a 5-collinear violation."""
    # Build a conflict graph: two indices conflict if adding both would create
    # a line with >=5 points. Actually, we need to check all lines.
    # Simpler: use a heuristic. Start with all covered, then remove points
    # that are part of 5-collinear violations.
    selected = set(covered_indices)
    changed = True
    while changed:
        changed = False
        # Check all lines
        line_counts = {}
        for i in selected:
            for j in selected:
                if i >= j:
                    continue
                k = _line_key(points[i], points[j])
                line_counts[k] = line_counts.get(k, 0) + 1
        # Find lines with >=5 points
        for k, c in line_counts.items():
            if c >= 10:  # 5 points give 10 pairs
                # Find which points are on this line
                line_pts = [i for i in selected if _line_key(points[i], points[0]) == k]
                # This is wrong; let me redo
                break
        # Better: for each line with >=5 points, remove one point
        for k, c in list(line_counts.items()):
            if c >= 10:
                # Find points on this line
                line_pts = []
                for i in selected:
                    # Check if point i is on line k
                    # k is (dx, dy, offset) where offset = x1*dy - y1*dx
                    # Line: dy*(x - x1) - dx*(y - y1) = 0 => dy*x - dx*y = offset
                    dx, dy, offset = k
                    x, y = points[i]
                    if dy * x - dx * y == offset:
                        line_pts.append(i)
                if len(line_pts) >= 5:
                    # Remove one point (the one with fewest other lines)
                    # Just remove the first one for simplicity
                    remove_idx = line_pts[0]
                    selected.discard(remove_idx)
                    changed = True
                    break
    return selected


def _evaluate_line_set(line_set, points):
    """Given a set of candidate lines and a set of points, determine which
    lines are 'satisfied' (contain exactly 4 of the points) and check validity.
    Returns (num_exact4, num_5plus, selected_point_indices)."""
    # Find all points that lie on at least one line in line_set
    covered = set()
    for line in line_set:
        for i in _points_on_line(line, points):
            covered.add(i)
    # Find maximum independent set (no 5 collinear)
    selected = _max_independent_set(covered, points)
    if len(selected) < 16:
        # Need exactly 16 points; if we can't get 16, this line-set is infeasible
        # with these points. Return a bad score.
        return -1000000, 1, set()
    # Count exact-4 lines among selected points
    n4, n5 = _score_points([points[i] for i in selected])
    return n4, n5, selected


def _geometric_local_search(points, max_time):
    """Fast geometric local search from the current program's approach."""
    random.seed(42)
    best_score, _ = _score_points(points)
    best_points = [p[:] for p in points]
    step_sizes = [1, 2, 3, 5, 8, 13, 21]
    deadline = time.time() + max_time
    while time.time() < deadline:
        improved = False
        for i in range(16):
            if time.time() > deadline:
                break
            orig = points[i][:]
            for step in step_sizes:
                for dx in range(-step, step + 1):
                    for dy in range(-step, step + 1):
                        if dx == 0 and dy == 0:
                            continue
                        new_x = orig[0] + dx
                        new_y = orig[1] + dy
                        if abs(new_x) > _BOUND or abs(new_y) > _BOUND:
                            continue
                        points[i] = [new_x, new_y]
                        # Check distinctness
                        valid = True
                        for j in range(16):
                            if j != i and points[j] == points[i]:
                                valid = False
                                break
                        if not valid:
                            points[i] = orig
                            continue
                        s, n5 = _score_points(points)
                        if n5 == 0 and s > best_score:
                            best_score = s
                            best_points = [p[:] for p in points]
                            improved = True
                        else:
                            points[i] = orig
                if time.time() > deadline:
                    break
            points[i] = orig
        if not improved:
            # Random perturbation
            if random.random() < 0.5:
                i = random.randint(0, 15)
                orig = points[i][:]
                for _ in range(30):
                    if time.time() > deadline:
                        break
                    new_x = random.randint(-_BOUND, _BOUND)
                    new_y = random.randint(-_BOUND, _BOUND)
                    points[i] = [new_x, new_y]
                    valid = True
                    for j in range(16):
                        if j != i and points[j] == points[i]:
                            valid = False
                            break
                    if not valid:
                        continue
                    s, n5 = _score_points(points)
                    if n5 == 0 and s > best_score:
                        best_score = s
                        best_points = [p[:] for p in points]
                        improved = True
                    else:
                        points[i] = orig
                points[i] = orig
            if not improved and best_score >= 10:
                break
    return best_points


def construct():
    start_time = time.time()
    deadline = _DEADLINE
    
    # Start with 4x4 grid as a strong baseline
    points = [[x, y] for x in range(4) for y in range(4)]
    best_score, _ = _score_points(points)
    best_points = [p[:] for p in points]
    
    if best_score < 0:
        return best_points
    
    # Generate candidate lines
    candidate_lines = _generate_candidate_lines()
    
    # Phase 1: Hypergraph search over line-sets
    # Start with an empty line-set and greedily add lines
    line_set = set()
    for line in candidate_lines:
        if time.time() > deadline - 30:
            break
        # Try adding this line
        new_line_set = line_set | {line}
        n4, n5, selected = _evaluate_line_set(new_line_set, points)
        if n5 == 0 and n4 > best_score:
            best_score = n4
            # Update points to the selected ones
            if len(selected) >= 16:
                # Take first 16 selected points
                selected_list = sorted(selected)[:16]
                points = [points[i][:] for i in selected_list]
                best_points = [p[:] for p in points]
                line_set = new_line_set
    
    # Phase 2: Geometric local search refinement
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        refined = _geometric_local_search(points, remaining_time - 5)
        s, n5 = _score_points(refined)
        if n5 == 0 and s > best_score:
            best_score = s
            best_points = refined
    
    # Ensure we return a valid configuration
    final_score, final_n5 = _score_points(best_points)
    if final_n5 > 0:
        # Fallback to 4x4 grid
        best_points = [[x, y] for x in range(4) for y in range(4)]
    
    return best_points

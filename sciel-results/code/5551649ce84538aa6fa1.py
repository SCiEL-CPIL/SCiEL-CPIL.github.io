import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 105.0  # safety margin before 120s

    # Base configuration: 6 horizontal lines of 3 points + 2 isolated points.
    # Known score = 14 (each line contributes at most 2, plus 2 isolated).
    base_pts = [
        [10, 0], [11, 0], [12, 0],
        [-12, 2], [-11, 2], [-10, 2],
        [10, 4], [11, 4], [12, 4],
        [-12, 6], [-11, 6], [-10, 6],
        [10, 8], [11, 8], [12, 8],
        [-12, 10], [-11, 10], [-10, 10],
        [50, 50], [-50, -50]
    ]

    def is_valid(points):
        """Check: 20 distinct integer points, |coord|<=100, no 4 collinear."""
        if len(points) != 20:
            return False
        if len(set(map(tuple, points))) != 20:
            return False
        for p in points:
            if abs(p[0]) > 100 or abs(p[1]) > 100:
                return False
        # No four collinear
        for comb in itertools.combinations(points, 4):
            (x1, y1), (x2, y2), (x3, y3), (x4, y4) = comb
            if (x2 - x1) * (y3 - y1) != (x3 - x1) * (y2 - y1):
                continue
            if (x2 - x1) * (y4 - y1) != (x4 - x1) * (y2 - y1):
                continue
            return False
        return True

    def precompute_collinear(points):
        """For each pair (i,j), return the set of indices k collinear with i,j."""
        n = len(points)
        collinear = [[set() for _ in range(n)] for _ in range(n)]
        for i in range(n):
            x1, y1 = points[i]
            for j in range(i + 1, n):
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                for k in range(n):
                    if k == i or k == j:
                        continue
                    x3, y3 = points[k]
                    if dx * (y3 - y1) == dy * (x3 - x1):
                        collinear[i][j].add(k)
                        collinear[j][i].add(k)
        return collinear

    def largest_no3_collinear(points):
        """Exact branch and bound: largest subset with no 3 collinear."""
        n = len(points)
        collinear = precompute_collinear(points)

        best = [0]
        used = [False] * n

        def backtrack(start, selected, size):
            if size > best[0]:
                best[0] = size
            # Upper bound: current size + remaining candidates
            if size + (n - start) <= best[0]:
                return
            for i in range(start, n):
                if not used[i]:
                    # Check if adding i creates a collinear triple
                    valid = True
                    for j in selected:
                        if i in collinear[i][j]:
                            valid = False
                            break
                    if valid:
                        selected.append(i)
                        backtrack(i + 1, selected, size + 1)
                        selected.pop()

        backtrack(0, [], 0)
        return best[0]

    # Verify base configuration
    if not is_valid(base_pts):
        # Fallback: parabola (score 20) - should never happen
        return [[x, x * x] for x in range(-10, 10)]

    best_pts = base_pts[:]
    best_score = largest_no3_collinear(base_pts)

    if time.time() > deadline:
        return best_pts

    # Local search: try single-point swaps to break lines of 3 and create
    # cross-line collinear triples that reduce the max independent set.
    random.seed(42)

    # Precompute for faster validity: track lines of 3 in the current config
    def get_lines_of_3(points):
        """Return list of sets of indices that form lines of exactly 3."""
        n = len(points)
        # Map from line (canonical form) to set of points on it
        line_map = {}
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                # Canonical line: (dy, dx) normalized by gcd, with offset
                g = 1
                a, b = abs(dy), abs(dx)
                while b:
                    a, b = b, a % b
                g = a
                ndx = dx // g
                ndy = dy // g
                # Canonical direction
                if ndx < 0 or (ndx == 0 and ndy < 0):
                    ndx, ndy = -ndx, -ndy
                # Offset: cross product of direction with point
                offset = ndy * x1 - ndx * y1
                key = (ndx, ndy, offset)
                if key not in line_map:
                    line_map[key] = set()
                line_map[key].add(i)
                line_map[key].add(j)
        # Filter to lines of exactly 3
        lines3 = [pts for pts in line_map.values() if len(pts) == 3]
        return lines3

    # Local search loop
    max_iter = 5000
    for it in range(max_iter):
        if time.time() > deadline:
            break
        if best_score <= 13:
            # Can't do better than 13 with this structure easily; try a bit more
            if best_score <= 12:
                break

        # Pick a random point to move (prefer points on lines of 3)
        lines3 = get_lines_of_3(best_pts)
        if lines3:
            # Pick a random line of 3, then a random point on it
            line = random.choice(lines3)
            idx = random.choice(list(line))
        else:
            idx = random.randint(0, 19)

        # Generate a random new position
        tries = 0
        while tries < 20:
            nx = random.randint(-100, 100)
            ny = random.randint(-100, 100)
            if not any(best_pts[i][0] == nx and best_pts[i][1] == ny for i in range(20)):
                break
            tries += 1
        else:
            continue

        cand = best_pts[:]
        cand[idx] = [nx, ny]

        if not is_valid(cand):
            continue

        s = largest_no3_collinear(cand)
        if s < best_score:
            best_pts = cand
            best_score = s

    return best_pts

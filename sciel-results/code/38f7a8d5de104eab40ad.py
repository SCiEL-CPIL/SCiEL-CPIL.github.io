"""No-isosceles set: local search with swaps and add/remove moves."""
import time
import random
import math

N = 64
DEADLINE = 290.0  # safety margin before 330s kill


def _violations_on_add(points, p):
    """Return number of isosceles triangles that would be created by adding p to points."""
    x, y = p
    count = 0
    for i in range(len(points)):
        xi, yi = points[i]
        d2_i = (x - xi) ** 2 + (y - yi) ** 2
        for j in range(i + 1, len(points)):
            xj, yj = points[j]
            d2_j = (x - xj) ** 2 + (y - yj) ** 2
            d2_ij = (xi - xj) ** 2 + (yi - yj) ** 2
            if d2_i == d2_j or d2_i == d2_ij or d2_j == d2_ij:
                count += 1
    return count


def _remove_point(points, idx):
    """Remove point at idx, return new list."""
    return points[:idx] + points[idx + 1:]


def _add_point(points, p):
    return points + [p]


def _check_no_isosceles(points):
    """Verify no three points form an isosceles triangle."""
    n = len(points)
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            for k in range(j + 1, n):
                xk, yk = points[k]
                d2_ij = (xi - xj) ** 2 + (yi - yj) ** 2
                d2_ik = (xi - xk) ** 2 + (yi - yk) ** 2
                d2_jk = (xj - xk) ** 2 + (yj - yk) ** 2
                if d2_ij == d2_ik or d2_ij == d2_jk or d2_ik == d2_jk:
                    return False
    return True


def _greedy_init():
    """Greedy construction: add points in random order if they don't create isosceles triangles."""
    random.seed(42)
    all_points = [(x, y) for x in range(N) for y in range(N)]
    random.shuffle(all_points)
    points = []
    for p in all_points:
        if _violations_on_add(points, p) == 0:
            points.append(p)
    return points


def _local_search(points, time_limit):
    """Local search with add, remove, and swap moves."""
    start = time.time()
    current = list(points)
    best = list(current)
    best_size = len(current)
    no_improve = 0
    max_no_improve = 500

    while time.time() - start < time_limit:
        moved = False
        # Try add moves
        all_points = [(x, y) for x in range(N) for y in range(N)]
        random.shuffle(all_points)
        for p in all_points:
            if p in current:
                continue
            if _violations_on_add(current, p) == 0:
                current.append(p)
                moved = True
                if len(current) > best_size:
                    best = list(current)
                    best_size = len(current)
                    no_improve = 0
                break

        if not moved:
            # Try swap moves: remove one point, add another
            if len(current) >= 2:
                idx = random.randrange(len(current))
                removed = current[idx]
                candidate = _remove_point(current, idx)
                # Try to add a random point
                for _ in range(200):
                    p = (random.randrange(N), random.randrange(N))
                    if p == removed:
                        continue
                    if p in candidate:
                        continue
                    if _violations_on_add(candidate, p) == 0:
                        candidate.append(p)
                        current = candidate
                        moved = True
                        if len(current) > best_size:
                            best = list(current)
                            best_size = len(current)
                            no_improve = 0
                        break
                if not moved:
                    current = candidate + [removed]  # restore

            # Try remove-then-add (2-opt style)
            if len(current) >= 1:
                idx = random.randrange(len(current))
                removed = current[idx]
                candidate = _remove_point(current, idx)
                for _ in range(200):
                    p = (random.randrange(N), random.randrange(N))
                    if p == removed:
                        continue
                    if p in candidate:
                        continue
                    if _violations_on_add(candidate, p) == 0:
                        candidate.append(p)
                        current = candidate
                        moved = True
                        if len(current) > best_size:
                            best = list(current)
                            best_size = len(current)
                            no_improve = 0
                        break
                if not moved:
                    current = candidate + [removed]  # restore

            if not moved:
                no_improve += 1
                if no_improve > max_no_improve:
                    # Perturbation: remove a random point and try to re-add
                    if len(current) > 1:
                        idx = random.randrange(len(current))
                        current = _remove_point(current, idx)
                        no_improve = 0

    return best


def construct():
    start = time.time()
    deadline = start + DEADLINE

    # Phase 1: Multiple greedy inits
    best = []
    best_size = 0
    random.seed(42)

    for trial in range(20):
        if time.time() > deadline - 60:
            break
        pts = _greedy_init()
        if len(pts) > best_size:
            best = pts
            best_size = len(pts)

    # Phase 2: Local search on best init
    if time.time() < deadline - 30:
        remaining = min(deadline - time.time() - 10, 200)
        improved = _local_search(best, remaining)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)

    # Phase 3: More local search restarts if time allows
    while time.time() < deadline - 20:
        remaining = min(deadline - time.time() - 10, 60)
        if remaining < 10:
            break
        # Try local search from a slightly perturbed version of best
        perturbed = list(best)
        if len(perturbed) > 2:
            idx = random.randrange(len(perturbed))
            perturbed = _remove_point(perturbed, idx)
        improved = _local_search(perturbed, remaining)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)

    return best

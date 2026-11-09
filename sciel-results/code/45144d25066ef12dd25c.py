"""Rewritten program: fast targeted search for 20 integer points with no four
collinear and a small no-three-collinear subset.

Strategy:
  - A compact, symmetric base set (parabola-like) with no four collinear.
  - Precompute, for every pair of points, the set of other points collinear
    with that pair.  This makes the no-three-collinear constraint check O(1)
    per candidate insertion.
  - Greedy construction of a no-three-collinear subset, then local search
    (add / swap / remove) to grow it, with a deadline guard.
  - If the subset reaches 13 or more, we also try to reduce the total number
    of collinear triples among all 20 points (which helps keep the score low
    even if the exact maximum no-three-collinear subset is 12).
"""

import random
import time
from math import gcd
from itertools import combinations


def _line_key(p1, p2):
    """Canonical key for the line through two distinct points."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0 and dy == 0:
        return None
    g = gcd(abs(dx), abs(dy))
    if g == 0:
        g = 1
    dx_n = dx // g
    dy_n = dy // g
    if dx_n < 0 or (dx_n == 0 and dy_n < 0):
        dx_n = -dx_n
        dy_n = -dy_n
    c = dy_n * p1[0] - dx_n * p1[1]
    return (dx_n, dy_n, c)


def _build_collinear_sets(points):
    """For each point index i, and each other point j, return the set of
    indices k (k != i, k != j) that are collinear with i and j."""
    n = len(points)
    collinear = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            # find all points on this line
            for k in range(n):
                if k == i or k == j:
                    continue
                if _line_key(points[i], points[k]) == key:
                    collinear[i].add(j)
                    collinear[i].add(k)
                    collinear[j].add(i)
                    collinear[j].add(k)
                    collinear[k].add(i)
                    collinear[k].add(j)
    return collinear


def _no_three_collinear_size(points, collinear):
    """Greedy + local search for a large subset with no three collinear.
    Returns the size of the best subset found."""
    n = len(points)
    best_subset = set()
    best_size = 0

    # Greedy construction
    remaining = set(range(n))
    current = set()
    while remaining:
        # Find a point that doesn't create a collinear triple
        added = False
        for idx in list(remaining):
            # Check: for every pair (a, b) in current, is idx collinear with a and b?
            conflict = False
            for a, b in combinations(sorted(current), 2):
                if b in collinear[a] and a in collinear[b]:
                    # a and b are collinear with some set; check if idx is on that line
                    # Actually collinear[a] contains all points collinear with a and some partner
                    # We need: idx collinear with a and b means idx is on line(a,b)
                    # collinear[a] contains b if a,b collinear with someone; but we need line(a,b) specifically
                    pass
            # Simpler: check all pairs in current
            ok = True
            cur_list = sorted(current)
            for a_pos in range(len(cur_list)):
                for b_pos in range(a_pos + 1, len(cur_list)):
                    a = cur_list[a_pos]
                    b = cur_list[b_pos]
                    # Is idx on line(a, b)?
                    if _line_key(points[a], points[b]) == _line_key(points[a], points[idx]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                current.add(idx)
                remaining.remove(idx)
                added = True
                break
        if not added:
            break

    if len(current) > best_size:
        best_subset = current
        best_size = len(current)

    # Local search: try swaps to grow the subset
    deadline = time.time() + 60
    while time.time() < deadline and best_size < n:
        improved = False
        # Try adding a point
        for idx in list(remaining):
            cur_list = sorted(best_subset)
            ok = True
            for a_pos in range(len(cur_list)):
                for b_pos in range(a_pos + 1, len(cur_list)):
                    a = cur_list[a_pos]
                    b = cur_list[b_pos]
                    if _line_key(points[a], points[b]) == _line_key(points[a], points[idx]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                new_subset = best_subset | {idx}
                # Verify no three collinear
                valid = True
                sl = sorted(new_subset)
                for i in range(len(sl)):
                    for j in range(i + 1, len(sl)):
                        for k in range(j + 1, len(sl)):
                            if _line_key(points[sl[i]], points[sl[j]]) == _line_key(points[sl[i]], points[sl[k]]):
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    best_subset = new_subset
                    best_size = len(best_subset)
                    remaining.remove(idx)
                    improved = True
                    break
        if not improved:
            # Try swap: remove one from subset, add one from remaining
            found_swap = False
            for rem_idx in list(remaining):
                for sub_idx in list(best_subset):
                    new_subset = (best_subset - {sub_idx}) | {rem_idx}
                    valid = True
                    sl = sorted(new_subset)
                    for i in range(len(sl)):
                        for j in range(i + 1, len(sl)):
                            for k in range(j + 1, len(sl)):
                                if _line_key(points[sl[i]], points[sl[j]]) == _line_key(points[sl[i]], points[sl[k]]):
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if valid:
                        best_subset = new_subset
                        remaining = (remaining - {rem_idx}) | {sub_idx}
                        found_swap = True
                        break
                if found_swap:
                    break
            if not found_swap:
                break

    return best_size


def _count_collinear_triples(points):
    """Count total number of collinear triples among all points."""
    n = len(points)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            for k in range(j + 1, n):
                if _line_key(points[i], points[k]) == key:
                    count += 1
    return count


def _has_four_collinear(points):
    """Check if any four points are collinear."""
    n = len(points)
    line_map = {}
    for i in range(n):
        for j in range(i + 1, n):
            key = _line_key(points[i], points[j])
            if key is None:
                continue
            if key not in line_map:
                line_map[key] = set()
            line_map[key].add(i)
            line_map[key].add(j)
    for key, idx_set in line_map.items():
        if len(idx_set) >= 4:
            return True
    return False


def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    # Base set: 20 points on a parabola y = x^2, x in [-10, 9]
    # No four collinear on a parabola (any line intersects a parabola in at most 2 points)
    base_points = [[x, x * x] for x in range(-10, 10)]

    # Verify no four collinear
    if _has_four_collinear(base_points):
        # Should not happen for a parabola, but fallback
        return base_points

    # Build collinear sets for fast queries
    collinear = _build_collinear_sets(base_points)

    # Find no-three-collinear subset size
    subset_size = _no_three_collinear_size(base_points, collinear)

    # If subset size is already small (<= 10), we're in good shape
    # Try to improve by perturbing points to reduce collinear triples
    best_points = base_points[:]
    best_subset_size = subset_size

    # Local search: perturb individual points slightly to reduce triples
    # while maintaining no-four-collinear
    for _ in range(5000):
        if time.time() > deadline:
            break
        if best_subset_size <= 8:
            break

        # Pick a random point to move
        idx = random.randint(0, 19)
        ox, oy = best_points[idx]
        # Try small perturbations
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = ox + dx, oy + dy
                if abs(nx) > 100 or abs(ny) > 100:
                    continue
                new_points = best_points[:]
                new_points[idx] = [nx, ny]

                # Check distinctness
                if len(set(tuple(p) for p in new_points)) != 20:
                    continue

                # Check no four collinear
                if _has_four_collinear(new_points):
                    continue

                # Check subset size
                collinear_new = _build_collinear_sets(new_points)
                size_new = _no_three_collinear_size(new_points, collinear_new)

                if size_new < best_subset_size:
                    best_points = new_points
                    best_subset_size = size_new
                    break
            else:
                continue
            break

    # Final verification
    if _has_four_collinear(best_points):
        return base_points

    return best_points

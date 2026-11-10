"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd, isqrt
from fractions import Fraction
from itertools import combinations

def construct():
    """Search for integral point sets with no three collinear and no four concyclic.

    Strategy (graph-clique + structural candidate generation):
      1. Vertices = integer lattice points in a bounded box; edge iff squared
         distance is a perfect square (=> rational distance).
      2. A valid set is a clique in this graph with no 3 collinear and no 4 concyclic.
      3. To add a point we do NOT brute-force the lattice: a new point must lie on
         the intersection of the circle around each existing point, so we generate
         only the finitely many lattice points at integer distance from ALL existing
         points, then filter by the geometric constraints.
    """
    t_start = time.time()
    DEADLINE = t_start + 142  # 8s safety margin before 150s kill

    def time_ok():
        return time.time() < DEADLINE

    # ---------- exact helpers ----------
    def is_square(n):
        if n < 0:
            return False
        r = isqrt(n)
        return r * r == n

    def is_rational_square(frac):
        if frac < 0:
            return False
        num, den = frac.numerator, frac.denominator
        return is_square(num) and is_square(den)

    def check_collinear(p1, p2, p3):
        (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
        return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0

    def check_concyclic(p1, p2, p3, p4):
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = p1, p2, p3, p4
        a1, b1, c1 = x1, y1, x1 * x1 + y1 * y1
        a2, b2, c2 = x2, y2, x2 * x2 + y2 * y2
        a3, b3, c3 = x3, y3, x3 * x3 + y3 * y3
        a4, b4, c4 = x4, y4, x4 * x4 + y4 * y4
        det = (a1 * (b2 * (c3 - c4) - c2 * (b3 - b4) + (b3 * c4 - c3 * b4))
             - b1 * (a2 * (c3 - c4) - c2 * (a3 - a4) + (a3 * c4 - c3 * a4))
             + c1 * (a2 * (b3 - b4) - b2 * (a3 - a4) + (a3 * b4 - b3 * a4))
             - (a2 * (b3 * c4 - c3 * b4) - b2 * (a3 * c4 - c3 * a4) + c2 * (a3 * b4 - b3 * a4)))
        return det == 0

    def validate_full(points):
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                d2 = dx * dx + dy * dy
                if not is_rational_square(Fraction(d2.numerator, d2.denominator)):
                    return False
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if check_collinear(points[i], points[j], points[k]):
                        return False
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        if check_concyclic(points[i], points[j], points[k], points[l]):
                            return False
        return True

    def validate_incremental(points, new_point):
        # distance to all existing
        for p in points:
            dx = new_point[0] - p[0]
            dy = new_point[1] - p[1]
            d2 = dx * dx + dy * dy
            if not is_rational_square(Fraction(d2.numerator, d2.denominator)):
                return False
        # collinearity with all pairs
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                if check_collinear(points[i], points[j], new_point):
                    return False
        # concyclicity with all triples
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if check_concyclic(points[i], points[j], points[k], new_point):
                        return False
        return True

    # ---------- structural candidate generation ----------
    def integer_points_at_distance(center, r):
        """All integer points at integer distance r from integer center."""
        cx, cy = int(center[0]), int(center[1])
        r2 = r * r
        out = []
        for dx in range(-r, r + 1):
            dy2 = r2 - dx * dx
            if dy2 < 0:
                continue
            dy = isqrt(dy2)
            if dy * dy != dy2:
                continue
            for s in ((1, -1) if dy != 0 else (0,)):
                for t in ((1, -1) if dx != 0 else (0,)):
                    out.append((cx + t * dx, cy + s * dy))
        return out

    def candidates_compatible_with_all(points, max_r):
        """Integer points at integer distance from EVERY existing point (within max_r of each).

        This is the key structural filter: a valid new point must be at rational
        distance from all existing points, so for integer points it must lie on the
        integer-radius circle of each existing point. Intersect those sets.
        """
        if not points:
            return set()
        # start from the circle of the first point (smallest set usually)
        cand = set(integer_points_at_distance(points[0], r) for r in range(1, max_r + 1))
        for p in points[1:]:
            if not cand:
                break
            allowed = set()
            for r in range(1, max_r + 1):
                allowed.update(integer_points_at_distance(p, r))
            cand = cand & allowed
        cand.difference_update(points)
        return cand

    # ---------- search ----------
    # known good 5-point seed
    best_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    point_set = set(best_points)
    best_size_ever = len(best_points)

    random.seed(12345)

    # Phase A: try to grow the seed by structural candidate generation, escalating radius.
    max_r = 40
    while time_ok() and len(best_points) < 8:
        improved = False
        cand = candidates_compatible_with_all(best_points, max_r)
        # also add a broader pool: integer points at integer distance from each existing point
        # (relax to pairwise, not all-at-once) to give the validator more to chew on
        for p in best_points:
            for r in range(1, max_r + 1):
                cand.update(integer_points_at_distance(p, r))
        cand.difference_update(point_set)
        cand_list = list(cand)
        random.shuffle(cand_list)
        for c in cand_list:
            if not time_ok():
                break
            if validate_incremental(best_points, c):
                best_points.append(c)
                point_set.add(c)
                best_size_ever = max(best_size_ever, len(best_points))
                improved = True
                break
        if improved:
            continue
        max_r += 15
        if max_r > 400:
            break

    # Phase B: clique search over a fresh bounded lattice box (different starting seed).
    # Build the integer-distance graph and greedily extract large cliques, then filter.
    def greedy_clique_search(box, target_size):
        """Greedily build a clique in the integer-distance graph, filtering geometry."""
        # candidate vertices
        verts = [(x, y) for x in range(-box, box + 1) for y in range(-box, box + 1)]
        random.shuffle(verts)
        # adjacency: integer squared distance
        # We do a greedy set-growth: keep a current set; a vertex is admissible if at
        # integer distance from all members and passes geometry checks.
        cur = []
        curset = set()
        for v in verts:
            if not time_ok():
                break
            if len(cur) >= target_size:
                break
            # integer distance to all current members
            ok = True
            for u in cur:
                dx = v[0] - u[0]
                dy = v[1] - u[1]
                if not is_square(dx * dx + dy * dy):
                    ok = False
                    break
            if not ok:
                continue
            # geometry
            if validate_incremental(cur, v):
                cur.append(v)
                curset.add(v)
        return cur

    for box in (40, 60, 90, 130):
        if not time_ok():
            break
        if len(best_points) >= 8:
            break
        for attempt in range(3):
            if not time_ok():
                break
            cl = greedy_clique_search(box, 8)
            if len(cl) > len(best_points):
                best_points = cl
                point_set = set(cl)
                best_size_ever = max(best_size_ever, len(cl))
                if len(best_points) >= 8:
                    break

    # Phase C: targeted 6th-point search on the seed using circle-circle intersections.
    # For a candidate to be at integer distance from two fixed integer points A,B, it lies
    # on the intersection of two circles. We already cover this via candidates_compatible_with_all,
    # but here we explicitly search a wider radius window focused on the seed's bounding box.
    if time_ok() and len(best_points) == 5:
        seed5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
        for max_r in range(30, 260, 10):
            if not time_ok():
                break
            cand = candidates_compatible_with_all(seed5, max_r)
            cand_list = list(cand)
            random.shuffle(cand_list)
            found = None
            for c in cand_list:
                if validate_incremental(seed5, c):
                    found = c
                    break
            if found is not None:
                best_points = seed5 + [found]
                point_set = set(best_points)
                best_size_ever = max(best_size_ever, len(best_points))
                break

    # Final validation; fall back to seed if anything is off.
    if not validate_full(best_points):
        best_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]

    # Format the result
    return {'D': 1, 'points': [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in best_points]}
# EVOLVE-BLOCK-END

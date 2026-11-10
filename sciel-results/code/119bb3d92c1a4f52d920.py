# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq

_P = 7
_Q = 49
_Q2 = 2401
_DEADLINE = time.time() + 600.0
_MARGIN = 30.0


def _build_tables():
    """Build field arithmetic, line membership, and the baseline Nikodym set."""
    p = _P
    k = 3  # alpha^2 = 3 (3 is a quadratic non-residue mod 7)

    def pt_idx(xa, xb, ya, yb):
        return ((xa * p + xb) * p * p + ya * p + yb)

    # Build line_points: list of arrays of point indices
    # Line types:
    #   0..48: vertical lines (fixed x, all y)
    #   49..2448: slanted lines (slope m, offset (oa,ob))
    line_points = []

    # Vertical lines: for each x in F_49, the line { (x, y) : y in F_49 }
    for xa in range(p):
        for xb in range(p):
            pts = []
            for t in range(p * p):
                ta, tb = t // p, t % p
                ya, yb = ta, tb
                pts.append(pt_idx(xa, xb, ya, yb))
            line_points.append(np.array(pts, dtype=np.int64))

    # Slanted lines: for each slope m in F_49 and offset (oa,ob) in F_49^2
    # Line: { (t, m*t + offset) : t in F_49 }
    for m in range(p * p):
        ma, mb = m // p, m % p
        for oa in range(p):
            for ob in range(p):
                pts = []
                for t in range(p * p):
                    ta, tb = t // p, t % p
                    # m*t = (ma+mb*alpha)*(ta+tb*alpha) = (ma*ta + mb*tb*3, ma*tb + mb*ta)
                    ya = (ma * ta + mb * tb * k + oa) % p
                    yb = (ma * tb + mb * ta + ob) % p
                    pts.append(pt_idx(ta, tb, ya, yb))
                line_points.append(np.array(pts, dtype=np.int64))

    n_lines = len(line_points)  # 49 + 49*49 = 2450

    # For each point, list of lines containing it
    point_lines = [[] for _ in range(_Q2)]
    for li, pts in enumerate(line_points):
        for pi in pts:
            point_lines[pi].append(li)

    # Baseline: S = { (x,y) : 2*xa + ya^2 - yb^2*3 == 0 (mod 7) AND ya NOT a square mod 7 }
    squares = set()
    for u in range(p):
        squares.add((u * u) % p)

    S = np.zeros(_Q2, dtype=bool)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2 * xa + ya * ya - yb * yb * k) % p == 0:
                        if ya not in squares:
                            S[pt_idx(xa, xb, ya, yb)] = True

    return p, k, line_points, point_lines, S


def nikodym_set() -> np.ndarray:
    p, k, line_points, point_lines, S = _build_tables()
    q = _Q
    q2 = _Q2
    n_lines = len(line_points)

    # Precompute line_S_count and cover_count
    line_S_count = np.zeros(n_lines, dtype=np.int64)
    for li in range(n_lines):
        pts = line_points[li]
        cnt = 0
        for pi in pts:
            if S[pi]:
                cnt += 1
        line_S_count[li] = cnt

    # cover_count[x] = number of lines L containing x such that:
    #   if x in S: S_count(L) == 1 and x is the S-point on L
    #   if x not in S: S_count(L) == 0
    cover_count = np.zeros(q2, dtype=np.int64)
    for li in range(n_lines):
        pts = line_points[li]
        s_count = 0
        s_point = -1
        for pi in pts:
            if S[pi]:
                s_count += 1
                s_point = pi
                if s_count > 1:
                    break
        if s_count == 0:
            for pi in pts:
                cover_count[pi] += 1
        elif s_count == 1:
            cover_count[s_point] += 1

    # Verify baseline validity
    if np.any(cover_count == 0):
        # Should not happen with correct baseline; fallback to all points
        result = np.zeros((q2, 2, 2), dtype=np.int64)
        for z in range(q2):
            x = z // (p * p)
            y = z % (p * p)
            xa, xb = x // p, x % p
            ya, yb = y // p, y % p
            result[z] = [[xa, xb], [ya, yb]]
        return result

    # Track zero-line count per point: number of lines through z with S_count == 0
    zero_line_count = np.zeros(q2, dtype=np.int64)
    for z in range(q2):
        for li in point_lines[z]:
            if line_S_count[li] == 0:
                zero_line_count[z] += 1

    def try_add(z):
        """Try adding z to S. Returns True if valid (and keeps the change)."""
        if zero_line_count[z] == 0:
            return False

        # Check that no other point loses its last covering line
        affected_lines = point_lines[z]
        for li in affected_lines:
            old_count = line_S_count[li]
            if old_count == 0:
                # L used to cover all its points; now covers only z
                for pi in line_points[li]:
                    if pi != z and cover_count[pi] <= 1:
                        return False
            elif old_count == 1:
                # L used to cover one S-point; now covers no one
                for pi in line_points[li]:
                    if S[pi]:
                        if cover_count[pi] <= 1:
                            return False
                        break
            # old_count >= 2: no change in coverage

        # All checks passed: commit the addition
        S[z] = True
        for li in affected_lines:
            old_count = line_S_count[li]
            line_S_count[li] += 1
            if old_count == 0:
                for pi in line_points[li]:
                    if pi != z:
                        cover_count[pi] -= 1
                        zero_line_count[pi] -= 1
            elif old_count == 1:
                for pi in line_points[li]:
                    if S[pi] and pi != z:
                        cover_count[pi] -= 1
                        break

        # Recompute z's cover_count
        new_z_cover = 0
        for li in affected_lines:
            if line_S_count[li] == 1:
                for pi in line_points[li]:
                    if S[pi]:
                        if pi == z:
                            new_z_cover += 1
                        break
        cover_count[z] = new_z_cover
        return True

    # Priority queue based greedy expansion
    # Use a min-heap keyed by (zero_line_count, cover_count, z)
    # Points with fewer zero-lines are more constrained, so they're harder to add
    # We want to add points that are "easiest" to add while maintaining validity

    def build_pq():
        """Build priority queue of candidate points."""
        pq = []
        for z in range(q2):
            if S[z] or zero_line_count[z] == 0:
                continue
            # Priority: lower zero_line_count means more constrained (harder)
            # We want to add points that won't break others, so prefer those with
            # higher zero_line_count (more "slack")
            # But also consider cover_count: lower cover_count means more critical
            # Use negative zero_line_count to prefer adding "safe" points first
            priority = (-int(zero_line_count[z]), int(cover_count[z]), z)
            heapq.heappush(pq, priority)
        return pq

    # Multiple passes with different strategies
    for pass_num in range(200):
        if time.time() > _DEADLINE - _MARGIN:
            break

        if pass_num % 3 == 0:
            # Strategy 1: Priority queue by (-zero_line_count, cover_count)
            pq = build_pq()
            added_any = False
            while pq:
                if time.time() > _DEADLINE - _MARGIN:
                    break
                neg_zlc, cc, z = heapq.heappop(pq)
                if S[z] or zero_line_count[z] == 0:
                    continue
                if try_add(z):
                    added_any = True
            if not added_any:
                break
        elif pass_num % 3 == 1:
            # Strategy 2: Sort by zero_line_count ascending (most constrained first)
            candidates = []
            for z in range(q2):
                if S[z] or zero_line_count[z] == 0:
                    continue
                candidates.append((int(zero_line_count[z]), z))
            candidates.sort()
            added_any = False
            for zlc, z in candidates:
                if time.time() > _DEADLINE - _MARGIN:
                    break
                if S[z] or zero_line_count[z] == 0:
                    continue
                if try_add(z):
                    added_any = True
            if not added_any:
                break
        else:
            # Strategy 3: Random order
            candidates = [z for z in range(q2) if not S[z] and zero_line_count[z] > 0]
            np.random.shuffle(candidates)
            added_any = False
            for z in candidates:
                if time.time() > _DEADLINE - _MARGIN:
                    break
                if S[z] or zero_line_count[z] == 0:
                    continue
                if try_add(z):
                    added_any = True
            if not added_any:
                break

    # Build output: N = all points not in S
    pts_list = []
    for z in range(q2):
        if not S[z]:
            x = z // (p * p)
            y = z % (p * p)
            xa, xb = x // p, x % p
            ya, yb = y // p, y % p
            pts_list.append([[xa, xb], [ya, yb]])

    if not pts_list:
        # Fallback: return all points
        for z in range(q2):
            x = z // (p * p)
            y = z % (p * p)
            xa, xb = x // p, x % p
            ya, yb = y // p, y % p
            pts_list.append([[xa, xb], [ya, yb]])

    result = np.array(pts_list, dtype=np.int64)
    return result


# EVOLVE-BLOCK-END

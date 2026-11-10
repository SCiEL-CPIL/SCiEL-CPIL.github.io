# EVOLVE-BLOCK-START
import numpy as np
import time
import random

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

    # Baseline: parabola y = x^2 (known valid Nikodym complement, size 49)
    S = np.zeros(_Q2, dtype=bool)
    for t in range(p * p):
        ta, tb = t // p, t % p
        # y = t^2 = (ta^2 + 2*ta*tb*alpha + tb^2*3, 2*ta*tb)
        ya = (ta * ta + tb * tb * k) % p
        yb = (2 * ta * tb) % p
        S[pt_idx(ta, tb, ya, yb)] = True

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
        # Fallback to all points
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

    # Track S size for scoring
    S_size = int(np.sum(S))
    best_S_size = S_size
    best_S = S.copy()
    best_line_S_count = line_S_count.copy()
    best_cover_count = cover_count.copy()
    best_zero_line_count = zero_line_count.copy()

    def try_add(z):
        """Try adding z to S. Returns True if valid (and keeps the change)."""
        if zero_line_count[z] == 0:
            return False

        # Check that no other point loses its last covering line
        affected_lines = point_lines[z]
        for li in affected_lines:
            old_count = line_S_count[li]
            if old_count == 0:
                for pi in line_points[li]:
                    if pi != z and cover_count[pi] <= 1:
                        return False
            elif old_count == 1:
                for pi in line_points[li]:
                    if S[pi]:
                        if cover_count[pi] <= 1:
                            return False
                        break

        # Commit the addition
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

    def revert_add(z):
        """Revert adding z to S (i.e., remove z from S)."""
        affected_lines = point_lines[z]
        for li in affected_lines:
            old_count = line_S_count[li]
            line_S_count[li] -= 1
            if old_count == 1:
                # Line now has 0 S-points, covers all its points
                for pi in line_points[li]:
                    if pi != z:
                        cover_count[pi] += 1
                        zero_line_count[pi] += 1
            elif old_count == 2:
                # Line now has 1 S-point
                for pi in line_points[li]:
                    if S[pi]:
                        cover_count[pi] += 1
                        break
                for pi in line_points[li]:
                    if not S[pi] and pi != z:
                        cover_count[pi] -= 1
                        zero_line_count[pi] -= 1

        S[z] = False

        # Recompute z's cover_count (z is now not in S)
        new_z_cover = 0
        for li in affected_lines:
            if line_S_count[li] == 0:
                new_z_cover += 1
        cover_count[z] = new_z_cover

    def try_remove(z):
        """Try removing z from S. Returns True if valid (and keeps the change)."""
        if not S[z]:
            return False

        affected_lines = point_lines[z]
        for li in affected_lines:
            old_count = line_S_count[li]
            if old_count == 1:
                # After removal, this line has 0 S-points, so it covers all its points
                # Need to ensure all points on this line (except z) still have cover > 0
                for pi in line_points[li]:
                    if pi != z and cover_count[pi] <= 1:
                        return False

        # Commit the removal
        S[z] = False
        for li in affected_lines:
            old_count = line_S_count[li]
            line_S_count[li] -= 1
            if old_count == 1:
                # Line now has 0 S-points, covers all its points
                for pi in line_points[li]:
                    cover_count[pi] += 1
                    zero_line_count[pi] += 1
            elif old_count == 2:
                # Line now has 1 S-point, covers only that point
                for pi in line_points[li]:
                    if S[pi]:
                        cover_count[pi] += 1
                        break
                for pi in line_points[li]:
                    if not S[pi]:
                        cover_count[pi] -= 1
                        zero_line_count[pi] -= 1

        # Recompute z's cover_count (z is now not in S)
        new_z_cover = 0
        for li in affected_lines:
            if line_S_count[li] == 0:
                new_z_cover += 1
        cover_count[z] = new_z_cover
        return True

    def revert_remove(z):
        """Revert removing z from S (i.e., add z back to S)."""
        affected_lines = point_lines[z]
        for li in affected_lines:
            old_count = line_S_count[li]
            line_S_count[li] += 1
            if old_count == 0:
                # Line now has 1 S-point (z)
                for pi in line_points[li]:
                    if pi != z:
                        cover_count[pi] -= 1
                        zero_line_count[pi] -= 1
            elif old_count == 1:
                # Line now has 2 S-points
                for pi in line_points[li]:
                    if S[pi] and pi != z:
                        cover_count[pi] -= 1
                        break

        S[z] = True

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

    # Greedy phase: add points to maximize S
    for pass_num in range(50):
        if time.time() > _DEADLINE - _MARGIN:
            break

        added_any = False
        candidates = [(int(zero_line_count[z]), z) for z in range(q2) if not S[z] and zero_line_count[z] > 0]
        candidates.sort(reverse=True)
        for zlc, z in candidates:
            if time.time() > _DEADLINE - _MARGIN:
                break
            if S[z] or zero_line_count[z] == 0:
                continue
            if try_add(z):
                added_any = True

        if not added_any:
            break

    # Update best
    current_size = int(np.sum(S))
    if current_size > best_S_size:
        best_S_size = current_size
        best_S = S.copy()
        best_line_S_count = line_S_count.copy()
        best_cover_count = cover_count.copy()
        best_zero_line_count = zero_line_count.copy()

    # Local search phase with simulated annealing
    T = 5.0
    T_min = 0.1
    cooling = 0.9995

    while time.time() < _DEADLINE - _MARGIN and T > T_min:
        z = random.randint(0, q2 - 1)

        if S[z]:
            # Try to remove
            if try_remove(z):
                new_size = int(np.sum(S))
                delta = new_size - best_S_size
                if delta >= 0 or random.random() < np.exp(delta / T):
                    if new_size > best_S_size:
                        best_S_size = new_size
                        best_S = S.copy()
                        best_line_S_count = line_S_count.copy()
                        best_cover_count = cover_count.copy()
                        best_zero_line_count = zero_line_count.copy()
                else:
                    # Revert
                    revert_remove(z)
            T *= cooling
        else:
            # Try to add
            if try_add(z):
                new_size = int(np.sum(S))
                delta = new_size - best_S_size
                if delta >= 0 or random.random() < np.exp(delta / T):
                    if new_size > best_S_size:
                        best_S_size = new_size
                        best_S = S.copy()
                        best_line_S_count = line_S_count.copy()
                        best_cover_count = cover_count.copy()
                        best_zero_line_count = zero_line_count.copy()
                else:
                    # Revert
                    revert_add(z)
            T *= cooling

    # Final greedy pass on the best solution found
    S = best_S.copy()
    line_S_count = best_line_S_count.copy()
    cover_count = best_cover_count.copy()
    zero_line_count = best_zero_line_count.copy()

    for pass_num in range(50):
        if time.time() > _DEADLINE - _MARGIN:
            break
        added_any = False
        candidates = [(int(zero_line_count[z]), z) for z in range(q2) if not S[z] and zero_line_count[z] > 0]
        candidates.sort(reverse=True)
        for zlc, z in candidates:
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
        for z in range(q2):
            x = z // (p * p)
            y = z % (p * p)
            xa, xb = x // p, x % p
            ya, yb = y // p, y % p
            pts_list.append([[xa, xb], [ya, yb]])

    result = np.array(pts_list, dtype=np.int64)
    return result


# EVOLVE-BLOCK-END

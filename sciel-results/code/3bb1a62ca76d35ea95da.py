# EVOLVE-BLOCK-START
import numpy as np
import time

_P = 7
_Q = 49
_Q2 = 2401
_T0 = time.time()
_DEADLINE = _T0 + 600.0
_MARGIN = 30.0


def _time_left():
    return _DEADLINE - time.time() - _MARGIN


def nikodym_set() -> np.ndarray:
    p = _P
    q = _Q
    q2 = _Q2
    k = 3  # alpha^2 = 3

    # Field operations using precomputed tables
    # Index: a * p + b for a + b*alpha
    add_table = np.zeros((q, q), dtype=np.int64)
    mul_table = np.zeros((q, q), dtype=np.int64)
    for x in range(q):
        xa, xb = x // p, x % p
        for y in range(q):
            ya, yb = y // p, y % p
            add_table[x, y] = ((xa + ya) % p) * p + ((xb + yb) % p)
            mul_table[x, y] = ((xa * ya + xb * yb * k) % p) * p + ((xa * yb + xb * ya) % p)

    # Point index: x * q + y
    def pt_idx(x, y):
        return x * q + y

    def pt_xy(idx):
        return idx // q, idx % q

    # Build all lines
    line_points = []

    # Vertical lines: 49 lines
    for x in range(q):
        pts = np.array([pt_idx(x, y) for y in range(q)], dtype=np.int64)
        line_points.append(pts)

    # Slanted lines: 49*49 = 2401 lines
    for m in range(q):
        for c in range(q):
            pts = np.array([pt_idx(t, int(add_table[mul_table[m, t], c])) for t in range(q)], dtype=np.int64)
            line_points.append(pts)

    n_lines = len(line_points)  # 2450

    # For each point, list of lines containing it
    point_lines = [[] for _ in range(q2)]
    for li in range(n_lines):
        for pi in line_points[li]:
            point_lines[pi].append(li)

    # Precompute line membership as a boolean matrix (n_lines x q2)
    line_member = np.zeros((n_lines, q2), dtype=np.int8)
    for li in range(n_lines):
        for pi in line_points[li]:
            line_member[li, pi] = 1

    # Start with S = graph of a random function (always valid)
    rng = np.random.default_rng(42)
    f_vals = rng.integers(0, q, size=q)
    S = np.zeros(q2, dtype=bool)
    for x in range(q):
        S[pt_idx(x, int(f_vals[x]))] = True

    # Line S-counts
    line_S_count = np.zeros(n_lines, dtype=np.int64)
    for li in range(n_lines):
        line_S_count[li] = np.sum(line_member[li] & S)

    # For each point, count how many lines through it have S-count 0 or 1
    # A point z is "covered" if there exists a line L through z with line_S_count[L] <= 1
    # and if line_S_count[L] == 1, the S-point on L is z.

    # Efficient validity check:
    # For each point z, check if there's a line through z with S-count <= 1 (and if 1, it's z)
    def is_valid():
        # For each line, find S-points on it
        # A point z is valid if:
        #   - z is in S: there exists a line L through z with exactly 1 S-point (which is z)
        #   - z not in S: there exists a line L through z with 0 S-points
        # Equivalently: for each z, min over lines L through z of (S_count[L] - (1 if z in S and z is the only S on L else 0)) <= 0
        # Simpler: for each z, check lines through z
        for z in range(q2):
            if S[z]:
                found = False
                for li in point_lines[z]:
                    if line_S_count[li] == 1:
                        # Check that z is the only S-point on this line
                        # line_S_count[li] == 1 and S[z] means z is the one
                        found = True
                        break
                if not found:
                    return False
            else:
                found = False
                for li in point_lines[z]:
                    if line_S_count[li] == 0:
                        found = True
                        break
                if not found:
                    return False
        return True

    # Verify initial validity
    if not is_valid():
        # Shouldn't happen for function graph, but fallback
        S[:] = False
        line_S_count[:] = 0

    # Greedy addition with lookahead
    # For each candidate point z not in S, determine if adding it keeps validity
    # Adding z to S: for each line L through z, line_S_count[L] increases by 1
    # This may break validity for:
    #   - Points on lines L where S_count was 0 (now 1): those points need another line with S_count 0
    #   - Points on lines L where S_count was 1 (now 2): those points (if in S) need another line with S_count 1
    #   - z itself: needs a line with S_count 1 (which is z)

    # Precompute for efficiency: for each point, store its lines
    # Use a different strategy: try to add points in batches, checking validity after each batch

    # First, identify "easy" additions: points z where adding z doesn't create any line with S_count > 1
    # i.e., all lines through z currently have S_count == 0
    # Such additions are always safe for z, but may affect other points

    # Let's do iterative greedy: repeatedly find the best point to add
    # Score for adding z: number of points that become "more constrained" (negative) 
    # minus number of points that become "easier" (positive)
    # Simplified: just try to add, check validity, keep if valid

    # To speed up validity check, maintain for each point the count of "good lines"
    # A good line for point z is:
    #   if z in S: line with S_count == 1 (and z is the S-point)
    #   if z not in S: line with S_count == 0
    # Point z is valid iff it has at least one good line

    good_line_count = np.zeros(q2, dtype=np.int64)
    for z in range(q2):
        cnt = 0
        for li in point_lines[z]:
            if S[z]:
                if line_S_count[li] == 1:
                    cnt += 1
            else:
                if line_S_count[li] == 0:
                    cnt += 1
        good_line_count[z] = cnt

    # A point is "critical" if good_line_count[z] == 1
    # Adding a point may reduce good_line_count for other points

    # Greedy: try to add points, prioritizing those that are "safe"
    # A point z is safe to add if:
    #   - After adding, good_line_count[z] >= 1 (z still has a good line)
    #   - For all points w affected, good_line_count[w] >= 1

    # Affected points: all points on lines through z
    # For a line L through z:
    #   If S_count[L] was 0: now 1. Points w on L (w != z, w not in S): their good lines lose L
    #   If S_count[L] was 1: now 2. Points w on L (w != z, w in S): their good lines lose L
    #   z itself: if z was not in S, z's good lines: previously counted lines with S_count 0.
    #     Now z is in S, so z needs lines with S_count 1. The lines through z that had S_count 0 now have S_count 1, so they become good for z.

    # This is complex. Let's just do trial-and-error with the full validity check,
    # but optimize the validity check.

    def check_valid_fast():
        # A point z is valid iff good_line_count[z] >= 1
        return np.all(good_line_count >= 1)

    # Update good_line_count when adding/removing a point
    def add_point(z):
        """Add z to S, update line_S_count and good_line_count. Returns True if valid."""
        S[z] = True
        for li in point_lines[z]:
            old_count = line_S_count[li]
            line_S_count[li] = old_count + 1
            # Update good_line_count for all points on this line
            for w in line_points[li]:
                if w == z:
                    continue
                # w's good line status for line li changes
                if S[w]:
                    # w in S: good if S_count == 1
                    if old_count == 1:
                        # Was good, now S_count = 2, not good
                        good_line_count[w] -= 1
                    elif old_count + 1 == 1:
                        # Was not good, now good (but old_count >= 0, so old_count+1==1 means old_count==0, but we're adding so old_count was 0, now 1)
                        # Wait, if w in S and old_count was 0, that means w is not on this line? No, w is on this line.
                        # If w in S and S_count was 0, that's a contradiction since w is in S and on the line.
                        # So this case shouldn't happen.
                        pass
                else:
                    # w not in S: good if S_count == 0
                    if old_count == 0:
                        # Was good, now S_count = 1, not good
                        good_line_count[w] -= 1
                    elif old_count + 1 == 0:
                        # Impossible
                        pass
        # Now update z's good_line_count
        # z is now in S, so z needs lines with S_count == 1
        # Recompute z's good_line_count
        z_good = 0
        for li in point_lines[z]:
            if line_S_count[li] == 1:
                z_good += 1
        good_line_count[z] = z_good
        return check_valid_fast()

    def remove_point(z):
        """Remove z from S, update line_S_count and good_line_count. Returns True if valid."""
        for li in point_lines[z]:
            old_count = line_S_count[li]
            line_S_count[li] = old_count - 1
            # Update good_line_count for all points on this line
            for w in line_points[li]:
                if w == z:
                    continue
                if S[w]:
                    if old_count == 1:
                        # Was good, now S_count = 0, not good
                        good_line_count[w] -= 1
                    elif old_count - 1 == 1:
                        # Was not good, now good
                        good_line_count[w] += 1
                else:
                    if old_count == 0:
                        # Impossible (old_count >= 1 since we're removing)
                        pass
                    elif old_count - 1 == 0:
                        # Was not good, now good
                        good_line_count[w] += 1
        # Update z's good_line_count
        # z is now not in S, so z needs lines with S_count == 0
        z_good = 0
        for li in point_lines[z]:
            if line_S_count[li] == 0:
                z_good += 1
        good_line_count[z] = z_good
        S[z] = False
        return check_valid_fast()

    # Greedy addition
    added = 0
    for _ in range(500):  # Limit iterations
        if _time_left() <= 0:
            break
        # Find best candidate
        best_z = -1
        best_score = -1
        # Try a random subset of candidates for speed
        candidates = [z for z in range(q2) if not S[z]]
        if len(candidates) > 200:
            candidates = list(rng.choice(candidates, size=200, replace=False))
        for z in candidates:
            # Estimate score: how many points are affected
            # Just try it
            if add_point(z):
                best_z = z
                best_score = 1
                break
            else:
                remove_point(z)
        if best_z == -1:
            break
        added += 1

    # Build output: N = all points not in S
    pts_list = []
    for z in range(q2):
        if not S[z]:
            x, y = pt_xy(z)
            xa, xb = x // p, x % p
            ya, yb = y // p, y % p
            pts_list.append([[xa, xb], [ya, yb]])

    result = np.array(pts_list, dtype=np.int64)
    return result


# EVOLVE-BLOCK-END

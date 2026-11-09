# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import sys

def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences for n=54 points and m=27 lines on integer lattice.
    Two-phase local search: alternate between optimizing lines (given points) and
    points (given lines), with incremental delta scoring for speed.
    """
    start_time = time.time()
    deadline = start_time + 585  # safety margin before 630s kill

    def normalize_line(A, B, C):
        g = math.gcd(abs(int(A)), math.gcd(abs(int(B)), abs(int(C))))
        if g == 0:
            return (0, 0, 0)
        A, B, C = int(A) // g, int(B) // g, int(C) // g
        if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
            A, B, C = -A, -B, -C
        return (A, B, C)

    def count_incidents(pts, lns):
        """Vectorized incidence count."""
        vals = (lns[:, 0:1].T * pts[:, 0:1] +
                lns[:, 1:2].T * pts[:, 1:2] +
                lns[:, 2:3].T)
        return int(np.sum(vals == 0))

    def has_duplicate_points(pts):
        seen = set()
        for p in pts:
            key = (int(p[0]), int(p[1]))
            if key in seen:
                return True
            seen.add(key)
        return False

    def has_duplicate_lines(lns):
        seen = set()
        for i in range(len(lns)):
            n = normalize_line(lns[i, 0], lns[i, 1], lns[i, 2])
            if n in seen:
                return True
            seen.add(n)
        return False

    # Baseline: k=3 grid construction
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

    assert points.shape == (54, 2)
    assert lines.shape == (27, 3)
    assert not has_duplicate_points(points)
    assert not has_duplicate_lines(lines)

    best_score = count_incidents(points, lines)
    best_points = points.copy()
    best_lines = lines.copy()

    # Build candidate sets
    candidate_xs = list(range(0, 14))
    candidate_ys = list(range(0, 40))
    candidate_points = set()
    for x in candidate_xs:
        for y in candidate_ys:
            candidate_points.add((x, y))
    candidate_points_list = list(candidate_points)

    # Candidate lines with small coefficients
    candidate_lines_set = set()
    for A in range(-8, 9):
        for B in range(-8, 9):
            for C in range(-40, 41):
                if A == 0 and B == 0:
                    continue
                n = normalize_line(A, B, C)
                if n != (0, 0, 0):
                    candidate_lines_set.add(n)
    candidate_lines_list = list(candidate_lines_set)

    # Precompute point-incidence for current lines (how many lines pass through each point)
    def compute_point_incidences(pts, lns):
        """Return array of length 54: number of lines through each point."""
        inc = np.zeros(54, dtype=np.int64)
        for li in range(27):
            A, B, C = int(lns[li, 0]), int(lns[li, 1]), int(lns[li, 2])
            for pi in range(54):
                if A * int(pts[pi, 0]) + B * int(pts[pi, 1]) + C == 0:
                    inc[pi] += 1
        return inc

    # Precompute line-incidence for current points (how many points on each line)
    def compute_line_incidences(pts, lns):
        """Return array of length 27: number of points on each line."""
        inc = np.zeros(27, dtype=np.int64)
        for li in range(27):
            A, B, C = int(lns[li, 0]), int(lns[li, 1]), int(lns[li, 2])
            for pi in range(54):
                if A * int(pts[pi, 0]) + B * int(pts[pi, 1]) + C == 0:
                    inc[li] += 1
        return inc

    # Local search: alternate phases
    improvement_found = True
    phase = 0  # 0 = optimize lines, 1 = optimize points

    while improvement_found:
        improvement_found = False
        phase = 1 - phase  # alternate

        if time.time() > deadline - 30:
            break

        if phase == 0:
            # Optimize lines: for each line, try replacing with a better candidate
            for li in range(27):
                if time.time() > deadline - 30:
                    break

                orig_line = tuple(int(x) for x in best_lines[li])
                orig_line_inc = 0
                A, B, C = int(best_lines[li, 0]), int(best_lines[li, 1]), int(best_lines[li, 2])
                for pi in range(54):
                    if A * int(best_points[pi, 0]) + B * int(best_points[pi, 1]) + C == 0:
                        orig_line_inc += 1

                best_delta = 0
                best_new_line = None

                for new_line in candidate_lines_list:
                    if new_line == orig_line:
                        continue
                    # Check duplicate
                    is_dup = False
                    for j in range(27):
                        if j != li and normalize_line(*tuple(int(x) for x in best_lines[j])) == new_line:
                            is_dup = True
                            break
                    if is_dup:
                        continue

                    # Count incidences for new line
                    new_line_inc = 0
                    A2, B2, C2 = new_line
                    for pi in range(54):
                        if A2 * int(best_points[pi, 0]) + B2 * int(best_points[pi, 1]) + C2 == 0:
                            new_line_inc += 1

                    delta = new_line_inc - orig_line_inc
                    if delta > best_delta:
                        best_delta = delta
                        best_new_line = new_line

                if best_new_line is not None and best_delta > 0:
                    new_lines = best_lines.copy()
                    new_lines[li] = np.array(best_new_line, dtype=np.int64)
                    best_score += best_delta
                    best_lines = new_lines
                    improvement_found = True

        else:
            # Optimize points: for each point, try replacing with a better candidate
            for pi in range(54):
                if time.time() > deadline - 30:
                    break

                orig_pt = tuple(int(x) for x in best_points[pi])
                orig_pt_inc = 0
                for li in range(27):
                    A, B, C = int(best_lines[li, 0]), int(best_lines[li, 1]), int(best_lines[li, 2])
                    if A * orig_pt[0] + B * orig_pt[1] + C == 0:
                        orig_pt_inc += 1

                best_delta = 0
                best_new_pt = None

                for new_pt in candidate_points_list:
                    if new_pt == orig_pt:
                        continue
                    # Check duplicate
                    if any(tuple(int(x) for x in best_points[j]) == new_pt for j in range(54) if j != pi):
                        continue

                    # Count incidences for new point
                    new_pt_inc = 0
                    for li in range(27):
                        A, B, C = int(best_lines[li, 0]), int(best_lines[li, 1]), int(best_lines[li, 2])
                        if A * new_pt[0] + B * new_pt[1] + C == 0:
                            new_pt_inc += 1

                    delta = new_pt_inc - orig_pt_inc
                    if delta > best_delta:
                        best_delta = delta
                        best_new_pt = new_pt

                if best_new_pt is not None and best_delta > 0:
                    new_points = best_points.copy()
                    new_points[pi] = np.array(best_new_pt, dtype=np.int64)
                    best_score += best_delta
                    best_points = new_points
                    improvement_found = True

    # Try 2-swap moves to escape local optima
    if time.time() < deadline - 60:
        for _ in range(200):  # limited 2-swap attempts
            if time.time() > deadline - 60:
                break

            # Try swapping two points
            pi1 = int(np.random.randint(54))
            pi2 = int(np.random.randint(53))
            if pi2 >= pi1:
                pi2 += 1

            new_pt1 = (int(np.random.choice(candidate_xs)), int(np.random.choice(candidate_ys)))
            new_pt2 = (int(np.random.choice(candidate_xs)), int(np.random.choice(candidate_ys)))

            # Check validity
            other_pts = set()
            for j in range(54):
                if j != pi1 and j != pi2:
                    other_pts.add(tuple(int(x) for x in best_points[j]))

            if new_pt1 in other_pts or new_pt2 in other_pts or new_pt1 == new_pt2:
                continue

            # Compute delta
            delta = 0
            for li in range(27):
                A, B, C = int(best_lines[li, 0]), int(best_lines[li, 1]), int(best_lines[li, 2])
                old1 = A * int(best_points[pi1, 0]) + B * int(best_points[pi1, 1]) + C
                old2 = A * int(best_points[pi2, 0]) + B * int(best_points[pi2, 1]) + C
                new1 = A * new_pt1[0] + B * new_pt1[1] + C
                new2 = A * new_pt2[0] + B * new_pt2[1] + C
                delta += (1 if new1 == 0 else 0) + (1 if new2 == 0 else 0)
                delta -= (1 if old1 == 0 else 0) + (1 if old2 == 0 else 0)

            if delta > 0:
                new_points = best_points.copy()
                new_points[pi1] = np.array(new_pt1, dtype=np.int64)
                new_points[pi2] = np.array(new_pt2, dtype=np.int64)
                best_score += delta
                best_points = new_points

            # Try swapping two lines
            li1 = int(np.random.randint(27))
            li2 = int(np.random.randint(26))
            if li2 >= li1:
                li2 += 1

            new_line1 = candidate_lines_list[int(np.random.randint(len(candidate_lines_list)))]
            new_line2 = candidate_lines_list[int(np.random.randint(len(candidate_lines_list)))]

            # Check duplicates
            other_lines = set()
            for j in range(27):
                if j != li1 and j != li2:
                    other_lines.add(normalize_line(*tuple(int(x) for x in best_lines[j])))

            if new_line1 in other_lines or new_line2 in other_lines or new_line1 == new_line2:
                continue

            # Compute delta
            delta = 0
            for pi in range(54):
                A1, B1, C1 = int(best_lines[li1, 0]), int(best_lines[li1, 1]), int(best_lines[li1, 2])
                A2, B2, C2 = int(best_lines[li2, 0]), int(best_lines[li2, 1]), int(best_lines[li2, 2])
                old1 = A1 * int(best_points[pi, 0]) + B1 * int(best_points[pi, 1]) + C1
                old2 = A2 * int(best_points[pi, 0]) + B2 * int(best_points[pi, 1]) + C2
                new1 = new_line1[0] * int(best_points[pi, 0]) + new_line1[1] * int(best_points[pi, 1]) + new_line1[2]
                new2 = new_line2[0] * int(best_points[pi, 0]) + new_line2[1] * int(best_points[pi, 1]) + new_line2[2]
                delta += (1 if new1 == 0 else 0) + (1 if new2 == 0 else 0)
                delta -= (1 if old1 == 0 else 0) + (1 if old2 == 0 else 0)

            if delta > 0:
                new_lines = best_lines.copy()
                new_lines[li1] = np.array(new_line1, dtype=np.int64)
                new_lines[li2] = np.array(new_line2, dtype=np.int64)
                best_score += delta
                best_lines = new_lines

    # Final verification
    assert best_points.shape == (54, 2)
    assert best_lines.shape == (27, 3)
    assert not has_duplicate_points(best_points)
    assert not has_duplicate_lines(best_lines)

    return best_points, best_lines


# EVOLVE-BLOCK-END

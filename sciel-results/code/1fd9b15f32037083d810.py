import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    deadline = 570.0  # safety margin before 630s kill

    p = 7
    k = 3  # alpha^2 = 3

    # Field operations
    def field_add(x, y):
        return ((x[0]+y[0])%p, (x[1]+y[1])%p)

    def field_mul(x, y):
        return ((x[0]*y[0] + k*x[1]*y[1])%p, (x[0]*y[1]+x[1]*y[0])%p)

    def field_neg(x):
        return ((-x[0])%p, (-x[1])%p)

    # Enumerate all field elements
    field_elements = [(a,b) for a in range(p) for b in range(p)]

    # All points in F_49^2
    all_points = [((xa,xb),(ya,yb)) for xa in range(p) for xb in range(p)
                  for ya in range(p) for yb in range(p)]
    point_to_idx = {pt: i for i, pt in enumerate(all_points)}
    n_points = 2401

    # Build all lines
    lines = []  # list of lists of point indices

    # Vertical lines: direction (0,1)
    for xa in range(p):
        for xb in range(p):
            line_pts = []
            for ya in range(p):
                for yb in range(p):
                    pt = ((xa,xb),(ya,yb))
                    line_pts.append(point_to_idx[pt])
            lines.append(line_pts)

    # Slanted lines: direction (1,m), y = m*x + c
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            for ca in range(p):
                for cb in range(p):
                    c = (ca, cb)
                    line_pts = []
                    for xa in range(p):
                        for xb in range(p):
                            x = (xa, xb)
                            mx = field_mul(m, x)
                            y = field_add(mx, c)
                            pt = (x, y)
                            line_pts.append(point_to_idx[pt])
                    lines.append(line_pts)

    n_lines = len(lines)  # 2450

    # For each point, which lines pass through it
    point_lines = [[] for _ in range(n_points)]
    for li, line in enumerate(lines):
        for pt_idx in line:
            point_lines[pt_idx].append(li)

    # Hermitian unital: 2*xa + ya^2 - 3*yb^2 = 0 mod 7
    def in_unital(pt_idx):
        xa, xb = all_points[pt_idx][0]
        ya, yb = all_points[pt_idx][1]
        return (2*xa + ya*ya - k*yb*yb) % p == 0

    # Squares mod 7
    squares_mod7 = {0, 1, 2, 4}

    # Initial S: unital points with ya NOT in squares
    # This is the complement of the blocking set
    S = set()
    for pt_idx in range(n_points):
        if in_unital(pt_idx):
            ya, yb = all_points[pt_idx][1]
            if ya not in squares_mod7:
                S.add(pt_idx)

    # Verify Nikodym property for N = F_49^2 \ S
    def verify_nikodym(S_set):
        """Check that every point has a line where all other points are in N = complement of S."""
        S_frozen = frozenset(S_set)
        for pt_idx in range(n_points):
            found = False
            for li in point_lines[pt_idx]:
                line = lines[li]
                valid = True
                for other in line:
                    if other != pt_idx and other in S_frozen:
                        valid = False
                        break
                if valid:
                    found = True
                    break
            if not found:
                return False
        return True

    assert verify_nikodym(S), "Initial S should be valid"

    # Now greedily grow S (shrink N) by adding points that don't break the property
    # A point p can be added to S iff for every line L through p,
    # there is still some other line L' through p (L' != L) such that
    # all points on L' except p are NOT in S (i.e., are in N).
    # Equivalently: after adding p to S, p must still have at least one
    # "clean" line (a line through p where all other points are in N).

    # Track for each point how many "clean" lines it has
    # A line L is clean for point x (x on L) if no other point on L is in S
    clean_lines = np.zeros(n_points, dtype=np.int32)
    for x in range(n_points):
        for li in point_lines[x]:
            line = lines[li]
            is_clean = True
            for other in line:
                if other != x and other in S:
                    is_clean = False
                    break
            if is_clean:
                clean_lines[x] += 1

    assert np.all(clean_lines >= 1), "Initial S invalid"

    # Greedy: try to add points to S
    # A point p can be added iff clean_lines[p] >= 2 (it will lose one clean line
    # for each line L through p that becomes non-clean, but we need at least 1 remaining)
    # Actually, adding p to S makes every line through p non-clean (since p is on them).
    # So p loses ALL its clean lines. But wait - that's not right either.
    # Let me reconsider: a line L through p is clean for p iff no OTHER point on L is in S.
    # After adding p to S, L is still clean for p if no other point on L is in S.
    # But L is no longer clean for OTHER points on L (since p is now in S).
    # So: adding p to S does NOT change clean_lines[p] (p's clean lines are determined
    # by other points on those lines, not by p itself).
    # But it DOES decrease clean_lines[other] for other points on lines through p.

    # So p can be added iff:
    # 1. clean_lines[p] >= 1 (p still has a clean line after addition - but this doesn't change)
    #    Actually, clean_lines[p] doesn't change, so we need clean_lines[p] >= 1 which is already true.
    # 2. For every other point q on a line through p: if that line was clean for q,
    #    q loses it. So q's clean_lines decreases. We need all q to still have >= 1.

    def can_add_to_S(p, S_set, clean_lines_arr):
        """Check if point p can be added to S without making any point have 0 clean lines."""
        for li in point_lines[p]:
            line = lines[li]
            # Check if this line is currently clean for any other point
            for other in line:
                if other != p:
                    # Is this line clean for 'other'?
                    # The line is clean for other iff no point on line (except other) is in S
                    # p is not yet in S, so check if any other point on line (except other) is in S
                    # But we're about to add p, so after adding, the line has p in S.
                    # So the line will NOT be clean for 'other' after we add p.
                    # But was it clean for 'other' before? If yes, other loses it.
                    # We need to check if other has other clean lines.
                    # Simpler: just check if clean_lines_arr[other] <= 1 for any other
                    # that would lose this line.
                    # A line is clean for 'other' iff no other point on line (except other) is in S.
                    # Before adding p: check if any point on line (except other, except p) is in S.
                    # If not, the line is clean for other, and other will lose it.
                    is_clean_for_other = True
                    for q in line:
                        if q != other and q != p and q in S_set:
                            is_clean_for_other = False
                            break
                    if is_clean_for_other and clean_lines_arr[other] <= 1:
                        return False
        return True

    # Greedy growth of S
    candidates = [pt for pt in range(n_points) if pt not in S]

    # Multiple passes
    max_passes = 10
    for pass_num in range(max_passes):
        if time.time() - start_time > deadline - 60:
            break

        added_any = False
        random.shuffle(candidates)

        for p in candidates:
            if time.time() - start_time > deadline - 60:
                break
            if p in S:
                continue
            if can_add_to_S(p, S, clean_lines):
                # Add p to S
                S.add(p)
                added_any = True
                # Update clean_lines for affected points
                for li in point_lines[p]:
                    line = lines[li]
                    for other in line:
                        if other != p:
                            # Check if this line was clean for other before
                            # (no point on line except other was in S before adding p)
                            was_clean = True
                            for q in line:
                                if q != other and q != p and q in S:
                                    was_clean = False
                                    break
                            if was_clean:
                                clean_lines[other] -= 1

        if not added_any:
            break

    # Final verification
    if not verify_nikodym(S):
        # Roll back - this shouldn't happen if can_add_to_S is correct
        # But just in case, rebuild from scratch
        S = set()
        for pt_idx in range(n_points):
            if in_unital(pt_idx):
                ya, yb = all_points[pt_idx][1]
                if ya not in squares_mod7:
                    S.add(pt_idx)

    # N = complement of S
    N_points = []
    for i, pt in enumerate(all_points):
        if i not in S:
            xa, xb = pt[0]
            ya, yb = pt[1]
            N_points.append([[xa, xb], [ya, yb]])

    return np.array(N_points, dtype=np.int64)

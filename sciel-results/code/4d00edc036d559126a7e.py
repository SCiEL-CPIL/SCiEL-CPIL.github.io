# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_{49}^2 using greedy incremental line-count tracking.
    """
    p = 7
    q = p * p  # 49
    start_time = time.time()
    deadline = start_time + 580.0  # safety margin

    # Field F_49 = F_7[alpha], alpha^2 = 3
    # Element represented as (a, b) meaning a + b*alpha
    # Precompute field multiplication table: mult[(a1,b1)][(a2,b2)] -> (a,b)
    def field_mul(e1, e2):
        a1, b1 = e1
        a2, b2 = e2
        # (a1 + b1*alpha)(a2 + b2*alpha) = a1*a2 + (a1*b2 + a2*b1)*alpha + b1*b2*alpha^2
        # alpha^2 = 3
        a = (a1 * a2 + 3 * b1 * b2) % p
        b = (a1 * b2 + a2 * b1) % p
        return (a, b)

    def field_add(e1, e2):
        return ((e1[0] + e2[0]) % p, (e1[1] + e2[1]) % p)

    def field_neg(e):
        return ((-e[0]) % p, (-e[1]) % p)

    def field_sub(e1, e2):
        return field_add(e1, field_neg(e2))

    # Precompute all field elements as tuples
    field_elems = [(a, b) for a in range(p) for b in range(p)]
    elem_to_idx = {e: i for i, e in enumerate(field_elems)}

    # Precompute multiplication table
    mult_table = np.zeros((q, q, 2), dtype=np.int64)
    for i, e1 in enumerate(field_elems):
        for j, e2 in enumerate(field_elems):
            r = field_mul(e1, e2)
            mult_table[i, j] = r

    # Precompute addition table
    add_table = np.zeros((q, q, 2), dtype=np.int64)
    for i, e1 in enumerate(field_elems):
        for j, e2 in enumerate(field_elems):
            r = field_add(e1, e2)
            add_table[i, j] = r

    # Points in F_49^2: each point is ((xa, xb), (ya, yb))
    # Index: xa*p + xb for x, ya*p + yb for y
    # Total points: q^2 = 2401
    point_count = q * q

    # Lines: for each point x and each direction v, the line is {x + t*v : t in F_49}
    # Directions: v = (1, m) for m in F_49, or v = (0, 1)
    # Total directions: q + 1 = 50

    # For each direction d (0..49), and each point, precompute the line (set of points)
    # Direction d: if d < q, v = (1, field_elems[d]); else v = (0, 1)
    # Line through point idx with direction d: {point + t*v : t in F_49}

    # Precompute line membership: for each direction d, for each line id, the set of point indices
    # A line is determined by (direction, intercept). For direction (1, m), intercept is y - m*x
    # For direction (0, 1), intercept is x

    # Let's think differently. For each direction d, there are q parallel lines partitioning F_49^2.
    # For direction d < q: v = (1, m) where m = field_elems[d]
    #   Line: points (x, y) such that y - m*x = c for some c in F_49
    #   There are q such lines, each with q points.
    # For direction d = q: v = (0, 1)
    #   Line: points (x, y) such that x = c for some c in F_49
    #   There are q such lines, each with q points.

    # Precompute for each direction d and each line index li, the list of point indices on that line
    # point index: (x_idx * q + y_idx) where x_idx = xa*p+xb, y_idx = ya*p+yb

    # Actually, let me use a different indexing.
    # Point (x, y) where x, y in F_49. Index = x_idx * q + y_idx.
    # x_idx = xa * p + xb, y_idx = ya * p + yb

    # For direction d < q: m = field_elems[d], line c: y - m*x = c
    # For direction d = q: line c: x = c

    # Precompute line_points[d][c] = list of point indices
    line_points = []
    for d in range(q + 1):
        lines_for_d = []
        if d < q:
            m = field_elems[d]
            m_idx = d
            for c_idx in range(q):
                c = field_elems[c_idx]
                pts = []
                for x_idx in range(q):
                    # y = m * x + c
                    mx = field_mul(field_elems[x_idx], m)
                    y = field_add(mx, c)
                    y_idx = elem_to_idx[y]
                    pts.append(x_idx * q + y_idx)
                lines_for_d.append(pts)
        else:
            # v = (0, 1), line: x = c
            for c_idx in range(q):
                c = field_elems[c_idx]
                pts = []
                for y_idx in range(q):
                    pts.append(c_idx * q + y_idx)
                lines_for_d.append(pts)
        line_points.append(lines_for_d)

    # For each point, precompute which (direction, line) pairs it belongs to
    # point_lines[point_idx] = list of (d, c_idx)
    point_lines = [[] for _ in range(point_count)]
    for d in range(q + 1):
        for c_idx in range(q):
            for pt in line_points[d][c_idx]:
                point_lines[pt].append((d, c_idx))

    # line_count[d][c_idx] = number of points in N on that line
    line_count = np.zeros((q + 1, q), dtype=np.int32)
    # free_count[d][c_idx] = number of points NOT in N on that line = q - line_count
    # A point x is "covered" by direction d, line c if all other points on that line are in N
    # i.e., line_count[d][c] >= q - 1 (since x itself may or may not be in N)
    # Actually: for point x on line (d, c), the condition is that all points on the line EXCEPT x are in N.
    # If x is in N: need line_count[d][c] >= q (all q points in N, including x)
    #   Wait no: "points other than x all belong to N" means q-1 other points must be in N.
    #   If x is in N: line_count[d][c] includes x, so we need line_count[d][c] - 1 >= q-1, i.e., line_count[d][c] >= q
    #   If x is NOT in N: line_count[d][c] does not include x, so we need line_count[d][c] >= q-1

    # For a point x, it's valid if there EXISTS a direction d and line c containing x such that:
    #   if x in N: line_count[d][c] >= q
    #   if x not in N: line_count[d][c] >= q - 1

    # Greedy: start with empty N, add points one by one.
    # When we add a point x to N, for each (d, c) in point_lines[x], line_count[d][c] increases by 1.
    # This may make some other points y (on the same lines) become "coverable".

    # A point y is coverable if there exists (d, c) in point_lines[y] such that:
    #   if y in N: line_count[d][c] >= q
    #   if y not in N: line_count[d][c] >= q - 1

    # We want to maximize |N| (complement is minimized, but score is complement size... wait)
    # Score = 2401 - |N|, higher is better. So we want |N| as SMALL as possible!
    # Wait, re-reading: "Metric (higher is better): the complement size 2401 - |N|"
    # So we want a SMALL Nikodym set!

    # A Nikodym set N: every point x in F_49^2 (in N or not) lies on a line where all other points are in N.
    # We want to minimize |N|.

    # Strategy: start with all points in N (|N| = 2401), then try to remove points.
    # A point x can be removed from N if after removal, x is still "covered":
    #   x is not in N, so we need some (d, c) containing x with line_count[d][c] >= q - 1
    #   After removing x, line_count[d][c] decreases by 1 for all (d, c) in point_lines[x].
    #   So we need: for some (d, c) in point_lines[x], (line_count[d][c] - 1) >= q - 1, i.e., line_count[d][c] >= q

    # Also, removing x must not break coverage of other points.
    # Other points y that were covered ONLY by lines (d, c) that contain x might lose coverage.

    # This is complex. Let me try a different approach.

    # Actually, let me re-read the problem more carefully.
    # "every point x of F_49^2 (in N or not) must lie on a line {x + t*v : t in F_49}, with v = (1, m) for some m in F_49 or v = (0, 1), whose points other than x all belong to N"

    # So for EVERY point x (whether in N or not), there must exist a direction such that all OTHER points on that line through x are in N.

    # If x is in N: the line has q points total, x is one of them, so q-1 others must be in N.
    #   This means line_count[d][c] >= q (all points on the line are in N, since x is in N and q-1 others are in N).
    #   Wait: line_count counts points in N on the line. If x is in N and q-1 others are in N, then line_count = q.
    #   So: line_count[d][c] == q.

    # If x is NOT in N: the line has q points, x is not in N, q-1 others must be in N.
    #   line_count[d][c] = q - 1 (the q-1 points in N, x is not counted).

    # So the condition is:
    # For each point x:
    #   If x in N: exists (d,c) in point_lines[x] with line_count[d][c] == q
    #   If x not in N: exists (d,c) in point_lines[x] with line_count[d][c] == q - 1

    # We want to minimize |N|.

    # Start with N = all points. line_count[d][c] = q for all (d,c).
    # Every point x in N has line_count == q for all its lines. Covered.
    # Try to remove points.

    # When we remove point x from N:
    #   For each (d, c) in point_lines[x]: line_count[d][c] decreases by 1.
    #   x is now not in N. x needs some (d, c) in point_lines[x] with line_count[d][c] == q - 1.
    #   After removal, for (d, c) in point_lines[x], line_count[d][c] = q - 1. So x is covered. Good.
    #   But other points y on those lines: if y was in N and was covered ONLY by lines that now have line_count < q, y loses coverage.
    #   If y was not in N and was covered ONLY by lines that now have line_count < q - 1, y loses coverage.

    # This is getting complex. Let me try a simpler heuristic:
    # Start with N = all points.
    # Repeatedly try to remove a point x such that:
    #   1. x is in N
    #   2. After removing x, all points are still covered.

    # To check condition 2 efficiently, we can track for each point y, the number of "witness" lines
    # that cover it. A witness line for y is a (d, c) in point_lines[y] such that:
    #   if y in N: line_count[d][c] == q
    #   if y not in N: line_count[d][c] == q - 1

    # If we remove x, only points y on the same lines as x are affected.
    # For each (d, c) in point_lines[x], line_count[d][c] decreases by 1.
    # This affects the witness status of points on line (d, c).

    # Let me implement this with careful bookkeeping.

    # Initialize: N = all points
    in_N = np.ones(point_count, dtype=bool)
    line_count[:] = q

    # witness_count[y] = number of witness lines for point y
    witness_count = np.zeros(point_count, dtype=np.int32)
    for y in range(point_count):
        for (d, c) in point_lines[y]:
            # y is in N, need line_count[d][c] == q
            if line_count[d, c] == q:
                witness_count[y] += 1

    # Initially, every point has q+1 witness lines (all directions).

    # Greedy removal:
    # Find a point x in N such that removing it keeps all points covered.
    # x can be removed if:
    #   - For x itself: after removal, x not in N, needs line_count[d][c] == q-1 for some (d,c) in point_lines[x]
    #     After removal, line_count[d][c] = q-1 for all (d,c) in point_lines[x]. So x is covered.
    #   - For other points y on lines through x: they must still have at least one witness line.

    # When we remove x:
    #   For each (d, c) in point_lines[x]:
    #     line_count[d, c] -= 1
    #     For each point y on line (d, c):
    #       if y == x: skip (handled separately)
    #       if y in N:
    #         old_witness = (line_count[d, c] + 1 == q)  # before removal
    #         new_witness = (line_count[d, c] == q)  # after removal, but line_count already decremented
    #         Actually, let me be more careful.

    # Let me restructure. Before removal, line_count[d, c] = L.
    # After removal, line_count[d, c] = L - 1.
    # For point y on line (d, c), y != x:
    #   if y in N:
    #     was_witness = (L == q)
    #     is_witness = (L - 1 == q) = (L == q + 1) = False (since L <= q)
    #     So if L == q, y loses this witness. If L < q, y didn't have this witness anyway.
    #   if y not in N:
    #     was_witness = (L == q - 1)
    #     is_witness = (L - 1 == q - 1) = (L == q)
    #     So if L == q - 1, y loses this witness. If L == q, y gains this witness (but y not in N, and L was q, meaning all points including... wait)

    # Hmm, this is getting tangled. Let me just do a direct check for candidate removals.

    # Simpler approach: try removing each point in N, check validity, if valid, remove it.
    # To speed up, only check points that are "removable" based on some heuristic.

    # Actually, with 2401 points and 50 lines per point, and checking all affected points...
    # When removing x, affected points are those on the 50 lines through x, minus x.
    # Each line has 49 points, so up to 50 * 48 = 2400 other points.
    # For each affected point, check if it still has a witness.
    # This is O(2400) per removal attempt, and we try up to 2401 removals.
    # Total: ~2401 * 2400 = 5.7M operations. Should be fine in Python with numpy? Maybe not.

    # Let me use the witness_count approach for efficiency.

    # Redo with witness_count tracking:

    in_N = np.ones(point_count, dtype=bool)
    line_count = np.full((q + 1, q), q, dtype=np.int32)

    # witness_count[y] = number of (d, c) in point_lines[y] that are witnesses for y
    # Witness for y: if y in N, line_count[d][c] == q; if y not in N, line_count[d][c] == q-1
    witness_count = np.zeros(point_count, dtype=np.int32)
    for y in range(point_count):
        wc = 0
        for (d, c) in point_lines[y]:
            if in_N[y]:
                if line_count[d, c] == q:
                    wc += 1
            else:
                if line_count[d, c] == q - 1:
                    wc += 1
        witness_count[y] = wc

    # Initially all in_N, all line_count = q, so witness_count = q+1 = 50 for all.

    removed = 0
    while True:
        if time.time() > deadline:
            break

        # Find a removable point
        found = False
        for x in range(point_count):
            if not in_N[x]:
                continue

            # Check if x can be removed
            # After removal:
            #   x not in N, needs some (d,c) in point_lines[x] with line_count[d][c] == q-1
            #   After removal, for (d,c) in point_lines[x], line_count[d][c] = q-1. So x is covered.
            #   Other points: need to check witness_count remains > 0.

            # Compute the effect of removing x
            # For each (d, c) in point_lines[x]:
            #   line_count[d, c] goes from q to q-1
            #   For points y on line (d, c), y != x:
            #     if y in N: was witness if line_count == q (yes, it was q), now line_count == q-1, not a witness.
            #       So y loses one witness.
            #     if y not in N: was witness if line_count == q-1 (no, it was q), now line_count == q-1, is a witness.
            #       So y gains one witness.

            # Since all points are currently in N (initially), and we only remove points:
            # When we remove x, for each (d, c) in point_lines[x]:
            #   line_count[d, c] = q -> q-1
            #   For y on line (d, c), y != x:
            #     if y in N: loses witness (line_count was q, now q-1)
            #     if y not in N: gains witness (line_count was q, now q-1... wait)

            # Hmm, but line_count might not be q for all lines anymore after some removals.
            # Let me handle the general case.

            # For each (d, c) in point_lines[x], let L = line_count[d, c] (current, before removal)
            # After removal, line_count[d, c] = L - 1
            # For y on line (d, c), y != x:
            #   if y in N:
            #     was_witness = (L == q)
            #     is_witness = (L - 1 == q) = False
            #     delta_witness = -1 if was_witness else 0
            #   if y not in N:
            #     was_witness = (L == q - 1)
            #     is_witness = (L - 1 == q - 1) = (L == q)
            #     delta_witness = +1 if is_witness and not was_witness else (-1 if was_witness and not is_witness else 0)
            #     = +1 if L == q else (-1 if L == q-1 else 0)

            # We need to check that for all affected y, witness_count[y] + delta_witness > 0

            # Let's collect affected points and their deltas
            affected = {}  # y -> total delta_witness
            for (d, c) in point_lines[x]:
                L = line_count[d, c]
                for y in line_points[d][c]:
                    if y == x:
                        continue
                    if y in affected:
                        pass
                    if in_N[y]:
                        delta = -1 if L == q else 0
                    else:
                        if L == q:
                            delta = 1  # was not witness (L==q, need q-1), now is witness (L-1==q-1)
                        elif L == q - 1:
                            delta = -1  # was witness (L==q-1), now not witness (L-1==q-2)
                        else:
                            delta = 0
                    affected[y] = affected.get(y, 0) + delta

            # Check if all affected y still have witness_count > 0
            valid = True
            for y, delta in affected.items():
                if witness_count[y] + delta <= 0:
                    valid = False
                    break

            if not valid:
                continue

            # Also check x itself: after removal, x not in N, needs line_count[d][c] == q-1 for some (d,c)
            # After removal, for (d,c) in point_lines[x], line_count[d][c] = L-1 where L was q (if x was the only... no)
            # Actually, x is in N, so for (d,c) in point_lines[x], the line had x in it.
            # line_count[d][c] includes x. After removal, line_count[d][c] = L - 1.
            # x needs some (d,c) with line_count[d][c] == q-1 after removal.
            # So we need some (d,c) in point_lines[x] with L - 1 == q-1, i.e., L == q.
            # Since x is in N and was covered, there exists (d,c) with L == q. So this is satisfied.

            # Perform removal
            in_N[x] = False
            for (d, c) in point_lines[x]:
                line_count[d, c] -= 1
                for y in line_points[d][c]:
                    if y == x:
                        continue
                    if in_N[y]:
                        if line_count[d, c] + 1 == q:  # was q before decrement
                            witness_count[y] -= 1
                    else:
                        if line_count[d, c] + 1 == q:  # was q, now q-1, gains witness
                            witness_count[y] += 1
                        elif line_count[d, c] + 1 == q - 1:  # was q-1, now q-2, loses witness
                            witness_count[y] -= 1
            removed += 1
            found = True
            break

        if not found:
            break

    # Build result
    result = []
    for idx in range(point_count):
        if in_N[idx]:
            x_idx = idx // q
            y_idx = idx % q
            xa = x_idx // p
            xb = x_idx % p
            ya = y_idx // p
            yb = y_idx % p
            result.append([[xa, xb], [ya, yb]])

    return np.array(result, dtype=np.int64)

# EVOLVE-BLOCK-END

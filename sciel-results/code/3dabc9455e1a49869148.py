# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 via a line-based greedy construction.

    Structural idea (from the inspiration): rather than a fixed algebraic curve,
    treat the Nikodym condition as a line-based constraint. For each point x and
    direction v, a line through x is "clean" if every other point on it is in N
    (i.e. NOT in the excluded set S). x is covered iff it has >=1 clean line.
    We greedily grow S (the excluded set) to maximize |S| while keeping every
    point covered. The metric is the complement size 2401 - |S|, so a larger S
    is better.

    Refinement machinery (from the current program): fast vectorized field
    arithmetic, a precomputed line table, and a deadline guard so the search
    always returns the best valid result found within the wall-clock budget.
    """
    start_time = time.time()
    deadline = start_time + 595  # 5s safety margin before the 600s kill

    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3 (a quadratic non-residue mod 7)

    # ---- Field F_49 = F_7[alpha], alpha^2 = k, element = a + b*alpha ----
    # index = a*p + b
    A = np.array([i // p for i in range(q)], dtype=np.int64)
    B = np.array([i % p for i in range(q)], dtype=np.int64)

    # Addition table: add[i, j] = index of (i + j)
    add = np.zeros((q, q), dtype=np.int64)
    for i in range(q):
        ai, bi = A[i], B[i]
        add[i, :] = ((ai + A) % p) * p + ((bi + B) % p)

    # Multiplication table: mul[i, j] = index of (i * j)
    mul = np.zeros((q, q), dtype=np.int64)
    for i in range(q):
        ai, bi = A[i], B[i]
        pa = (ai * A + bi * B * k) % p
        pb = (ai * B + bi * A) % p
        mul[i, :] = pa * p + pb

    # Negation: neg[i] = index of (-i)
    neg = (-(A * p + B)) % (p * p)  # = (p - a) % p * p + (p - b) % p
    neg = np.array([((p - A[i]) % p) * p + ((p - B[i]) % p) for i in range(q)], dtype=np.int64)

    # ---- Directions: v = (1, m) for m in F_49 (49) and v = (0, 1) (1) ----
    # For direction (1, m): point at offset t on line through base b is
    #   (x_b + t, y_b + m*t)
    # For direction (0, 1): point at offset t is (x_b, y_b + t)
    #
    # We build a canonical line table: for each direction, partition the 2401
    # points into 49 lines of 49 points. A line through base point b with
    # direction d contains points { b + t*v_d : t in F_49 }.

    # Total points: 2401. Point index = x_idx * q + y_idx where x_idx, y_idx in 0..48.
    total_pts = q * q  # 2401

    # For each direction, we need to assign each point to a line and know the
    # 49 points on each line. We represent lines by (direction, line_id).
    #
    # For direction (1, m): the line through (x, y) is uniquely determined by
    #   the "invariant" c = y - m*x (in F_49). Points on the same line share c.
    #   There are 49 distinct c values -> 49 lines.
    # For direction (0, 1): invariant is x. 49 lines.
    #
    # Build: for each direction d, a 2401-element array line_id_of_point[d, pt]
    # and for each line, the list of 49 point indices.

    n_dirs = q + 1  # 50
    # line_id_of_point: shape (n_dirs, total_pts)
    line_id_of_point = np.zeros((n_dirs, total_pts), dtype=np.int64)
    # For each direction and line_id, store the 49 points.
    # lines[d, line_id, t] = point index at offset t
    lines = np.zeros((n_dirs, q, q), dtype=np.int64)

    for d in range(n_dirs):
        if d < q:
            # direction (1, m) where m = d
            m = d
            # For each base point (x_idx, y_idx), the line invariant is y - m*x
            # We iterate over all 2401 points, compute invariant, group by invariant.
            # invariant = y_idx - m * x_idx (in F_49, using field ops)
            # x_idx and y_idx are field element indices.
            # x = x_idx, y = y_idx. m*x = mul[m, x_idx]. y - m*x = add[y_idx, neg[mul[m, x_idx]]]
            # But we need this for each point. Let's vectorize over all points.
            # Point index pt = x_idx * q + y_idx
            x_idxs = (np.arange(total_pts) // q)  # shape (2401,)
            y_idxs = (np.arange(total_pts) % q)   # shape (2401,)
            # m * x for each point: mul[m, x_idxs]
            mx = mul[m, x_idxs]  # shape (2401,)
            # y - m*x: add[y_idxs, neg[mx]]
            inv = add[y_idxs, neg[mx]]  # shape (2401,), values in 0..48
            line_id_of_point[d, :] = inv
            # Now for each line_id (0..48), gather the 49 points.
            # For a given invariant c, the points are those with inv == c.
            # For each such point, we also need to order them by t (offset).
            # The point at offset t on the line through base b (where b has invariant c)
            # is b + t*v. But we can just collect all points with inv == c.
            for c in range(q):
                pts_on_line = np.where(inv == c)[0]  # 49 points
                # Order them by their x-coordinate (since direction is (1, m),
                # x determines t uniquely: x = x_b + t)
                # x_idx of each point:
                x_coords = pts_on_line // q
                order = np.argsort(x_coords)
                lines[d, c, :] = pts_on_line[order]
        else:
            # direction (0, 1): invariant is x_idx
            x_idxs = (np.arange(total_pts) // q)
            line_id_of_point[d, :] = x_idxs
            for c in range(q):
                pts_on_line = np.where(x_idxs == c)[0]
                # Order by y-coordinate
                y_coords = pts_on_line % q
                order = np.argsort(y_coords)
                lines[d, c, :] = pts_on_line[order]

    # ---- Greedy construction of S (excluded set) ----
    # State: S is a boolean array of shape (total_pts,)
    # For each point, clean_count[pt] = number of directions where the line
    # through pt in that direction has all other 48 points NOT in S.
    #
    # Initially S is empty, so every line is clean. clean_count = 50 for all.
    #
    # When we add a point p to S:
    #   For each direction d, p lies on line (d, line_id_of_point[d, p]).
    #   For each other point x on that line, the line is no longer clean for x
    #   (since p is now in S). So clean_count[x] decreases by 1.
    #   If any x has clean_count[x] drop to 0, adding p would make x uncovered -> invalid.
    #
    # Greedy: repeatedly pick the point p (not in S) whose addition causes the
    # largest total decrease in clean_count (i.e., p lies on the most lines that
    # are currently the sole clean line for some other point), subject to not
    # making any point uncovered.
    #
    # For efficiency, we track for each point x, how many clean lines it has.
    # Adding p to S "kills" the line (d, l) for all x on that line. A point x
    # becomes uncovered if it had exactly 1 clean line and that line is killed.
    #
    # Let's define: for each line (d, l), let on_line[d, l] be the set of points.
    # When we add p, we kill 50 lines (one per direction through p).
    # For each killed line, all points on it lose one clean line.
    #
    # We maintain:
    #   in_S: bool array (total_pts,)
    #   clean_count: int array (total_pts,)
    #   For each point x, the set of clean lines through x (as (d, l) pairs).
    #
    # To check feasibility of adding p:
    #   For each direction d, line l = line_id_of_point[d, p].
    #   For each x on line (d, l), x != p:
    #       if clean_count[x] == 1 and the only clean line for x is (d, l):
    #           -> adding p would uncover x. Infeasible.
    #
    # To make this fast, we maintain for each point x, the list of clean lines.
    # But that's 50 * 2401 = 120050 entries, manageable.
    #
    # Actually, let's use a simpler scoring: the "impact" of adding p is the
    # number of points x (x != p) for which p's addition reduces clean_count[x].
    # We want to maximize this impact while keeping all clean_count[x] >= 1.
    #
    # Let's just do a greedy: try to add points in an order that maximizes
    # impact, checking feasibility each time.

    in_S = np.zeros(total_pts, dtype=bool)
    clean_count = np.full(total_pts, n_dirs, dtype=np.int64)

    # For each point x, track which lines through x are clean.
    # clean_lines[x] = set of (d, l) where line (d, l) contains x and all other
    # points on it are not in S.
    # Initially all 50 lines through each point are clean.
    # Represent as a 2D boolean array: clean_line_flag[x, d] = True if the line
    # through x in direction d is clean.
    clean_line_flag = np.ones((total_pts, n_dirs), dtype=bool)

    # For each line (d, l), track how many points on it are in S.
    # line_s_count[d, l] = number of points in S on line (d, l).
    # A line is clean for point x on it iff line_s_count[d, l] == (1 if x in S else 0)
    #   i.e., the only point in S on the line is x itself (or none, if x not in S).
    # Wait: the Nikodym condition requires that for point x, the line through x
    # in direction d has all points OTHER than x in N (not in S). So the line
    # is clean for x iff no point on the line other than x is in S.
    # This means: if x is in S, the line is clean for x iff line_s_count[d, l] == 1 (only x).
    # If x is not in S, the line is clean for x iff line_s_count[d, l] == 0.
    #
    # Let's track line_s_count[d, l] = number of points in S on line (d, l).
    line_s_count = np.zeros((n_dirs, q), dtype=np.int64)

    # When we add point p to S:
    #   For each direction d:
    #       l = line_id_of_point[d, p]
    #       line_s_count[d, l] += 1
    #       # Now for each other point x on line (d, l):
    #       #   The line (d, l) is no longer clean for x (since p is now in S and x != p).
    #       #   So clean_line_flag[x, d] = False, clean_count[x] -= 1
    #   Also for p itself:
    #       For each direction d, the line through p in direction d:
    #       #   Before adding p, the line had some s_count. After, s_count += 1.
    #       #   The line is clean for p iff s_count_after == 1 (only p in S on the line).
    #       #   So if s_count_after == 1, clean_line_flag[p, d] stays True (or becomes True).
    #       #   If s_count_after > 1, clean_line_flag[p, d] = False.
    #
    # Let's implement this step by step.

    # Precompute for each point p, the 50 lines through it: (d, l) for d in 0..49.
    # lines_through[p, d] = l = line_id_of_point[d, p]
    lines_through = line_id_of_point.T  # shape (total_pts, n_dirs)

    # Precompute for each line (d, l), the list of points on it.
    # We already have lines[d, l, :] = 49 point indices.

    # For efficient iteration, let's also build: for each point p, for each direction d,
    # the other 48 points on the line through p in direction d.
    # other_points[p, d, :] = 48 point indices (excluding p)
    # This is 2401 * 50 * 48 = 5.7M entries, ~46MB. Acceptable.
    other_points = np.zeros((total_pts, n_dirs, q - 1), dtype=np.int64)
    for d in range(n_dirs):
        for l in range(q):
            pts = lines[d, l, :]  # 49 points
            for i in range(q):
                p = pts[i]
                # other points: all pts except pts[i]
                other = np.concatenate([pts[:i], pts[i+1:]])
                other_points[p, d, :] = other

    # Now the greedy loop.
    # Score of adding p: number of (x, d) pairs where x != p, x is on the line
    # through p in direction d, and clean_line_flag[x, d] is currently True.
    # This is the number of clean line instances that would be killed.
    #
    # We want to maximize this score subject to feasibility:
    #   For all x, after adding p, clean_count[x] >= 1.
    #   Equivalently: for all x != p, if clean_count[x] == 1 and the only clean
    #   line for x is one of the 50 lines through p, then infeasible.
    #
    # Let's compute the score and feasibility for all candidate p at once.

    def compute_scores_and_feasibility():
        """For each point p not in S, compute the score (impact) and feasibility."""
        scores = np.zeros(total_pts, dtype=np.int64)
        feasible = np.ones(total_pts, dtype=bool)
        feasible[in_S] = False  # can't add points already in S

        for p in range(total_pts):
            if in_S[p]:
                continue
            score = 0
            is_feasible = True
            for d in range(n_dirs):
                l = lines_through[p, d]
                # Points on this line other than p:
                others = other_points[p, d, :]
                # For each other point x, if clean_line_flag[x, d] is True,
                # then adding p kills this clean line for x.
                # Count how many such x there are.
                clean_others = clean_line_flag[others, d]
                n_clean = np.count_nonzero(clean_others)
                score += n_clean
                # Feasibility: if any x has clean_count[x] == 1 and the only
                # clean line for x is (d, l), then killing it uncovers x.
                # Check: for x in others, if clean_count[x] == 1 and
                # clean_line_flag[x, d] is True and no other direction d' has
                # clean_line_flag[x, d'] True, then infeasible.
                # This is expensive to check per x. Let's defer to a simpler check:
                # After tentatively killing these lines, check if any x drops to 0.
            scores[p] = score

        return scores, feasible

    # The above is O(2401 * 50 * 49) per iteration, which is ~5.9M operations.
    # With ~2400 iterations, that's ~14 billion operations. Too slow in Python.
    #
    # Let's use a vectorized approach.
    #
    # Alternative: instead of trying all p each iteration, use a simpler greedy:
    # randomly sample candidates and pick the best feasible one.
    #
    # Or: use a different strategy. Since the problem is small (2401 points),
    # let's try a simpler approach: add points in a fixed order (e.g., by x, y)
    # and check feasibility. This is O(2401 * 50 * 49) total, which is fine.

    # Let's do a simple greedy: iterate through all points in a fixed order,
    # and add a point if it's feasible. This is fast and gives a valid result.

    # For a better result, let's try multiple random orderings and keep the best.

    rng = np.random.default_rng(42)
    best_S = None
    best_size = 0

    n_attempts = 20  # number of random orderings to try

    for attempt in range(n_attempts):
        if time.time() > deadline:
            break

        in_S = np.zeros(total_pts, dtype=bool)
        clean_count = np.full(total_pts, n_dirs, dtype=np.int64)
        clean_line_flag = np.ones((total_pts, n_dirs), dtype=bool)
        line_s_count = np.zeros((n_dirs, q), dtype=np.int64)

        order = rng.permutation(total_pts)

        for p in order:
            if time.time() > deadline:
                break

            # Check feasibility of adding p
            feasible = True
            score = 0
            for d in range(n_dirs):
                l = lines_through[p, d]
                others = other_points[p, d, :]
                # Count clean lines killed
                clean_others = clean_line_flag[others, d]
                n_clean = np.count_nonzero(clean_others)
                score += n_clean
                # Check if any x would become uncovered
                # x is uncovered if clean_count[x] == 1 and the only clean line is (d, l)
                # For x in others where clean_line_flag[x, d] is True:
                #   if clean_count[x] == 1, then (d, l) is the only clean line for x.
                #   Killing it would uncover x.
                #   So if any such x exists, infeasible.
                if n_clean > 0:
                    # Check if any of these x has clean_count == 1
                    x_candidates = others[clean_others]
                    if np.any(clean_count[x_candidates] == 1):
                        feasible = False
                        break

            if not feasible:
                continue

            # Add p to S
            in_S[p] = True
            for d in range(n_dirs):
                l = lines_through[p, d]
                line_s_count[d, l] += 1
                # Update clean_line_flag for all points on this line
                others = other_points[p, d, :]
                # For each x in others, the line (d, l) is no longer clean
                # (since p is now in S and x != p)
                clean_line_flag[others, d] = False
                clean_count[others] -= 1
                # For p itself: the line is clean for p iff line_s_count[d, l] == 1
                if line_s_count[d, l] == 1:
                    clean_line_flag[p, d] = True
                else:
                    clean_line_flag[p, d] = False
                    # clean_count[p] was decremented? No, we didn't decrement p.
                    # Actually, p's clean count should reflect the number of clean lines.
                    # Let's recompute clean_count[p] at the end.
            # Recompute clean_count[p]
            clean_count[p] = np.count_nonzero(clean_line_flag[p, :])

        size = np.count_nonzero(in_S)
        if size > best_size:
            best_size = size
            best_S = in_S.copy()

    # If no points were added (shouldn't happen), fallback to a single point.
    if best_S is None or best_size == 0:
        best_S = np.zeros(total_pts, dtype=bool)
        best_S[0] = True
        best_size = 1

    # Convert best_S to the required output format.
    # Points in N are those NOT in S.
    # For each point index pt, x_idx = pt // q, y_idx = pt % q
    # x_idx = xa * p + xb, y_idx = ya * p + yb
    # Output: array of shape (k, 2, 2) where k = 2401 - best_size

    n_points = total_pts - best_size
    result = np.zeros((n_points, 2, 2), dtype=np.int64)
    idx = 0
    for pt in range(total_pts):
        if not best_S[pt]:
            x_idx = pt // q
            y_idx = pt % q
            xa = x_idx // p
            xb = x_idx % p
            ya = y_idx // p
            yb = y_idx % p
            result[idx] = [[xa, xb], [ya, yb]]
            idx += 1

    return result

# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 630s kill

    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3 (quadratic non-residue mod 7)
    n_points = q * q  # 2401
    n_dirs = q + 1  # 50 directions

    # ---------- Field F_49 arithmetic, vectorized over the 49 elements ----------
    # element index i -> (a, b) for a + b*alpha, with i = a*p + b
    idx_a = np.array([i // p for i in range(q)], dtype=np.int64)
    idx_b = np.array([i % p for i in range(q)], dtype=np.int64)

    # addition table: add[i,j] = index of i+j
    add_tbl = np.zeros((q, q), dtype=np.int64)
    for i in range(q):
        for j in range(q):
            a = (idx_a[i] + idx_a[j]) % p
            b = (idx_b[i] + idx_b[j]) % p
            add_tbl[i, j] = a * p + b

    # multiplication table: mul[i,j] = index of i*j
    mul_tbl = np.zeros((q, q), dtype=np.int64)
    for i in range(q):
        for j in range(q):
            a = (idx_a[i] * idx_a[j] + idx_b[i] * idx_b[j] * k) % p
            b = (idx_a[i] * idx_b[j] + idx_b[i] * idx_a[j]) % p
            mul_tbl[i, j] = a * p + b

    # negation: neg[i] = index of -i
    neg_tbl = np.zeros(q, dtype=np.int64)
    for i in range(q):
        a = (-idx_a[i]) % p
        b = (-idx_b[i]) % p
        neg_tbl[i] = a * p + b

    # ---------- Enumerate the 50 directions ----------
    # direction d in 0..48: v = (1, m_d) where m_d = field element d
    # direction 49: v = (0, 1)
    # For each direction d and each "offset" o (49 of them), we get a line of 49 points.
    # Total lines = 50 * 49 = 2450.

    # We precompute, for each point x (0..2400), the 50 lines through it,
    # and for each such line, the bitmask (over the 48 other points) of which
    # points on that line are "in S" (excluded).
    #
    # Point index: point (x_field, y_field) -> idx = x_field * q + y_field
    # x_field = xa*p+xb, y_field = ya*p+yb

    # For each direction d, and each point x, compute the other 48 points on the line.
    # line_pts[d, x, t] for t in 0..48 gives the index of the t-th other point.
    # We store as a flat array: line_pts shape (n_dirs, n_points, q-1)
    # This is 50 * 2401 * 48 = 5.76M ints -> ~46 MB as int64. OK.

    line_pts = np.empty((n_dirs, n_points, q - 1), dtype=np.int64)

    # direction vectors: dir_x[d], dir_y[d] as field indices
    dir_x = np.empty(n_dirs, dtype=np.int64)
    dir_y = np.empty(n_dirs, dtype=np.int64)
    for d in range(q):
        dir_x[d] = 1 * p + 0  # field element 1
        dir_y[d] = d          # field element d
    dir_x[q] = 0  # field element 0
    dir_y[q] = 1 * p + 0  # field element 1

    # For each direction d, generate the 49 lines by picking 49 representative points
    # (one per parallel class). For direction (1, m), the lines are parameterized by
    # the "intercept" y - m*x = c. For direction (0,1), lines are x = c.
    #
    # For each direction d, we compute for every point x the line it belongs to,
    # and the ordered list of points on that line.

    for d in range(n_dirs):
        dx = dir_x[d]
        dy = dir_y[d]
        # For each of the 49 parameter values t (field element), the line is
        # { base + t * v } where base is a chosen representative.
        # Instead, for each point x, the line through x with direction d is
        # { x + t * v : t in F_49 }. The 48 other points are x + t*v for t != 0.
        for x in range(n_points):
            x_field = x // q
            y_field = x % q
            # iterate t in 1..48 (skip t=0)
            for ti in range(1, q):
                # t_field = ti (index ti corresponds to field element ti)
                # offset = t * v
                off_x = mul_tbl[ti, dx]
                off_y = mul_tbl[ti, dy]
                nx = add_tbl[x_field, off_x]
                ny = add_tbl[y_field, off_y]
                line_pts[d, x, ti - 1] = nx * q + ny

    # ---------- Greedy construction with bitmask tracking ----------
    # S = set of excluded points. We track for each point x and each direction d
    # whether the line through x in direction d is "clean" (no point of S on it
    # other than possibly x itself).
    #
    # clean_mask[x] is a bitmask over the 50 directions; bit d set means the
    # line through x in direction d is currently clean.
    #
    # When we add a point p to S:
    #   For each direction d, the line through p in direction d now contains p (in S).
    #   For every OTHER point x on that line, the line through x in direction d is
    #   no longer clean (because it contains p in S). So we clear bit d in clean_mask[x].
    #   Additionally, for p itself, after adding p to S, the line through p in direction d
    #   is "clean for p" only if no OTHER point on that line is in S. But since we just
    #   added p, and the line contains p, the line is NOT clean for p (p is in S, and
    #   the Nikodym condition requires the OTHER points to be in N, which they are, but
    #   p itself being in S doesn't matter for p's own witness — wait, re-read).
    #
    # Nikodym condition: for every point x (in N or not), there exists a direction d
    # such that ALL points on the line {x + t*v : t in F_49} OTHER THAN x belong to N.
    # So for point x, the witness line must have all 48 other points in N (not in S).
    # x itself can be in S or not; it doesn't matter.
    #
    # So clean_mask[x] bit d = 1 means: all 48 other points on the line through x in
    # direction d are in N (i.e., not in S). This is independent of whether x is in S.
    #
    # When we add p to S:
    #   For each direction d, consider the line L through p in direction d.
    #   L contains p and 48 other points.
    #   For each other point x on L: the line through x in direction d is the same line L.
    #   Since p is now in S, and p is on L and p != x, the line L now has a point in S
    #   other than x. So clean_mask[x] bit d must be cleared.
    #   For p itself: the line through p in direction d is L. The other 48 points on L
    #   are unchanged (none of them just got added to S). So whether L is clean for p
    #   depends on whether any of those 48 other points are in S. That hasn't changed
    #   by adding p. So clean_mask[p] bit d is unchanged.
    #
    # Wait, that's not right either. clean_mask[p] bit d was tracking whether all 48
    # OTHER points on L are in N. Adding p to S doesn't affect those 48 other points.
    # So yes, clean_mask[p] is unchanged.
    #
    # But we also need: after adding p, is p still "covered"? p is covered if
    # clean_mask[p] has at least one bit set. Since clean_mask[p] is unchanged,
    # p remains covered iff it was covered before. But p was in N before (not in S),
    # and we're adding it to S. The Nikodym condition applies to ALL points, including
    # those in S. So p must still be covered. If clean_mask[p] was non-zero before,
    # it's still non-zero, so p is still covered. Good.
    #
    # The danger is: some other point x on L might have had clean_mask[x] with ONLY
    # bit d set. After clearing bit d, clean_mask[x] becomes 0, meaning x is no longer
    # covered. That would violate the Nikodym condition. So we must check: for every
    # other point x on L, if clean_mask[x] == (1 << d) before clearing, then after
    # clearing it becomes 0 -> invalid. So we can only add p if for all x on L (x != p),
    # clean_mask[x] has at least one bit set OTHER than d (or clean_mask[x] == 0 is
    # already a problem, but we maintain the invariant that all points are covered).
    #
    # Actually, we maintain the invariant that at all times, every point has
    # clean_mask != 0 (is covered). When we add p, we clear bit d in clean_mask[x]
    # for all x on L, x != p. We must ensure none of those x had clean_mask[x] == (1<<d).
    #
    # This is the key check. Let's implement it.

    # Precompute: for each point p and direction d, the list of other points on the line.
    # That's line_pts[d, p, :] which has 48 entries.

    # We'll do multiple random restarts. In each restart:
    # 1. Initialize S = empty, clean_mask = all 50 bits set for all points.
    # 2. Randomly order all 2401 points.
    # 3. For each point p in order, try to add it to S:
    #    - For each direction d (0..49):
    #      - Check all 48 other points x on line_pts[d, p, :]:
    #        if clean_mask[x] == (1 << d), then adding p would uncover x -> fail.
    #    - If no direction fails, add p to S and update clean_mask for all affected x.
    #
    # This check is O(50 * 48) = 2400 per candidate, times 2401 candidates = ~5.7M ops
    # per restart. With numpy, we can vectorize the check.

    # Let's precompute line_pts as a numpy array and use vectorized operations.
    # line_pts shape: (n_dirs, n_points, q-1) = (50, 2401, 48)

    # For a given p, the affected points are line_pts[:, p, :] which is (50, 48).
    # We need to check: for each d, is there any x in line_pts[d, p, :] with
    # clean_mask[x] == (1 << d)?
    #
    # clean_mask is shape (n_points,) with values 0..(1<<50)-1.
    # (1 << d) for d in 0..49.
    #
    # We can do: affected = line_pts[:, p, :]  # shape (50, 48)
    # cm_affected = clean_mask[affected]  # shape (50, 48)
    # dir_bits = np.array([1 << d for d in range(n_dirs)])  # shape (50,)
    # conflict = np.any(cm_affected == dir_bits[:, None], axis=1)  # shape (50,)
    # if not np.any(conflict): safe to add.

    # This is fast with numpy. Let's implement.

    dir_bits = np.array([1 << d for d in range(n_dirs)], dtype=np.uint64)

    best_S = set()
    best_size = 0
    rng = random.Random(42)

    def try_greedy(order):
        """Try to build S by processing points in the given order. Return S."""
        S = set()
        # clean_mask: uint64 array, bit d set means line through x in dir d is clean
        clean_mask = np.full(n_points, (1 << n_dirs) - 1, dtype=np.uint64)
        in_S = np.zeros(n_points, dtype=bool)

        for p in order:
            if in_S[p]:
                continue
            # Check if adding p is safe
            affected = line_pts[:, p, :]  # shape (50, 48)
            cm_affected = clean_mask[affected]  # shape (50, 48)
            # For each direction d, check if any x has clean_mask[x] == (1 << d)
            conflict = np.any(cm_affected == dir_bits[:, None], axis=1)  # shape (50,)
            if np.any(conflict):
                continue
            # Safe to add p
            in_S[p] = True
            S.add(p)
            # Update clean_mask: for each direction d, clear bit d in all affected x
            # affected shape (50, 48), dir_bits shape (50,)
            # We need: for each d, for each x in affected[d, :], clean_mask[x] &= ~(1 << d)
            # Vectorized:
            for d in range(n_dirs):
                xs = affected[d, :]
                clean_mask[xs] &= ~dir_bits[d]

        return S

    # Run multiple random restarts
    n_restarts = 30
    for r in range(n_restarts):
        if time.time() > deadline - 30:
            break
        order = list(range(n_points))
        rng.shuffle(order)
        S = try_greedy(order)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
            # print(f"Restart {r}: |S| = {len(S)}")

    # ---------- Local search refinement ----------
    # Try to add more points that were skipped, in a different order.
    # Or try to remove some points from S and re-add others.
    # For now, let's try a second pass: process points in reverse order,
    # or in a different random order, starting from the current S.

    # Actually, let's try a "second pass" where we start from best_S and try
    # to add the remaining points.
    if time.time() < deadline - 30 and len(best_S) > 0:
        in_S = np.zeros(n_points, dtype=bool)
        for p in best_S:
            in_S[p] = True
        clean_mask = np.full(n_points, (1 << n_dirs) - 1, dtype=np.uint64)
        # Rebuild clean_mask from scratch based on current S
        for p in best_S:
            affected = line_pts[:, p, :]
            for d in range(n_dirs):
                xs = affected[d, :]
                clean_mask[xs] &= ~dir_bits[d]
        # Now try to add remaining points
        remaining = [p for p in range(n_points) if not in_S[p]]
        rng.shuffle(remaining)
        for p in remaining:
            if time.time() > deadline - 10:
                break
            affected = line_pts[:, p, :]
            cm_affected = clean_mask[affected]
            conflict = np.any(cm_affected == dir_bits[:, None], axis=1)
            if np.any(conflict):
                continue
            in_S[p] = True
            best_S.add(p)
            for d in range(n_dirs):
                xs = affected[d, :]
                clean_mask[xs] &= ~dir_bits[d]

    # ---------- Build the result ----------
    # N = all points NOT in best_S
    in_S = np.zeros(n_points, dtype=bool)
    for p in best_S:
        in_S[p] = True

    # N points: those with in_S[p] == False
    n_indices = np.where(~in_S)[0]
    points = np.zeros((len(n_indices), 2, 2), dtype=np.int64)
    for i, idx in enumerate(n_indices):
        x_field = idx // q
        y_field = idx % q
        xa = x_field // p
        xb = x_field % p
        ya = y_field // p
        yb = y_field % p
        points[i] = [[xa, xb], [ya, yb]]

    # Ensure at least 1 point
    if len(points) == 0:
        points = np.zeros((1, 2, 2), dtype=np.int64)

    return points

# EVOLVE-BLOCK-END

import numpy as np
import time
import random
from math import gcd

def normalize_line(A, B, C):
    if A == 0 and B == 0:
        return None
    g = gcd(abs(A), gcd(abs(B), abs(C)))
    if g == 0:
        g = 1
    A //= g; B //= g; C //= g
    if A < 0 or (A == 0 and B < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def count_incidences(points, lines):
    P = points.astype(np.int64)
    L = lines.astype(np.int64)
    px = P[:, 0].reshape(-1, 1)
    py = P[:, 1].reshape(-1, 1)
    A = L[:, 0].reshape(1, -1)
    B = L[:, 1].reshape(1, -1)
    C = L[:, 2].reshape(1, -1)
    return int(np.sum(A * px + B * py + C == 0))

def line_through(p1, p2):
    dx = int(p2[0]) - int(p1[0])
    dy = int(p2[1]) - int(p1[1])
    A, B, C = dy, -dx, dx * int(p1[1]) - dy * int(p1[0])
    return normalize_line(A, B, C)

def build_grid_lines():
    lines = []
    for i in range(9):
        lines.append((1, 0, -i))
    for j in range(9):
        lines.append((0, 1, -j))
    for c in range(-4, 0):
        lines.append((1, -1, -c))
    for c in range(0, 5):
        lines.append((1, 1, -c))
    return np.array(lines, dtype=np.int64)

def grid_points_all():
    pts = []
    for x in range(9):
        for y in range(9):
            pts.append((x, y))
    return np.array(pts, dtype=np.int64)

def candidate_pool():
    pts = []
    for x in range(-30, 31):
        for y in range(-30, 31):
            pts.append((x, y))
    return np.array(pts, dtype=np.int64)

def incidence_matrix(points, lines):
    """Returns (N, M) bool array: True if point i lies on line j."""
    P = points.astype(np.int64)
    L = lines.astype(np.int64)
    px = P[:, 0].reshape(-1, 1)
    py = P[:, 1].reshape(-1, 1)
    A = L[:, 0].reshape(1, -1)
    B = L[:, 1].reshape(1, -1)
    C = L[:, 2].reshape(1, -1)
    return (A * px + B * py + C) == 0

def best_points_for_lines(lines, pool, k=54):
    """Select k points from pool maximizing incidences with lines."""
    M = incidence_matrix(pool, lines)  # (|pool|, 27)
    counts = M.sum(axis=1)
    idx = np.argsort(counts)[::-1][:k]
    return pool[idx]

def local_search(points, lines, pool, time_limit, seed=0):
    """
    Fast local search on (points, lines).
    Moves:
      1. Replace one point with a pool point (recompute I cheaply).
      2. Replace one line with a line through two current points (recompute I cheaply).
      3. Re-optimize all points for current lines.
    Uses precomputed incidence matrix where possible.
    """
    random.seed(seed)
    N = len(points)
    M = len(lines)
    cur_pts = points.copy()
    cur_lns = lines.copy()
    cur_inc = incidence_matrix(cur_pts, cur_lns)  # (54, 27) bool
    cur_I = int(cur_inc.sum())

    best_pts = cur_pts.copy()
    best_lns = cur_lns.copy()
    best_I = cur_I

    start = time.time()

    def recompute_inc():
        return incidence_matrix(cur_pts, cur_lns)

    # Precompute pool incidence for current lines (updated when lines change)
    pool_inc = incidence_matrix(pool, cur_lns)  # (|pool|, 27)

    def rebuild_pool_inc():
        nonlocal pool_inc
        pool_inc = incidence_matrix(pool, cur_lns)

    # Map pool points to indices for fast lookup
    pool_set = {tuple(p): i for i, p in enumerate(pool)}

    # Point counts and line counts
    pt_counts = cur_inc.sum(axis=1)  # (54,)
    ln_counts = cur_inc.sum(axis=0)  # (27,)

    no_improve = 0
    max_no_improve = 200

    while time.time() - start < time_limit:
        if no_improve > max_no_improve:
            # Random restart: re-optimize points
            new_pts = best_points_for_lines(cur_lns, pool, 54)
            new_inc = incidence_matrix(new_pts, cur_lns)
            new_I = int(new_inc.sum())
            if new_I > cur_I:
                cur_pts = new_pts
                cur_inc = new_inc
                cur_I = new_I
                pt_counts = cur_inc.sum(axis=1)
                ln_counts = cur_inc.sum(axis=0)
                no_improve = 0
                if cur_I > best_I:
                    best_I = cur_I
                    best_pts = cur_pts.copy()
                    best_lns = cur_lns.copy()
            no_improve = 0
            continue

        move = random.random()

        if move < 0.50:
            # Replace a point
            # Pick a low-count point to replace
            worst_pt = int(np.argmin(pt_counts))
            # Try a random pool point
            for _ in range(30):
                cand_idx = random.randint(0, len(pool) - 1)
                cand = pool[cand_idx]
                # Check if cand is already in cur_pts
                if any(tuple(cur_pts[i]) == tuple(cand) for i in range(N)):
                    continue
                # Delta: gain from cand - loss from worst_pt
                gain = int(pool_inc[cand_idx].sum())
                loss = int(pt_counts[worst_pt])
                delta = gain - loss
                if delta > 0:
                    # Accept
                    old_pt = cur_pts[worst_pt].copy()
                    cur_pts[worst_pt] = cand
                    # Update incidence row
                    cur_inc[worst_pt] = pool_inc[cand_idx]
                    cur_I += delta
                    pt_counts = cur_inc.sum(axis=1)
                    ln_counts = cur_inc.sum(axis=0)
                    no_improve = 0
                    if cur_I > best_I:
                        best_I = cur_I
                        best_pts = cur_pts.copy()
                        best_lns = cur_lns.copy()
                    break

        elif move < 0.80:
            # Replace a line
            worst_ln = int(np.argmin(ln_counts))
            for _ in range(30):
                i1, i2 = random.sample(range(N), 2)
                nl = line_through(cur_pts[i1], cur_pts[i2])
                if nl is None:
                    continue
                nl_tup = tuple(nl)
                if any(tuple(cur_lns[j]) == nl_tup for j in range(M)):
                    continue
                # Compute new line's incidence with current points
                A, B, C = nl
                new_col = (A * cur_pts[:, 0] + B * cur_pts[:, 1] + C) == 0
                gain = int(new_col.sum())
                loss = int(ln_counts[worst_ln])
                delta = gain - loss
                if delta > 0:
                    old_ln = cur_lns[worst_ln].copy()
                    cur_lns[worst_ln] = nl
                    cur_inc[:, worst_ln] = new_col
                    cur_I += delta
                    pt_counts = cur_inc.sum(axis=1)
                    ln_counts = cur_inc.sum(axis=0)
                    rebuild_pool_inc()
                    no_improve = 0
                    if cur_I > best_I:
                        best_I = cur_I
                        best_pts = cur_pts.copy()
                        best_lns = cur_lns.copy()
                    break

        else:
            # Re-optimize all points for current lines
            new_pts = best_points_for_lines(cur_lns, pool, 54)
            new_inc = incidence_matrix(new_pts, cur_lns)
            new_I = int(new_inc.sum())
            if new_I > cur_I:
                cur_pts = new_pts
                cur_inc = new_inc
                cur_I = new_I
                pt_counts = cur_inc.sum(axis=1)
                ln_counts = cur_inc.sum(axis=0)
                no_improve = 0
                if cur_I > best_I:
                    best_I = cur_I
                    best_pts = cur_pts.copy()
                    best_lns = cur_lns.copy()
            else:
                no_improve += 1
        # Small cooling: occasionally accept neutral moves to escape local optima
        if random.random() < 0.02:
            no_improve = max(0, no_improve - 50)

    return best_lns, best_pts, best_I

def try_diagonal_variants(pool, time_limit):
    """Try different combinations of diagonal constants to find a better grid."""
    best_I = -1
    best_pts = None
    best_lns = None

    # Base: 9V + 9H
    base_lines = []
    for i in range(9):
        base_lines.append((1, 0, -i))
    for j in range(9):
        base_lines.append((0, 1, -j))

    # Try various sets of 4 slope-1 and 5 slope-1 diagonals
    d1_options = [
        [-4, -3, -2, -1],
        [-5, -4, -3, -2],
        [-3, -2, -1, 0],
        [-4, -3, -2, 0],
        [-5, -4, -3, -1],
        [-4, -2, -1, 0],
    ]
    dm1_options = [
        [0, 1, 2, 3, 4],
        [-1, 0, 1, 2, 3],
        [1, 2, 3, 4, 5],
        [-2, -1, 0, 1, 2],
        [0, 1, 2, 3, 5],
        [-1, 0, 1, 2, 4],
    ]

    for d1s in d1_options:
        for dm1s in dm1_options:
            if time.time() > time_limit:
                return best_lns, best_pts, best_I
            lines = list(base_lines)
            for c in d1s:
                lines.append((1, -1, -c))
            for c in dm1s:
                lines.append((1, 1, -c))
            lns = np.array(lines, dtype=np.int64)
            pts = best_points_for_lines(lns, pool, 54)
            I = count_incidences(pts, lns)
            if I > best_I:
                best_I = I
                best_pts = pts.copy()
                best_lns = lns.copy()

    return best_lns, best_pts, best_I

def szemeredi_trotter_incidences():
    deadline = time.time() + 600  # 30s margin

    pool = candidate_pool()

    # Start with the known good grid
    grid_lns = build_grid_lines()
    grid_pts = best_points_for_lines(grid_lns, pool, 54)
    grid_I = count_incidences(grid_pts, grid_lns)

    best_I = grid_I
    best_pts = grid_pts.copy()
    best_lns = grid_lns.copy()

    # Try diagonal variants
    if time.time() < deadline - 100:
        v_lns, v_pts, v_I = try_diagonal_variants(pool, deadline - 100)
        if v_I > best_I:
            best_I = v_I
            best_pts = v_pts.copy()
            best_lns = v_lns.copy()

    # Local search from best
    if time.time() < deadline - 30:
        tl = min(200, deadline - time.time() - 30)
        if tl > 10:
            s_lns, s_pts, s_I = local_search(best_pts, best_lns, pool, tl, seed=42)
            if s_I > best_I:
                best_I = s_I
                best_pts = s_pts.copy()
                best_lns = s_lns.copy()

    # Another local search with different seed
    if time.time() < deadline - 20:
        tl = min(150, deadline - time.time() - 20)
        if tl > 10:
            s_lns, s_pts, s_I = local_search(best_pts, best_lns, pool, tl, seed=12345)
            if s_I > best_I:
                best_I = s_I
                best_pts = s_pts.copy()
                best_lns = s_lns.copy()

    # Final: re-optimize points for best lines
    final_pts = best_points_for_lines(best_lns, pool, 54)
    final_I = count_incidences(final_pts, best_lns)
    if final_I > best_I:
        best_I = final_I
        best_pts = final_pts.copy()

    # Validate: 54 unique points
    unique_pts, idx = np.unique(best_pts, axis=0, return_index=True)
    if len(unique_pts) != 54:
        best_pts = grid_pts
        best_lns = grid_lns
    else:
        best_pts = unique_pts[idx]

    # Validate: 27 unique lines
    unique_lns = []
    for l in best_lns:
        nl = normalize_line(int(l[0]), int(l[1]), int(l[2]))
        if nl and nl not in unique_lns:
            unique_lns.append(nl)
    if len(unique_lns) < 27:
        best_lns = grid_lns
        best_pts = best_points_for_lines(best_lns, pool, 54)
    else:
        best_lns = np.array(unique_lns[:27], dtype=np.int64)

    return best_pts, best_lns

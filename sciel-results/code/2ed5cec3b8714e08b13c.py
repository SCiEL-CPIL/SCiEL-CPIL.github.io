# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict

DEADLINE = 630.0
START_TIME = time.time()


def _time_left() -> float:
    return DEADLINE - (time.time() - START_TIME)


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 where F_49 = F_7[alpha], alpha^2 = 3.
    Uses reverse greedy removal with incremental validity tracking.
    """
    p = 7
    k = 3  # alpha^2 = 3

    def field_mul(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        a3 = (a1 * a2 + k * b1 * b2) % p
        b3 = (a1 * b2 + a2 * b1) % p
        return (a3, b3)

    def field_sub(p1, p2):
        a1, b1 = p1
        a2, b2 = p2
        return ((a1 - a2) % p, (b1 - b2) % p)

    # Enumerate all points
    all_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    all_points.append(((xa, xb), (ya, yb)))

    n_total = len(all_points)  # 2401

    # Directions: (1, m) for all m in F_49, plus (0, 1)
    directions = []
    for ma in range(p):
        for mb in range(p):
            directions.append(((1, 0), (ma, mb)))
    directions.append(((0, 1), (0, 0)))
    n_dirs = len(directions)  # 50

    # Precompute line neighbors: line_neighbors[d][i] = list of other points on same line
    line_neighbors = []
    for d_idx in range(n_dirs):
        vx, vy = directions[d_idx]
        line_map = {}
        for pt_idx, pt in enumerate(all_points):
            (x, y) = pt
            if vx == (1, 0):
                mx = field_mul(vy, x)
                c = field_sub(y, mx)
            else:
                c = x
            if c not in line_map:
                line_map[c] = []
            line_map[c].append(pt_idx)
        neigh = [[] for _ in range(n_total)]
        for c, pts_on_line in line_map.items():
            for i in pts_on_line:
                neigh[i] = [j for j in pts_on_line if j != i]
        line_neighbors.append(neigh)

    # Precompute: for each point i and direction d, which other points are on the line
    # Also precompute reverse: for each point j and direction d, which points i share that line with j
    # reverse_line[d][j] = list of points i such that j is in line_neighbors[d][i]
    # i.e., i and j are on the same line in direction d
    reverse_line = []
    for d_idx in range(n_dirs):
        rev = [[] for _ in range(n_total)]
        for i in range(n_total):
            for j in line_neighbors[d_idx][i]:
                rev[j].append(i)
        reverse_line.append(rev)

    # Base construction: unital-based
    squares = set()
    for u in range(p):
        squares.add((u * u) % p)

    S_base = set()
    for pt_idx, pt in enumerate(all_points):
        (xa, xb), (ya, yb) = pt
        in_unital = (2 * xa + ya * ya - k * yb * yb) % p == 0
        in_y_special = ya in squares
        if in_unital and not in_y_special:
            S_base.add(pt_idx)

    # Multiple restarts with different random seeds for candidate ordering
    best_S_size = len(S_base)
    best_S = set(S_base)

    def greedy_add(S_init, seed, time_limit):
        """Greedy: add points to S while maintaining validity using incremental tracking."""
        S = set(S_init)
        in_S = np.zeros(n_total, dtype=bool)
        for s in S:
            in_S[s] = True

        # s_count_on_line[d][i] = number of points in S on the line through i in direction d (excluding i itself)
        # Actually, we need: for point i and direction d, how many points j != i on line(i,d) are in S
        # Let's use: line_s_count[d][i] = count of S-points on line through i in direction d, excluding i
        line_s_count = np.zeros((n_dirs, n_total), dtype=np.int32)

        # Initialize line_s_count based on initial S
        for d_idx in range(n_dirs):
            for i in range(n_total):
                cnt = 0
                for j in line_neighbors[d_idx][i]:
                    if in_S[j]:
                        cnt += 1
                line_s_count[d_idx, i] = cnt

        # A point i is "valid" for direction d if line_s_count[d][i] == 0
        # A point i is "valid overall" if there exists d with line_s_count[d][i] == 0

        # Track for each point how many directions are still clean
        clean_dirs_count = np.zeros(n_total, dtype=np.int32)
        for i in range(n_total):
            clean_dirs_count[i] = np.sum(line_s_count[:, i] == 0)

        # Candidate points: those not in S
        rng = np.random.default_rng(seed)
        candidates = [i for i in range(n_total) if not in_S[i]]
        rng.shuffle(candidates)

        added = 0
        for cand in candidates:
            if time.time() - START_TIME > time_limit:
                break

            # Check if cand can be added: it needs at least one clean direction
            if clean_dirs_count[cand] == 0:
                continue

            # Try adding cand to S
            # For each direction d, for each point i that shares a line with cand in direction d:
            #   line_s_count[d][i] increases by 1
            #   if it was 0, clean_dirs_count[i] decreases by 1

            # Save state for rollback
            affected = []  # list of (d_idx, i) where line_s_count changed

            for d_idx in range(n_dirs):
                # Points i that share a line with cand in direction d
                for i in reverse_line[d_idx][cand]:
                    if line_s_count[d_idx, i] == 0:
                        clean_dirs_count[i] -= 1
                        affected.append((d_idx, i, 0))  # was 0
                    else:
                        affected.append((d_idx, i, line_s_count[d_idx, i]))
                    line_s_count[d_idx, i] += 1

            # Check if any point lost all clean directions
            # Only points that had clean_dirs_count go from 1 to 0 are problematic
            # Actually we need to check if any point now has clean_dirs_count == 0
            # But we only changed clean_dirs_count for points in reverse_line[d][cand]
            # Let's check if any affected point has clean_dirs_count == 0
            valid = True
            for d_idx, i, old_val in affected:
                if clean_dirs_count[i] == 0:
                    valid = False
                    break

            if valid:
                # Commit: add cand to S
                in_S[cand] = True
                S.add(cand)
                added += 1
            else:
                # Rollback
                for d_idx, i, old_val in affected:
                    line_s_count[d_idx, i] -= 1
                    if line_s_count[d_idx, i] == 0:
                        clean_dirs_count[i] += 1

        return S

    # Try multiple seeds
    num_restarts = 20
    time_per_restart = 25.0  # seconds per restart
    total_time_budget = 500.0  # leave 130s for finalization

    for r in range(num_restarts):
        if _time_left() < 100:
            break
        seed = 42 + r * 137
        S_result = greedy_add(S_base, seed, START_TIME + total_time_budget - _time_left())
        if len(S_result) > best_S_size:
            best_S_size = len(S_result)
            best_S = S_result

    # Build final result
    points = []
    for pt_idx in range(n_total):
        if pt_idx not in best_S:
            pt = all_points[pt_idx]
            (xa, xb), (ya, yb) = pt
            points.append([[xa, xb], [ya, yb]])

    arr = np.array(points, dtype=np.int64)
    return arr


# EVOLVE-BLOCK-END

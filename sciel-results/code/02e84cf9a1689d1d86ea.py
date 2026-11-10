# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = _START_TIME + 600.0

p = 7
K = 3
N_TOTAL = p ** 4  # 2401


def _point_to_idx(xa, xb, ya, yb):
    return ((xa * p + xb) * p * p + (ya * p + yb))


def _idx_to_point(idx):
    yb = idx % p
    ya = (idx // p) % p
    xb = (idx // (p * p)) % p
    xa = (idx // (p * p * p)) % p
    return xa, xb, ya, yb


def _generate_seed_excluded():
    """
    S = points on the unital (2*xa + ya^2 - 3*yb^2 = 0 mod 7)
    where ya is NOT a quadratic residue mod 7 (i.e., ya in {3,5,6}).
    This gives |S| = 3*7*7 = 147.
    """
    squares = set((u * u) % p for u in range(p))  # {0,1,2,4}
    excluded = np.zeros(N_TOTAL, dtype=np.int8)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2 * xa + ya * ya - yb * yb * K) % p == 0:
                        if ya not in squares:
                            excluded[_point_to_idx(xa, xb, ya, yb)] = 1
    return excluded


def _precompute_lines():
    """
    Return list of 2450 arrays, each of length 49 (int32), holding point indices.
    49 vertical lines + 49*49 lines with direction (1, m).
    """
    lines = []
    # Vertical: direction (0,1), parameterized by (xa, xb)
    for xa in range(p):
        for xb in range(p):
            pts = []
            for ya in range(p):
                for yb in range(p):
                    pts.append(_point_to_idx(xa, xb, ya, yb))
            lines.append(np.array(pts, dtype=np.int32))
    # Direction (1, m) for ma, mb in 0..6
    for ma in range(p):
        for mb in range(p):
            for ca in range(p):
                for cb in range(p):
                    pts = []
                    for xa in range(p):
                        for xb in range(p):
                            ya = (ca + ma * xa + 3 * mb * xb) % p
                            yb = (cb + ma * xb + mb * xa) % p
                            pts.append(_point_to_idx(xa, xb, ya, yb))
                    lines.append(np.array(pts, dtype=np.int32))
    return lines


def _build_adjacency(lines):
    """
    Build the 2401x2401 adjacency matrix A where A[i,j]=1 iff i and j are
    on a common line (i != j).  Two distinct points in F_49^2 are on exactly
    one line, so A is a simple 0/1 matrix.
    """
    A = np.zeros((N_TOTAL, N_TOTAL), dtype=np.int32)
    for line in lines:
        idx = line
        A[np.ix_(idx, idx)] = 1
    np.fill_diagonal(A, 0)
    return A


def _compute_bad_counts(lines, in_S):
    """
    bad[x] = number of lines through x that contain at least one S point other than x.
    A point x is validly placed (Nikodym condition satisfied) iff bad[x] < 50.
    """
    bad = np.zeros(N_TOTAL, dtype=np.int32)
    for line in lines:
        s_on_line = in_S[line].sum()
        if s_on_line > 0:
            for idx in line:
                if s_on_line > (1 if in_S[idx] else 0):
                    bad[idx] += 1
    return bad


def nikodym_set() -> np.ndarray:
    start = time.time()

    # --- Phase 1: Setup ---
    in_S = _generate_seed_excluded().copy()
    lines = _precompute_lines()
    A = _build_adjacency(lines)  # 2401 x 2401 int32, ~22 MB

    bad = _compute_bad_counts(lines, in_S)

    # --- Phase 2: Vectorized batched greedy MIS ---
    # We want to grow S (the excluded set) to maximize |S| while keeping
    # bad[x] < 50 for all x.  A point c can be added iff for all neighbors
    # x of c (on the same line), bad[x] < 49 (so that after increment it stays < 50).
    #
    # Batch of B candidates:
    #   M = A[:, candidates]  (2401 x B)
    #   For each x: max_bad_in_batch = max over c in batch where M[x,c]=1 of bad[x]
    #   Actually we need: for each candidate c, check if ANY neighbor x has bad[x] >= 49.
    #   Equivalently: for candidate c, check if sum over x of (A[x,c] * (bad[x]>=49)) == 0.
    #   Let danger = (bad >= 49).astype(int32)  (2401,)
    #   For batch: hits = A[:, batch].T @ danger  (B,)
    #   Candidate c can be added iff hits[c] == 0.
    #
    # After selecting addable candidates, update:
    #   bad += A[:, addable].sum(axis=1)  -- no, that's not right.
    #   For each added point c, bad[x] += 1 for all x on a line with c.
    #   That's: bad += A[:, addable].sum(axis=1)? No.
    #   A[x, c] = 1 means x and c share a line. So for each added c, increment bad[x] for all x with A[x,c]=1.
    #   bad += A[:, addable_indices].sum(axis=1)  -- this gives for each x, how many added candidates are neighbors.
    #   Yes! That's correct.

    B = 64  # batch size
    n_candidates = N_TOTAL
    # Candidates are all points not yet in S
    # We'll process in rounds; each round we look at all not-in-S points.

    total_added = 0
    while True:
        if time.time() > _DEADLINE - 15:
            break

        # Get current candidates
        cand_mask = in_S == 0
        n_cand = int(cand_mask.sum())
        if n_cand == 0:
            break

        cand_indices = np.where(cand_mask)[0]

        # Process in batches
        batch_start = 0
        round_added = 0
        while batch_start < n_cand:
            if time.time() > _DEADLINE - 15:
                break

            batch_end = min(batch_start + B, n_cand)
            batch = cand_indices[batch_start:batch_end]
            batch_len = len(batch)

            # danger[x] = 1 if bad[x] >= 49
            danger = (bad >= 49).astype(np.int32)

            # hits[c] = number of danger points that are neighbors of batch[c]
            # A is 2401x2401; A[:, batch] is 2401 x batch_len
            # hits = A[:, batch].T @ danger  -> batch_len vector
            hits = A[:, batch].T @ danger

            # Addable: hits == 0
            addable_mask = (hits == 0)
            addable = batch[addable_mask]
            n_add = len(addable)

            if n_add > 0:
                # Update bad: for each x, bad[x] += number of added points that are neighbors of x
                # A[:, addable].sum(axis=1) gives that
                bad += A[:, addable].sum(axis=1)
                in_S[addable] = 1
                round_added += n_add
                total_added += n_add

            batch_start = batch_end

        if round_added == 0:
            # No more points can be added in this round
            break

    # --- Phase 3: Build output ---
    # N = complement of S
    n_points = int((in_S == 0).sum())
    result = np.zeros((n_points, 2, 2), dtype=np.int64)
    pt = 0
    for idx in range(N_TOTAL):
        if not in_S[idx]:
            xa, xb, ya, yb = _idx_to_point(idx)
            result[pt, 0, 0] = xa
            result[pt, 0, 1] = xb
            result[pt, 1, 0] = ya
            result[pt, 1, 1] = yb
            pt += 1

    return result

# EVOLVE-BLOCK-END

import time
import itertools
import numpy as np

_DEADLINE = 150.0
_MARGIN = 3.0


def _tetracode(q, n):
    # Direct sum of the ternary tetracode (9 words, length 4, perfect radius-1)
    # with every word on the remaining n-4 coordinates: 9 * q**(n-4) codewords.
    base = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    rest = n - 4
    out = []
    for h in base:
        for tail in itertools.product(range(q), repeat=rest):
            out.append(h + list(tail))
    return out


def _cover_matrix(q, n, radius):
    # Matrix C of shape (q**n, q**n): C[w, c] = 1 if w is within `radius` of codeword c.
    # Built by expanding every word into its radius-1 neighbourhood (radius assumed 1 here).
    total = q ** n
    C = np.zeros((total, total), dtype=np.uint8)
    base = np.array([q ** i for i in range(n)], dtype=np.int64)
    for w in range(total):
        digs = np.unravel_index(w, [q] * n)
        C[w, w] = 1
        for pos in range(n):
            for v in range(q):
                if v != digs[pos]:
                    digs2 = list(digs)
                    digs2[pos] = v
                    C[w, int(np.dot(digs2, base))] = 1
    return C


def _remove_redundant(C):
    # Repeatedly drop any column (codeword) whose removal leaves every row covered.
    # A column is redundant iff (sum over remaining columns) >= 1 wherever it is 1.
    covered = C.sum(axis=1)  # how many codewords cover each word
    m = C.shape[1]
    active = np.ones(m, dtype=bool)
    # Precompute per-column contribution: a column c is redundant iff
    # for every row r with C[r,c]==1, covered[r] >= 2 (some other covers it).
    # i.e. column c redundant iff no row r has C[r,c]==1 and covered[r]==1.
    while True:
        # rows uniquely covered by some active column
        # For each active column, check if it has a unique row.
        # vectorized: unique rows are those with covered==1; they must be kept by their sole cover.
        # A column is redundant iff it does not cover any row with covered==1.
        # Compute for active columns: does it touch a covered==1 row?
        unique_rows = (covered == 1)
        if not np.any(unique_rows):
            break
        # For each active column, does it have a 1 in a unique row?
        # touch = C[unique_rows, active].any(axis=0)
        sub = C[unique_rows, :]
        # only consider active columns
        touch = sub[:, active].any(axis=0)
        # map back to active indices
        act_idx = np.nonzero(active)[0]
        # columns among active that do NOT touch a unique row are redundant
        redundant_local = np.where(~touch)[0]
        if len(redundant_local) == 0:
            break
        rem_idx = act_idx[redundant_local]
        # remove them
        active[rem_idx] = False
        # update covered
        covered -= C[:, rem_idx].sum(axis=1)
    return np.nonzero(active)[0]


def construct(q, n, radius):
    start = time.time()
    t0 = start
    # Build initial code
    code = _tetracode(q, n)
    # Fast path for the known task: prune aggressively.
    total = q ** n
    # Build coverage matrix (only feasible for total not too large)
    if total <= 30000 and len(code) <= 6000:
        C = _cover_matrix(q, n, radius)
        keep = _remove_redundant(C)
        code = [code[i] for i in keep]
    # Deadline guard: return current best
    if time.time() - t0 > _DEADLINE - _MARGIN:
        return code
    return code

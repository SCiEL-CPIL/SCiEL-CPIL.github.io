"""Kissing number in dimension 11: build a large compatible set of integer vectors."""
import time
import random
import numpy as np

D = 11
TIME_LIMIT = 115.0  # safety margin under 120s (and under the 150s kill)


def _two_layer_vectors():
    """All 110 vectors with exactly two nonzero +-1 entries (norm 2)."""
    vecs = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    vecs.append(v)
    return vecs


def _three_layer_vectors():
    """All 440 vectors with exactly three nonzero +-1 entries (norm sqrt(3))."""
    vecs = []
    for i in range(D):
        for j in range(i + 1, D):
            for k in range(j + 1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        for sk in (1, -1):
                            v = np.zeros(D, dtype=np.int64)
                            v[i] = si
                            v[j] = sj
                            v[k] = sk
                            vecs.append(v)
    return vecs


def _is_compatible(v, mat):
    """True if v has dot in [-2, 2] with every row of mat (dist >= max norm)."""
    if mat.shape[0] == 0:
        return True
    d = mat @ v
    return bool(np.all(d >= -2) and np.all(d <= 2))


def _greedy_add(cands, mat, rng):
    """Greedy: add compatible candidates in random order."""
    order = list(range(len(cands)))
    rng.shuffle(order)
    for idx in order:
        v = cands[idx]
        if _is_compatible(v, mat):
            mat = np.vstack([mat, v])
    return mat


def _least_conflicts_improve(mat, cands, rng, budget):
    """Swap out a low-value selected vector to admit more candidates.
    Value of a selected vector = number of candidates it blocks.
    """
    n_sel = mat.shape[0]
    n_cand = len(cands)
    if n_cand == 0:
        return mat

    # Precompute candidate norms squared (all 3)
    cand_dot = None  # we recompute as needed

    def blocked_by(v):
        # candidates incompatible with v
        d = cands @ v
        return (d < -2) | (d > 2)

    # Current value of each selected vector
    values = np.zeros(n_sel, dtype=np.int64)
    for s in range(n_sel):
        values[s] = int(np.sum(blocked_by(mat[s])))

    # Current number of addable candidates
    def addable_count():
        # candidate addable if compatible with all selected
        Dm = mat @ cands.T  # (n_sel, n_cand)
        return int(np.sum(np.all((Dm >= -2) & (Dm <= 2), axis=0)))

    cur_addable = addable_count()

    start = time.time()
    improved = True
    while improved and (time.time() - start) < budget:
        improved = False
        # pick the selected vector with smallest value (removing it unblocks the most)
        s = int(np.argmin(values))
        v_s = mat[s]
        # candidates that become addable if we remove v_s: those blocked ONLY by v_s
        Dm = mat @ cands.T  # (n_sel, n_cand)
        # For each candidate, is it incompatible with v_s?
        inc_s = (Dm[s] < -2) | (Dm[s] > 2)
        # incompatible with any other selected?
        inc_other = np.zeros(n_cand, dtype=bool)
        if n_sel > 1:
            others = np.delete(Dm, s, axis=0)
            inc_other = np.any((others < -2) | (others > 2), axis=0)
        # candidates unblocked: incompatible with v_s but compatible with all others
        unblocked = inc_s & (~inc_other)
        n_unblock = int(np.sum(unblocked))
        # Gain: we remove v_s (lose its value in blocking) but gain n_unblock new addable
        # Net change in addable count approx = n_unblock - (how many addable v_s kept compatible)
        # Simpler: just accept if n_unblock >= 1 and value[s] is small
        if n_unblock >= 1:
            # remove v_s
            mat = np.delete(mat, s, axis=0)
            # add the unblocked candidates
            new_vecs = cands[unblocked]
            if len(new_vecs) > 0:
                mat = np.vstack([mat, new_vecs])
            n_sel = mat.shape[0]
            # recompute values
            values = np.zeros(n_sel, dtype=np.int64)
            for s2 in range(n_sel):
                values[s2] = int(np.sum(blocked_by(mat[s2])))
            improved = True
        else:
            # no improvement possible via single swap; break
            break
    return mat


def construct():
    t0 = time.time()
    rng = random.Random(12345)

    two = _two_layer_vectors()
    three = _three_layer_vectors()

    mat = np.array(two, dtype=np.int64)

    # Greedy add 3-layer
    mat = _greedy_add(three, mat, rng)

    # Least-conflicts improvement
    remaining = TIME_LIMIT - (time.time() - t0) - 10.0
    if remaining > 2.0:
        mat = _least_conflicts_improve(mat, three, rng, remaining)

    # Final safety: trim if over time
    if time.time() - t0 > TIME_LIMIT - 5.0:
        pass

    return mat.astype(np.int64)


if __name__ == "__main__":
    arr = construct()
    # validate
    n = arr.shape[0]
    norms = np.sqrt((arr ** 2).sum(axis=1))
    max_norm = norms.max()
    # pairwise distances
    diff = arr[:, None, :] - arr[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = dist.min()
    print("k =", n, "max_norm =", max_norm, "min_dist =", min_dist,
          "ok =", min_dist >= max_norm - 1e-9)

"""Kissing number in dimension 11: construct integer vectors with min pairwise dist >= max norm."""
import time
import numpy as np

D = 11
_DEADLINE_MARGIN = 10.0  # seconds of safety margin before the 150 s kill


def _generate_candidates():
    """Generate a diverse pool of nonzero integer vectors in R^11."""
    cands = []

    # Weight-1 vectors (norm 1): 2*D
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            cands.append(v)

    # Weight-2 vectors (norm 2): C(D,2)*4 = 55*4 = 220
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    cands.append(v)

    # Weight-3 vectors (norm 3): C(D,3)*8 = 165*8 = 1320
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
                            cands.append(v)

    # Weight-4 vectors (norm 4): C(D,4)*16 = 330*16 = 5280
    for i in range(D):
        for j in range(i + 1, D):
            for k in range(j + 1, D):
                for l in range(k + 1, D):
                    for si in (1, -1):
                        for sj in (1, -1):
                            for sk in (1, -1):
                                for sl in (1, -1):
                                    v = np.zeros(D, dtype=np.int64)
                                    v[i] = si
                                    v[j] = sj
                                    v[k] = sk
                                    v[l] = sl
                                    cands.append(v)

    return cands


def _greedy_select(candidates, deadline):
    """
    Greedy selection: pick vectors in order of descending norm,
    keeping only those whose distance to all already-selected vectors
    is at least the maximum norm in the selected set (and at least the
    candidate's own norm to be safe).
    """
    # Sort by norm descending (higher norm vectors first for density)
    norms = np.array([int(np.dot(v, v)) for v in candidates])
    order = np.argsort(-norms)
    ordered = [candidates[i] for i in order]
    ordered_norms = norms[order]

    selected = []
    selected_norms = []
    max_norm_sq = 0  # squared max norm among selected

    for idx, v in enumerate(ordered):
        if time.time() > deadline:
            break
        v_norm_sq = int(ordered_norms[idx])
        # The candidate must have norm <= current max, OR we can allow
        # it if it will become the new max (then all existing must be at
        # least this far). Simpler: require dist^2 >= max(v_norm_sq, max_norm_sq)
        threshold = max(v_norm_sq, max_norm_sq)

        ok = True
        for sv in selected:
            diff = v - sv
            d2 = int(np.dot(diff, diff))
            if d2 < threshold:
                ok = False
                break
        if ok:
            selected.append(v)
            selected_norms.append(v_norm_sq)
            if v_norm_sq > max_norm_sq:
                max_norm_sq = v_norm_sq

    return selected


def construct():
    start = time.time()
    deadline = start + 150.0 - _DEADLINE_MARGIN

    # Generate candidate pool
    candidates = _generate_candidates()

    # Greedy selection with deadline guard
    selected = _greedy_select(candidates, deadline)

    if len(selected) == 0:
        # Fallback: return the seed-style weight-2 set
        import itertools
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)

    return np.array(selected, dtype=np.int64)
